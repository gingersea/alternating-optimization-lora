"""
DeepSpeed integration engine for the AltOpt trainer.

Provides ZeRO-optimized training with configurable stages:
  - ZeRO-1: Optimizer state partitioning across GPUs (reduces optimizer memory)
  - ZeRO-2: + Gradient partitioning (further reduction)
  - ZeRO-3: + Parameter partitioning (near-linear scaling with GPU count)

For 2× RTX 5090 (32GB each, 64GB total), ZeRO-2 is the sweet spot for
7B models — it offloads optimizer states and gradients while keeping
parameters local, avoiding the communication overhead of ZeRO-3.

DeepSpeed ZeRO Memory Breakdown (Llama-2-7B, bf16):
  ┌─────────────┬──────────┬──────────┬──────────┬──────────┐
  │             │ No ZeRO  │ ZeRO-1   │ ZeRO-2   │ ZeRO-3   │
  ├─────────────┼──────────┼──────────┼──────────┼──────────┤
  │ Weights     │ 14 GB    │ 14 GB    │ 14 GB    │  7 GB    │
  │ Gradients   │ 14 GB    │ 14 GB    │  7 GB    │  0.7 GB  │
  │ Opt States  │ 28 GB    │ 14 GB    │ 14 GB    │  1.4 GB  │
  │ Activations │  8 GB    │  8 GB    │  8 GB    │  8 GB    │
  │ **Total**   │**64 GB** │**50 GB** │**43 GB** │**17 GB** │
  └─────────────┴──────────┴──────────┴──────────┴──────────┘

Key design decisions:
  - ZeRO-2 as default: best tradeoff between memory and speed for 7B on 2 GPUs
  - bf16 mixed precision: 2× speedup vs fp32, native on RTX 5090
  - Activation checkpointing: further 40% memory reduction
  - Gradient accumulation: enables larger effective batch sizes
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

_DEEPSPEED_AVAILABLE = False
try:
    import deepspeed
    from deepspeed.runtime.zero.stage_1_and_2 import (
        estimate_zero2_model_states_mem_needs_all_live,
    )
    from deepspeed.runtime.zero.stage3 import (
        estimate_zero3_model_states_mem_needs_all_live,
    )
    _DEEPSPEED_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class DeepSpeedConfig:
    """
    DeepSpeed configuration for the AltOpt trainer.

    Mirrors the relevant subset of DeepSpeed's JSON config format
    with Pythonic defaults suitable for 7B model post-training.
    """

    # ── ZeRO Optimization ──
    zero_stage: int = 2
    """ZeRO stage: 0=disabled, 1=optimizer state, 2=+gradients, 3=+parameters."""

    offload_optimizer: bool = False
    """Offload optimizer states to CPU (ZeRO-2/3). Reduces GPU memory at cost of speed.

    Kept false by default — let the trainer override when needed (AdamW state needs it;
    SGD state is light enough to stay on GPU).
    """

    offload_param: bool = False
    """Offload parameters to CPU (ZeRO-3). Extreme memory savings, slow."""

    # ── Mixed Precision ──
    fp16_enabled: bool = False
    """Enable fp16 mixed precision (for legacy GPUs without bf16)."""

    bf16_enabled: bool = True
    """Enable bf16 mixed precision (recommended for A100/H100/RTX 5090)."""

    # ── Gradient Management ──
    train_micro_batch_size_per_gpu: int = 2
    """Per-GPU micro batch size (must be integer, DeepSpeed 0.19+ rejects 'auto')."""

    gradient_accumulation_steps: int = 1
    """Accumulate gradients over N steps before optimizer step. Simulates larger batch."""

    gradient_clipping: float = 1.0
    """Max gradient norm for clipping."""

    # ── Activation Checkpointing ──
    activation_checkpointing: bool = True
    """Enable DeepSpeed's activation checkpointing (partition activations across GPUs)."""

    partition_activations: bool = False
    """ZeRO-2 only: partition activations across GPUs for extra memory savings."""

    # ── Communication ──
    communication_data_type: str = "bfp16"
    """Data type for all-reduce communication. bfp16 is faster than fp32."""

    stage3_prefetch_bucket_size: int = 5e7
    """ZeRO-3: parameter prefetch bucket size (bytes)."""

    stage3_param_persistence_threshold: int = 1e6
    """ZeRO-3: params smaller than this (in elements) stay on all GPUs."""

    # ── Logging ──
    wall_clock_breakdown: bool = False
    """Detailed timing breakdown for each training step."""

    def to_dict(self) -> dict:
        """Convert to DeepSpeed JSON config dictionary.

        DeepSpeed 0.19+ does not accept 'auto' for batch sizes — must be integers.
        """
        cfg: dict[str, Any] = {
            "train_micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.gradient_clipping,
            "wall_clock_breakdown": self.wall_clock_breakdown,
            "zero_allow_untested_optimizer": True,
            "zero_force_ds_cpu_optimizer": False,
        }

        # Mixed precision
        if self.bf16_enabled:
            cfg["bf16"] = {"enabled": True}
        elif self.fp16_enabled:
            cfg["fp16"] = {
                "enabled": True,
                "loss_scale": 0,  # Dynamic loss scaling
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            }

        # ZeRO stage
        if self.zero_stage == 0:
            pass  # No ZeRO config needed
        elif self.zero_stage == 1:
            cfg["zero_optimization"] = {
                "stage": 1,
            }
        elif self.zero_stage == 2:
            cfg["zero_optimization"] = {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu" if self.offload_optimizer else "none",
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
            }
            if self.partition_activations:
                cfg["activation_checkpointing"] = {
                    "partition_activations": True,
                    "cpu_checkpointing": False,
                }
        elif self.zero_stage == 3:
            cfg["zero_optimization"] = {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu" if self.offload_optimizer else "none",
                },
                "offload_param": {
                    "device": "cpu" if self.offload_param else "none",
                },
                "stage3_prefetch_bucket_size": self.stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": self.stage3_param_persistence_threshold,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 2e8,
            }

        return cfg

    def save(self, path: str) -> None:
        """Save DeepSpeed config to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("DeepSpeed config saved to %s", path)


class DeepSpeedEngine:
    """
    Wraps a HuggingFace model with DeepSpeed for memory-efficient training.

    Handles the complexity of initializing DeepSpeed with HuggingFace models,
    managing the engine lifecycle, and providing hooks for the AltOpt trainer
    to integrate seamlessly.

    Usage:
        engine = DeepSpeedEngine(model, ds_config)
        engine.initialize(dataloader)
        for batch in dataloader:
            loss = engine.step(batch)
    """

    def __init__(
        self,
        model: nn.Module,
        config: DeepSpeedConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ):
        if not _DEEPSPEED_AVAILABLE:
            raise ImportError(
                "DeepSpeed is required. Install: pip install deepspeed"
            )

        self.model = model
        self.config = config
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._engine = None
        self._initialized = False

    @property
    def engine(self):
        """The underlying DeepSpeed engine (deepspeed.DeepSpeedEngine)."""
        if self._engine is None:
            raise RuntimeError("DeepSpeedEngine not initialized. Call initialize() first.")
        return self._engine

    def initialize(
        self,
        dataloader=None,
        model_parameters=None,
        training_data=None,
        config_params: Optional[dict] = None,
    ):
        """
        Initialize the DeepSpeed engine.

        Args:
            dataloader: PyTorch DataLoader for training (used to infer batch sizes).
            model_parameters: Optional explicit parameter filter.
            training_data: Optional dataset (for auto batch size inference).
            config_params: Optional overrides for the DeepSpeed config dict.
        """
        # Single-process multi-GPU via DeepSpeed ZeRO: WORLD_SIZE tells DeepSpeed
        # how many GPUs to shard across in a single process. Requires mpi4py +
        # OpenMPI for NCCL collective ops. GPU count auto-detected via CUDA_VISIBLE.
        gpu_count = torch.cuda.device_count()
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", str(gpu_count))

        # Move model to CPU so DeepSpeed can manage device placement
        self.model = self.model.cpu()

        ds_config = self.config.to_dict()

        # Apply overrides
        if config_params:
            ds_config.update(config_params)

        params = model_parameters or filter(
            lambda p: p.requires_grad, self.model.parameters()
        )

        self._engine, self._optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=params,
            optimizer=self._optimizer,
            lr_scheduler=self._lr_scheduler,
            config_params=ds_config,
            training_data=training_data,
        )

        self._initialized = True
        logger.info(
            "DeepSpeed initialized: ZeRO-%d, bf16=%s, fp16=%s",
            self.config.zero_stage,
            self.config.bf16_enabled,
            self.config.fp16_enabled,
        )

        return self._engine

    def step(self, batch: dict[str, torch.Tensor]) -> float:
        """
        Execute one training step through DeepSpeed.

        DeepSpeed handles:
          - Forward pass with mixed precision (bf16/fp16)
          - Backward pass with scaled loss
          - Gradient all-reduce across GPUs
          - Optimizer step with ZeRO partitioning
          - Gradient clipping
          - Learning rate scheduling

        Args:
            batch: Model inputs (input_ids, attention_mask, labels, etc.)

        Returns:
            loss value as Python float.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before step()")

        device = self._engine.device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        outputs = self._engine(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        self._engine.backward(loss)
        self._engine.step()

        return loss.item() if isinstance(loss, torch.Tensor) else loss

    def forward_only(self, batch: dict[str, torch.Tensor]):
        """Forward pass without backward (for evaluation)."""
        if not self._initialized:
            raise RuntimeError("Call initialize() before forward_only()")
        device = self._engine.device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.no_grad():
            return self._engine(**batch)

    def save_checkpoint(self, save_dir: str, tag: Optional[str] = None):
        """
        Save DeepSpeed checkpoint (model + optimizer + scheduler states).
        """
        self._engine.save_checkpoint(save_dir, tag=tag)
        logger.info("DeepSpeed checkpoint saved to %s (tag=%s)", save_dir, tag)

    def load_checkpoint(
        self, load_dir: str, tag: Optional[str] = None, load_optimizer_states: bool = True
    ):
        """
        Load DeepSpeed checkpoint.
        """
        _, client_state = self._engine.load_checkpoint(
            load_dir, tag=tag, load_optimizer_states=load_optimizer_states
        )
        logger.info("DeepSpeed checkpoint loaded from %s", load_dir)
        return client_state

    @staticmethod
    def estimate_memory(
        model: nn.Module,
        zero_stage: int = 2,
        num_gpus: int = 2,
    ) -> dict:
        """
        Estimate memory consumption with DeepSpeed ZeRO.

        Uses DeepSpeed's built-in memory estimator for accurate predictions.

        Args:
            model: The model to estimate for.
            zero_stage: ZeRO stage (1, 2, or 3).
            num_gpus: Number of GPUs available.

        Returns:
            dict with per-GPU memory breakdown in GB.
        """
        if not _DEEPSPEED_AVAILABLE:
            return {"error": "DeepSpeed not available for estimation"}

        if zero_stage == 3:
            result = estimate_zero3_model_states_mem_needs_all_live(
                model, num_gpus=num_gpus, num_nodes=1
            )
        else:
            result = estimate_zero2_model_states_mem_needs_all_live(
                model, num_gpus=num_gpus, num_nodes=1
            )

        # Convert to GB
        memory_breakdown = {}
        for key, value in result.items():
            if isinstance(value, (int, float)):
                memory_breakdown[key] = round(value / (1024 ** 3), 2)

        return memory_breakdown

    @property
    def global_steps(self) -> int:
        """Number of optimizer steps taken."""
        if self._engine is not None:
            return self._engine.global_steps
        return 0

    @property
    def global_samples(self) -> int:
        """Number of samples processed."""
        if self._engine is not None:
            return self._engine.global_samples
        return 0


def estimate_zero_memory(
    model: nn.Module,
    zero_stage: int = 2,
    num_gpus: int = 2,
    batch_size: int = 4,
    seq_length: int = 2048,
) -> dict[str, float]:
    """
    Estimate total training memory with DeepSpeed ZeRO.

    Combines DeepSpeed's model state estimator with activation memory
    heuristics to provide a realistic total.

    Args:
        model: The model.
        zero_stage: ZeRO stage (1, 2, 3).
        num_gpus: Number of GPUs.
        batch_size: Per-GPU batch size.
        seq_length: Sequence length.

    Returns:
        dict with detailed memory breakdown.
    """
    from .model_utils import estimate_training_memory_gb

    # Model + optimizer states estimate
    model_breakdown = estimate_training_memory_gb(
        model, batch_size, seq_length, optimizer_type="adamw",
        use_gradient_checkpointing=True,
    )

    # DeepSpeed ZeRO savings
    if zero_stage == 0:
        zero_saving_factor = 1.0
    elif zero_stage == 1:
        zero_saving_factor = 0.6  # Optimizer state partitioned
    elif zero_stage == 2:
        zero_saving_factor = 0.45  # + Gradients partitioned
    elif zero_stage == 3:
        zero_saving_factor = 0.25  # + Parameters partitioned

    total = model_breakdown["total_estimated_gb"] * zero_saving_factor

    return {
        **model_breakdown,
        "zero_stage": zero_stage,
        "num_gpus": num_gpus,
        "zero_memory_factor": zero_saving_factor,
        "estimated_with_zero_gb": round(total, 2),
        "per_gpu_gb": round(total, 2),
    }
