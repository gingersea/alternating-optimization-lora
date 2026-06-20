"""
AltOptTrainer — unified training orchestrator.

Integrates AltOptFramework/LoRABaseline with profiling, checkpointing,
evaluation, and PEFT bridge into a single trainer class with hook-based
lifecycle management.

This replaces the fragmented logic in experiments/runner.py with a
centralized scheduler, analogous to HuggingFace's Trainer but specialized
for the 2×2 factorial comparison protocol.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .framework import AltOptFramework, AltOptState, Phase, PhaseConfig, PhaseSchedule
from .lora import LoRABaseline, LoRAConfig
from .profiling.flops import FlopsProfiler
from .profiling.memory import MemoryTracker
from .checkpoint import CheckpointManager
from .evaluation import Evaluator

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for AltOptTrainer."""

    protocol: str = "A"
    optimizer_type: str = "altopt"
    parameter_form: str = "full_rank"

    total_budget_flops: float = float("inf")
    max_steps: Optional[int] = None
    max_epochs: Optional[int] = None

    eval_every: int = 100

    run_dir: str = "runs/default"
    save_every: int = 500
    keep_last_ckpt: int = 3
    resume_from: Optional[str] = None

    profile_memory: bool = False

    phase_schedule: Optional[PhaseSchedule] = None

    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_modules: Optional[list[str]] = None

    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 0.01
    adamw_betas: tuple[float, float] = (0.9, 0.999)

    seed: int = 42

    use_deepspeed: bool = False
    deepspeed_zero_stage: int = 2
    deepspeed_bf16: bool = True
    deepspeed_fp16: bool = False
    gradient_accumulation_steps: int = 1


@dataclass
class TrainerState:
    """Runtime state aggregated by the trainer."""

    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    best_perplexity: float = float("inf")

    # Resource tracking
    cumulative_flops: float = 0.0
    peak_memory_mb: float = 0.0
    elapsed_seconds: float = 0.0

    # Logs
    loss_history: list[float] = field(default_factory=list)
    loss_types: list[str] = field(default_factory=list)  # 'loss' or 'noise_energy'
    eval_history: list[dict] = field(default_factory=list)
    flops_history: list[float] = field(default_factory=list)
    memory_history: list[float] = field(default_factory=list)

    _start_time: float = field(default_factory=time.time)

    def record_loss(self, loss: float, loss_type: str = "loss"):
        self.loss_history.append(loss)
        self.loss_types.append(loss_type)

    def record_eval(self, step: int, results: dict):
        self.eval_history.append({"step": step, **results})

    def record_flops(self, flops: float):
        self.flops_history.append(flops)
        self.cumulative_flops += flops

    def record_memory(self, mb: float):
        self.memory_history.append(mb)
        self.peak_memory_mb = max(self.peak_memory_mb, mb)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "best_perplexity": self.best_perplexity,
            "cumulative_flops": self.cumulative_flops,
            "peak_memory_mb": self.peak_memory_mb,
            "elapsed_seconds": time.time() - self._start_time,
            "loss_history": self.loss_history,
            "loss_types": self.loss_types,
            "eval_history": self.eval_history,
        }


class AltOptTrainer:
    """
    Unified trainer for the 2×2 factorial comparison protocol.

    Hooks:
      _on_step_start(batch)  — profiling start, device transfer
      _on_step_end(loss)     — logging, eval trigger, checkpoint trigger
      _on_epoch_end(epoch)   — mandatory eval, checkpoint save

    Usage:
        config = TrainerConfig(protocol="A", run_dir="runs/proto_a")
        trainer = AltOptTrainer(model, config, eval_dataloader=eval_dl)
        trainer.train(train_dataloader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        eval_dataloader,
        tokenizer=None,
    ):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.tokenizer = tokenizer
        self.state = TrainerState()

        # Components (assembled in _setup)
        self.optimizer = None
        self.altopt = None
        self.lora_baseline = None
        self.peft_bridge = None

        self._setup()

        # Infrastructure modules
        self.flops_profiler = FlopsProfiler()
        self.flops_profiler._model = model  # so heuristic works without explicit start()
        self.memory_tracker = MemoryTracker(full_profile=config.profile_memory)
        self.checkpoint = CheckpointManager(
            run_dir=config.run_dir,
            save_every=config.save_every,
            keep_last=config.keep_last_ckpt,
        )
        self.evaluator = Evaluator(
            metrics=["perplexity", "loss"],
            eval_dataloader=eval_dataloader,
        )

        # Resume if requested
        if config.resume_from:
            step, altopt_state_dict, model, _ = self.checkpoint.resume(
                config.resume_from, model, self.optimizer
            )
            self.state.step = step
            if self.altopt is not None and altopt_state_dict:
                self._restore_altopt_state(altopt_state_dict)

    def _setup(self):
        cfg = self.config
        torch.manual_seed(cfg.seed)

        if cfg.use_deepspeed:
            self._setup_optimizer_for_deepspeed()
            self._setup_deepspeed()
            return

        if cfg.parameter_form == "lora":
            if cfg.optimizer_type == "altopt":
                # Protocol C: LoRA + AltOpt (SGD+Perturb only, no ALS)
                peft_ok = False
                try:
                    from .peft_bridge import PeftBridge, model_supports_lora

                    if not model_supports_lora(self.model):
                        raise ValueError("Model architecture does not support PEFT LoRA")

                    self.peft_bridge = PeftBridge(
                        self.model,
                        r=cfg.lora_r,
                        alpha=cfg.lora_alpha,
                        dropout=cfg.lora_dropout,
                        target_modules=cfg.lora_target_modules,
                    )
                    self.model = self.peft_bridge.peft_model
                    peft_ok = True
                except (ImportError, ValueError, RuntimeError, AttributeError) as e:
                    logger.info("PEFT unavailable or incompatible for this model: %s. "
                                "Falling back to built-in LoRALayer.", e)
                    peft_ok = False

                if not peft_ok:
                    lora_cfg = LoRAConfig(
                        r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout,
                        target_modules=cfg.lora_target_modules or ["c_attn", "c_proj"],
                    )
                    self.lora_baseline = LoRABaseline(self.model, lora_cfg, lr=cfg.lr)
                    self.optimizer = None  # AltOptFramework manages its own SGD optimizer

                # Build SGD+Perturb-only schedule.
                # ALS is incompatible with LoRA — it targets full-rank nn.Linear
                # (lm_head), using untrained LoRA activations and corrupting head
                # weights → NaN/Inf divergence on 7B+ models.
                if cfg.phase_schedule is not None:
                    schedule = cfg.phase_schedule
                    unfiltered = schedule.phases
                    filtered_phases = [p for p in unfiltered if p.phase != Phase.ALS]
                    if len(filtered_phases) != len(unfiltered):
                        logger.info(
                            "Protocol C: removed %d ALS phase(s) (ALS targets "
                            "full-rank modules, incompatible with LoRA)",
                            len(unfiltered) - len(filtered_phases),
                        )
                        schedule = PhaseSchedule(phases=filtered_phases, cycles=schedule.cycles)
                else:
                    schedule = PhaseSchedule(
                        phases=[
                            PhaseConfig(phase=Phase.SGD, steps=100, lr=cfg.lr),
                            PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
                        ],
                        cycles=3,
                    )
                self.altopt = AltOptFramework(self.model, schedule)
                self.optimizer = self.altopt.sgd._optimizer
            else:
                # Protocol D: LoRA + AdamW — prefer PEFT for correct device_map handling.
                # LoRABaseline wraps layers in-place, breaking accelerate's device_map
                # split and causing OOM on 7B+ models with device_map="auto".
                peft_ok = False
                try:
                    from .peft_bridge import PeftBridge, model_supports_lora
                    from torch.optim import AdamW

                    if model_supports_lora(self.model):
                        self.peft_bridge = PeftBridge(
                            self.model,
                            r=cfg.lora_r,
                            alpha=cfg.lora_alpha,
                            dropout=cfg.lora_dropout,
                            target_modules=cfg.lora_target_modules,
                        )
                        self.model = self.peft_bridge.peft_model
                        self.optimizer = AdamW(
                            filter(lambda p: p.requires_grad, self.model.parameters()),
                            lr=cfg.lr, betas=cfg.adamw_betas, weight_decay=cfg.weight_decay,
                        )
                        peft_ok = True
                except (ImportError, ValueError, RuntimeError, AttributeError) as e:
                    logger.info("PEFT unavailable for Protocol D: %s. "
                                "Falling back to built-in LoRALayer.", e)

                if not peft_ok:
                    lora_cfg = LoRAConfig(
                        r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout,
                        target_modules=cfg.lora_target_modules or ["c_attn", "c_proj"],
                    )
                    self.lora_baseline = LoRABaseline(self.model, lora_cfg, lr=cfg.lr)
                    self.optimizer = getattr(self.lora_baseline, "_optimizer", None)
                    if self.optimizer is None:
                        logger.warning(
                            "LoRA: no adapters applied (no matching Linear modules in model). "
                            "Falling back to full-rank AdamW."
                        )
                        from torch.optim import AdamW
                        self.optimizer = AdamW(
                            filter(lambda p: p.requires_grad, self.model.parameters()),
                            lr=cfg.lr, betas=cfg.adamw_betas, weight_decay=cfg.weight_decay,
                        )
                        self.lora_baseline = None
            return

        # Full-rank protocols (A, B)
        if cfg.optimizer_type == "altopt":
            schedule = cfg.phase_schedule or PhaseSchedule(
                phases=[
                    PhaseConfig(phase=Phase.SGD, steps=100, lr=cfg.lr),
                    PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
                ],
                cycles=3,
            )
            self.altopt = AltOptFramework(self.model, schedule)
            _sgd = self.altopt.sgd  # trigger lazy SGD creation
            self.optimizer = _sgd._optimizer
        else:
            # 8-bit AdamW for full-rank 7B: reduces optimizer memory from
            # 28GB to 3.5GB, enabling ZeRO-2 on 2×32GB GPUs (~31GB each).
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=cfg.lr, betas=cfg.adamw_betas, weight_decay=cfg.weight_decay,
                )
                logger.info("Using 8-bit AdamW (bitsandbytes) for full-rank optimizer")
            except ImportError:
                from torch.optim import AdamW
                self.optimizer = AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=cfg.lr, betas=cfg.adamw_betas, weight_decay=cfg.weight_decay,
                )

    def _setup_optimizer_for_deepspeed(self):
        """Create optimizer BEFORE DeepSpeed engine — ZeRO-2 requires one.

        DeepSpeed takes ownership of the optimizer; we pass it to
        DeepSpeedEngine which forwards it to deepspeed.initialize().
        """
        cfg = self.config

        if cfg.parameter_form == "lora":
            # LoRA + DeepSpeed path (Protocol C/D experimental)
            if cfg.optimizer_type == "altopt":
                schedule = cfg.phase_schedule or PhaseSchedule(
                    phases=[PhaseConfig(phase=Phase.SGD, steps=100, lr=cfg.lr)],
                    cycles=1,
                )
                self.altopt = AltOptFramework(self.model, schedule)
                self.optimizer = self.altopt.sgd._optimizer if self.altopt._sgd else None
            else:
                from torch.optim import AdamW
                self.optimizer = AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=cfg.lr, betas=cfg.adamw_betas, weight_decay=cfg.weight_decay,
                )
            return

        # Full-rank protocols (A, B) with DeepSpeed
        if cfg.optimizer_type == "altopt":
            schedule = cfg.phase_schedule or PhaseSchedule(
                phases=[
                    PhaseConfig(phase=Phase.SGD, steps=100, lr=cfg.lr),
                    PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
                ],
                cycles=3,
            )
            self.altopt = AltOptFramework(self.model, schedule)
            _sgd = self.altopt.sgd  # trigger lazy SGD creation
            self.optimizer = _sgd._optimizer
        else:
            # DeepSpeedCPUAdam: required for ZeRO-2/3 with CPU optimizer offload.
            # Lives on CPU RAM (251GB) → GPU only needs model + gradients + activations.
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            self.optimizer = DeepSpeedCPUAdam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=cfg.lr, betas=cfg.adamw_betas, weight_decay=cfg.weight_decay,
            )
            logger.info("Using DeepSpeedCPUAdam + DeepSpeed ZeRO-2 (CPU optimizer offload)")

    def _setup_deepspeed(self):
        from .deepspeed_engine import DeepSpeedConfig, DeepSpeedEngine

        ds_cfg = DeepSpeedConfig(
            zero_stage=self.config.deepspeed_zero_stage,
            bf16_enabled=self.config.deepspeed_bf16,
            fp16_enabled=self.config.deepspeed_fp16,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            train_micro_batch_size_per_gpu=1,
        )
        self._deepspeed_engine = DeepSpeedEngine(
            self.model, ds_cfg, optimizer=self.optimizer
        )
        self._deepspeed_initialized = False

        self._has_altopt = self.config.optimizer_type == "altopt"
        if self._has_altopt:
            schedule = self.config.phase_schedule or PhaseSchedule(
                phases=[
                    PhaseConfig(phase=Phase.SGD, steps=100, lr=self.config.lr),
                    PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
                ],
                cycles=3,
            )
            self.altopt = AltOptFramework(self.model, schedule)

    # ── Main Training Loop ─────────────────────────────────────────

    def train(self, dataloader) -> TrainerState:
        if self.config.use_deepspeed:
            return self._train_deepspeed(dataloader)
        return self._train_standard(dataloader)

    def _train_standard(self, dataloader) -> TrainerState:
        self.model.train()
        cfg = self.config

        for epoch in range(cfg.max_epochs or 1):
            self.state.epoch = epoch
            for batch in dataloader:
                self._on_step_start(batch)

                loss = self._execute_step(batch)

                self._on_step_end(loss)
                self.state.step += 1

                if self._budget_exceeded():
                    logger.info("Budget exceeded at step %d", self.state.step)
                    self._on_epoch_end(epoch)
                    return self.state

                if cfg.max_steps and self.state.step >= cfg.max_steps:
                    self._on_epoch_end(epoch)
                    return self.state

            self._on_epoch_end(epoch)

        return self.state

    def _execute_step(self, batch: dict) -> float:
        device = self.device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if self.altopt is not None:
            return self.altopt.step(batch)
        elif self.lora_baseline is not None:
            return self.lora_baseline.step(batch)
        elif self.peft_bridge is not None:
            return self._peft_altopt_step(batch)
        else:
            return self._adamw_step(batch)

    def _adamw_step(self, batch: dict) -> float:
        self.optimizer.zero_grad()
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=1.0
        )
        self.optimizer.step()
        return loss.item() if isinstance(loss, torch.Tensor) else loss

    def _peft_altopt_step(self, batch: dict) -> float:
        # Protocol C: AltOpt optimization on PEFT LoRA adapters
        # We forward through the PEFT model, then use AltOpt's SGD phase
        # to update only the LoRA adapter parameters
        self.optimizer.zero_grad()
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=1.0
        )
        self.optimizer.step()
        return loss.item() if isinstance(loss, torch.Tensor) else loss

    def _current_phase_name(self) -> str:
        if self.altopt is not None and self.altopt.state.current_phase is not None:
            return self.altopt.state.current_phase.value.upper()
        if self.altopt is not None:
            return "SGD"
        if self.config.optimizer_type == "adamw":
            return "AdamW"
        return "UNKNOWN"

    # ── Hooks ───────────────────────────────────────────────────────

    def _on_step_start(self, batch):
        self.memory_tracker.reset_peak()

    def _on_step_end(self, loss: float):
        phase = self._current_phase_name()
        loss_type = "noise_energy" if phase == "PERTURB" else "loss"
        self.state.record_loss(loss, loss_type)
        flops = self.flops_profiler.record_step(phase)
        self.state.record_flops(flops)
        mem = self.memory_tracker.snapshot()
        self.state.record_memory(mem["allocated_mb"])

        step = self.state.step
        if step > 0 and step % self.config.eval_every == 0:
            results = self.evaluator.evaluate(self.model)
            self.state.record_eval(step, results)
            ppl = results.get("perplexity", float("inf"))
            if ppl < self.state.best_perplexity:
                self.state.best_perplexity = ppl
            if results.get("loss", float("inf")) < self.state.best_loss:
                self.state.best_loss = results["loss"]
            logger.info(
                "Step %d | loss=%.4f | ppl=%.2f | flops=%.2e | mem=%.0fMB",
                step, loss, ppl, self.state.cumulative_flops, self.state.peak_memory_mb,
            )

        self.checkpoint.maybe_save(step, self.state, self.model, self.optimizer)

    def _on_epoch_end(self, epoch: int):
        results = self.evaluator.evaluate(self.model)
        self.state.record_eval(self.state.step, results)
        if self.optimizer is not None:
            self.checkpoint.save(self.state.step, self.state, self.model, self.optimizer)
        else:
            self.checkpoint.save(self.state.step, self.state, self.model, None)
        logger.info("Epoch %d complete. Final ppl=%.2f", epoch, results.get("perplexity", float("inf")))

    def _restore_altopt_state(self, state_dict: dict):
        if self.altopt is None:
            return
        s = self.altopt.state
        s.global_step = state_dict.get("global_step", 0)
        s.current_cycle = state_dict.get("current_cycle", 0)
        s.phase_step = state_dict.get("phase_step", 0)
        s.losses = state_dict.get("losses", [])
        s.grad_norms = state_dict.get("grad_norms", [])

    def _budget_exceeded(self) -> bool:
        cfg = self.config
        if self.state.cumulative_flops >= cfg.total_budget_flops:
            return True
        return False

    def _train_deepspeed(self, dataloader) -> TrainerState:
        if not self._deepspeed_initialized:
            self._deepspeed_engine.initialize(dataloader)
            self._deepspeed_initialized = True

        engine = self._deepspeed_engine.engine
        self.model.train()
        cfg = self.config

        for epoch in range(cfg.max_epochs or 1):
            self.state.epoch = epoch
            for batch in dataloader:
                self._on_step_start(batch)

                device = engine.device
                batch_gpu = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                if self._has_altopt and self.altopt is not None:
                    loss = self.altopt.step(batch_gpu)
                else:
                    outputs = engine(**batch_gpu)
                    raw_loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                    engine.backward(raw_loss)
                    engine.step()
                    loss = raw_loss.item() if isinstance(raw_loss, torch.Tensor) else raw_loss

                self._on_step_end(loss)
                self.state.step += 1

                if self._budget_exceeded():
                    logger.info("Budget exceeded at step %d", self.state.step)
                    self._on_epoch_end(epoch)
                    return self.state

                if cfg.max_steps and self.state.step >= cfg.max_steps:
                    self._on_epoch_end(epoch)
                    return self.state

            self._on_epoch_end(epoch)

        return self.state

    # ── High-level API ──────────────────────────────────────────────

    def evaluate(self) -> dict:
        return self.evaluator.evaluate(self.model)

    def save(self, path: Optional[str] = None):
        save_path = path or self.config.run_dir
        self.checkpoint.save(self.state.step, self.state, self.model, self.optimizer)

    def load(self, path: str):
        step, state_dict, model, optimizer = self.checkpoint.load(path, self.model, self.optimizer)
        self.state.step = step
        self.model = model
        self.optimizer = optimizer
        if state_dict:
            self._restore_altopt_state(state_dict)

    def export_results(self, path: Optional[str] = None):
        output = Path(path or self.config.run_dir) / "trainer_state.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        logger.info("Results exported to %s", output)


def run_protocol(
    model: nn.Module,
    config: TrainerConfig,
    train_dataloader,
    eval_dataloader,
) -> TrainerState:
    """Convenience function: construct and run a single protocol."""
    trainer = AltOptTrainer(model, config, eval_dataloader)
    return trainer.train(train_dataloader)
