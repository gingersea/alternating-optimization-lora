"""
LoRA (Low-Rank Adaptation) baseline implementation.

Provides a clean, self-contained LoRA implementation for fair comparison
with the alternating optimization framework. Compatible with HuggingFace
PEFT conventions but implemented independently for full control over
the comparison protocol.

Key interfaces:
  - LoRAConfig: specifies rank r, alpha, target modules, dropout
  - LoRABaseline: wraps a model with LoRA adapters and AdamW optimizer
  - Supports both standard AdamW and custom optimizer injection
    (critical for Protocol C: LoRA-structured AltOpt)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    r: int = 8                          # Rank
    alpha: float = 16.0                 # Scaling factor
    dropout: float = 0.0                # Dropout probability
    target_modules: list[str] = field(  # Which modules to adapt
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: str = "none"                  # Bias handling: "none", "all", "lora_only"

    @property
    def scaling(self) -> float:
        """LoRA scaling factor: alpha / r."""
        return self.alpha / self.r


class LoRALayer(nn.Module):
    """
    Low-rank adaptation applied to a single linear layer.

    W'x = Wx + (alpha/r) * B A x

    where A ∈ ℝ^{r × d_in}, B ∈ ℝ^{d_out × r}, initialized:
      - A ~ Kaiming uniform
      - B ~ zeros
    """

    def __init__(self, base_layer: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.base_layer = base_layer
        self.config = config

        d_in = base_layer.in_features
        d_out = base_layer.out_features
        r = config.r

        # Freeze base weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # LoRA parameters (created on same device and dtype as base layer)
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, d_in, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(d_out, r, device=device, dtype=dtype))
        self.scaling = config.scaling

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize A with Kaiming, B with zeros."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Wx + (alpha/r) * B A x"""
        base_out = self.base_layer(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling


import math  # noqa: E402 (import needed for reset_parameters)


class LoRABaseline:
    """
    LoRA baseline for fair comparison.

    Wraps a pretrained model with LoRA adapters on specified modules,
    provides an AdamW optimizer (or custom optimizer), and exposes
    the same step() interface as AltOptFramework.

    Supports custom optimizer injection for Protocol C:
      - Standard: LoRA + AdamW (Protocol D)
      - Custom:   LoRA + AltOpt optimizer (Protocol C)
    """

    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-4,
    ):
        self.model = model
        self.lora_config = lora_config
        self.lr = lr

        # Apply LoRA adapters
        self._lora_modules: dict[str, LoRALayer] = {}
        self._apply_lora()

        # Set up optimizer
        if optimizer is not None:
            self._optimizer = optimizer
        else:
            self._optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
            )

        self.losses: list[float] = []
        self.grad_norms: list[float] = []

    def _apply_lora(self) -> None:
        """Replace target linear layers with LoRA-augmented versions."""
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Check if this module matches target patterns
            base_name = name.split(".")[-1]
            if base_name not in self.lora_config.target_modules:
                continue

            # Wrap with LoRA
            lora_layer = LoRALayer(module, self.lora_config)

            # Replace in parent
            parent = self.model
            *path, leaf = name.split(".")
            for part in path:
                parent = getattr(parent, part)
            setattr(parent, leaf, lora_layer)

            self._lora_modules[name] = lora_layer

        n_adapted = len(self._lora_modules)
        n_params = self.num_trainable_params()
        logger.info(
            "LoRA applied to %d modules, %d trainable parameters (r=%d)",
            n_adapted, n_params, self.lora_config.r
        )

    def step(self, batch: dict[str, torch.Tensor]) -> float:
        """One training step: forward, backward, optimize."""
        self._optimizer.zero_grad()

        device = next(self.model.parameters()).device
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        outputs = self.model(**batch_on_device)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            max_norm=1.0,
        )

        # Track grad norm
        grad_norm = self._compute_grad_norm()
        self.grad_norms.append(grad_norm)

        self._optimizer.step()

        loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
        self.losses.append(loss_val)
        return loss_val

    def _compute_grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    def num_trainable_params(self) -> int:
        """Number of trainable parameters (LoRA adapters only)."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_lora_params(self) -> dict[str, torch.Tensor]:
        """Extract only LoRA parameters."""
        return {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def merge_and_unload(self) -> nn.Module:
        """
        Merge LoRA weights into base weights and remove adapters.

        W_merged = W + (alpha/r) * B @ A

        Returns the model with LoRA adapter modules removed.
        """
        for name, lora_module in self._lora_modules.items():
            base = lora_module.base_layer
            delta = (lora_module.lora_B @ lora_module.lora_A) * lora_module.scaling
            base.weight.data += delta

            # Replace LoRALayer with original Linear
            parent = self.model
            *path, leaf = name.split(".")
            for part in path:
                parent = getattr(parent, part)
            setattr(parent, leaf, base)

        self._lora_modules.clear()
        return self.model
