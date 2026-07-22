"""
Stochastic Gradient Descent phase optimizer.

Provides fine-grained, per-sample (or micro-batch) gradient-based updates
that capture cross-block interactions missed by the ALS block-wise solver.

Key differences from standard optimizers:
  - Operates on the SAME parameters as ALS (full-rank by default)
  - Tracks gradient norm for convergence monitoring
  - Supports micro-batch accumulation for memory-constrained settings
  - Designed to alternate with ALS and Perturbation phases
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SGDPhaseOptimizer:
    """
    SGD optimizer for the SGD phase of alternating optimization.

    Thin wrapper around torch.optim.SGD with additional tracking for
    gradient norms and seamless alternation with ALS blocks.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
        micro_batch_size: Optional[int] = None,
    ):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.micro_batch_size = micro_batch_size

        self.last_grad_norm: float = 0.0

        # Create optimizer over trainable parameters
        # foreach=False avoids multi-tensor kernel buffer allocations on constrained GPUs
        self._optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            foreach=False,
        )

    def set_lr(self, lr: float) -> None:
        """Update learning rate (e.g., between cycles)."""
        self.lr = lr
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def step(self, batch: dict[str, torch.Tensor]) -> float:
        """
        Perform one SGD step.

        Args:
            batch: dict with model inputs (passed directly to model.forward)

        Returns:
            loss: float scalar loss
        """
        self._optimizer.zero_grad()

        # Forward pass
        device = next(self.model.parameters()).device
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        outputs = self.model(**batch_on_device)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        # Backward
        loss.backward()

        # Track gradient norm
        self.last_grad_norm = self._compute_grad_norm()

        # Optional gradient clipping
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            max_norm=1.0,
        )

        self._optimizer.step()

        return loss.item() if isinstance(loss, torch.Tensor) else loss

    def _compute_grad_norm(self) -> float:
        """Compute total gradient L2 norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        return total_norm ** 0.5

    def state_dict(self) -> dict:
        """Get optimizer state for checkpointing."""
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state from checkpoint."""
        self._optimizer.load_state_dict(state_dict)


class LARSPhaseOptimizer(SGDPhaseOptimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) for the SGD phase.

    Standard SGD:  θ -= lr × g         (same lr for all layers)
    LARS:          θ -= η × ‖θ‖/‖g‖ × g  (per-layer lr)

    This is specifically designed to counteract the residual amplification
    problem in Protocol A: when ALS modifies lm_head on a deep model,
    gradients at shallow layers are amplified ~8.7× (28-layer residual chain).
    Standard SGD with global gradient clipping dilutes shallow-layer updates.
    LARS scales each layer's learning rate by its own weight-to-gradient ratio,
    so shallow layers get proportionally larger updates despite gradient clipping.

    Args:
        model: nn.Module
        lr: base learning rate (η in the LARS formula)
        momentum: momentum coefficient
        weight_decay: weight decay coefficient (λ)
        trust_coefficient: max allowed ‖θ‖/‖g‖ ratio (prevents explosion)
        epsilon: numerical stability
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
        trust_coefficient: float = 0.001,
        epsilon: float = 1e-8,
    ):
        self.trust_coefficient = trust_coefficient
        self.epsilon = epsilon
        super().__init__(model, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def step(self, batch: dict[str, torch.Tensor]) -> float:
        self._optimizer.zero_grad()

        device = next(self.model.parameters()).device
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        outputs = self.model(**batch_on_device)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        loss.backward()

        self.last_grad_norm = self._compute_grad_norm()

        # No global gradient clipping — LARS's ‖θ‖/‖g‖ scaling IS the per-layer clip.
        # Global clipping can shrink ‖g‖ → 0 for some layers, making
        # trust_coefficient * ‖θ‖ / ‖g‖ → ∞ and exploding updates.

        # LARS: per-layer adaptive learning rate with max_local_lr cap
        max_local_lr = self.lr * 10.0  # update never exceeds 10× base lr

        for group in self._optimizer.param_groups:
            lr = group["lr"]
            mom = group.get("momentum", 0)
            wd = group.get("weight_decay", 0)

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data.clone()
                if wd != 0:
                    d_p.add_(p.data, alpha=wd)

                param_norm = p.data.norm(2)
                grad_norm = d_p.norm(2)
                denominator = grad_norm + wd * param_norm + self.epsilon

                if param_norm > 0 and grad_norm > 0 and denominator > 0:
                    local_lr = lr * self.trust_coefficient * param_norm / denominator
                    local_lr = min(local_lr, max_local_lr)
                else:
                    local_lr = lr

                if mom > 0:
                    state = self._optimizer.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.data)
                    buf = state["momentum_buffer"]
                    buf.mul_(mom).add_(d_p, alpha=local_lr)
                    p.data.add_(buf, alpha=-1)
                else:
                    p.data.add_(d_p, alpha=-local_lr)

        return loss.item() if isinstance(loss, torch.Tensor) else loss
