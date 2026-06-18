"""
Core alternating optimization framework orchestrator.

Coordinates ALS, SGD, and Perturbation phases according to a configurable
schedule. The framework is optimizer-agnostic at the phase level — each phase
can be swapped for alternative implementations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from .als import ALSBlockSolver
from .sgd import SGDPhaseOptimizer
from .perturbation import PerturbationScheduler

logger = logging.getLogger(__name__)


class Phase(Enum):
    """Identifies which phase the framework is currently executing."""
    ALS = "als"
    SGD = "sgd"
    PERTURB = "perturb"


@dataclass
class PhaseConfig:
    """Configuration for a single phase in the alternating schedule."""
    phase: Phase
    steps: int
    block_size: Optional[int] = None  # ALS: rows per block
    lr: Optional[float] = None        # SGD: learning rate
    noise_scale: Optional[float] = None  # Perturb: noise std


@dataclass
class PhaseSchedule:
    """Ordered sequence of phases to alternate through."""
    phases: list[PhaseConfig]
    cycles: int = 1  # How many times to repeat the full sequence

    @classmethod
    def default_schedule(cls, d_model: int) -> "PhaseSchedule":
        return cls(
            phases=[
                PhaseConfig(phase=Phase.SGD, steps=100, lr=1e-4),
                PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
            ],
            cycles=3,
        )


@dataclass
class AltOptState:
    """Mutable state tracked across optimization steps."""
    global_step: int = 0
    current_cycle: int = 0
    phase_step: int = 0
    current_phase: Optional[Phase] = None
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)

    def record_loss(self, loss: float) -> None:
        self.losses.append(loss)

    def record_grad_norm(self, norm: float) -> None:
        self.grad_norms.append(norm)


class AltOptFramework:
    """
    Alternating Optimization Framework for LLM post-training.

    Orchestrates three phases:
      1. ALS  — block-wise exact least-squares solving (matrix inversion per block)
      2. SGD  — per-sample stochastic gradient descent for fine-grained refinement
      3. Perturb — parameter-space stochastic perturbation to escape local minima

    Usage:
        model = AutoModelForCausalLM.from_pretrained(...)
        framework = AltOptFramework(model, schedule)
        for batch in dataloader:
            loss = framework.step(batch)
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: PhaseSchedule,
        als_solver: Optional[ALSBlockSolver] = None,
        sgd_optimizer: Optional[SGDPhaseOptimizer] = None,
        perturbation: Optional[PerturbationScheduler] = None,
        loss_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.schedule = schedule
        self.state = AltOptState()

        # Phase components (lazy-init if not provided)
        self._als = als_solver
        self._sgd = sgd_optimizer
        self._perturb = perturbation
        self._loss_fn = loss_fn

        self._phase_index: int = 0
        self._cycle_count: int = 0

    # ── Lazy component initialization ──────────────────────────────

    @property
    def als(self) -> ALSBlockSolver:
        if self._als is None:
            self._als = ALSBlockSolver(self.model)
        return self._als

    @property
    def sgd(self) -> SGDPhaseOptimizer:
        if self._sgd is None:
            self._sgd = SGDPhaseOptimizer(self.model)
        return self._sgd

    @property
    def perturb(self) -> PerturbationScheduler:
        if self._perturb is None:
            self._perturb = PerturbationScheduler(self.model)
        return self._perturb

    # ── Main step ──────────────────────────────────────────────────

    def step(self, batch: dict[str, torch.Tensor]) -> float:
        """
        Execute one step of the current phase.

        Returns:
            loss: float loss value for logging
        """
        if self._phase_index >= len(self.schedule.phases):
            self._cycle_count += 1
            if self._cycle_count >= self.schedule.cycles:
                logger.info("All cycles complete.")
                return 0.0
            self._phase_index = 0

        config = self.schedule.phases[self._phase_index]
        self.state.current_phase = config.phase
        self.state.current_cycle = self._cycle_count

        loss = self._execute_phase(config, batch)

        self.state.record_loss(loss)
        self.state.global_step += 1
        self.state.phase_step += 1

        # Advance phase if completed
        if self.state.phase_step >= config.steps:
            self.state.phase_step = 0
            self._phase_index += 1

        return loss

    def _execute_phase(self, config: PhaseConfig, batch: dict[str, torch.Tensor]) -> float:
        if config.phase == Phase.ALS:
            return self.als.solve_block(batch, block_size=config.block_size or 1024)
        elif config.phase == Phase.SGD:
            self.sgd.set_lr(config.lr or 1e-4)
            loss = self.sgd.step(batch)
            self.state.record_grad_norm(self.sgd.last_grad_norm)
            return loss
        elif config.phase == Phase.PERTURB:
            return self.perturb.apply_noise(scale=config.noise_scale or 1e-3)
        else:
            raise ValueError(f"Unknown phase: {config.phase}")

    # ── Full training loop ─────────────────────────────────────────

    def fit(
        self,
        dataloader: Any,
        max_steps: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> AltOptState:
        """
        Run full alternating optimization over a dataloader.

        Args:
            dataloader: iterable yielding batches (dict[str, Tensor])
            max_steps: optional cap on total steps
            callback: called with (self.state, loss) after each step

        Returns:
            AltOptState with logged losses and grad norms
        """
        steps = 0
        for batch in dataloader:
            loss = self.step(batch)
            if callback is not None:
                callback(self.state, loss)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        return self.state

    # ── State queries ──────────────────────────────────────────────

    def get_parameters(self) -> dict[str, torch.Tensor]:
        """Return a snapshot of current trainable parameters."""
        return {name: param.detach().clone()
                for name, param in self.model.named_parameters()
                if param.requires_grad}

    def compute_flops_estimate(self) -> dict[str, float]:
        """Rough FLOPs estimate for accounting purposes."""
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # ALS: O(b³) per block, SGD: O(d²) per backward, Perturb: O(d)
        return {
            "total_params": total_params,
            "als_per_block": 0.0,  # filled after first ALS step
            "sgd_per_step": 3.0 * total_params,  # heuristic: 3× params per backward
        }
