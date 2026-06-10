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

    protocol: str = "A"                   # A, B, C, D
    optimizer_type: str = "altopt"
    parameter_form: str = "full_rank"

    # Budget
    total_budget_flops: float = float("inf")
    max_steps: Optional[int] = None
    max_epochs: Optional[int] = None

    # Evaluation
    eval_every: int = 100

    # Checkpoint
    run_dir: str = "runs/default"
    save_every: int = 500
    keep_last_ckpt: int = 3
    resume_from: Optional[str] = None

    # Profiling
    profile_memory: bool = False

    # Phase schedule (for AltOpt)
    phase_schedule: Optional[PhaseSchedule] = None

    # LoRA (for protocols C, D)
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_modules: Optional[list[str]] = None

    # Optimizer hyperparams
    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 0.01
    adamw_betas: tuple[float, float] = (0.9, 0.999)

    # Seed
    seed: int = 42


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
    eval_history: list[dict] = field(default_factory=list)
    flops_history: list[float] = field(default_factory=list)
    memory_history: list[float] = field(default_factory=list)

    _start_time: float = field(default_factory=time.time)

    def record_loss(self, loss: float):
        self.loss_history.append(loss)

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

        if cfg.parameter_form == "lora":
            if cfg.optimizer_type == "altopt":
                # Protocol C: LoRA + AltOpt — try PEFT bridge first, fall back to built-in
                try:
                    from .peft_bridge import PeftBridge
                    self.peft_bridge = PeftBridge(
                        self.model,
                        r=cfg.lora_r,
                        alpha=cfg.lora_alpha,
                        dropout=cfg.lora_dropout,
                        target_modules=cfg.lora_target_modules,
                    )
                    self.model = self.peft_bridge.peft_model
                except ImportError:
                    logger.info("peft not available, using built-in LoRALayer + AltOpt")
                    lora_cfg = LoRAConfig(
                        r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout,
                        target_modules=cfg.lora_target_modules or ["c_attn", "c_proj"],
                    )
                    self.lora_baseline = LoRABaseline(self.model, lora_cfg, lr=cfg.lr)
                    self.optimizer = self.lora_baseline._optimizer

                # For both PEFT and built-in paths, create AltOpt framework
                # on the (now LoRA-wrapped) model.
                # Note: ALS solver won't find nn.Linear modules in LoRA mode,
                # so it gracefully skips. SGD+perturb alternation still applies.
                schedule = cfg.phase_schedule or PhaseSchedule(
                    phases=[
                        PhaseConfig(phase=Phase.SGD, steps=100, lr=cfg.lr),
                        PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
                    ],
                    cycles=3,
                )
                self.altopt = AltOptFramework(self.model, schedule)
            else:
                # Protocol D: LoRA + AdamW
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
                    PhaseConfig(phase=Phase.ALS, steps=1, block_size=1024),
                    PhaseConfig(phase=Phase.SGD, steps=100, lr=cfg.lr),
                    PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
                ],
                cycles=3,
            )
            self.altopt = AltOptFramework(self.model, schedule)
            self.optimizer = self.altopt.sgd._optimizer if self.altopt._sgd else None
        else:
            from torch.optim import AdamW
            self.optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=cfg.lr, betas=cfg.adamw_betas, weight_decay=cfg.weight_decay,
            )

    # ── Main Training Loop ─────────────────────────────────────────

    def train(self, dataloader) -> TrainerState:
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
        self.state.record_loss(loss)
        phase = self._current_phase_name()
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
