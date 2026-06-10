"""
Checkpoint manager for training state persistence.

Handles:
  - Full training snapshot save (model, optimizer, AltOptState, config)
  - Resume from checkpoint with hash-verified integrity
  - Multi-run directory management (scan, list, select)
  - Automatic old checkpoint cleanup (keep_last policy)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Save and restore complete training state.

    Directory structure:
        {run_dir}/checkpoints/step_{N:05d}/
            model_weights.pt
            optimizer_state.pt
            altopt_state.json
            config.yaml
            metadata.json
    """

    def __init__(
        self,
        run_dir: str,
        save_every: int = 500,
        keep_last: int = 3,
    ):
        self.run_dir = Path(run_dir)
        self.save_every = save_every
        self.keep_last = keep_last
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def maybe_save(
        self,
        step: int,
        state,
        model: nn.Module,
        optimizer,
        config: Optional[dict] = None,
    ):
        if step > 0 and step % self.save_every == 0:
            self.save(step, state, model, optimizer, config)

    def save(
        self,
        step: int,
        state,
        model: nn.Module,
        optimizer,
        config: Optional[dict] = None,
    ):
        ckpt_dir = self.run_dir / f"checkpoints" / f"step_{step:05d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), ckpt_dir / "model_weights.pt")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer_state.pt")

        state_dict = self._serialize_state(state)
        with open(ckpt_dir / "altopt_state.json", "w") as f:
            json.dump(state_dict, f, indent=2)

        if config is not None:
            with open(ckpt_dir / "config.yaml", "w") as f:
                yaml.safe_dump(config, f)

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "python_version": os.sys.version,
        }
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                metadata["git_commit"] = result.stdout.strip()
        except Exception:
            pass

        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self._write_manifest(ckpt_dir)
        self._cleanup_old()

        logger.info("Checkpoint saved: step=%d, dir=%s", step, ckpt_dir)

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer=None,
    ) -> tuple[int, dict, nn.Module, object]:
        ckpt_dir = Path(path)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

        if not self._verify_manifest(ckpt_dir):
            logger.warning("Checkpoint integrity check failed for %s", ckpt_dir)

        model.load_state_dict(torch.load(ckpt_dir / "model_weights.pt", map_location="cpu"))
        if optimizer is not None and (ckpt_dir / "optimizer_state.pt").exists():
            optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer_state.pt", map_location="cpu"))

        with open(ckpt_dir / "altopt_state.json") as f:
            state_dict = json.load(f)

        with open(ckpt_dir / "metadata.json") as f:
            metadata = json.load(f)

        step = metadata.get("step", 0)
        logger.info("Checkpoint loaded: step=%d from %s", step, ckpt_dir)
        return step, state_dict, model, optimizer

    def resume(
        self,
        run_id: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        runs_base_dir: str = "runs",
    ) -> tuple[int, dict, nn.Module, torch.optim.Optimizer]:
        run_dir = Path(runs_base_dir) / run_id
        checkpoints_dir = run_dir / "checkpoints"

        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"No checkpoints found for run: {run_id}")

        ckpt_dirs = sorted(checkpoints_dir.iterdir(), key=lambda p: p.name, reverse=True)
        if not ckpt_dirs:
            raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

        latest = ckpt_dirs[0]
        return self.load(str(latest), model, optimizer)

    def list_runs(self, base_dir: str = "runs") -> list[str]:
        base = Path(base_dir)
        if not base.exists():
            return []
        return sorted([
            d.name for d in base.iterdir()
            if d.is_dir() and (d / "checkpoints").exists()
        ])

    def latest_checkpoint(self, base_dir: str = "runs") -> Optional[str]:
        runs = self.list_runs(base_dir)
        if not runs:
            return None
        latest_run = runs[-1]
        ckpt_dir = Path(base_dir) / latest_run / "checkpoints"
        ckpts = sorted(ckpt_dir.iterdir(), key=lambda p: p.name, reverse=True)
        return str(ckpts[0]) if ckpts else None

    def _cleanup_old(self):
        ckpts_dir = self.run_dir / "checkpoints"
        if not ckpts_dir.exists():
            return
        ckpts = sorted(ckpts_dir.iterdir(), key=lambda p: p.name)
        while len(ckpts) > self.keep_last:
            old = ckpts.pop(0)
            import shutil
            shutil.rmtree(old)
            logger.debug("Removed old checkpoint: %s", old)

    def _write_manifest(self, ckpt_dir: Path):
        hashes = {}
        for fpath in ckpt_dir.iterdir():
            if fpath.name == "manifest.json":
                continue
            sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
            hashes[fpath.name] = sha
        with open(ckpt_dir / "manifest.json", "w") as f:
            json.dump(hashes, f, indent=2)

    def _serialize_state(self, state) -> dict:
        if hasattr(state, "to_dict"):
            return state.to_dict()
        return {
            "global_step": getattr(state, "global_step", 0),
            "current_cycle": getattr(state, "current_cycle", 0),
            "phase_step": getattr(state, "phase_step", 0),
            "current_phase": getattr(state, "current_phase", None),
            "losses": getattr(state, "losses", []),
            "grad_norms": getattr(state, "grad_norms", []),
        }

    def _verify_manifest(self, ckpt_dir: Path) -> bool:
        manifest_path = ckpt_dir / "manifest.json"
        if not manifest_path.exists():
            return True
        with open(manifest_path) as f:
            expected = json.load(f)
        for fname, expected_hash in expected.items():
            fpath = ckpt_dir / fname
            if not fpath.exists():
                return False
            actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                return False
        return True
