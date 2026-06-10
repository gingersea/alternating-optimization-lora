"""
CUDA memory tracking for resource-normalized comparison.

Provides per-step and cumulative memory statistics using
torch.cuda.memory_stats() for precise peak tracking.

Two modes:
  - Light: only tracks peak allocated/reserved (default, near-zero overhead)
  - Full: per-step time series for Pareto frontier plotting (enabled via
    environment variable ALTOPT_PROFILE_MEMORY=1)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_MEM_PROFILE_ENABLED = os.environ.get("ALTOPT_PROFILE_MEMORY", "0") == "1"


class MemoryTracker:
    """
    Tracks GPU memory usage across optimization steps.

    Uses torch.cuda.memory_stats() which reads directly from the CUDA
    allocator — more precise than torch.cuda.max_memory_allocated() alone
    because it distinguishes allocated vs reserved memory.
    """

    def __init__(self, full_profile: bool = False):
        self.full_profile = full_profile or _MEM_PROFILE_ENABLED
        self._history: list[dict] = []
        self._peak_allocated_mb: float = 0.0
        self._peak_reserved_mb: float = 0.0

    def reset_peak(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def snapshot(self) -> dict:
        if not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0}

        stats = torch.cuda.memory_stats()
        allocated = stats.get("allocated_bytes.all.peak", 0) / (1024 * 1024)
        reserved = stats.get("reserved_bytes.all.peak", 0) / (1024 * 1024)

        self._peak_allocated_mb = max(self._peak_allocated_mb, allocated)
        self._peak_reserved_mb = max(self._peak_reserved_mb, reserved)

        snapshot = {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "peak_allocated_mb": self._peak_allocated_mb,
            "peak_reserved_mb": self._peak_reserved_mb,
        }

        if self.full_profile:
            snapshot["active_mb"] = stats.get("active_bytes.all.current", 0) / (1024 * 1024)
            snapshot["inactive_mb"] = stats.get("inactive_split_bytes.all.current", 0) / (1024 * 1024)
            self._history.append(snapshot)

        return snapshot

    def time_series(self) -> list[dict]:
        return self._history

    def summary(self) -> dict:
        return {
            "peak_allocated_mb": self._peak_allocated_mb,
            "peak_reserved_mb": self._peak_reserved_mb,
            "n_snapshots": len(self._history),
        }

    def reset(self):
        self._history.clear()
        self._peak_allocated_mb = 0.0
        self._peak_reserved_mb = 0.0
        self.reset_peak()
