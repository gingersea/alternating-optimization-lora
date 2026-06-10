"""Tests for profiling modules (FlopsProfiler, MemoryTracker)."""

import pytest
import torch
import torch.nn as nn

from altopt.profiling.flops import FlopsProfiler
from altopt.profiling.memory import MemoryTracker


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 16)

    def forward(self, x):
        return self.linear(x)


class TestFlopsProfiler:
    def test_initialization(self):
        profiler = FlopsProfiler()
        assert profiler.cumulative() == 0.0

    def test_heuristic_flops_with_model(self):
        profiler = FlopsProfiler()
        model = SimpleModel()
        profiler.start(model)
        flops = profiler._heuristic_flops()
        assert flops > 0  # model has params

    def test_heuristic_flops_without_model(self):
        profiler = FlopsProfiler()
        assert profiler._heuristic_flops() == 0.0

    def test_step_flops_empty(self):
        profiler = FlopsProfiler()
        result = profiler.step_flops()
        assert result["total"] == 0.0

    def test_phase_breakdown_empty(self):
        profiler = FlopsProfiler()
        assert profiler.phase_breakdown() == {}

    def test_phase_tracking(self):
        profiler = FlopsProfiler()
        model = SimpleModel()
        profiler.start(model)
        # Simulate phase recording
        profiler.record_phase(0, "ALS")
        profiler.record_phase(1, "SGD")
        profiler.record_phase(2, "SGD")
        assert len(profiler._phase_labels) == 3

    def test_reset(self):
        profiler = FlopsProfiler()
        model = SimpleModel()
        profiler.start(model)
        _ = profiler._heuristic_flops()
        profiler.reset()
        assert profiler.cumulative() == 0.0
        assert len(profiler._history) == 0
        assert len(profiler._phase_labels) == 0


class TestMemoryTracker:
    def test_initialization(self):
        tracker = MemoryTracker()
        assert tracker._peak_allocated_mb == 0.0

    def test_snapshot_no_cuda(self):
        tracker = MemoryTracker()
        result = tracker.snapshot()
        assert "allocated_mb" in result
        assert "reserved_mb" in result
        if not torch.cuda.is_available():
            assert result["allocated_mb"] == 0.0

    def test_reset_peak_no_error(self):
        tracker = MemoryTracker()
        tracker.reset_peak()  # should not raise

    def test_summary(self):
        tracker = MemoryTracker()
        summary = tracker.summary()
        assert "peak_allocated_mb" in summary
        assert "peak_reserved_mb" in summary
        assert summary["n_snapshots"] == 0

    def test_time_series_empty(self):
        tracker = MemoryTracker(full_profile=True)
        assert tracker.time_series() == []

    def test_full_profile_collects_history(self):
        tracker = MemoryTracker(full_profile=True)
        tracker.snapshot()
        tracker.snapshot()
        if torch.cuda.is_available():
            assert len(tracker.time_series()) == 2
        else:
            assert len(tracker.time_series()) == 0

    def test_reset(self):
        tracker = MemoryTracker(full_profile=True)
        tracker.snapshot()
        tracker.reset()
        assert tracker._peak_allocated_mb == 0.0
        assert len(tracker.time_series()) == 0
