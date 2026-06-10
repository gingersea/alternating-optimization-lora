"""Tests for CheckpointManager."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from altopt.checkpoint import CheckpointManager
from altopt.trainer import TrainerState


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)


@pytest.fixture
def state():
    s = TrainerState()
    s.record_loss(0.5)
    s.record_loss(0.3)
    return s


@pytest.fixture
def tmp_run_dir():
    d = tempfile.mkdtemp()
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


class TestCheckpointManager:
    def test_initialization_creates_dir(self, tmp_run_dir):
        run_dir = os.path.join(tmp_run_dir, "test_run")
        ckpt = CheckpointManager(run_dir=run_dir)
        assert os.path.exists(run_dir)

    def test_save_creates_checkpoint(self, tmp_run_dir, model, optimizer, state):
        run_dir = os.path.join(tmp_run_dir, "test_run")
        ckpt = CheckpointManager(run_dir=run_dir, save_every=1)
        ckpt.save(step=100, state=state, model=model, optimizer=optimizer)

        ckpt_dir = os.path.join(run_dir, "checkpoints", "step_00100")
        assert os.path.exists(ckpt_dir)
        assert os.path.exists(os.path.join(ckpt_dir, "model_weights.pt"))
        assert os.path.exists(os.path.join(ckpt_dir, "optimizer_state.pt"))
        assert os.path.exists(os.path.join(ckpt_dir, "altopt_state.json"))
        assert os.path.exists(os.path.join(ckpt_dir, "metadata.json"))

    def test_load_restores_model(self, tmp_run_dir, model, optimizer, state):
        run_dir = os.path.join(tmp_run_dir, "test_run")
        ckpt = CheckpointManager(run_dir=run_dir)

        # Modify model before save
        with torch.no_grad():
            model.linear.weight.fill_(42.0)

        ckpt.save(step=50, state=state, model=model, optimizer=optimizer)

        # Create fresh model
        new_model = SimpleModel()
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)

        ckpt_path = os.path.join(run_dir, "checkpoints", "step_00050")
        step, state_dict, loaded_model, loaded_optimizer = ckpt.load(
            ckpt_path, new_model, new_optimizer
        )

        assert step == 50
        assert torch.allclose(loaded_model.linear.weight, torch.full_like(loaded_model.linear.weight, 42.0))

    def test_maybe_save_triggers_at_interval(self, tmp_run_dir, model, optimizer, state):
        run_dir = os.path.join(tmp_run_dir, "test_run")
        ckpt = CheckpointManager(run_dir=run_dir, save_every=10)

        ckpt.maybe_save(step=5, state=state, model=model, optimizer=optimizer)
        assert not os.path.exists(os.path.join(run_dir, "checkpoints", "step_00005"))

        ckpt.maybe_save(step=10, state=state, model=model, optimizer=optimizer)
        assert os.path.exists(os.path.join(run_dir, "checkpoints", "step_00010"))

    def test_list_runs_empty(self, tmp_run_dir):
        ckpt = CheckpointManager(run_dir=os.path.join(tmp_run_dir, "dummy"))
        runs = ckpt.list_runs(base_dir=tmp_run_dir)
        assert runs == []

    def test_list_runs_finds_runs(self, tmp_run_dir, model, optimizer, state):
        run_dir = os.path.join(tmp_run_dir, "run_001")
        ckpt = CheckpointManager(run_dir=run_dir)
        ckpt.save(step=10, state=state, model=model, optimizer=optimizer)

        runs = ckpt.list_runs(base_dir=tmp_run_dir)
        assert "run_001" in runs

    def test_cleanup_old_checkpoints(self, tmp_run_dir, model, optimizer, state):
        run_dir = os.path.join(tmp_run_dir, "test_run")
        ckpt = CheckpointManager(run_dir=run_dir, save_every=1, keep_last=2)

        for step in [10, 20, 30, 40]:
            ckpt.save(step=step, state=state, model=model, optimizer=optimizer)

        ckpts_dir = os.path.join(run_dir, "checkpoints")
        remaining = os.listdir(ckpts_dir)
        assert len(remaining) == 2
        assert "step_00030" in remaining
        assert "step_00040" in remaining

    def test_load_nonexistent_checkpoint(self, tmp_run_dir, model, optimizer):
        ckpt = CheckpointManager(run_dir=os.path.join(tmp_run_dir, "dummy"))
        with pytest.raises(FileNotFoundError):
            ckpt.load("/nonexistent/path", model, optimizer)

    def test_save_with_config(self, tmp_run_dir, model, optimizer, state):
        run_dir = os.path.join(tmp_run_dir, "test_run")
        ckpt = CheckpointManager(run_dir=run_dir)
        ckpt.save(step=1, state=state, model=model, optimizer=optimizer,
                  config={"protocol": "A", "lr": 1e-4})

        ckpt_dir = os.path.join(run_dir, "checkpoints", "step_00001")
        assert os.path.exists(os.path.join(ckpt_dir, "config.yaml"))
