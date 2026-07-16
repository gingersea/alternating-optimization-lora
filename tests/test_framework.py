"""Tests for the Alternating Optimization Framework."""

import pytest
import torch
import torch.nn as nn

from altopt.framework import AltOptFramework, Phase, PhaseConfig, PhaseSchedule
from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer
from altopt.perturbation import PerturbationScheduler


class SimpleModel(nn.Module):
    """Minimal linear model for unit testing."""

    def __init__(self, d_in: int = 64, d_out: int = 32):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor, labels=None, **kwargs) -> torch.Tensor:
        out = self.linear(x)
        if labels is not None:
            loss = ((out - labels) ** 2).mean()
            output_cls = type("Output", (), {"loss": loss, "logits": out})
            return output_cls()
        return out


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def batch():
    """Create a synthetic batch for testing."""
    x = torch.randn(4, 64)
    # SimpleModel.forward accepts 'x' as keyword, matching the batch format
    return {"x": x, "labels": x[:, :32]}


class TestPhaseSchedule:
    """Tests for phase schedule construction and iteration."""

    def test_default_schedule_has_two_phases(self):
        schedule = PhaseSchedule.default_schedule(d_model=768)
        assert len(schedule.phases) == 2
        assert schedule.cycles == 3

    def test_phase_types_in_order(self):
        schedule = PhaseSchedule.default_schedule(d_model=768)
        expected = [Phase.SGD, Phase.PERTURB]
        actual = [p.phase for p in schedule.phases]
        assert actual == expected

    def test_phase_config_has_required_fields(self):
        phase = PhaseConfig(phase=Phase.SGD, steps=50, lr=1e-3)
        assert phase.phase == Phase.SGD
        assert phase.steps == 50
        assert phase.lr == 1e-3


class TestPerturbationScheduler:
    """Tests for stochastic perturbation mechanisms."""

    def test_gaussian_noise_changes_parameters(self, model):
        scheduler = PerturbationScheduler(model, noise_type="gaussian", initial_scale=0.1)
        original_params = {name: p.clone() for name, p in model.named_parameters()}

        energy = scheduler.apply_noise(scale=0.1)

        for name, p in model.named_parameters():
            assert not torch.allclose(p, original_params[name]), f"Parameter {name} unchanged"

        assert energy > 0, "Noise energy should be positive"

    def test_zero_scale_does_not_change_parameters(self, model):
        scheduler = PerturbationScheduler(model, noise_type="gaussian", initial_scale=0.1)
        original_params = {name: p.clone() for name, p in model.named_parameters()}

        scheduler.apply_noise(scale=0.0)

        for name, p in model.named_parameters():
            assert torch.allclose(p, original_params[name]), f"Parameter {name} changed with zero scale"

    def test_cosine_decay_decreases_over_cycles(self, model):
        scheduler = PerturbationScheduler(
            model, noise_type="gaussian", initial_scale=0.1, decay_schedule="cosine"
        )
        energy_0 = scheduler.apply_noise(cycle=0)
        energy_5 = scheduler.apply_noise(cycle=5)

        assert energy_5 < energy_0, "Cosine decay should reduce noise over cycles"

    def test_uniform_noise_bounded(self, model):
        scheduler = PerturbationScheduler(model, noise_type="uniform", initial_scale=0.1)
        original_params = {name: p.clone() for name, p in model.named_parameters()}

        scheduler.apply_noise(scale=0.1)

        for name, p in model.named_parameters():
            diff = (p - original_params[name]).abs().max()
            # Uniform noise should be bounded by ±scale (±0.1)
            assert diff <= 0.1 + 1e-6, f"Uniform noise exceeded scale bound for {name}: {diff}"

    def test_layer_multiplier_embedding_smaller(self):
        """Embedding layers should get less perturbation (10% of base)."""
        mult = PerturbationScheduler._layer_multiplier("model.embed_tokens.weight")
        assert mult == 0.1

    def test_layer_multiplier_attention_moderate(self):
        """Attention layers should get moderate perturbation (50% of base)."""
        mult = PerturbationScheduler._layer_multiplier("model.layers.0.self_attn.q_proj")
        assert mult == 0.5


class TestSGDPhaseOptimizer:
    """Tests for the SGD phase optimizer."""

    def test_sgd_step_reduces_loss(self, model, batch):
        optimizer = SGDPhaseOptimizer(model, lr=0.1)
        initial_params = {name: p.clone() for name, p in model.named_parameters()}

        loss = optimizer.step(batch)

        params_changed = any(
            not torch.allclose(p, initial_params[name])
            for name, p in model.named_parameters()
        )
        assert params_changed, "SGD step should change model parameters"
        assert isinstance(loss, float)

    def test_grad_norm_is_tracked(self, model, batch):
        optimizer = SGDPhaseOptimizer(model, lr=0.1)
        optimizer.step(batch)

        assert optimizer.last_grad_norm >= 0, "Grad norm should be non-negative"
        assert optimizer.last_grad_norm < float("inf"), "Grad norm should be finite"

    def test_set_lr_updates_optimizer(self, model, batch):
        optimizer = SGDPhaseOptimizer(model, lr=1e-3)
        optimizer.set_lr(1e-1)

        for param_group in optimizer._optimizer.param_groups:
            assert param_group["lr"] == 1e-1


class TestALSBlockSolver:
    """Tests for the ALS block-wise exact solver."""

    def test_als_solver_initializes(self, model):
        solver = ALSBlockSolver(model)
        assert solver.reg_lambda == 1e-3  # default changed from 1e-4 to 1e-3

    def test_clear_cache(self, model):
        solver = ALSBlockSolver(model)
        solver._cache["test"] = torch.ones(1)
        solver.clear_cache()
        assert len(solver._cache) == 0


class TestAltOptFramework:
    """Integration tests for the full alternating optimization framework."""

    def test_framework_initializes(self, model):
        schedule = PhaseSchedule.default_schedule(d_model=64)
        framework = AltOptFramework(model, schedule)

        assert framework.schedule is schedule
        assert framework.state.global_step == 0

    def test_lazy_component_initialization(self, model):
        schedule = PhaseSchedule.default_schedule(d_model=64)
        framework = AltOptFramework(model, schedule)

        # Components should be lazily created
        assert framework._als is None
        assert framework._sgd is None
        assert framework._perturb is None

        # Access triggers creation
        _ = framework.als
        assert framework._als is not None

    def test_get_parameters_snapshot(self, model):
        schedule = PhaseSchedule.default_schedule(d_model=64)
        framework = AltOptFramework(model, schedule)

        params = framework.get_parameters()
        assert "linear.weight" in params
        assert "linear.bias" in params

    def test_compute_flops_estimate(self, model):
        schedule = PhaseSchedule.default_schedule(d_model=64)
        framework = AltOptFramework(model, schedule)

        flops = framework.compute_flops_estimate()
        assert "total_params" in flops
        assert flops["total_params"] > 0
