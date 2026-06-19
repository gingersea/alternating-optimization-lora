"""Tests for AltOptTrainer and infrastructure modules."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from altopt.trainer import AltOptTrainer, TrainerConfig, TrainerState
from altopt.framework import Phase, PhaseConfig, PhaseSchedule


class TinyModel(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.linear(input_ids.float())
        loss = x.mean() if labels is None else ((x - labels.float()) ** 2).mean()
        output_cls = type("Output", (), {"loss": loss, "logits": x})
        return output_cls()


def make_dataloader(n_samples=16, d_model=64, batch_size=4):
    data = []
    for _ in range(n_samples):
        x = torch.randn(d_model)
        data.append({"input_ids": x, "attention_mask": torch.ones(d_model), "labels": x})
    from torch.utils.data import DataLoader, TensorDataset
    return data  # return raw list for simplicity


def make_eval_dataloader(n_samples=8, d_model=64, batch_size=4):
    return make_dataloader(n_samples, d_model, batch_size)


class TestTrainerConfig:
    def test_default_config(self):
        cfg = TrainerConfig()
        assert cfg.protocol == "A"
        assert cfg.optimizer_type == "altopt"
        assert cfg.parameter_form == "full_rank"

    def test_custom_config(self):
        cfg = TrainerConfig(protocol="D", optimizer_type="adamw", parameter_form="lora", lora_r=4)
        assert cfg.protocol == "D"
        assert cfg.lora_r == 4


class TestTrainerState:
    def test_initial_state(self):
        state = TrainerState()
        assert state.step == 0
        assert state.best_loss == float("inf")
        assert state.cumulative_flops == 0.0

    def test_record_loss(self):
        state = TrainerState()
        state.record_loss(0.5)
        state.record_loss(0.3)
        assert state.loss_history == [0.5, 0.3]

    def test_record_eval(self):
        state = TrainerState()
        state.record_eval(10, {"perplexity": 20.0, "loss": 3.0})
        assert state.eval_history[0]["step"] == 10
        assert state.eval_history[0]["perplexity"] == 20.0

    def test_record_memory_updates_peak(self):
        state = TrainerState()
        state.record_memory(100)
        state.record_memory(200)
        state.record_memory(150)
        assert state.peak_memory_mb == 200.0

    def test_to_dict(self):
        state = TrainerState()
        state.record_loss(0.5)
        d = state.to_dict()
        assert "step" in d
        assert "loss_history" in d
        assert d["loss_history"] == [0.5]


class TestAltOptTrainer:
    def test_trainer_initializes_protocol_a(self):
        model = TinyModel()
        eval_data = make_eval_dataloader()
        cfg = TrainerConfig(
            protocol="A", optimizer_type="altopt", parameter_form="full_rank",
            max_steps=5, run_dir=tempfile.mkdtemp(),
        )
        trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_data)
        assert trainer.altopt is not None
        assert trainer.config.protocol == "A"

    def test_trainer_initializes_protocol_b(self):
        model = TinyModel()
        eval_data = make_eval_dataloader()
        cfg = TrainerConfig(
            protocol="B", optimizer_type="adamw", parameter_form="full_rank",
            max_steps=5, run_dir=tempfile.mkdtemp(),
        )
        trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_data)
        assert trainer.optimizer is not None
        assert trainer.altopt is None

    def test_trainer_initializes_protocol_d_lora(self):
        model = TinyModel()
        eval_data = make_eval_dataloader()
        cfg = TrainerConfig(
            protocol="D", optimizer_type="adamw", parameter_form="lora",
            lora_r=2, lora_target_modules=["linear"],
            max_steps=5, run_dir=tempfile.mkdtemp(),
        )
        trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_data)
        assert trainer.lora_baseline is not None

    def test_trainer_initializes_protocol_c_lora_altopt(self):
        """Protocol C: LoRA + AltOpt — should have altopt framework, no ALS phase."""
        model = TinyModel()
        eval_data = make_eval_dataloader()
        cfg = TrainerConfig(
            protocol="C", optimizer_type="altopt", parameter_form="lora",
            lora_r=2, lora_target_modules=["linear"],
            max_steps=5, run_dir=tempfile.mkdtemp(),
        )
        trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_data)
        assert trainer.altopt is not None
        # Verify schedule does NOT include ALS phase for LoRA
        if trainer.altopt is not None:
            phase_types = [pc.phase for pc in trainer.altopt.schedule.phases]
            from altopt.framework import Phase
            assert Phase.ALS not in phase_types, \
                f"Protocol C should not use ALS phase, got {phase_types}"

    def test_all_protocols_initialize(self):
        """All 4 protocols (A, B, C, D) must initialize without error."""
        protocols = [
            ("A", "altopt", "full_rank"),
            ("B", "adamw", "full_rank"),
            ("C", "altopt", "lora"),
            ("D", "adamw", "lora"),
        ]
        for proto, opt, param_form in protocols:
            model = TinyModel()
            eval_data = make_eval_dataloader()
            cfg = TrainerConfig(
                protocol=proto, optimizer_type=opt, parameter_form=param_form,
                lora_r=2, lora_target_modules=["linear"],
                max_steps=2, run_dir=tempfile.mkdtemp(), seed=42,
            )
            try:
                trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_data)
                assert trainer.config.protocol == proto
                assert trainer.config.optimizer_type == opt
                assert trainer.config.parameter_form == param_form
            except Exception as e:
                pytest.fail(f"Protocol {proto} ({opt}/{param_form}) failed: {e}")

    def test_train_loop_protocol_a(self):
        model = TinyModel()
        train_data = make_dataloader(n_samples=16)
        eval_data = make_eval_dataloader()
        cfg = TrainerConfig(
            protocol="A", optimizer_type="altopt", parameter_form="full_rank",
            max_steps=3, eval_every=10, run_dir=tempfile.mkdtemp(), seed=42,
        )
        trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_data)

        from torch.utils.data import DataLoader
        train_dl = DataLoader(train_data, batch_size=4)
        state = trainer.train(train_dl)

        assert state.step == 3
        assert len(state.loss_history) == 3
        assert state.cumulative_flops > 0

    def test_train_loop_protocol_b(self):
        model = TinyModel()
        train_data = make_dataloader(n_samples=16)
        eval_data = make_eval_dataloader()
        cfg = TrainerConfig(
            protocol="B", optimizer_type="adamw", parameter_form="full_rank",
            max_steps=3, run_dir=tempfile.mkdtemp(), seed=42,
        )
        trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_data)

        from torch.utils.data import DataLoader
        train_dl = DataLoader(train_data, batch_size=4)
        state = trainer.train(train_dl)

        assert state.step == 3
        assert len(state.loss_history) == 3
        assert state.cumulative_flops > 0

    def test_export_results(self):
        model = TinyModel()
        eval_data = make_eval_dataloader()
        tmpdir = tempfile.mkdtemp()
        cfg = TrainerConfig(protocol="A", max_steps=1, run_dir=tmpdir)
        trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_data)

        from torch.utils.data import DataLoader
        train_data = make_dataloader(n_samples=4)
        train_dl = DataLoader(train_data, batch_size=4)
        trainer.train(train_dl)
        trainer.export_results()

        import json
        result_path = os.path.join(tmpdir, "trainer_state.json")
        assert os.path.exists(result_path)
        with open(result_path) as f:
            data = json.load(f)
        assert "loss_history" in data

    def test_evaluate(self):
        model = TinyModel()
        eval_data = make_eval_dataloader()
        cfg = TrainerConfig(protocol="A", max_steps=1, run_dir=tempfile.mkdtemp())
        trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_data)
        results = trainer.evaluate()
        assert "perplexity" in results
        assert "loss" in results

    def test_flops_budget_triggers_stop(self):
        model = TinyModel()
        eval_data = make_eval_dataloader()
        train_data = make_dataloader(n_samples=32)
        cfg = TrainerConfig(
            protocol="B", optimizer_type="adamw", parameter_form="full_rank",
            total_budget_flops=100, max_steps=1000, run_dir=tempfile.mkdtemp(), seed=42,
        )
        trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_data)
        from torch.utils.data import DataLoader
        train_dl = DataLoader(train_data, batch_size=4)
        state = trainer.train(train_dl)
        # Should stop before max_steps due to FLOPs budget
        assert state.step < 1000
