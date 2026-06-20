# Qwen2.5-7B Experiment Bug Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three bugs blocking Qwen2.5-7B 2×2 factorial experiment — Protocol D CUDA device mismatch, Protocol C NaN/Inf divergence, and enable Protocol A+B via DeepSpeed ZeRO-2.

**Architecture:** Three independent fixes: (1) `altopt/lora.py` device alignment in `LoRALayer.__init__`, (2) `altopt/trainer.py` + `experiments/run_7b_gpu.py` for Protocol C — strip ALS phases when parameter_form is LoRA, (3) `altopt/deepspeed_engine.py` completion + re-enable A+B protocols in runner. Each fix is testable independently.

**Tech Stack:** PyTorch 2.6+, PEFT, DeepSpeed ZeRO-2, bitsandbytes 8-bit AdamW, transformers, Qwen2.5-7B

## Global Constraints

- All changes must preserve backward compat with GPT-2, OPT-125m, and other architectures
- Protocol C must not modify base model weights (LoRA-only)
- DeepSpeed ZeRO-2 must run on 2× RTX 5090 (32GB each)
- Existing 67 tests must continue passing
- Fixes must each have a targeted unit test

---

## Root Cause Summary

### Protocol D: CUDA device mismatch
`altopt/lora.py:72-73` — `nn.Parameter(torch.empty(r, d_in))` defaults to CPU, but base layer is on CUDA via `device_map="auto"`. Forward pass: `x @ lora_A.T` crashes because x is CUDA, lora_A is CPU.

### Protocol C: NaN/Inf
Three interacting issues:
1. ALS solver targets `lm_head` (full-rank nn.Linear) but model is LoRA-wrapped — solves full-rank head using untrained LoRA activations → corrupts head weights
2. `trainer.py` PEFT bridge path doesn't strip ALS from user-provided schedule for LoRA
3. Perturb noise on untrained LoRA adapters amplifies instability from corrupted lm_head

### Protocol A+B: DeepSpeed not integrated
Uncommitted `deepspeed_engine.py` has partial env-var setup + model.cpu(). Experiment runner skips A+B entirely.

---

## Task 1: Fix LoRALayer device alignment

**Files:**
- Modify: `altopt/lora.py:67-74`
- Test: `tests/test_lora.py` (add 2 tests)

**Interfaces:**
- Consumes: base_layer (nn.Linear, may be on any device)
- Produces: LoRALayer with lora_A, lora_B on same device as base_layer

- [ ] **Step 1: Write failing tests — lora params on base device + CUDA forward**

Add to `tests/test_lora.py` in class `TestLoRALayer`:

```python
def test_lora_params_on_base_device(self):
    """LoRA parameters must be on the same device as the base layer."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    base = nn.Linear(64, 32).to("cuda")
    config = LoRAConfig(r=4)
    lora = LoRALayer(base, config)

    assert lora.lora_A.device == base.weight.device, \
        f"lora_A on {lora.lora_A.device}, expected {base.weight.device}"
    assert lora.lora_B.device == base.weight.device, \
        f"lora_B on {lora.lora_B.device}, expected {base.weight.device}"

def test_forward_cuda_no_device_mismatch(self):
    """Forward pass must work when base layer is on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    base = nn.Linear(64, 32).to("cuda")
    config = LoRAConfig(r=4)
    lora = LoRALayer(base, config)

    x = torch.randn(8, 64, device="cuda")
    out = lora(x)
    assert out.shape == (8, 32)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/room115/alternating-optimization-lora
python -m pytest tests/test_lora.py::TestLoRALayer::test_lora_params_on_base_device -xvs
# Expected: FAIL — lora_A on cpu, expected cuda:0

python -m pytest tests/test_lora.py::TestLoRALayer::test_forward_cuda_no_device_mismatch -xvs
# Expected: FAIL — RuntimeError: mat2 is on cpu, different from other tensors on cuda:0
```

- [ ] **Step 3: Fix `LoRALayer.__init__` — create params on base device**

In `altopt/lora.py`, lines 71-73, replace:

```python
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.empty(d_out, r))
```

With:

```python
        device = base_layer.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, d_in, device=device))
        self.lora_B = nn.Parameter(torch.empty(d_out, r, device=device))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_lora.py -xvs
# Expected: ALL PASS (including 2 new CUDA tests)
```

- [ ] **Step 5: Commit**

```bash
git add altopt/lora.py tests/test_lora.py
git commit -m "fix: LoRALayer params now created on base layer device

nn.Parameter(torch.empty(r, d_in)) defaults to CPU, causing CUDA device
mismatch when the wrapped nn.Linear is on GPU via device_map='auto'.
Now uses base_layer.weight.device for lora_A/lora_B creation.

Adds two tests: device parity check and CUDA forward pass.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Fix Protocol C — remove ALS from LoRA+AltOpt schedule

**Files:**
- Modify: `altopt/trainer.py:197-260` (entire Protocol C block in _setup)
- Modify: `experiments/run_7b_gpu.py:99-112` (build_altopt_schedule)
- Modify: `experiments/run_7b_gpu.py:158` (call site, pass param_form)
- Test: `tests/test_trainer.py` (add test for Protocol C initialization)

**Interfaces:**
- Consumes: `run_protocol(protocol_label, opt_type, param_form, seed, n_steps)` from main()
- Produces: Protocol C uses SGD+Perturb-only schedule, no ALS phase

- [ ] **Step 1: Write failing test — Protocol C trainer init**

Add to `tests/test_trainer.py` in class `TestAltOptTrainer`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_trainer.py::TestAltOptTrainer::test_trainer_initializes_protocol_c_lora_altopt -xvs
# Expected: FAIL — default schedule includes ALS
```

- [ ] **Step 3: Fix `trainer.py` — replace entire Protocol C section**

Replace lines 197-260 in `altopt/trainer.py` (the full `if cfg.parameter_form == "lora":` block) with:

```python
        if cfg.parameter_form == "lora":
            if cfg.optimizer_type == "altopt":
                # Protocol C: LoRA + AltOpt (SGD+Perturb only, no ALS)
                peft_ok = False
                try:
                    from .peft_bridge import PeftBridge, model_supports_lora

                    if not model_supports_lora(self.model):
                        raise ValueError("Model architecture does not support PEFT LoRA")

                    self.peft_bridge = PeftBridge(
                        self.model,
                        r=cfg.lora_r,
                        alpha=cfg.lora_alpha,
                        dropout=cfg.lora_dropout,
                        target_modules=cfg.lora_target_modules,
                    )
                    self.model = self.peft_bridge.peft_model
                    peft_ok = True
                except (ImportError, ValueError, RuntimeError) as e:
                    logger.info("PEFT unavailable or incompatible for this model: %s. "
                                "Falling back to built-in LoRALayer.", e)
                    peft_ok = False

                if not peft_ok:
                    lora_cfg = LoRAConfig(
                        r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout,
                        target_modules=cfg.lora_target_modules or ["c_attn", "c_proj"],
                    )
                    self.lora_baseline = LoRABaseline(self.model, lora_cfg, lr=cfg.lr)
                    self.optimizer = None  # AltOptFramework manages its own SGD optimizer

                # Build SGD+Perturb-only schedule.
                # ALS is incompatible with LoRA — it targets full-rank nn.Linear
                # (lm_head), using untrained LoRA activations and corrupting head
                # weights → NaN/Inf divergence on 7B+ models.
                if cfg.phase_schedule is not None:
                    schedule = cfg.phase_schedule
                    unfiltered = schedule.phases
                    filtered_phases = [p for p in unfiltered if p.phase != Phase.ALS]
                    if len(filtered_phases) != len(unfiltered):
                        logger.info(
                            "Protocol C: removed %d ALS phase(s) (ALS targets "
                            "full-rank modules, incompatible with LoRA)",
                            len(unfiltered) - len(filtered_phases),
                        )
                        schedule = PhaseSchedule(phases=filtered_phases, cycles=schedule.cycles)
                else:
                    schedule = PhaseSchedule(
                        phases=[
                            PhaseConfig(phase=Phase.SGD, steps=100, lr=cfg.lr),
                            PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
                        ],
                        cycles=3,
                    )
                self.altopt = AltOptFramework(self.model, schedule)
                self.optimizer = self.altopt.sgd._optimizer
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
```

- [ ] **Step 4: Fix `run_7b_gpu.py` — schedule builder accepts param_form**

Replace `build_altopt_schedule` (lines 99-112) in `experiments/run_7b_gpu.py`:

```python
def build_altopt_schedule(n_steps: int, param_form: str = "full_rank") -> PhaseSchedule:
    """Build ASP phase schedule for n_steps total.

    For full_rank: ALS -> SGD -> Perturb cycles.
    For lora: SGD -> Perturb only (ALS targets full-rank modules,
    incompatible with LoRA parameterization).
    """
    sgd_per_cycle = max(10, n_steps // 4)
    # Account for phases per cycle when computing n_cycles
    if param_form == "lora":
        n_phases_per_cycle = 2  # SGD + Perturb
    else:
        n_phases_per_cycle = 3  # ALS + SGD + Perturb
    n_cycles = max(1, n_steps // (sgd_per_cycle + n_phases_per_cycle))
    phases = []
    if param_form != "lora":
        phases.append(PhaseConfig(phase=Phase.ALS, steps=1, block_size=2048))
    phases.append(PhaseConfig(phase=Phase.SGD, steps=sgd_per_cycle, lr=5e-5))
    phases.append(PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=5e-4))
    return PhaseSchedule(phases=phases, cycles=n_cycles)
```

Replace the call at line 158:

```python
        config.phase_schedule = build_altopt_schedule(n_steps, param_form)
```

- [ ] **Step 5: Run all tests to verify**

```bash
python -m pytest tests/ -xvs
# Expected: ALL PASS
```

- [ ] **Step 6: Commit**

```bash
git add altopt/trainer.py experiments/run_7b_gpu.py tests/test_trainer.py
git commit -m "fix: Protocol C uses SGD+Perturb-only schedule (no ALS for LoRA)

ALS solver targets full-rank nn.Linear modules (lm_head), which is
incoherent with LoRA parameterization and causes NaN/Inf divergence
on 7B models. Protocol C now strips ALS phases from any schedule
when parameter_form='lora'. Trainer also auto-strips ALS from
user-provided schedules in Protocol C.

Fixes pre-existing optimizer wiring bug: non-PEFT fallback path
used lora_baseline._optimizer for checkpointing, but training
routed through AltOptFramework with its own SGD optimizer.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Enable Protocol A+B — DeepSpeed ZeRO-2 for full-rank 7B training

**Files:**
- Modify: `altopt/deepspeed_engine.py:249-262` (initialize method, already partially done in working tree — verify correctness)
- Modify: `experiments/run_7b_gpu.py:220-234` (main, re-enable A+B protocols + conditional DeepSpeed)
- Test: `tests/test_deepspeed.py` (new file, lightweight config tests — no GPU needed)

**Interfaces:**
- Consumes: `DeepSpeedEngine.initialize(dataloader)` with model on CPU
- Produces: Functional ZeRO-2 training for full-rank Qwen2.5-7B

- [ ] **Step 1: Write DeepSpeed config tests**

Create `tests/test_deepspeed.py`:

```python
"""Tests for DeepSpeed engine configuration (no GPU needed)."""
import json
import torch
import torch.nn as nn
from altopt.deepspeed_engine import DeepSpeedConfig


class TestDeepSpeedConfig:
    def test_zero_stage_2_bf16(self):
        cfg = DeepSpeedConfig(zero_stage=2, bf16_enabled=True)
        d = cfg.to_dict()
        assert d["bf16"]["enabled"] is True
        assert d["zero_optimization"]["stage"] == 2
        assert d["zero_optimization"]["reduce_scatter"] is True

    def test_zero_stage_3_fp16(self):
        cfg = DeepSpeedConfig(zero_stage=3, bf16_enabled=False, fp16_enabled=True)
        d = cfg.to_dict()
        assert "fp16" in d
        assert d["fp16"]["enabled"] is True
        assert d["zero_optimization"]["stage"] == 3

    def test_zero_stage_0_no_zero_section(self):
        cfg = DeepSpeedConfig(zero_stage=0)
        d = cfg.to_dict()
        assert "zero_optimization" not in d

    def test_save_load_roundtrip(self, tmp_path):
        cfg = DeepSpeedConfig(zero_stage=2)
        path = tmp_path / "ds_config.json"
        cfg.save(str(path))
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["zero_optimization"]["stage"] == 2

    def test_defaults(self):
        cfg = DeepSpeedConfig()
        assert cfg.zero_stage == 2
        assert cfg.bf16_enabled is True
        assert cfg.gradient_clipping == 1.0
```

- [ ] **Step 2: Run config tests**

```bash
python -m pytest tests/test_deepspeed.py -xvs
# Expected: ALL PASS (5 tests, no GPU or DeepSpeed import needed)
```

- [ ] **Step 3: Verify DeepSpeed env setup in working tree is correct**

The uncommitted changes in `altopt/deepspeed_engine.py` (lines 252-261) are:

```python
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", str(max(1, torch.cuda.device_count())))
        os.environ.setdefault("LOCAL_RANK", "0")

        # Move model to CPU so DeepSpeed can manage device placement
        self.model = self.model.cpu()
```

These changes are correct — DeepSpeed requires full control over device placement, and single-process launcher mode needs these env vars. No additional changes needed.

- [ ] **Step 4: Re-enable Protocol A+B + conditional DeepSpeed in experiment runner**

In `experiments/run_7b_gpu.py`, replace the `protocols` list in `main()` (around line 230):

```python
    # Full-rank protocols (A, B) need DeepSpeed ZeRO-2 (2 GPUs).
    # LoRA protocols (C, D) have only ~590K trainable params, single GPU suffices.
    protocols = [
        ("A", "altopt", "full_rank"),
        ("B", "adamw", "full_rank"),
        ("C", "altopt", "lora"),
        ("D", "adamw", "lora"),
    ]
```

In `run_protocol`, add conditional DeepSpeed based on `param_form`. Update the TrainerConfig creation (around lines 150-155):

```python
    use_ds = (param_form == "full_rank")  # DeepSpeed only for full-rank 7B

    config = TrainerConfig(
        protocol=protocol_label,
        optimizer_type=opt_type,
        parameter_form=param_form,
        max_steps=n_steps,
        lr=5e-5 if param_form == "full_rank" else 1e-4,
        run_dir=str(OUT_DIR / f"ckpt_{label}"),
        seed=seed,
        eval_every=EVAL_EVERY,
        save_every=SAVE_EVERY,
        use_deepspeed=use_ds,
        deepspeed_zero_stage=2,
        deepspeed_bf16=True,
        gradient_accumulation_steps=GRAD_ACCUM,
        lora_r=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
```

Also add `import gc` to the top of `run_protocol` if not already present (needed for cleanup `finally` block).

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -xvs
# Expected: ALL PASS
```

- [ ] **Step 6: Commit**

```bash
git add altopt/deepspeed_engine.py experiments/run_7b_gpu.py tests/test_deepspeed.py
git commit -m "feat: enable Protocol A+B full-rank 7B training via DeepSpeed ZeRO-2

Completes DeepSpeed integration: env var setup for single-process NCCL,
model.cpu() for DeepSpeed device management, re-enables Protocols A+B
in Qwen2.5-7B experiment runner. Full-rank protocols use DeepSpeed
ZeRO-2 on 2 GPUs; LoRA protocols (C, D) use single-GPU (~590K
trainable params, ~9GB VRAM).

Adds DeepSpeed config tests (no GPU needed).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Integration test — verify full 2×2 trainer initialization

**Files:**
- Test: `tests/test_trainer.py` (add integration test)

**Interfaces:**
- Produces: Confidence that all 4 protocol trainer paths initialize correctly

- [ ] **Step 1: Add integration test**

Add to `tests/test_trainer.py` in class `TestAltOptTrainer`:

```python
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
```

- [ ] **Step 2: Run integration test**

```bash
python -m pytest tests/test_trainer.py::TestAltOptTrainer::test_all_protocols_initialize -xvs
# Expected: PASS
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_trainer.py
git commit -m "test: integration test for all 4 protocol trainer paths

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Full verification

- [ ] **Step 1: Run complete test suite**

```bash
cd /home/room115/alternating-optimization-lora
python -m pytest tests/ -v --tb=short 2>&1
# Expected: 70+ tests, ALL PASS
```

- [ ] **Step 2: Verify import integrity**

```bash
python -c "
from altopt.trainer import AltOptTrainer, TrainerConfig
from altopt.lora import LoRABaseline, LoRAConfig, LoRALayer
from altopt.framework import AltOptFramework, Phase, PhaseConfig, PhaseSchedule
from altopt.sgd import SGDPhaseOptimizer
from altopt.perturbation import PerturbationScheduler
from altopt.deepspeed_engine import DeepSpeedConfig, DeepSpeedEngine
print('All imports OK')
"
```

---

## Run Order & Dependency Summary

```
Task 1 (LoRA device fix) ─┐
Task 2 (Protocol C fix)  ─┼─ independent, can run in parallel
Task 3 (DeepSpeed A+B)   ─┘
Task 4 (integration test)── depends on 1, 2, 3
Task 5 (full verification)── depends on 4
```

Tasks 1-3 are fully independent. Tasks 4-5 are sequential verification.

## After All Fixes — Run the Experiment

Once all fixes are committed and tests pass, run:

```bash
cd /home/room115/alternating-optimization-lora
python experiments/run_7b_gpu.py
```

This runs 4 protocols × 3 seeds × 800 steps on Qwen2.5-7B (~1-2 hours per run, 12 runs total). Full-rank protocols (A, B) use DeepSpeed ZeRO-2 across 2 GPUs. LoRA protocols (C, D) use single GPU.
