# P1 Experiment Baseline — Environment & Pre-Flight Check

**Date**: 2026-07-15
**Purpose**: Fixed baseline for all P1 experiments. All P1 results report this snapshot as the environment reference.

---

## 1. Hardware

| Component | Value |
|-----------|-------|
| GPU 0 | NVIDIA GeForce RTX 5090, 32,607 MiB, ~32 GiB |
| GPU 1 | NVIDIA GeForce RTX 5090, 32,607 MiB, ~32 GiB |
| Total GPU memory | 64 GiB |
| CPU RAM | ~251 GB |

## 2. Software

| Package | Version |
|---------|---------|
| Python | 3.12 |
| PyTorch | 2.9.0+cu128 |
| CUDA | 12.8 |
| NVIDIA driver | 595.71.05 |
| DeepSpeed | 0.19.2 |

## 3. Test Suite (Pre-P1 Snapshot)

```
122 passed, 2 failed
FAILED tests/test_trainer.py::TestAltOptTrainer::test_train_loop_protocol_b
FAILED tests/test_trainer.py::TestAltOptTrainer::test_flops_budget_triggers_stop
```
- Both failures pre-existing: bitsandbytes 8-bit AdamW requires GPU tensors, tests run on CPU.
- P0 test fix applied: `test_als_solver_initializes` reg_lambda 1e-4 → 1e-3.

## 4. P0 Claims Audit (Final)

| Status | Count |
|--------|-------|
| `replicated` | C3, C6 (plus OPT/Qwen subset of C1/C7) |
| `transcribed` | C2, C5, C8, C9, C10, C13, C14 |
| `inferred` | C11 |
| `predicted` | C12 |
| Retracted | C4 (removed from paper) |

## 5. Eval Protocol (Canonical)

- **N_EVAL=200** (~12,640 tokens): primary protocol for all 7B results
- **Full test set** (~298,938 tokens): Protocol B only (`full_test_eval.json`, recovered)
- **Small models**: per-experiment (50–100 samples), not comparable to 7B

## 6. Git Baseline

```
7a0ae13 Rewrite README.md and todo.md — coherent, aligned with v0.7.1 Major Revision
```
