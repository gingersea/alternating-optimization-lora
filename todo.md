# Qwen2.5-7B 2×2 Factorial — Complete Work Plan

**Date**: 2026-06-20
**Status**: Phase A (Protocol C+D) ✅ → Phase B (Protocol B full-rank) ✅ → Phase C (analysis + push) 🔄

---

## Phase A: Protocol C+D (LoRA) — ✅ DONE (2026-06-20 ~05:00)

- [x] Fix LoRALayer device alignment
- [x] Fix Protocol C ALS strip for LoRA
- [x] Fix LoRALayer dtype=bf16
- [x] Fix Protocol D PeftBridge device_map
- [x] Run Protocol C × 3 seeds × 800 steps — PPL 135.36 ± 9.1
- [x] Run Protocol D × 3 seeds × 800 steps — PPL 10.41 ± 0.01
- [x] Push results to GitHub

---

## Phase B: Protocol B Full-Rank — ✅ DONE (2026-06-20 ~18:30)

### B1-B2: Environment
- [x] CUDA driver upgraded to 595.71.05
- [x] bitsandbytes libnvJitLink.so.13 LD_LIBRARY_PATH fix
- [x] DeepSpeed 0.19.2 + DS_SKIP_CUDA_CHECK=1
- [x] DeepSpeedCPUAdam + CPU optimizer offload
- [x] OpenMPI installed (libopenmpi-dev)

### B3-B5: Protocol B Experiment
- [x] Run Protocol B (AdamW+full-rank) × 3 seeds × 800 steps
- [x] Seed 42: PPL 1.25, 54min
- [x] Seed 123: PPL 1.24, 54min
- [x] Seed 456: PPL 1.25, 52min
- [x] **Mean PPL: 1.25 ± 0.01**

### B6: Protocol A Attempts (6 rounds, all failed)
- [x] v1-v2: device_map="auto" OOM
- [x] v3-v4: DeepSpeed + 8-bit AdamW → ZeRO compatibility issues
- [x] v5: CPU offload + fp32 AdamW → DeepSpeedCPUAdam required
- [x] v6: DeepSpeedCPUAdam → CUDA 12.8/13.0 mismatch
- [x] v7: DS_SKIP_CUDA_CHECK → intermediate-layer ALS hallucinates (‖ΔW‖/‖W‖ > 10⁶)
- [x] v8: lm_head-only ALS → SGD optimizer DeepSpeed incompatibility
- [x] **Root cause confirmed**: single-process DeepSpeed can't shard model params
- [x] ALS depth-boundary fixes applied to code (not tested on 7B)

---

## Phase C: Analysis & Publication — 🔄 IN PROGRESS

### C1: Documentation
- [x] Experiment registry (`docs/experiment-registry.md`) — full inventory
- [x] Updated alignment audit (`docs/alignment_audit.md`) — v2026-06-20
- [x] Updated README — Phase B results + status
- [x] Updated todo.md

### C2: Git
- [x] Commit ALS depth-boundary fixes
- [x] Commit experiment runner scripts
- [x] Commit results (JSON)
- [ ] Push to GitHub

### C3: Next Actions (ranked)
1. OPT-125m Protocol A @ 704s (complete 4/4 in small model)
2. 7B Protocol A' (SGD+Perturb, no ALS) as approximate opt comparison
3. Fix eval sample size (N_EVAL=200 → full test set)
4. Update paper draft v0.6

---

## 2×2 Factorial Matrix

| | AltOpt (ASP) | AdamW |
|---|---|---|
| **LoRA** | C: 135.36 ± 9.1 ✅ | D: 10.41 ± 0.01 ✅ |
| **Full-rank** | A: blocked ❌ | **B: 1.25 ± 0.01** ✅ |

### Key Metrics
- **B vs D**: 8.3× improvement (full-rank >> LoRA)
- **D vs C**: 13.0× improvement (AdamW >> AltOpt on LoRA)
- **Fresh baseline**: PPL 105.56 (same eval set)
- **A-B interaction**: not computable on 7B

### Cross-Architecture
- **5 architectures**: GPT-2, OPT-125m, TinyLlama, Qwen-0.5B, Qwen2.5-7B
- **Depth boundary**: ≤24L converges, ≥28L diverges (4/4 confirmed)
- **A-B gap scaling**: ∝ exp(0.077·L), superlinear

---

*Last updated: 2026-06-20*
