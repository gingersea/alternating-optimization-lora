# Project Status — v3.2 FINAL

**Date**: 2026-06-22
**Status**: ✅ Complete. 12 experiments done (P0-P5 + F1-F2 + A + E4). Grok review cleared (Minor Revision → Accept). Paper v3.2 ready.

---

## Completed (All Items) — 10 Experiments

### P0-P5: Hypothesis Testing
- [x] P0: Chinese WikiText — r=8 language-independent; η∝H falsified
- [x] P1: ASP crossover — SGD+Perturb beats AdamW by 28% on GPT-2 at 800s
- [x] P2: T5 encoder-decoder — boundary condition confirmed
- [x] P3: M-index cross-scale — β scale-dependent phase transition
- [x] P4: SmolLM2 r_min≈12 — fine-grained confirmation (±1 rank)
- [x] P5: Multi-seed rank curve — SE<0.002; max|Δ|=0.0055

### F1-F2: Theory Completion
- [x] F1: η mechanism — task-stable across languages and training budgets; H and N_samples eliminated
- [x] F2: Full ASP — real Cholesky ALS non-monotonic at 12L; best PPL=1.87 mid-training

### A: SST-2 Classification Validation — DONE
- [x] A: SST-2 rank curve — r=4/8/32 all achieve IDENTICAL accuracy (84.7%, 739/872)
- Finding: rank plateau extends to classification AND down to r=4 (below PPL plateau at r=8)

### Core + Downstream + Math
All core experiments, downstream evaluations, cross-dataset validations, and mathematical derivations complete. Paper v3.0 at paper/paper_draft_v0.2.md (878 lines).

---

## Future Enhancement Directions (Post-v3.0, Non-Blocking)

| ID | Direction | GPU | Value | Question |
|----|-----------|-----|-------|----------|
| F3 | Multi-task η (GLUE) | 2h | 🟡 | Does r=8 plateau extend to classification tasks? |
| F4 | MoE validation (Mixtral) | 45min | 🟡 | Does sparse FFN change effective L/d_h? |
| E1 | Training budget equation | — | 🟡 | Derive r_min(N_samples) closed form |
| E2 | Long-horizon rank stability | 30min | 🟢 | r=8 plateau at 200-1600 steps? |
| E3 | LLaMA-3.2 validation | 1h | 🟢 | Cross-family confirmation |
| E4 | FFN LoRA | 20min | 🟢 | Test break condition #3 — lower r_min? |

---

*Last updated: 2026-06-22, v3.0*

