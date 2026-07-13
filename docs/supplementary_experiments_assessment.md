# Supplementary Experiment Feasibility Assessment

**Date**: 2026-07-13
**Hardware**: 2× NVIDIA RTX 5090 (32GB each), both idle

---

## E1: Training Budget Equation — $r_{\min}(N_{\text{samples}})$

**Question**: Derive closed-form relationship between minimum LoRA rank and training sample count.

**Approach**: Theoretical derivation, no GPU required. The rank sufficiency law $r_{\min} = \eta \cdot L/d_h$ was tested at $N = 400, 800, 1600$ and the plateau held constant at all three. However, the formal relationship between $N_{\text{samples}}$ and $r_{\min}$ has not been derived — the falsification of $\eta \propto 1/N$ only eliminated one candidate scaling form.

**Feasibility**: ✅ **VERY HIGH** — Pure math, no experiments needed.
- Could extend the residual stream derivation (§6.4 Component 1) to incorporate a finite-sample correction term
- The key insight: for very small $N$, the empirical $X^\top X$ matrix underestimates the true covariance, creating an effective rank ceiling that saturates at some $N_{\text{crit}}$
- Derivation effort: ~2-4 hours
- **Recommendation**: Do it. Adds theoretical depth with zero GPU cost.

---

## F3: Multi-Task $\eta$ (GLUE Benchmark)

**Question**: Does the $r=8$ plateau extend to classification tasks beyond SST-2?

**Approach**: Run LoRA rank curve ($r=4, 8, 32$) on additional GLUE tasks (e.g., MNLI, QQP, QNLI, RTE) using Qwen2.5-0.5B.

**Feasibility**: 🟡 **MODERATE** — Requires GPU, but not much.
- Qwen2.5-0.5B with LoRA fits easily on one RTX 5090 (~2GB VRAM)
- Each rank curve (3 ranks × 1 model × 1 task) takes ~5-10 minutes
- GLUE tasks vary widely in size: RTE (~2.5K train) is fast; MNLI (~393K train) could take hours
- The SST-2 result already confirms classification plateau; adding 2-3 more tasks would strengthen the claim
- **Recommendation**: Do a targeted subset (RTE + QNLI, ~1h GPU total) if time permits. Not blocking for submission — SST-2 already provides classification evidence.

---

## F4: MoE Validation (Mixtral)

**Question**: Does sparse FFN routing in MoE models change the effective $L/d_h$ ratio?

**Approach**: Run LoRA rank curve on Mixtral-8×7B or a smaller MoE model, comparing against the rank sufficiency law prediction.

**Feasibility**: 🔴 **LOW — Not feasible with current hardware.**
- Mixtral-8×7B requires ~90GB VRAM in bf16 (far exceeds 2×32GB RTX 5090)
- 4-bit quantization (QLoRA-style) could fit it in ~25GB, but would introduce quantization as a confound
- Smaller MoE models (e.g., Qwen2.5-MoE-0.5B or custom small MoE) are not readily available
- The paper already lists MoE as a known boundary condition (§6.7) — this is documented, not missing
- **Recommendation**: **Skip for current submission.** List as explicit boundary condition/future work. Only pursue if a reviewer specifically asks for MoE validation and a quantized approach is acceptable.

---

## Summary

| ID | Question | Feasibility | GPU Needed | Effort | Recommendation |
|----|----------|------------|------------|--------|----------------|
| E1 | Training budget equation | ✅ Very High | None | 2-4h | **Do it** — pure theory, zero cost |
| F3 | Multi-task GLUE η | 🟡 Moderate | ~1h | ~3h total | Nice-to-have, not blocking |
| F4 | MoE validation (Mixtral) | 🔴 Low | >90GB VRAM | — | **Skip** — hardware infeasible, already documented as boundary |

**Bottom line**: E1 is worth doing now (theoretical, no GPU). F3 adds credibility but isn't blocking. F4 is a hardware impossibility — document the limitation clearly and move on.
