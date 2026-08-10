# A-SYNC CONSTANT 96-Cycle — Extended Training Results

> **Date**: 2026-07-24
> **Model**: Qwen2.5-7B (28L)
> **Config**: sync=0.05 constant, lr=2e-4 constant, no decay, no perturb
> **Runtime**: 2.12 hours (7,641s)
> **Data**: `runs/a_sync_96cycle_7b.json`
> **Plot**: `docs/figures/a_sync_96cycle_analysis.png`

---

## 1. The Question

The 48-cycle run converged at C44 to PPL 7.6 and appeared to plateau. Two competing hypotheses:

1. **True asymptote**: PPL 7.6 is the A-SYNC-specific fixed point (body capacity limit given lm_head-only ALS guidance)
2. **Pseudo-plateau**: the apparent plateau was an artifact of limited cycles — the power-law tail continues improving slowly

**Answer**: Hypothesis 2 is correct. **The C44 plateau was NOT a true asymptote.**

---

## 2. Result

| Run | Cycles | Best PPL | Improvement |
|-----|--------|----------|-------------|
| CONSTANT 24c | 24 | 9.04 | — |
| CONSTANT 48c | 48 | 7.64 | −1.40 vs 24c |
| **CONSTANT 96c** | **96** | **6.82** | **−0.82 vs 48c** |

Final PPL: **59.1 → 6.82** (8.7× improvement over baseline).

The 96c trajectory broke through the 48c "plateau" (7.6) and continued declining monotonically to 6.82 — with no sign of stopping:

```
C48: 7.5 → C54: 7.4 → C60: 7.3 → C66: 7.2 → C72: 7.1 → C78: 7.0 → C84: 6.9 → C90: 6.9 → C96: 6.82
```

---

## 3. Convergence Rate Decay

The per-cycle improvement decays roughly as a power law:

| Window | dPPL/dcycle |
|--------|-------------|
| C1–8    | −5.59 |
| C9–16   | −0.55 |
| C17–24  | −0.14 |
| C25–32  | −0.086 |
| C33–40  | −0.054 |
| C41–48  | −0.035 |
| C49–56  | −0.024 |
| C57–64  | −0.015 |
| C65–72  | −0.013 |
| C73–80  | −0.016 |
| C81–88  | −0.013 |

The rate never reaches zero — even at C88 the model is still improving by ~0.013 PPL/cycle. This is **consistent with the auto-decay theory**: ALS δ magnitude shrinks as the body improves, but never exactly vanishes, so the injection continues providing weak directional guidance.

---

## 4. Asymptote Fit

Fitting $PPL(c) = ppl_{inf} + A \cdot c^{-\alpha}$ on the full trajectory:

| Run | $ppl_{inf}$ | $\alpha$ | Projected PPL at C200 |
|-----|-------------|----------|----------------------|
| 24c | −30.9 | 0.30 | (unreliable, too few points) |
| 48c | −1.1 | 0.57 | 2.13 |
| **96c** | **3.48** | **0.70** | **4.98** |

The 96c fit (most data, most reliable) predicts a **true asymptote of PPL ≈ 3.5** with a power-law exponent α=0.70. At C200, the model should reach approximately **PPL 5.0**.

---

## 5. Gap to AdamW

| Metric | Value |
|--------|-------|
| AdamW 7B (800 steps) | PPL 1.25 |
| A-SYNC 96c | PPL 6.82 |
| **Current gap** | **5.45×** |
| Projected gap at C200 | ~4.0× (PPL 5.0) |
| Projected asymptote | PPL 3.5 (2.8× gap) |

The gap to AdamW is narrowing (6.1× at 48c → 5.45× at 96c) but remains substantial. The A-SYNC asymptote (~3.5) suggests a **fundamental quality floor** distinct from the model's capacity — likely because ALS only guides lm_head, leaving body layers to pure SGD.

---

## 6. Implications

1. **C44 was a pseudo-plateau, not the true limit** — extended training continues improving. The 48c "converged at C44" claim should be revised: convergence is power-law, not asymptotic.

2. **The auto-decay theory is validated**: constant injection strength + natural δ decay produces monotonic power-law convergence — exactly what the fixed-point argument predicts.

3. **The quality ceiling is A-SYNC-specific, not model-limited**: PPL 6.82 at 96c is far from AdamW's 1.25, and the fit asymptote (~3.5) suggests ALS-guidance-only is the bottleneck.

4. **Next directions confirmed**:
   - EMA/Aligned variants on 7B (reduce noise in late power-law tail)
   - Multi-layer ALS (guide body layers, potentially breaking the PPL 3.5 floor)
   - 128-192c runs to test the PPL 5.0 projection at C200

---

## 7. Reproducibility

```bash
# Reproduce
python experiments/_a_sync_96cycle_7b.py          # 2.1h on 2×RTX 5090

# Analyze
python experiments/_analyze_96cycle.py             # merged plot + fits

# Data
runs/a_sync_96cycle_7b.json
docs/figures/a_sync_96cycle_analysis.png
```
