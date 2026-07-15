# Mechanism Notes — Component Attribution & Depth Instability

**Date**: 2026-07-15
**Status**: Documentation draft — consolidates existing codebase mechanistic understanding.
**Related**: [causal_depth_boundary.md](causal_depth_boundary.md), [math-analysis.md](math-analysis.md), [claims-audit.md](claims-audit.md)

---

## 1. Component Attribution: ALS vs SGD vs Perturbation

### 1.1 Why ASP Components Should NOT Be Analyzed as a Single Factor

The ASP (ALS+SGD+Perturbation) protocol bundles three distinct mechanisms:

| Component | Mechanism | Blocking Question |
|-----------|-----------|-------------------|
| ALS | Block-wise exact least-squares solve on `nn.Linear` weights | What value does the closed-form solve add over gradient-based alternatives? |
| SGD | Gradient-based fine-grained convergence | How much of ASP's performance is just SGD with different initialization? |
| Perturbation | Parameter-space noise injection | Does perturbation help escape local minima, or does it just add variance? |

**Current gap**: All three are bundled as a single "ASP" factor. No experiment separates their individual contributions.

### 1.2 Nested Ablation Design (Pre-registered)

To isolate component effects, a nested ablation on a stable small model (e.g., OPT-125m at 12 layers) should measure:

1. **SGD-only baseline** — parameter-matched steps, no ALS, no perturbation
2. **ALS+SGD** — ALS once, then SGD recovery, no perturbation
3. **SGD+Perturbation** — cyclic perturbation without ALS
4. **Full ASP** — ALS+SGD+Perturbation (current Protocol A/C)

**Expected information gain**: Quantifies the marginal contribution of each component; enables falsifiable prediction of which component drives which observed behavior (non-monotonic convergence, overfitting resistance, depth instability).

### 1.3 Low-Rank ALS — Symmetric Protocol C

The [`solve_low_rank_block`](altopt/als.py:459) method has been implemented but not used in a symmetric 2×2 design. To close the quasi-factorial gap:

- Replace Protocol C's "SGD+Perturb only" with true low-rank ALS
- This makes (A-B)-(C-D) a clean interaction term (optimizer × parameter form)
- Old Protocol C results should be relabeled "C-SGD" (SGD+Perturb without ALS)

**Status**: Low-rank ALS solver exists (`torch.linalg.solve` + lstsq fallback), works on 0.5B and 7B. Needs experimental protocol only.

---

## 2. Depth Instability Mechanism

### 2.1 Phenomenological → Structural

**Current state**: "Models ≤24L converge, ≥28L diverge" is an observed pattern on 8 architectures. The causal mechanism is hypothesized but not experimentally verified.

**Causal theory** ([causal_depth_boundary.md](causal_depth_boundary.md)): ALS acts as a hard intervention (Pearl's do-calculus) on layer `l`'s weights. The intervention deviation propagates through residual connections as:

$$\delta_{k+1} = (I + J_{f_k}) \cdot \delta_k$$

where $J_{f_k}$ is the per-layer Jacobian. After $L - l$ residual hops:

$$\|\delta_L\| \leq \|\delta_l\| \cdot \prod_{k=l}^{L-1} \|I + J_{f_k}\|$$

### 2.2 Falsifiable Predictions

| Prediction | Test Method | Status |
|-----------|-------------|--------|
| Within-family depth sweep (e.g., Qwen2.5-0.5B/1.5B/3B/7B) should show continuous degradation with depth | Run same ALS config across model sizes in same family | **Not tested** |
| Per-layer activation drift increases exponentially with depth | Hook-based activation norm tracking during ALS cycle | **Not tested** |
| Increasing SGD recovery steps should push L\* boundary right | Vary SGD steps per ALS cycle at same model depth | **Not tested** |
| Layer-skipping (only update last N layers) should prevent divergence | Configurable skip_early_ratio sweep | Partially tested (skip_early_ratio=0.5 in implementation) |

### 2.3 Implementation Mechanisms Already in Code

From [`altopt/als.py`](altopt/als.py):

| Mechanism | Location | Purpose | Known Limitation |
|-----------|----------|---------|-----------------|
| `_depth_aware_step_size` | L141-158 | EMA α decays with distance from output: α(l) = step_size · exp(-β·(1-dist)) | Floor at 0.005; insufficient at L≥28 |
| `_should_skip_layer` | L195-208 | Skip first `skip_early_ratio` fraction of layers | Only halves chain length, not enough at depth |
| `_norm_check_and_clip` | L160-193 | Rollback if ‖ΔW‖/‖W‖ > clip_catastrophic | Reactive; divergence already occurred |
| `clip_catastrophic` | L77 | Hard abort threshold for per-cycle divergence | Catches extreme cases, doesn't prevent |

**Key insight**: All existing mechanisms are _reactive_ (detect and abort) and _static_ (fixed thresholds). No _adaptive controller_ exists that adjusts ALS aggressiveness based on real-time activation drift. This is the P2 "safe ALS controller" concept.

### 2.4 Measurement Gap

Current experiments log only final PPL/NaN. For mechanistic understanding, per-cycle diagnostics needed:

- Per-layer activation drift: ‖h_k^ALS - h_k‖ / ‖h_k‖
- Per-layer weight update ratio: ‖ΔW‖ / ‖W‖
- Spectral norm of Jacobian product through residual chain
- SGD recovery time per cycle (time to return to pre-ALS loss)

**Status**: None of these are instrumented in current experiment scripts.

---

## 3. Implicit Regularization — Verification Gaps

### 3.1 Train-Eval Parity Claim

**Existing evidence**: ASP maintains train≈eval at 1,200 steps while AdamW overfits (single model, single dataset).

**Gaps** (from todo.md P1.3):
1. Single dataset (WikiText-2) — needs at least one more dataset (C4)
2. No early-stopped AdamW baseline — ASP may just be "slower to overfit"
3. No weight-decay or dropout matched baseline
4. Best-validation checkpoint comparison missing — fixed-endpoint comparison favors slow optimizers

### 3.2 PAC-Bayes Derivation

Paper Appendix A derives: GenGap ≤ √(‖θ‖² + log(1/δ)) / (2σ²_eff N)
- ASP: σ²_eff ≈ 780
- AdamW: σ²_AdamW ~ 10⁻⁶

This is a _descriptive_ model (fits observed gap) rather than a _predictive_ one. To upgrade to predictive:
- Measure σ²_eff from per-step weight trajectories
- Predict GenGap before measuring eval loss
- Verify across architectures

---

## 4. Summary: What Constitutes "Mechanism-Complete"

| Dimension | Current | Target |
|-----------|---------|--------|
| Component attribution | Bundled (ASP = single factor) | Nested ablation isolates ALS/SGD/Perturb contributions |
| Depth mechanism | Phenomenological pattern (≤24 vs ≥28) | Causal: Jacobian spectra measured, within-family sweep confirms |
| Implicit regularization | Single dataset, descriptive PAC-Bayes | Multi-dataset, predictive, matched baselines |
| Measurement | PPL/NaN only | Per-layer diagnostics (drift, spectrum, recovery time) |

**Next step**: P1.1 nested ablation is the highest-information experiment. Can be run on OPT-125m (~15 min). P1.2 within-family depth sweep requires access to multiple model sizes in the same family (Qwen2.5-0.5B through 7B).
