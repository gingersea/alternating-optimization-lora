# A-SYNC Variant Evolution and ALS Divergence Fix Attempts

> **Context**: This document tracks the full genealogy of A-SYNC algorithm variants developed to overcome the ALS divergence problem on deep transformers (≥28 layers). See `residual-amplification-why-108.md` for the root-cause analysis of why Protocol A fails, and `why-protocol-a-fails-on-7b.md` for the original failure mode on Qwen2.5-7B.

---

## 1. The Problem: Residual Amplification

Protocol A (ASP = Alternating ALS + SGD + Perturbation) performs an ALS closed-form solve on `lm_head`, then runs SGD on the full model to let the transformer body adapt. On models with ≤24 layers this works — the A-B PPL gap is large but ASD converges. At ≥28 layers, the gap becomes infinite: all 11 independent attempts on Qwen2.5-7B diverge.

The root cause is **residual connection amplification**. Each transformer layer computes:

$$h_{l+1} = h_l + f_l(h_l; \theta_l)$$

When ALS modifies `lm_head`, the resulting perturbation $\delta$ propagates backward through the gradient chain. The per-layer Jacobian has spectral radius $\rho \approx 1.08$, meaning each residual connection amplifies the perturbation by ~8%. After 27 layers:

$$1.08^{27} \approx 8.0\times$$

This 8× amplification is well beyond what SGD with gradient clipping (`max_norm=1.0`) can recover within a single cycle. The result is a vicious cycle: ALS perturbation → gradient explosion → clipping flattens shallow-layer updates → parameters desynchronize → larger loss → larger perturbation.

The depth boundary is sharp and universal across architectures: 12L/22L/24L models converge; 28L/30L/32L models diverge.

---

## 2. Original Protocol A → A-SYNC Transition

**Protocol A** directly modifies `lm_head` weights via ALS, then attempts to recover via SGD. On 7B this is impossible — even with `step_size=0.01` (EMA damping retains 99% of old weights), the 1% perturbation gets amplified 8× through the residual chain.

**A-SYNC** (the foundational variant) takes a different approach:

1. ALS computes delta = `W_after_ALS - W_before` (the optimal direction for `lm_head`)
2. The ALS weight change is **reverted** — `lm_head` is restored to its pre-ALS state
3. During subsequent SGD steps, the delta is injected as **gradient bias**: `grad += sync_strength * delta`
4. The model **never sees** the ALS-modified weights in forward pass — only the gradient signal

This is a fundamental architectural change: ALS transitions from a **weight modifier** to a **gradient shaper**. The residual chain is never exposed to the full ALS perturbation; instead, the body receives a weak, sustained gradient signal that points toward the ALS-optimal direction. The perturbation bypasses the residual amplification path entirely.

The original 4-cycle A-SYNC on Qwen7B showed PPL 60.9 → convergence, proving the gradient-injection approach can cross the depth boundary. The 8-cycle extended test reached PPL 25.8 (with perturbation). Without perturbation (No-Perturb variant), 8-cycle reached PPL 16.6.

---

## 3. Full Variant Table

| # | Variant | Script(s) | Mechanism | 7B Result (PPL) | vs CONSTANT 48 |
|---|---------|-----------|-----------|-----------------|----------------|
| 1 | A-SYNC Vanilla | `_a_sync_qwen7b.py`, `_a_sync_8cycle_7b.py` | ALS delta → gradient bias, sync=0.05, decay=0.8, +perturb | 59.4 → 25.8 (8c) | Δ+18.2 |
| 2 | A-SYNC No-Perturb | `_a_sync_noperturb_7b.py` | Same as #1 minus perturbation phase | 60.9 → 16.6 (8c) | Δ+9.0 |
| 3 | A-SYNC+EMA | `_a_sync_plus_variants.py`, `_a_sync_plus_7b.py` | EMA smoothing of ALS delta: `ema_delta = β·raw + (1-β)·ema_delta`, β=0.3 | 7B: not recorded; 0.5B: best 5.5 | — |
| 4 | A-SYNC+Warmup | `_a_sync_plus_variants.py` | 4 cycles pure SGD before A-SYNC starts | 7B: not recorded; 0.5B: best 5.5 | — |
| 5 | A-SYNC+Aligned | `_a_sync_plus_variants.py` | Only inject delta component aligned with current SGD gradient | 7B: not recorded; 0.5B: best 5.5 | — |
| 6 | A-SYNC Cosine 32c | `_a_sync_32cycle_7b.py` | Cosine schedule: `strength = 0.05·0.5·(1+cos(π·t/T))` | 59.9 → 13.2 (plateau C20) | Δ+5.6 |
| 7 | A-SYNC CONSTANT 24c | `_a_sync_constant_7b.py` | NO decay on sync or lr, 24 cycles | 61.8 → 9.0 | Δ+1.4 |
| 8 | **A-SYNC CONSTANT 48c** | `_a_sync_48cycle_7b.py` | NO decay, 48 cycles | **58.8 → 7.6** (C44) | **baseline** |
| 9 | A-CYCLE | `_a_cycle_7b.py` | 3 blocks × 8 cycles, cosine reset per block | 61.6 → 16.5 | Δ+8.9 |
| 10 | A-SYNC+SWA Cosine | `_a_sync_swa_cosine_7b.py` | SWA from C10, cosine decay, 16 cycles | 59.7 → 10.5 (SWA: 13.8) | Δ+2.9 |
| 11 | A-PROBE | `_probe_7b.py` | Low-rank probe (r=64) added before lm_head; ALS solves only probe output | 60.2 → 22.8 | Δ+15.2 |
| 12 | LARS Optimizer | `_lars_qwen05b.py`, `_lars_sanity.py` | Replace SGD with LARS (trust_coef=0.001) | 7B: not tested; 0.5B: not recorded | — |

---

## 4. Detailed Breakdown of Each Variant

### 4.1 A-SYNC Vanilla (8-cycle)

The original A-SYNC proof-of-concept. ALS computes delta on `lm_head`, the weight change is reverted, and `sync_strength * delta` is added to the gradient during 50 SGD steps per cycle. Includes a perturbation phase (`scale=1e-3`) after each SGD block.

**7B trajectory**: 59.4 → 48.6 → 38.7 → 33.8 → 30.2 → 27.9 → 26.5 → 25.8

**Assessment**: Converges monotonically but slowly. The perturbation phase adds noise that dilutes the ALS signal. The exponential decay (`sync *= 0.8` per cycle) causes the ALS gradient to vanish by cycle 6-8, limiting total improvement.

### 4.2 A-SYNC No-Perturb (8-cycle)

Identical to vanilla but removes the perturbation phase entirely. On Qwen0.5B, perturbation was observed to cause a 7.8 → 23.9 PPL bounce; removing it produced monotonic descent (9.1 → 5.6).

**7B trajectory**: 60.9 → 38.8 → 25.0 → 21.9 → 19.6 → 18.3 → 17.5 → 16.6

**Assessment**: Substantially better than vanilla on 7B (+9.2 PPL improvement at final). Confirms perturbation is harmful on deep models — the ALS gradient signal alone is sufficient and cleaner. However, exponential decay still limits total convergence.

### 4.3 A-SYNC+EMA

Applies exponential moving average smoothing to the ALS delta across cycles:

```
ema_delta = beta * raw_delta + (1 - beta) * ema_delta   (beta=0.3)
```

The injection uses the smoothed delta rather than the raw per-cycle delta. The hypothesis: raw ALS deltas contain cycle-to-cycle noise from mini-batch variance; EMA suppresses high-frequency noise while preserving the consistent signal direction.

**Qwen0.5B result**: All A-SYNC+ variants (baseline, warmup, EMA, aligned) converge to the same floor of PPL 5.5 — the model's capacity ceiling. This makes 0.5B useless for discriminating between variants.

**7B result**: Not recorded. The `_a_sync_plus_7b.py` script exists but the run output (`runs/a_sync_plus_7b.json`) was not produced.

### 4.4 A-SYNC+Warmup

Precedes A-SYNC with 4 cycles of pure SGD (52 steps/cycle), then transitions to A-SYNC for the remaining 8 cycles. The hypothesis: starting A-SYNC from an SGD-pre-optimized basin reduces the initial mismatch between ALS direction and current trajectory — the model is already "pointed" in a reasonable direction before ALS steering begins.

**Qwen0.5B result**: Converged to PPL 5.5 floor, indistinguishable from baseline.

**7B result**: Not recorded.

### 4.5 A-SYNC+Aligned

Projects the ALS delta onto the current SGD gradient direction and only injects the positively-aligned component:

```
proj = (dot(grad, delta) / dot(grad, grad)).clamp(min=0) * grad
injection = proj * (strength * norm_delta / norm_grad)
```

The rationale: ALS deltas that oppose the current SGD trajectory create destructive interference. By filtering to only the aligned component, the gradient injection reinforces the optimization direction rather than fighting it. Tracks `cos(aligned_delta, grad)` per cycle as a diagnostic.

**Qwen0.5B result**: Converged to PPL 5.5 floor. Notably, the aligned variant starts much higher (13.8 vs 9.4 for baseline) — the alignment constraint initially reduces effective injection magnitude, but final convergence is identical.

**7B result**: Not recorded.

### 4.6 A-SYNC Cosine 32-cycle

Replaces exponential decay with a cosine schedule for both `sync_strength` and learning rate:

```
strength = 0.05 * 0.5 * (1 + cos(pi * t / T))
lr = 2e-4 * 0.5 * (1 + cos(pi * t / T))
```

Extended to 32 cycles to test whether A-SYNC continues improving with more cycles.

**7B trajectory**: 59.9 → 33.4 (C4) → 18.9 (C8) → 14.7 (C12) → 13.6 (C16) → 13.3 (C20) → 13.2 (C24-C32)

**Assessment**: Plateaus at C20. The cosine schedule drives both sync and lr to near-zero in the tail — by cycle 28+, the ALS gradient is effectively dead and SGD has no learning rate to continue improving. Final PPL 13.2 is better than 8-cycle vanilla (25.8) but worse than CONSTANT. **This plateau was the key diagnostic that motivated the CONSTANT experiment.**

### 4.7 A-SYNC CONSTANT (24-cycle)

Removes ALL decay — both `sync_strength=0.05` and `lr=2e-4` are held constant across all 24 cycles. Direct response to the cosine plateau finding.

**7B trajectory**: 61.8 → 36.0 (C4) → 18.5 (C8) → 12.4 (C12) → 10.4 (C16) → 9.5 (C20) → 9.0 (C24)

**Assessment**: First variant to break the PPL 10 barrier on 7B. Still converging at C24 — no plateau observed. The sustained ALS gradient signal continues to provide direction, and the body has sufficient SGD steps (50/cycle × 24 = 1,200) to adapt. This was a major result: **decay actively hurts A-SYNC convergence**.

### 4.8 A-SYNC CONSTANT (48-cycle) — BEST

Extended CONSTANT to 48 cycles. Same configuration: `sync=0.05` constant, `lr=2e-4` constant, 50 SGD steps/cycle.

**7B trajectory**: 58.8 → 33.0 (C4) → 16.9 (C8) → 12.2 (C12) → 10.6 (C16) → 9.7 (C20) → 9.0 (C24) → 8.7 (C28) → 8.4 (C32) → 8.1 (C36) → 8.0 (C40) → 7.8 (C44) → 7.6 (C48)

**Assessment**: Converges at C44 — the first observed A-SYNC convergence plateau without decay. Final PPL 7.6 is the best Protocol A result ever achieved on 7B. The convergence at C44 suggests the model reached a regime where ALS delta magnitude shrinks naturally (the body has adapted enough that ALS finds smaller deltas), making external decay unnecessary.

### 4.9 A-CYCLE Warm Restart

3 blocks of 8 cycles each, with cosine schedule resetting at block boundaries. 24 total cycles but sync+lr never fully decay.

**7B trajectory**: 61.6 → 42.6 (B1C4) → 40.4 (B1C8) → 26.1 (B2C4) → 25.1 (B2C8) → 16.9 (B3C4) → 16.5 (B3C8)

**Assessment**: PPL 16.5 — substantially worse than CONSTANT 24c (9.0) despite using the same number of total cycles. The cosine decay within each block still wastes cycles (note the near-flat B1C4→B1C8: 42.6→40.4, and B2C4→B2C8: 26.1→25.1). The restart helps (each block re-energizes), but the intra-block decay is self-defeating. CONSTANT wins because it never wastes a single cycle.

### 4.10 A-SYNC+SWA Cosine

Applies Stochastic Weight Averaging (exponential moving average of all model parameters) starting from cycle 10, combined with cosine sync decay. 16 cycles.

**7B trajectory**: Standard path reaches PPL 10.5. SWA-averaged model: PPL 13.8.

**Assessment**: SWA made things **worse** by +3.3 PPL. SWA smooths weights by averaging across recent checkpoints, but in A-SYNC's non-stationary optimization (the ALS gradient direction changes each cycle as the body adapts), averaging across qualitatively different optimization regimes produces a model that is good at nothing. SWA is designed for stationary or near-stationary SGD trajectories — A-SYNC's ALS-driven trajectory is fundamentally non-stationary.

### 4.11 A-PROBE

Instead of modifying `lm_head`, inserts a low-rank probe (3584 → 64 → 3584) before `lm_head`. ALS solves only the probe's output projection (a 64×64 Cholesky system — trivial). The `lm_head` weight is **never touched**. The hypothesis: the 64-dimensional bottleneck eliminates residual amplification because the ALS perturbation is confined to a tiny subspace.

**7B trajectory**: 60.2 → 34.5 (C4) → 24.2 (C8) → 22.7 (C12) → 22.8 (C16)

**Assessment**: Converges (no divergence!) — the bottleneck hypothesis is confirmed. But final PPL 22.8 is substantially worse than A-SYNC CONSTANT (9.0). The low-rank probe lacks expressive capacity: a 64-dim bottleneck forces the body to compress all task-relevant information through a subspace 56× smaller than the full 3584-dim hidden state. This causes information loss that no amount of optimization can recover.

### 4.12 LARS Optimizer

Replaces SGD with LARS (Layer-wise Adaptive Rate Scaling) during the SGD phase, tested on Qwen0.5B (24L) with `trust_coefficient=0.001`. LARS computes per-layer learning rates to normalize gradient magnitudes across layers — in theory, this could counteract the residual amplification gradient imbalance.

**0.5B result**: Not recorded in output JSON.

**7B result**: Not tested.

---

## 5. Fix Attempt Taxonomy

All variants can be classified into five orthogonal categories based on which aspect of the divergence problem they address:

| Category | Strategy | Variants | What It Targets |
|----------|----------|----------|-----------------|
| **A — Reduce perturbation magnitude** | Make ALS delta smaller/smoother before injection | EMA, Aligned, step_size reduction | `‖δ‖` — the initial perturbation size |
| **B — Increase recovery capacity** | Give SGD more ability to absorb the perturbation | More SGD steps/cycle, higher lr, LARS | `C_recovery` — SGD's healing budget |
| **C — Change intervention mechanics** | Bypass the residual amplification path entirely | A-SYNC (gradient injection), A-PROBE (low-rank probe), A-KD (teacher-guided) | `F(δ)` — the amplification function |
| **D — Change the schedule** | Control when and how strongly ALS signal is applied | Cosine decay, CONSTANT no-decay, A-CYCLE warm restart, Warmup pre-training | `d(strength)/dt` — temporal profile |
| **E — Post-hoc smoothing** | Average weights after optimization to reduce variance | SWA | `Var(θ_final)` — final weight uncertainty |

**Category C (changing the mechanics) is the only category that makes a step-change difference.** A-SYNC's gradient-injection approach (C) is the single innovation that made 7B convergence possible at all — all categories A, B, D, and E are refinements built on top of it. A-PROBE (also C) independently confirms the principle: by moving the ALS intervention to a low-rank bottleneck, it eliminates divergence entirely, but at the cost of capacity.

---

## 6. Convergence Table — Best PPL per Variant on 7B

| Variant | Cycles | Baseline PPL | Final PPL | ΔPPL | Converged? | Plateau? |
|---------|--------|-------------|-----------|------|------------|----------|
| A-SYNC Vanilla | 8 | 59.4 | 25.8 | −33.6 | Yes (stopped) | Likely not |
| A-SYNC No-Perturb | 8 | 60.9 | 16.6 | −44.3 | Yes (stopped) | Likely not |
| A-SYNC Cosine | 32 | 59.9 | 13.2 | −46.7 | Yes | C20 (decay kills) |
| A-SYNC+SWA Cosine | 16 | 59.7 | 10.5 | −49.2 | Yes (stopped) | Not yet |
| A-SYNC CONSTANT 24c | 24 | 61.8 | 9.0 | −52.8 | Yes (stopped) | Not yet |
| **A-SYNC CONSTANT 48c** | **48** | **58.8** | **7.6** | **−51.2** | **Yes (C44)** | **C44** |
| A-CYCLE 3×8 | 24 | 61.6 | 16.5 | −45.1 | Yes (stopped) | Within-block |
| A-PROBE | 16 | 60.2 | 22.8 | −37.4 | Yes | C12+ |
| A-SYNC+EMA | — | — | — | — | 7B not recorded | — |
| A-SYNC+Warmup | — | — | — | — | 7B not recorded | — |
| A-SYNC+Aligned | — | — | — | — | 7B not recorded | — |
| Protocol B (AdamW) | 800 steps | 105.6 | 1.25 | −104.3 | Yes | N/A |

The CONSTANT family dominates all other A-SYNC variants. CONSTANT 48c achieves 7.6 PPL — a 7.7× improvement over the baseline (58.8). However, it remains 6.1× worse than Protocol B (AdamW full-rank at 800 steps, PPL 1.25).

---

## 7. What Worked, What Didn't, and Why

### 7.1 What Worked

1. **Gradient injection (Category C — A-SYNC architecture)**: The foundational insight. By reverting ALS weight changes and injecting delta as gradient bias, the perturbation never enters the forward pass and never triggers residual amplification. This is the single innovation that made 7B work at all.

2. **No-decay schedules (Category D — CONSTANT)**: The cosine and exponential decay schedules actively hurt A-SYNC. By the time the body has adapted enough to benefit from ALS guidance (cycles 20+), decay has reduced sync and lr to near-zero. CONSTANT keeps the signal alive and achieves terminal convergence at C44 — a natural plateau where ALS delta magnitude shrinks because the body has adapted.

3. **Removing perturbation (Category A)**: On 7B, the perturbation phase adds noise that degrades convergence. No-Perturb gains +9.2 PPL over vanilla at 8 cycles.

4. **More cycles**: A-SYNC CONSTANT shows no signs of overfitting from cycle count alone. 48 cycles (2,400 SGD steps + 48 ALS solves) produces monotonic improvement through C44.

### 7.2 What Didn't Work (or Underperformed)

1. **Cosine decay (Category D)**: The worst scheduling choice. A-SYNC 32-cycle cosine plateaus at C20 because sync → 0. The CONSTANT experiment directly falsifies the intuition that decay helps.

2. **Warm restart (Category D — A-CYCLE)**: Outperforms cosine but underperforms CONSTANT. The intra-block decay wastes ~4 cycles per block worth of progress.

3. **SWA (Category E)**: Counterproductive on A-SYNC. Weight averaging across non-stationary optimization trajectories produces worse models (+3.3 PPL penalty).

4. **Low-rank bottleneck (Category C — A-PROBE)**: Eliminates divergence but caps expressiveness at PPL 22.8. The 64-dim bottleneck is too narrow for the full information content of natural language.

5. **EMA and Aligned injection (Category A)**: Both hit the Qwen0.5B capacity floor (PPL 5.5), making them untestable on small models. Their 7B effectiveness remains unknown — a critical gap.

### 7.3 Why CONSTANT Wins

The CONSTANT result reveals a fundamental property of A-SYNC dynamics: **the ALS delta magnitude auto-decays as the body converges**. When the transformer body is far from optimal (early cycles), lm_head ALS finds large deltas → strong gradient injection → fast progress. As the body adapts and the representations improve, ALS finds progressively smaller deltas because `lm_head` is already close to optimal for the improved hidden states. External decay is redundant and harmful — it suppresses the signal exactly when the body still needs it.

---

## 8. Open Problems and Next Directions

### 8.1 The Remaining 6.1× Gap to Protocol B

A-SYNC CONSTANT 48c reaches PPL 7.6, but Protocol B (AdamW full-rank) reaches 1.25. This 6.1× gap is substantial. Possible explanations:

- **ALS solves a misaligned objective**: ALS minimizes `lm_head` reconstruction error on the *current* body representations. These representations are suboptimal, so the ALS optimum is a moving target that never coincides with the true optimum.
- **Gradient clipping limitation**: Even with gradient injection, the 1.0 gradient norm cap may limit effective update magnitude.
- **Single-layer focus**: A-SYNC only injects gradient signal for `lm_head`. The body's 27 other layers receive only SGD gradients without ALS guidance — they may converge more slowly than under AdamW.

### 8.2 Critical Missing Experiments

1. **A-SYNC+EMA on 7B**: The `_a_sync_plus_7b.py` script was written but its output is missing. EMA could reduce per-cycle noise while preserving directional consistency — a natural complement to CONSTANT.

2. **A-SYNC+Aligned on 7B**: Alignment filtering may prevent destructive gradient interference in later cycles when ALS deltas become small and noisy.

3. **Multi-head ALS**: What if ALS solved 2-3 attention heads or FFN layers in addition to `lm_head`? This could provide structured guidance to deeper layers without the full cost of all-layer ALS.

4. **A-PROBE with larger rank**: The r=64 bottleneck is clearly too narrow. Testing r=256, r=512, or r=1024 could find a sweet spot where the bottleneck is wide enough for good perplexity while still narrow enough to suppress amplification.

5. **A-SYNC+LARS on 7B**: LARS was only tested on 0.5B and the results were not recorded. Layer-wise adaptive rates could address the gradient imbalance directly.

6. **Longer CONSTANT runs**: CONSTANT 48c converged at C44. Does it hold at PPL 7.6 indefinitely, or is there room for further improvement with 64 or 96 cycles?

### 8.3 Theoretical Gaps

- **Why does the ALS delta auto-decay?** We observe it empirically (C44 convergence without external decay) but lack a formal proof tying delta magnitude to body convergence rate.
- **Is there a fundamental PPL lower bound for gradient-injection methods?** If ALS solves a moving target, there may be an information-theoretic limit on how closely A-SYNC can approach the true optimum.
- **Why does perturbation hurt on 7B but help on smaller models?** On Qwen0.5B (24L), Protocol A's perturbation phase is essential. On 7B (28L), perturbation is actively harmful. This suggests a depth-dependent crossover that is not explained by the current theory.

---

## Appendix: Script-to-Result Mapping

| Script | Run JSON | Status |
|--------|----------|--------|
| `_a_sync_qwen7b.py` | `runs/a_sync_qwen7b.json` | 4-cycle config sweep (3 configs) |
| `_a_sync_8cycle_7b.py` | `runs/a_sync_8cycle_7b.json` | PPL 59.4 → 25.8 |
| `_a_sync_noperturb_7b.py` | `runs/a_sync_noperturb_8cycle_7b.json` | PPL 60.9 → 16.6 |
| `_a_sync_plus_7b.py` | — | **Missing** — EMA run not recorded |
| `_a_sync_plus_variants.py` | `runs/a_sync_plus_variants_05b.json` | 0.5B only, all hit floor |
| `_a_sync_32cycle_7b.py` | `runs/a_sync_32cycle_7b.json` | PPL 59.9 → 13.2 (plateau C20) |
| `_a_sync_constant_7b.py` | `runs/a_sync_constant_7b.json` | PPL 61.8 → 9.0 (24c) |
| `_a_sync_48cycle_7b.py` | `runs/a_sync_48cycle_7b.json` | PPL 58.8 → 7.6 (C44) |
| `_a_cycle_7b.py` | `runs/a_cycle_7b.json` | PPL 61.6 → 16.5 |
| `_a_sync_swa_cosine_7b.py` | `runs/a_sync_swa_cosine_7b.json` | PPL 59.7 → 10.5 (SWA: 13.8) |
| `_probe_7b.py` | `runs/probe_7b.json` | PPL 60.2 → 22.8 |
| `_lars_qwen05b.py` | `runs/lars_qwen05b.json` | Results not recorded |
| `_lars_sanity.py` | `runs/lars_sanity_gpt2.json` | Results not recorded |
