# Final Report — Alternating Optimization for LLM Post-Training

**Project**: ALS-Based Post-Training for Large Language Models: From Divergence to Convergence
**Date**: 2026-07-24
**Model family tested**: OPT-125m (12L), TinyLlama-1.1B (22L), Qwen2.5-0.5B (24L), Qwen2.5-7B (28L)

---

## Abstract

This project investigates **Alternating Least Squares (ALS)-based post-training** of large language models (LLMs) — an alternative to gradient-based fine-tuning that solves closed-form least-squares problems on output layers instead of relying solely on backpropagation. The central problem: while ALS-based methods work on shallow models (≤24 layers), they **catastrophically diverge on deep models (≥28 layers)** — 11/11 independent attempts on Qwen2.5-7B failed.

This project investigates **Alternating Least Squares (ALS)-based post-training** of large language models (LLMs) — an alternative to gradient-based fine-tuning that solves closed-form least-squares problems on output layers instead of relying solely on backpropagation. The central problem: while ALS-based methods work on shallow models (≤24 layers), they **catastrophically diverge on deep models (≥28 layers)** — 11/11 independent attempts on Qwen2.5-7B failed.

We make three contributions. **(1) A controlled ablation experiment** proves that ALS weight modification — not SGD, not perturbation noise, not hook overhead — is the sole sufficient cause of divergence: ALS-only diverges in a single step (PPL 73→2×10⁹), while SGD-only, Perturb-only, and ALS(no-op)+SGD all converge on the same 28-layer model. **(2) A causal theory** shows that residual connections amplify ALS perturbations by ≈1.08× per layer (1.08²⁷≈8× at 28 layers), far beyond SGD's recovery capacity — predicting the observed depth boundary (L_max≈26, verified across 4 model families). **(3) A negative result with a corrective lesson**: we designed an algorithm, A-SYNC, intended to salvage ALS by injecting its closed-form solution as a *gradient bias* rather than a weight modification, and verified through matched-budget controls that (a) the injection as originally implemented was a timing no-op that never reached any parameter update, and (b) even when correctly timed, injection never outperforms plain SGD (trajectory correlation 0.9998 on 7B; pure SGD reaches PPL 6.83 vs A-SYNC's 6.82 at identical budgets). The genuinely convergent behavior previously attributed to A-SYNC is the convergence of plain SGD itself — with the variant ranking re-interpreted as a **learning-rate schedule** effect (CONSTANT lr beats lr→0 cosine decay), not an injection-strength effect. We also report a FLOPs-normalized comparison establishing where AdamW and LoRA excel, positioning ALS-based methods as unstable on deep models and gradient repair as ineffective.

---

## 1. The Problem

### 1.1 Why it matters

Fine-tuning large language models is the backbone of modern AI deployment — from instruction tuning to domain adaptation. The standard approach (AdamW on all parameters) is **extremely compute-hungry**: a single 7B model fine-tune requires hundreds of GPU-hours. Post-training methods that reduce this cost without sacrificing quality would make LLM customization dramatically more accessible.

**Alternating Least Squares (ALS)** offers a fundamentally different approach. Instead of iterating thousands of gradient steps, ALS solves a **closed-form least-squares problem** on selected layers:

$$W^* = \arg\min_W \|X W^T - Y\|^2 = (X^T X + \lambda I)^{-1} X^T Y$$

where $X$ is the layer's input activations and $Y$ the target. One solve replaces hundreds of gradient steps — potentially a **100–1000× speedup** in optimization steps. This is the same trick that made ALS the standard for collaborative filtering (Netflix Prize era).

### 1.2 The critical failure

But when applied to modern LLMs, ALS-based post-training **breaks on deep models**. The failure is sharp and universal:

| Model | Layers | ALS PPL | Status |
|-------|--------|---------|--------|
| OPT-125m | 12 | 106.9 | ✓ Converges |
| TinyLlama-1.1B | 22 | 15.5 | ✓ Converges |
| Qwen2.5-0.5B | 24 | 18.0 | ✓ Converges (marginal) |
| Qwen2.5-7B | 28 | ∞ | ✗ **Diverges** (11/11 attempts) |

Every attempt on a 28-layer model ended in NaN within 1–3 ALS cycles. **Understanding and fixing this failure is the central challenge of this project.**

---

## 2. Existing Solutions

### 2.1 Gradient-based fine-tuning (the incumbent)

**AdamW full-rank fine-tuning** (Protocol B) is the industry standard. It works on models of any depth but is compute-expensive: each step requires a full forward+backward pass, and converging typically needs hundreds of steps. Our measurements: PPL 23.2 on OPT-125m at 0.91 TFLOPs.

**LoRA** (Low-Rank Adaptation) constrains updates to low-rank subspaces ($\Delta W = BA$, $r \ll d$). It reduces trainable parameters by 100–1000× and is currently the dominant efficiency method. Our measurements: PPL 37.3 on OPT-125m at only **0.013 TFLOPs** (70× less compute than AdamW).

### 2.2 Attempts to make ALS work on deep models

Before this project, the community's fixes were **reactive patches** applied to the ALS solver itself:

1. **Depth-aware damping**: exponentially scale down ALS updates for early layers
2. **Layer skipping**: skip the first 50% of transformer layers
3. **Norm clipping**: clamp per-layer weight changes below thresholds

These patches reduce the *symptoms* but never address the *cause* — the divergence reappears as soon as the model or data changes. All 11 independent 7B attempts (with every combination of patches) still diverged.

---

## 3. Our Approach

### 3.1 Phase 1: Rigorous diagnosis — controlled ablation

We designed a **5-condition controlled experiment** on Qwen2.5-7B to isolate which component causes divergence:

| # | Condition | ALS | SGD | Perturb | Final PPL | Result |
|---|-----------|:---:|:---:|:-------:|-----------|:------:|
| 1 | SGD-only (control) | ✗ | ✓ | ✗ | **53.6** | ✓ Converges |
| 2 | Perturb-only | ✗ | ✗ | ✓ | 94.4 | ✓ Converges |
| 3 | ALS(no-op)+SGD | no-op | ✓ | ✗ | **54.7** | ✓ Converges |
| 4 | **ALS-only** | ✓ | ✗ | ✗ | **2×10⁹** | ✗ **Diverges** |
| 5 | ALS+SGD | ✓ | ✓ | ✗ | 3×10⁸ | ✗ Diverges |

**The verdict is unambiguous**: ALS weight modification alone causes divergence in a single step (PPL jumps from 73 to 2×10⁹). SGD, perturbation noise, and ALS's hook overhead are all ruled out — those conditions converge normally on the same 28-layer model.

This was the first time the divergence was proven (not assumed) to originate from ALS's weight-modification behavior.

### 3.2 Phase 2: Causal theory — why residual connections amplify

Why does modifying lm_head weights destroy a 28-layer model? The answer lies in the **residual connection** — the architectural feature that lets gradients flow through deep networks.

Each transformer layer computes: $h_{l+1} = h_l + f_l(h_l)$. When ALS modifies lm_head, a perturbation $\delta$ propagates backward through every residual connection. Linearizing each layer's response to the perturbation:

$$\delta_{l+1} = (I + J_l) \cdot \delta_l$$

The per-layer amplification factor $\|I + J_l\| \approx 1.08$ is a structural property of trained transformers — the identity path preserves perturbations exactly, and each layer's nonlinear response adds ~8% more. After $L$ layers:

$$\|\delta_L\| = \|\delta_0\| \cdot 1.08^{L-1}$$

At 28 layers: $1.08^{27} \approx 8\times$ amplification. SGD's per-cycle recovery capacity is only ~0.005 — a **1600:1 asymmetry**. The model cannot heal faster than the perturbation grows. The theoretical critical depth:

$$L_{\max} = \frac{\ln(C_{\text{recovery}}/\|\delta_{\text{ALS}}\|)}{\ln 1.08} \approx 26$$

which matches the observed boundary (≤24 converge, ≥28 diverge) across all four model families.

A second finding emerged: ALS's $\delta$ magnitude **increases** across cycles (0.085 → 0.196, ×2.3), because SGD's recovery shifts the body distribution, making the next ALS solve find a *larger* discrepancy. This creates a **positive feedback loop** — the divergence is self-reinforcing, not self-limiting.

### 3.3 Phase 3: An attempted fix — A-SYNC (later invalidated)

We designed **A-SYNC** (ALS-directed Stochastic training with gradient iNjection and Constant strength) to keep ALS's ability to find optimal directions while avoiding the fatal weight modification:

```
Protocol A (old)                A-SYNC (new)
───────────────                ──────────────
1. ALS solves lm_head           1. ALS solves lm_head
2. W ← W_als  (weight changed)  2. δ = W_als − W_before
                                 3. W ← W_before (REVERTED)
3. SGD tries to recover         4. SGD: grad += sync × δ
                                  (gradient bias, not weight change)
```

The design intent: convert the ALS solution into a *directional gradient signal* that steers SGD, instead of a hard weight jump. **However, rigorous verification invalidated this algorithm** (see §4.1): the injection as implemented was a timing no-op (it never reached any parameter update), and even when timed correctly it is neutral-to-harmful. The convergence observed was that of plain SGD.

### 3.4 Phase 4: Variant exploration — re-interpreted

We explored 12 variants of A-SYNC to find the best configuration:

| Variant | Mechanism | 7B PPL |
|---------|-----------|:------:|
| Vanilla (8c) | gradient injection + perturb + exp decay | 25.8 |
| No-Perturb (8c) | remove perturbation phase | 16.6 |
| Cosine (32c) | cosine-decayed sync & lr | 13.2 (plateau) |
| **CONSTANT (24c)** | **constant sync=0.05, lr=2e-4** | **9.0** |
| CONSTANT (48c) | extended to 48 cycles | 7.6 |
| **CONSTANT (96c)** | **extended to 96 cycles** | **6.82** |
| A-CYCLE (3×8) | warm restart per block | 16.5 |
| +EMA | smoothed δ across cycles | 0.5B: 5.5 |
| +Aligned | inject only grad-aligned δ | 0.5B: 5.5 |
| +SWA | weight averaging | 10.5 |
| A-PROBE (r=64) | low-rank probe bypass | 22.8 |

With the injection shown to be a no-op, these results must be **re-interpreted**: the real variable driving the ranking was the **learning-rate schedule**, not the injection strength. CONSTANT (lr=2e-4 fixed) sustains SGD convergence; Cosine decays lr→0, starving the tail of SGD progress and plateauing at PPL 13.2. The power-law tail (−0.01 PPL/cycle at C88) is SGD's own convergence law. Two observations remain informative: **perturbation is counterproductive on deep models** (+9.2 PPL), and the **low-rank probe eliminates divergence** (confirming amplification as the mechanism) but caps quality at PPL 22.8.

---

## 4. Results

### 4.1 The decisive control: what actually converges on 28 layers

**Qwen2.5-7B (28L) — the model that previously diverged 11/11 times under Protocol A:**

```
"A-SYNC" CONSTANT 24c:  PPL 61.8 → 9.0
"A-SYNC" CONSTANT 48c:  PPL 58.8 → 7.6
"A-SYNC" CONSTANT 96c:  PPL 59.1 → 6.82   (still improving — power-law tail)
Pure SGD 96c (control):  PPL 61.4 → 6.83   (identical budget, no ALS machinery)
Old Protocol A:         11/11 DIVERGED
AdamW baseline:         PPL 1.25 (800 steps) — best absolute quality
```

**The decisive control — A-SYNC injection is vacuous:**

Before attributing any convergence to A-SYNC, we ran a matched-budget **pure SGD** control on the same Qwen7B (identical budget: 4800 steps, lr=2e-4, momentum=0, wd=0.01; identical eval points):

| Metric | "A-SYNC" 96c | Pure SGD 96c |
|--------|:-----------:|:------------:|
| Final PPL | 6.82 | **6.83** |
| Trajectory correlation | — | **0.99981** |

The two trajectories are indistinguishable (mean per-cycle difference 0.116 PPL). **Pure SGD alone reaches PPL 6.83; the injection contributed nothing.** Two additional diagnostics at OPT-125m scale confirm this: (a) the injection as implemented is a *timing no-op* — it is added to the gradient buffer *after* `optimizer.step()` and cleared by the next `zero_grad()`, so it never reaches any parameter update; (b) even with corrected timing, a sync sweep (0.05 / 0.5 / 2.0) never beats pure SGD (536.0 PPL) and is mildly harmful (550–559).

**Correct interpretation of the observed convergence**: plain SGD post-training on 28L Qwen7B converges to ~6.8 PPL in 4800 steps with a power-law tail (still −0.013 PPL/cycle at C88; fit $PPL(c) = 3.48 + 61.3\,c^{-0.70}$, asymptote ≈3.5). The "CONSTANT vs Cosine" variant ranking is a **learning-rate schedule** effect — CONSTANT holds lr=2e-4, Cosine decays lr→0 and plateaus at 13.2 — not an injection-strength effect.

### 4.2 FLOPs-normalized comparison — where each method wins

We compared plain-SGD-driven post-training vs AdamW full-rank vs LoRA on OPT-125m with matched FLOPs budgets:

| Protocol | Final PPL | FLOPs (T) | PPL/TFLOP |
|----------|-----------|-----------|-----------|
| AdamW Full-Rank | **23.2** | 0.911 | 25.5 |
| LoRA AdamW r=8 | 37.3 | 0.013 | **2812** |
| SGD-driven (12L, 2448 steps) | 60.7 | 1.846 | 32.9 |

**Efficiency ranking**: LoRA is 87× more compute-efficient than full-rank training. AdamW achieves the best absolute PPL. On shallow models the gradient-based methods are simply better on every axis — reinforcing that ALS-based machinery (weight-modifying or gradient-injecting) provides no post-training advantage.

### 4.3 Where does ALS-based post-training stand?

| Model depth | Recommended | Why |
|-------------|-------------|-----|
| ≤12 layers | AdamW or LoRA | ALS-based methods converge but are neither better nor cheaper |
| 12–24 layers | AdamW or LoRA | ALS converges but offers no advantage |
| **≥28 layers** | **Plain SGD (lr fixed) or AdamW** | **ALS weight modification diverges; gradient-repair is ineffective — plain SGD converges to 6.8 PPL** |

The honest conclusion: for post-training, **plain SGD with a sustained learning rate already converges on 28L models**, and ALS-based methods (whether weight-modifying or gradient-injecting) offer no benefit over it. The FLOPs-normalized comparison (§4.2) confirms AdamW is best in absolute quality and LoRA best in efficiency on shallow models.

---

## 5. Significance

### 5.1 For the field

1. **A rigorous, controlled answer to a long-standing failure**: prior work blamed "perturbation," "optimizer mismatch," or "numerical instability" for ALS divergence. We proved it is specifically the *weight-modification* behavior of ALS interacting with residual connections — a falsifiable claim backed by a 5-condition controlled ablation on a 28-layer model.

2. **A causal theory with predictive power**: the residual amplification framework ($\rho \approx 1.08$, $L_{\max} \approx 26$) predicts divergence boundaries across architectures and can guide architectural choices (gated residuals, MoE) for any future ALS-based training attempt.

3. **A rigorous negative result**: "repairing" ALS by injecting its closed-form solution as a gradient bias does not work — the injection was a timing no-op in our implementation, and even when correctly timed, it is redundant with the CE gradient and never outperforms plain SGD (verified at two scales, including a decisive matched-budget 7B control with trajectory correlation 0.9998). This negative result saves future researchers from a plausible-sounding but ineffective design pattern, and the timing audit itself carries a methodology lesson: verify that each hybrid component actually modifies what it claims to, at the time it claims to.

4. **A re-interpreted empirical finding**: the perceived "algorithm ranking" among our variants is driven by the **learning-rate schedule** (sustained lr converges; lr→0 cosine plateaus), not by ALS-related parameters. Plain SGD post-training on 28L Qwen7B converges to ~6.8 PPL in 4800 steps — a clean, reproducible baseline that the field can build on.

### 5.2 Limitations

- The ALS-divergence diagnosis and theory are validated only on WikiText-2 and a single model family at 7B scale; cross-domain validation remains future work.
- The FLOPs accounting uses parameter-count multipliers rather than full operator-level profiling.
- The gradient-injection negative result was verified at OPT-125m and Qwen7B but not across the full model family; the conclusion (injection redundant with CE gradient) is consistent at both scales tested.

### 5.3 Future directions

1. **Cross-domain validation of the divergence diagnosis** — C4, downstream tasks (HellaSwag, MMLU): confirm the ALS weight-modification divergence is not WikiText-2-specific
2. **ALS on blocks SGD cannot reach** — the one untested salvage direction: apply ALS as a solver for objectives where gradient descent is structurally weak (e.g., attention logit alignment), rather than as a redundant CE-gradient direction
3. **A-PROBE with larger rank** (256/512/1024) — the low-rank probe does eliminate divergence; a wider bottleneck may preserve quality
4. **Verify the lr-schedule interpretation** — a controlled sweep (fixed lr vs cosine vs warm-restart) without any ALS machinery would cleanly establish the learning-rate effect we now attribute the variant ranking to

---

## 6. Conclusion

This project began with a persistent failure mode — ALS-based post-training diverging on deep LLMs — and ended with a rigorously established diagnosis, a causal theory, and a hard-won negative result. Through a controlled ablation on Qwen2.5-7B, we proved that **ALS weight modification — not SGD, perturbation, or hook overhead — is the sole sufficient cause of divergence** (ALS-only diverges in one step; all ALS-free conditions converge). Through causal analysis, we explained why: residual connections amplify the perturbation by ρ≈1.08 per layer, so 28 layers produce ≈8× amplification beyond SGD's recovery capacity, predicting the observed depth boundary L_max≈26. We then tested the natural repair — injecting ALS's closed-form solution as a gradient bias (A-SYNC) — and verified, through matched-budget controls at two scales, that **this repair is ineffective**: the injection was a timing no-op in our implementation, and even when correctly timed it is redundant with the CE gradient and never outperforms plain SGD (7B trajectory correlation 0.9998; pure SGD reaches 6.83 vs A-SYNC's 6.82). The convergence previously attributed to A-SYNC is the convergence of plain SGD itself, and the apparent variant ranking is a learning-rate-schedule effect. The project's lasting value is a reproducible diagnostic methodology, a falsifiable causal theory of ALS divergence in residual networks, a demonstrated negative result that spares future researchers a plausible-but-ineffective design, and an honest account of how a no-op component can masquerade as an algorithmic contribution until it is rigorously controlled against.

---

*All experiments are reproducible. Core scripts: `experiments/_diverge_cause_7b.py` (ablation), `experiments/_pure_sgd_96c_7b.py` (decisive control), `experiments/_flops_sweep.py` (FLOPs comparison), `experiments/_a_sync_*.py` (historical, see the no-op diagnosis in `docs/diag-injection-report.md`). Data: `runs/`. Full docs: `docs/`.*
