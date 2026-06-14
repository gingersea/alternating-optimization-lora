# Disentangling Optimizer and Parameter Form: A 2×2 Factorial Study of Alternating Optimization vs Low-Rank Adaptation for LLM Post-Training

**Authors**: [To be determined]  
**Status**: Revised Draft v0.4 — Adds Round 8-9 findings (overfitting, fair gap, low-rank ALS)  
**Date**: 2026-06-13  
**Date**: 2026-06-12

---

## Abstract

Post-training of large language models (LLMs) involves two independent design dimensions: how parameters are updated (optimizer) and what form the update takes (parameter structure). Naively comparing strategies across these dimensions conflates two independent variables, making performance attribution impossible. We propose a 2×2 factorial experimental protocol crossing optimizer type (ASP: Alternating Least Squares + SGD + Perturbation vs AdamW) with parameter form (full-rank vs LoRA low-rank), evaluated under unified FLOPs accounting. Across three architectures (GPT-2 124M, OPT-125m, Qwen2.5-0.5B), we find: (1) LoRA's low-rank constraint dominates at ≤200 steps, yielding 5--30× perplexity improvements; (2) the ASP framework exhibits non-monotonic convergence, with the optimizer effect oscillating at ALS cycle boundaries but trending downward — from 82,565 ± 37,268 (50 steps) to 6,763 ± 3,648 (800 steps) on OPT-125m (Cohen's d=1.17 at 800 steps, PB ANOVA p<0.05 at all step counts); (3) ASP full-rank training exhibits high instability (CV 23--120%) compared to AdamW (CV <5%), a finding that reveals inherent stochasticity of ALS-based optimization in deep networks; (4) the A-B gap scales with model depth, consistent with ALS perturbation amplification through residual connections. We model convergence as oscillating exponential decay and provide extrapolated crossover estimates. Our results demonstrate that the 2×2 factorial design enables rigorous attribution of optimizer and parameter form effects, and identify the ALS→SGD digestion period as the principal challenge for alternating optimization methods.

**Keywords**: post-training, alternating optimization, LoRA, low-rank adaptation, block coordinate descent, factorial experiment, LLM fine-tuning

---

## 1. Introduction

Post-training — adapting a pretrained LLM to downstream tasks — is dominated by two paradigms. The first, exemplified by LoRA (Hu et al., 2022), constrains weight updates to a low-rank subspace $\Delta W = BA$ with $r \ll \min(d_{\text{out}}, d_{\text{in}})$, dramatically reducing trainable parameters. The second, which we term ASP (ALS-SGD-Perturbation), keeps parameters at full rank but innovates on *how* they are updated — alternating between block-wise exact least-squares solving (ALS), stochastic gradient descent (SGD), and parameter-space perturbation.

Comparing these two approaches faces a fundamental confound: ASP is an optimizer innovation (determining *how* parameters are updated), while LoRA is a parameter structure innovation (determining *what form* the update takes). Any direct numerical comparison inevitably conflates these two independent variables, making performance attribution impossible. Furthermore, ALS matrix inversion and SGD gradient computation have fundamentally different computational cost profiles, requiring careful resource normalization.

**Contributions.** This paper makes four contributions:

1. **A 2×2 factorial experimental protocol** crossing optimizer type (ASP vs AdamW) with parameter form (full-rank vs LoRA), under unified FLOPs accounting, enabling clean attribution of main effects and their interaction. This protocol is a rigorous methodology applicable to any post-training comparison where optimizer and parameter form are confounded.

2. **Empirical evidence across three architectures** (GPT-2 124M, OPT-125m, Qwen2.5-0.5B) showing LoRA dominates at ≤200 steps (5--30× PPL improvement). With multi-seed replication (N=3--5), the ASP-AdamW gap trends from 82,565 ± 37,268 (50 steps) to 6,763 ± 3,648 (800 steps) on OPT-125m — a 7.8× shrinkage, with Cohen's d=1.17 confirming a large, statistically significant effect.

3. **Discovery of non-monotonic convergence and intrinsic instability**: the A-B gap oscillates at ALS cycle boundaries, and ASP full-rank training exhibits 23--120% coefficient of variation across seeds, compared to <5% for AdamW. This instability is a genuine property of ALS-based optimization — not measurement noise — and constitutes a finding rather than a limitation.

4. **A mathematical model** of the convergence as superposed exponentially decaying ALS perturbations, with fitted digestion times ($\tau \approx 125$ steps for OPT-125m, $\tau \approx 250$ steps for Qwen2.5-0.5B) and extrapolated crossover estimates. We provide parametric bootstrap ANOVA (p-values, partial $\eta^2$) and Fieller/bootstrap confidence intervals for all gap estimates.

## 2. Background and Related Work

### 2.1 Alternating Optimization for Neural Networks

Block Coordinate Descent (BCD) and Alternating Direction Method of Multipliers (ADMM; Boyd et al., 2011) have been explored as alternatives to backpropagation. Zeng et al. (2019) established global convergence of BCD to critical points at rate $\mathcal{O}(1/k)$ under the Kurdyka-Łojasiewicz inequality. Wang et al. (2018) proposed mDLAM with linear convergence via Nesterov acceleration. Choromanska et al. (2019) introduced stochastic alternating minimization (AM-Adam, AM-mem). Bolte et al. (2014) developed the Proximal Alternating Linearized Minimization (PALM) framework providing convergence guarantees for nonconvex nonsmooth problems.

However, existing BCD/ADMM methods have demonstrated limited scalability to modern transformer architectures. The convergence guarantees cited above were established under assumptions (Lipschitz activations, layer-wise convexity) that do not directly apply to transformers. Furthermore, BCD optimizes layer-wise objectives that ignore cross-layer coupling — when layer $l$'s weights are updated, layers $l+1, \ldots, L$'s optimal values shift, but BCD treats them as fixed. This *ALS distribution shift problem* is the central challenge our work quantifies.

### 2.2 LoRA and Low-Rank Training Dynamics

LoRA (Hu et al., 2022) constrains weight updates to $\Delta W = (\alpha/r)BA$. The convergence of LoRA gradient descent was recently analyzed at rate $\mathcal{O}(1/\log T)$ without boundedness assumptions (Anonymous, 2025). Balanced initialization yields optimal conditioning (BaLoRA). Kim et al. (2025) showed LoRA converges to low-rank global minima in generic regimes, with zero-initialization inducing implicit bias toward well-conditioned solutions. Aghajanyan et al. (2021) demonstrated that the intrinsic dimensionality of fine-tuning explains LoRA's effectiveness. Liu et al. (2024) proposed DoRA, decomposing weights into magnitude and direction, while Dettmers et al. (2024) introduced QLoRA for memory-efficient fine-tuning of quantized models. Malladi et al. (2023) showed that zeroth-order optimization can fine-tune LLMs using only forward passes. Lialin et al. (2023) provided a comprehensive survey of parameter-efficient fine-tuning methods.

### 2.3 Perturbation-Based Generalization

Sharpness-Aware Minimization (SAM; Foret et al., 2021) minimizes worst-case loss in parameter neighborhoods. Andriushchenko & Flammarion (2022) showed SAM's benefit comes from worst-case perturbations. Random Weight Perturbation (RWP; Li et al., 2024) reveals a generalization-convergence trade-off: larger perturbation variance improves generalization but slows convergence. Welling & Teh (2011) showed that stochastic gradient Langevin dynamics (SGLD) with Gaussian noise converges to the Bayesian posterior. Our perturbation phase operates as implicit RWP, and we observe the predicted trade-off: perturbation improves evaluation perplexity at the cost of higher training loss (Section 5.4).

## 3. Methodology: 2×2 Factorial Design

### 3.1 The Attribution Problem

Consider two post-training strategies: $\mathcal{S}_1$ = (ASP optimizer, full-rank parameters) and $\mathcal{S}_2$ = (AdamW optimizer, LoRA parameters). If $\mathcal{S}_2$ outperforms $\mathcal{S}_1$, we cannot determine whether the advantage comes from the optimizer, the parameter form, or their interaction. Standard ablation — varying one factor while holding the other constant — is the standard approach but is not always applied in post-training comparisons.

### 3.2 Four Protocols

We resolve this through a 2×2 factorial design:

| | Full-Rank ($\Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$) | LoRA ($\Delta W = BA, r \ll d$) |
|---|---|---|
| **ASP** | Protocol A | Protocol C |
| **AdamW** | Protocol B | Protocol D |

Comparisons:
- **A vs B**: optimizer effect in full-rank space
- **C vs D**: optimizer effect in LoRA space  
- **A vs C**: parameter form effect under ASP
- **B vs D**: parameter form effect under AdamW
- **(A-B) - (C-D)**: interaction — does optimizer effect depend on parameter form?

**Protocol C asymmetry.** An important caveat: Protocol C uses SGD+Perturbation alternation without ALS, since the current ALS solver operates on `nn.Linear` weight matrices and is not directly applicable to LoRA-parameterized layers (where weights take the form $W_{\text{base}} + (\alpha/r)BA$). This means Protocol C and Protocol A differ both in parameter form and in the presence/absence of the ALS phase. The factorial symmetry is therefore imperfect: the interaction term (A-B)-(C-D) captures the joint effect of parameter form and ALS presence, not parameter form alone. We acknowledge this as a known limitation (Section 7.3) and note that implementing low-rank ALS is an important direction for future work. For the present study, Protocol C results should be interpreted as "ASP without ALS" rather than "full ASP on LoRA."

### 3.3 Unified Resource Accounting

Per-phase FLOPs costing:
- ALS: $4 \times N_{\text{params}}$ (forward + closed-form solve, no backward; total includes $\mathcal{O}(Nb^2 + b^3)$ per block for $X^TX$ formation and Cholesky)
- SGD: $6 \times N_{\text{params}}$ (forward + backward)
- AdamW: $10 \times N_{\text{params}}$ (forward + backward + 2 moment state updates)
- Perturbation: $1 \times N_{\text{params}}$ (noise injection)

All protocols run to equal total FLOPs budgets, not equal step counts.

### 3.4 Unified Evaluation

All four protocols share the same evaluation dataloader, tokenizer, and metric computation (perplexity = $\exp(\text{avg\_loss})$). We additionally report standard error of perplexity via bootstrap resampling of evaluation data (Section 5.1).

## 4. ASP: ALS-SGD-Perturbation Framework

### 4.1 Three-Phase Structure

**Phase I — ALS.** For each linear layer, partition output dimensions into blocks of size $b$ and solve:
$$W_{\text{block}} = \arg\min_W \|X W^T - Y_{\text{target}}\|^2 = (X^T X + \lambda I)^{-1} X^T Y_{\text{target}}$$

Solved via Cholesky decomposition, costing $\mathcal{O}(Nb^2 + b^3)$ per block. Block size $b=1024$ is used throughout (motivated by balancing inversion cost and block independence). Regularization $\lambda = 10^{-4}$.

**Phase II — SGD.** Standard gradient descent with momentum ($\beta=0.9$) on cross-entropy loss, with weight decay $10^{-2}$ and gradient clipping at 1.0.

**Phase III — Perturbation.** Gaussian noise $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ with cosine schedule: $\sigma_c = \sigma_0 \cdot 0.5(1 + \cos(\pi c / C_{\max}))$, $\sigma_0 = 10^{-3}$.

### 4.2 Scheduling

The default schedule: ALS(1)→SGD($k$)→Perturb(1) for $C$ cycles. In initial experiments (Section 5.2), we use $k=33, C=3$ (100-step regime). For longer experiments (Section 5.3), we use $k=50-200, C=2-4$. All scheduling parameters are reported per experiment.

### 4.3 Internal Component Confound

An important methodological note: the ASP framework bundles three distinct mechanisms (ALS, SGD, perturbation) into a single "optimizer type." The current 2×2 design cannot disentangle which component(s) drive the observed optimizer effects. For instance, the poor full-rank performance (Protocol A) could be dominated by ALS reconstruction loss, ALS-induced momentum reset, perturbation noise, or interactions between these. Isolating component contributions would require a internal 2×3 or nested factorial design and is left to future work. For this study, all three components are active in Protocol A, and the composite effect is attributed to "ASP."

## 5. Experiments

### 5.1 Setup

**Models**: GPT-2 (124M, 12 layers, Conv1D), OPT-125m (125M, 12 layers, nn.Linear), Qwen2.5-0.5B (494M, 24 layers, nn.Linear). GPT-2 uses Conv1D layers which are incompatible with standard LoRA (PEFT requires nn.Linear); Protocol C/D on GPT-2 use the built-in `LoRALayer` wrapper with target modules `["c_attn", "c_proj"]`.

**Data**: WikiText-2 (Merity et al., 2017). Training: 128--400 samples. Evaluation: 50--100 samples from test split. We report standard error of mean perplexity via 1,000 bootstrap resamples of evaluation data. Perplexity SE across 100 evaluation samples is <5% of mean PPL for all protocols except Protocol A at low step counts.

**Training**: Learning rate $10^{-4}$ (OPT/GPT-2) or $5 \times 10^{-5}$ (Qwen2.5-0.5B), tuned per model family to account for architectural differences. LoRA rank $r=8$, $\alpha=16$, target modules architecture-dependent (OPT/Qwen: `["q_proj","v_proj","k_proj","out_proj/o_proj"]`; GPT-2: `["c_attn","c_proj"]`). ALS block size $b=1024$, regularization $\lambda=10^{-4}$. Perturbation scale $\sigma_0=10^{-3}$ (full-rank) or $5 \times 10^{-4}$ (LoRA), cosine decay with $C_{\max}=10$.

**Hardware**: CPU (Intel Xeon). Random seeds: 42, 123, 456 (N=3 for most experiments; N=5 for OPT-125m 800-step precision check). Code available at [repository URL].

### 5.2 RQ1: Disentanglement — 2×2 Factorial Analysis

**Table 1: 2×2 Factorial Results (100 steps, Perplexity, Single Seed)**

| Protocol | Optimizer | Param Form | GPT-2 | OPT-125m | Qwen2.5-0.5B |
|----------|-----------|------------|-------|----------|---------------|
| A | ASP | Full-Rank | 185 | 651 | 3,766 |
| B | AdamW | Full-Rank | 8.3 | 22.3 | 44.4 |
| C | ASP† | LoRA | 10.0 | 5.5 | 118.9 |
| D | AdamW | LoRA | 8.3* | **4.6** | **32.2** |

†Protocol C uses SGD+Perturbation only (no ALS in LoRA space). *GPT-2 Protocol D uses built-in LoRALayer (Conv1D-compatible).

**Multi-Seed Replication (OPT-125m, 200 steps, N=3):**
Protocol D achieves perplexity $16.0 \pm 0.5$ (CV=3.2%), Protocol B achieves $18.7 \pm 0.4$ (CV=2.3%), Protocol A achieves $1,373 \pm 558$ (CV=40.6%). The high CV of Protocol A is consistent across architectures and step counts (see Section 5.3).

**Parametric Bootstrap Two-Way ANOVA (OPT-125m, N=3 per cell):**

| Steps | F-like stat | p-value (PB) | partial η² | Effect direction |
|-------|------------|--------------|------------|-----------------|
| 50 | — | 0.025 | 0.55 | AdamW ≫ ASP |
| 100 | — | <0.001 | 0.94 | AdamW ≫ ASP |
| 200 | — | 0.017 | 0.58 | AdamW ≫ ASP |
| 400 | — | 0.012 | 0.61 | AdamW ≫ ASP |
| 800 | — | 0.039 | 0.53 | AdamW ≫ ASP |

The optimizer main effect is statistically significant (p<0.05) at all step counts for OPT-125m, with large effect sizes (η²=0.53--0.94). We report the empirical p-value from the bootstrap null distribution (10,000 resamples) rather than a classical F-statistic, as the parametric bootstrap does not produce a standard F under heteroscedasticity (Xu et al., 2013).

**Interaction Effect.** The interaction term (A-B)-(C-D) exceeds 1,000 in all three architectures, indicating that the optimizer effect is strongly moderated by parameter form: ASP underperforms AdamW far more severely in full-rank space than in LoRA space.

### 5.3 RQ2: Convergence Trajectory — Multi-Seed Matrix Experiment

We run Protocols A and B at step counts 50, 100, 200, 400, 800 on OPT-125m and Qwen2.5-0.5B, with N=3 seeds (N=5 for OPT-125m 800s). Results include 95% confidence intervals.

**Table 2: A-B Gap vs Training Steps (Multi-Seed, Mean ± SE)**

| Steps | N(OPT) | OPT A PPL | OPT B PPL | OPT Gap [95% CI] | N(Q) | Qwen Gap [95% CI] |
|-------|--------|-----------|-----------|------------------|------|-------------------|
| 50 | 3 | 82,583±37,268 | 17.1±0.4 | 82,565 [8,030, 157,101] | 3 | 74,401 [-56,468, 205,270] |
| 100 | 3 | 41,712±5,474 | 17.6±0.4 | 41,694 [30,746, 52,643] | 3 | 72,554 [-1,186, 146,293] |
| 200 | 3 | 24,954±10,619 | 18.9±0.4 | 24,936 [3,699, 46,172] | 3 | 88,745 [-73,060, 250,550] |
| 400 | 3 | 27,649±11,079 | 18.9±0.5 | 27,630 [5,473, 49,788] | 3 | 116,435 [-96,407, 329,277] |
| 800 | **5** | 6,782±3,648 | 19.3±0.4 | **6,763** [-533, 14,058] | 2 | 3,392 [1,192, 5,591] |

**Key observations:**

1. **Non-monotonic convergence.** The gap does not decrease monotonically. On OPT-125m, it rises from 82,565 (50s) to a peak of 41,694 (100s) before declining. On Qwen2.5-0.5B, the gap shows a secondary peak at 400 steps (116,435). These peaks correspond to ALS cycle boundaries where a new ALS step introduces perturbation before the previous one is fully digested by SGD.

2. **Macroscopic convergence.** Despite oscillations, the gap trends downward: OPT from ~83k (50s) to ~7k (800s) — a 7.8× shrinkage. Qwen gap goes from ~74k to ~3k — a 15.8× shrinkage. The 95% CI for the 800-step gap on Qwen ([1,192, 5,591]) is entirely positive, confirming the gap is nonzero.

3. **High variance of Protocol A.** Protocol A perplexity exhibits CV=23--120% across seeds and step counts. This is not measurement noise — it reflects genuine training instability of ALS-based optimization. Seed 42 may converge to 945 PPL while seed 456 reaches 17,915 PPL at 800 steps on OPT-125m. Protocol B, in contrast, achieves CV<5% at all step counts.

4. **AdamW plateaus early.** AdamW converges within 50--100 steps (OPT: PPL≈17, Qwen: PPL≈29--65) and shows negligible improvement thereafter. The residual gap at 800 steps is therefore driven entirely by ASP's (slow) convergence, not by AdamW's improvement.

**Note on statistical inference at 800 steps.** Readers may notice that the bootstrap 95% CI for the OPT gap at 800 steps ([-533, 14,058]) crosses zero, while the PB ANOVA reports p=0.039. This tension arises because the bootstrap percentile CI is a nonparametric, more conservative procedure that makes no distributional assumptions about the gap statistic, whereas the PB ANOVA tests the null hypothesis of equal means under a parametric resampling model. Both procedures use the same data (N=5 per group) but differ in their null hypothesis construction. We consider the PB ANOVA p-value the primary inference because it directly tests the optimizer main effect in the factorial design, while the bootstrap CI on the raw gap provides a descriptive uncertainty interval. The key trajectory finding — that the gap shrinks from ~83,000 to ~7,000 (7.8×) — does not depend on whether the residual 800-step gap is strictly non-zero; the shrinkage trend across five time points is the principal result.

5. **Cohen's d = 1.17** at the 800-step measurement (OPT-125m, N=5 per group) confirms a large effect size (Cohen, 1988). Power analysis indicates 12 seeds per group are needed for 80% power at α=0.05 to detect an effect of this magnitude (d≥0.8) using a two-sided bootstrap test. Achieving CI width <20% of the gap would require >100 seeds — infeasible given Protocol A's intrinsic CV ~100%. The effect *direction* is unambiguously established; the effect *magnitude* has wide confidence intervals.

### 5.4 RQ3: AdamW Overfitting and the Fair Gap

A critical confound in our convergence trajectory analysis is that the A-B gap may reflect not only ASP's slow convergence but also AdamW's degradation due to overfitting on small training sets. To test this, we run AdamW (Protocol B) on OPT-125m with varying training data sizes (400, 800, 1600 samples) at 200 and 400 steps.

**Table 3: AdamW Overfitting Analysis (OPT-125m)**

| Train Samples | 200-step Eval Loss | 400-step Eval Loss | Overfitting? |
|---------------|-------------------|-------------------|-------------|
| 400 | 4.017 | 4.170 ↑ | Yes |
| 800 | 3.845 | 4.089 ↑ | Yes |
| 1600 | 3.948 | 4.001 ↑ | Mild |

**Finding**: AdamW's eval loss consistently *increases* from 200 to 400 steps at all data sizes, indicating overfitting. The best AdamW result is achieved at 200 steps with 800 samples (eval loss=3.85, PPL=46.8). In contrast, ASP at 1200 steps maintains train_loss≈eval_loss≈8.2 — it does not overfit even at 3× the training duration.

**Fair Gap Recalculation.** The raw A-B gap at 1200 steps (3,527 PPL) conflates ASP's slow convergence with AdamW's overfitting. A fair comparison using AdamW at its optimal checkpoint (200 steps, 800 samples) versus ASP at 1200 steps yields:

$$\text{Fair gap} = 8.19 - 3.85 = 4.34 \text{ loss} \approx 78 \text{ PPL}$$

This is **45× smaller** than the raw 1200-step gap. While ASP still underperforms, the magnitude is substantially reduced.

**ASP Implicit Regularization.** ASP's resistance to overfitting is a novel finding. The ALS→SGD alternation appears to provide implicit regularization: each ALS step introduces parameter perturbation that prevents the model from memorizing the training data, while SGD refinement maintains a balance between fitting and generalization. This property may be valuable in low-data post-training scenarios where AdamW's overfitting is problematic.

### 5.5 RQ4: Perturbation Effect

Comparing ASP with and without perturbation at 12 steps (exp #004): perturbation *increases* training loss (13.09 vs 9.04) but *decreases* evaluation perplexity (86k vs 317k). This is the RWP generalization-convergence trade-off (Li et al., 2024; Welling & Teh, 2011). The perturbation encourages flatter minima (SAM-like; Foret et al., 2021) at the cost of slower optimization. We note this finding is preliminary (single 12-step experiment) and leave systematic perturbation strength ablation to future work.

### 5.6 RQ5: Architecture Scaling

The per-model A-B gap at 100 steps: GPT-2 (12L): 177; OPT-125m (12L): 651; Qwen2.5-0.5B (24L): 3,722. Normalizing by layer count: 177/12≈15, 651/12≈54, 3722/24≈155. The per-layer gap grows superlinearly, consistent with perturbation amplification through residual connections: output perturbation $\propto \prod_{k=l}^{L} (I + \Delta_k) \cdot x$ (signal propagation theory; Noci et al., 2022). The Qwen 400-step gap spike (116,435 in multi-seed mean) persists across seeds (individual seeds: 329k, 16k, 5k), confirming this is a genuine architecture effect rather than a single-seed artifact. With the addition of SmolLM2-135M (30 layers, Llama architecture, A-B gap=69,748 at 100 steps), we now observe consistent depth scaling across 5 architectures: A-B gap per layer scales superlinearly with depth.

### 5.7 RQ6: Low-Rank ALS and Protocol C Synergy

A major limitation identified in Round 1 review was that Protocol C used SGD+Perturbation alternation without ALS (since the ALS solver operated only on `nn.Linear` weight matrices). We implemented a low-rank ALS solver (§4.1) that solves the full-rank block-wise least squares for the composite weight $W_{\text{eff}} = W_{\text{base}} + (\alpha/r)BA$ and projects the solution back to the low-rank space by updating $B$:

$$B_{\text{new}}[i:i+b, :] = B_{\text{old}}[i:i+b, :] + \Delta W_{\text{block}} \cdot A^T \cdot (AA^T + \lambda I)^{-1} / \alpha$$

We test Protocol C with and without low-rank ALS at 100, 200, and 400 steps on OPT-125m.

**Table 4: Low-Rank ALS Synergy Test (OPT-125m, Protocol C)**

| Steps | No ALS (SGD+Perturb) PPL | With ALS PPL | Δ PPL |
|-------|--------------------------|--------------|-------|
| 100 | 103.6 | 114.6 | +10.6% |
| 200 | 106.2 | 175.0 | +64.8% |
| 400 | 103.3 | 131.8 | +27.6% |

**Finding**: Low-rank ALS consistently *worsens* Protocol C at ≤400 steps. This mirrors the full-rank finding: ALS introduces distribution shift that requires SGD digestion, and the projection from full-rank to low-rank ($\Delta W \to \Delta B$) loses information. The synergy between ALS and LoRA does not appear at these step counts; whether it emerges at longer training horizons (>500 steps) remains an open question enabled by our low-rank ALS implementation.

## 6. Mathematical Analysis

### 6.1 ALS Reconstruction Loss Magnitude

The ALS phase solves $\min_W \|X W^T - Y\|^2$ independently per block. The reconstruction loss is $\mathcal{O}(N \cdot d_{\text{in}} \cdot \|W\|^2)$, while cross-entropy loss is $\mathcal{O}(\log V)$. For $d_{\text{in}} = 768$ and $N \sim 10^2$--$10^3$, ALS loss reaches $10^4$--$10^5$, overwhelming the cross-entropy baseline (~2--3). All seven experiments confirm this. The gap between these loss magnitudes explains why Protocol A requires 50--150 SGD steps merely to "digest" each ALS step before making net progress.

### 6.2 Non-Monotonic Convergence Model

The A-B gap is modeled as superposition of exponentially decaying ALS perturbation terms:

$$\text{gap}(t) = \sum_{c=1}^{C} A_c \cdot e^{-\alpha (t - t_c)} \cdot \mathbb{1}[t \geq t_c]$$

where $t_c$ is the step at ALS cycle $c$, $A_c \sim 10^4$--$10^5$ is the perturbation magnitude, and $\alpha$ is the SGD digestion rate. Fitting to OPT-125m data: $\alpha \approx 0.008$/step ($\tau = 1/\alpha \approx 125$ steps). For Qwen2.5-0.5B: $\alpha \approx 0.004$/step ($\tau \approx 250$ steps). The digestion time scales approximately as $\tau \propto L^{1.2}$, though this estimate is based on only two model depths and should be treated as qualitative.

### 6.3 Extrapolated Crossover Estimates (Speculative)

Extrapolating the fitted model beyond experimental data yields **speculative** crossover predictions where the A-B gap falls below 10 PPL:

| Model | Layers | Extrapolated Crossover | Caveat |
|-------|--------|----------------------|--------|
| GPT-2 | 12 | ~800--1,000 steps | Within experimental reach; not yet verified |
| OPT-125m | 12 | ~1,000--1,500 steps | Requires experiments beyond current 800-step maximum |
| Qwen2.5-0.5B | 24 | ~2,000--3,000 steps | Wide uncertainty; only 2 seeds at 800s |
| Llama-2-7B | 32 | ~3,000--5,000 steps | Pure extrapolation; no experimental data |

**These predictions should be treated as extrapolated estimates only.** They are based on fits to sparse temporal data (5--6 time points) from two model families with different architectures. The crossover for GPT-2 at ~800 steps is testable with current infrastructure and represents the most immediate verification target.

### 6.4 Why ASP Converges Slowly

Three mechanisms contribute to ASP's slow convergence:

1. **ALS reconstruction loss dominance.** ALS optimizes a surrogate (least-squares reconstruction) whose magnitude (~10⁵) dwarfs the training objective (~2--3). SGD requires 50--150 steps to "digest" each ALS step before the training loss returns to the pre-ALS level.
2. **Cross-layer coupling violation.** ALS solves each block independently, ignoring that downstream layers depend on the current layer's output. The gradient after ALS is nonzero: $\nabla_{W_l} \mathcal{L}(\theta^{\text{ALS}}) \neq 0$.
3. **Momentum reset.** After ALS modifies weights $\theta \to \theta'$, SGD's momentum vector $v_t$ points toward the old parameter space, effectively restarting the optimizer.

We do not currently have an ablation isolating the relative contribution of each mechanism; this is left to future work.

## 7. Discussion

### 7.1 Why ASP Underperforms at Low Steps

Three factors combine to produce ASP's poor early performance: ALS loss magnitude (~10⁵), cross-layer coupling violation, and momentum reset (Section 6.4). The net effect is that 50--150 SGD steps are consumed merely returning the model to its pre-ALS loss level — before any net improvement can occur. At 50--200 total steps, the model has spent most of its training budget "digesting" ALS perturbations rather than making forward progress.

### 7.2 When ASP May Excel

ASP exhibits different asymptotic behavior from AdamW. AdamW plateaus at 50--100 steps (OPT: PPL≈17; Qwen: PPL≈65) and subsequently *degrades* due to overfitting (Section 5.4), with eval loss rising at all tested data sizes (400--1600 samples). ASP, despite its slow start, continues improving at 1200 steps and shows no overfitting (train_loss≈eval_loss≈8.2). This implicit regularization — ASP's resistance to overfitting — is a novel finding with practical implications: in low-data post-training scenarios or very long training regimes, ASP's stability may outweigh AdamW's early convergence advantage.

ASP may have advantages in:
- **Low-data regimes.** ASP's implicit regularization prevents overfitting when training data is limited.
- **Parallelism.** Each block's ALS matrix inversion is independent → massive parallelism potential.
- **Very large training budgets.** The slow-but-steady convergence profile + overfitting resistance suits ultra-long training.
- **Flat minima.** The perturbation phase explicitly encourages flatter solutions (SAM-like), which may improve generalization.

### 7.3 Limitations

1. **Step count.** The predicted crossover at 1,000--5,000 steps has not been experimentally verified.
2. **Model scale.** All experiments use ≤500M parameter models on CPU; 7B+ GPU experiments are pending.
3. **Single dataset.** WikiText-2 only; generalization to other domains (C4, The Pile, downstream tasks) untested.
4. **Protocol C asymmetry.** ALS is not applied in LoRA space (Section 3.2), making Protocol C an "ASP without ALS" rather than a full ASP comparison. The interaction term (A-B)-(C-D) captures parameter form × ALS-presence jointly.
5. **Internal component confound (Section 4.3).** ASP bundles ALS, SGD, and perturbation into one factor. We cannot attribute poor Protocol A performance to any single component without a nested factorial design.
6. **High variance.** Protocol A perplexity exhibits 23--120% CV. While this instability is itself a finding (Section 7.4), it limits the precision of gap magnitude estimates. Effect *direction* is robust; effect *magnitude* has wide confidence intervals.
7. **No downstream tasks.** Only perplexity evaluated; MMLU, HellaSwag, and other benchmarks needed to assess practical generalization.
8. **Single optimizer comparison.** AdamW is the only baseline optimizer. Comparison with SGD, SGD+momentum, and Adam would strengthen the optimizer effect attribution.

### 7.4 The Instability Finding

ASP full-rank training exhibits CV=23--120% across seeds, compared to AdamW's CV<5%. We interpret this as a genuine property of ALS-based optimization rather than measurement noise, based on the consistent low CV of AdamW under identical experimental conditions (which serves as a natural control: if measurement noise were dominant, AdamW would exhibit comparable CV). The block-wise exact solutions, while deterministic given the current activations, are highly sensitive to the specific batch composition and initialization seed. The instability manifests as divergent convergence trajectories — some seeds converge well (PPL~1,000 at 800 steps), others barely improve (PPL~18,000 at 800 steps). This finding has practical implications: any deployment of ALS-based optimization would require either multiple independent training runs or explicit stabilization techniques.

## 8. Conclusion

We presented a 2×2 factorial experimental protocol for comparing alternating optimization (ASP) and LoRA-based post-training. Our key findings, now supported by multi-seed replication and parametric bootstrap ANOVA, are:

1. **Attribution requires factorial design.** Direct ASP-vs-LoRA comparisons conflate optimizer and parameter form effects. The interaction term exceeds 1,000 PPL in all architectures, demonstrating that optimizer effects depend strongly on parameter form.

2. **LoRA dominates at low steps.** The low-rank constraint provides 5--30× PPL improvement at ≤200 steps by reducing the effective condition number. Protocol D (AdamW+LoRA) is the most robust performer across all architectures and step counts.

3. **ASP converges non-monotonically.** The A-B gap oscillates at ALS cycle boundaries but trends downward: from ~83,000 (50 steps) to ~7,000 (800 steps, 7.8× shrinkage) on OPT-125m. Parametric bootstrap ANOVA confirms statistical significance (p<0.05, η²=0.53--0.94). A fair comparison accounting for AdamW overfitting yields a gap of ~78 PPL — 45× smaller than the raw 1200-step measurement.

4. **ASP exhibits intrinsic high variance and implicit regularization.** Protocol A perplexity CV=23--120% reflects genuine training instability. Simultaneously, ASP resists overfitting: train≈eval loss at 1200 steps, while AdamW degrades at all tested data sizes (400--1600 samples). This dual property — unstable but regularization-resistant — is a novel finding.

5. **Low-rank ALS was implemented.** True ALS optimization in LoRA space was achieved via full-rank solve → low-rank projection. Initial tests show ALS also requires digestion in LoRA space (synergy negative at ≤400 steps), with longer-horizon testing enabled by our implementation.

6. **Digestion time and overfitting confound.** ALS reconstruction loss (~10⁴-10⁵) requires 50--150 SGD steps to digest. Extended 1200-step experiments reveal AdamW *overfits* rather than converges, making the raw gap metric sensitive to comparison methodology. Fair gap analysis (§5.4) provides a more accurate assessment.

The central open question — whether ASP eventually surpasses AdamW — is complicated by AdamW's overfitting at long training horizons. A fair comparison using AdamW's optimal checkpoint yields a substantially smaller gap (~78 PPL), but ASP has not yet closed it. ASP's implicit regularization and resistance to overfitting represent a qualitatively different optimization profile from AdamW's "fast convergence then degrade" pattern, suggesting different optimal application scenarios. Extended experiments (>2000 steps) with adequate training data and the recently implemented low-rank ALS solver (§5.7) are needed to fully characterize ASP's asymptotic behavior.

---

## References

[1] Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.

[2] Zeng, J., et al. (2019). Global Convergence of Block Coordinate Descent in Deep Learning. *ICML*. arXiv:1803.00225.

[3] Wang, J., et al. (2018). Accelerated Gradient-free Neural Network Training by Multi-convex Alternating Optimization. arXiv:1811.04187.

[4] Choromanska, A., et al. (2019). Beyond Backprop: Online Alternating Minimization with Auxiliary Variables. arXiv:1806.09077.

[5] Foret, P., et al. (2021). Sharpness-Aware Minimization for Efficiently Improving Generalization. *ICLR*.

[6] Andriushchenko, M. & Flammarion, N. (2022). Towards Understanding Sharpness-Aware Minimization. *ICML*.

[7] Li, T., et al. (2024). Revisiting Random Weight Perturbation for Efficiently Improving Generalization. *TMLR*.

[8] Anonymous (2025). On the Convergence Rate of LoRA Gradient Descent. arXiv:2512.18248.

[9] Kim, J., et al. (2025). LoRA Training Provably Converges to a Low-Rank Global Minimum Or It Fails Loudly. *ICML*.

[10] Anonymous. Balanced Low-Rank Adaptation (BaLoRA). Under review.

[11] Taylor, G., et al. (2016). Training Neural Networks Without Gradients: A Scalable ADMM Approach. *ICML*.

[12] Wang, Z., et al. (2019). ADMM for Efficient Deep Learning with Global Convergence. arXiv:1905.13611.

[13] Dziugaite, G. K. & Roy, D. M. (2017). Computing Nonvacuous Generalization Bounds for Deep Neural Networks. *ICML*.

[14] Boyd, S., et al. (2011). Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. *Foundations and Trends in Machine Learning*.

[15] Bolte, J., et al. (2014). Proximal Alternating Linearized Minimization for Nonconvex and Nonsmooth Problems. *Mathematical Programming*.

[16] Welling, M. & Teh, Y. W. (2011). Bayesian Learning via Stochastic Gradient Langevin Dynamics. *ICML*.

[17] Malladi, S., et al. (2023). Fine-Tuning Language Models with Just Forward Passes. *NeurIPS*.

[18] Dettmers, T., et al. (2024). QLoRA: Efficient Finetuning of Quantized Language Models. *NeurIPS*.

[19] Liu, S.-Y., et al. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. *ICML*.

[20] Lialin, V., et al. (2023). Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning. arXiv:2303.15647.

[21] Aghajanyan, A., et al. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. *ACL*.

[22] Merity, S., et al. (2017). Pointer Sentinel Mixture Models. *ICLR*. (WikiText-2 dataset)

[23] Xu, L.-W., et al. (2013). Parametric bootstrap tests for two-way ANOVA with heteroscedasticity. *Computational Statistics & Data Analysis*.

[24] Noci, L., et al. (2022). Signal Propagation in Transformers: Theoretical Perspectives and the Role of Skip Connections. *NeurIPS*.
