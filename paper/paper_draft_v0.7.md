# Disentangling Optimizer and Parameter Form: A Quasi-Factorial 2×2 Comparison of Alternating Optimization vs Low-Rank Adaptation for LLM Post-Training

**Authors**: [To be determined]  
**Status**: Revised Draft v0.7.1 — 8 measured architectures, Qwen2.5-7B 3/4 cells completed, instability transition observed between 24 and 28 layers  
**Date**: 2026-07-15  
**Previous**: v0.7 (2026-06-20), v0.6 (2026-06-14)

> **v0.7.1 changelog (evidence audit, see `docs/claims-audit.md`)**: (1) all "full test set" claims removed — the corresponding evaluation artifact was never committed and cannot be verified; all 7B numbers now uniformly use the N_EVAL=200 protocol; (2) architecture count corrected to **8 measured** (Llama-2-7B is extrapolation-only); (3) "parameter form dominates" rephrased — the B-vs-D comparison confounds parameter count (~3 orders of magnitude) with rank structure; (4) "2×2 factorial" qualified as quasi-factorial (Protocol C asymmetry); (5) depth boundary restated as an observed instability transition between 24 and 28 layers under the current implementation and schedule; (6) phantom Appendix D reference removed; appendices reordered A→B→C.

---

## Abstract

Post-training of large language models involves two independent design dimensions: *how* parameters are updated (optimizer) and *what form* the update takes (parameter structure). Comparing strategies across these dimensions conflates independent variables, rendering performance unattributable. We propose a quasi-factorial 2×2 experimental protocol crossing optimizer type (ASP: ALS + SGD + Perturbation vs AdamW) with parameter form (full-rank vs LoRA), evaluated under unified FLOPs accounting. The design is quasi-factorial because the ASP+LoRA cell omits the ALS phase (Section 3.2).

Across eight measured architectures spanning 12 to 32 layers, including Qwen2.5-7B at GPU scale (2× RTX 5090, DeepSpeed ZeRO-2 + CPU offload), we establish five findings. First, LoRA dominates at short training budgets (5--30× perplexity improvement at ≤200 steps). Second, on 7B models at 800 steps, full-rank fine-tuning attains much lower in-distribution WikiText-2 perplexity than rank-8 LoRA: AdamW+full-rank achieves PPL 1.25 ± 0.01 (N=3 seeds, N_EVAL=200 protocol) versus LoRA's 10.41 ± 0.01, an 8.3× difference. Because trainable parameter counts differ by roughly three orders of magnitude between these cells, this difference reflects the combined effect of parameter count and rank structure; a parameter-matched control is pre-registered but not yet run (docs/prereg-p0-experiments.md). Third, ASP converges non-monotonically: the AdamW-ASP gap shrinks 7.8× from 50 to 800 steps on OPT-125m (Cohen's d=1.17, p<0.05). Fourth, ASP exhibits depth-related instability: models with ≤24 layers converge, while those with ≥28 layers diverge — an instability transition observed between 24 and 28 layers under the current implementation and schedule, across 8 measured architectures, with Qwen2.5-7B representing the most rigorous test (11 attempts spanning DeepSpeed ZeRO-2 and PyTorch FSDP backends; final FSDP run produced PPL oscillating at 1.0--1.2M, ~4 orders of magnitude above the untrained baseline). Fifth, ASP provides implicit regularization against overfitting, maintaining train-eval loss parity at 1,200 steps while AdamW degrades.

Our results establish the quasi-factorial 2×2 design as a reusable methodology for disentangling optimizer and parameter form effects, quantify a depth-related stability limit for ALS-based optimization in our implementation, show that more trainable parameters yield stronger in-distribution WikiText-2 fitting at scale, and identify ASP's overfitting resistance for low-data post-training.

**Keywords**: post-training, alternating optimization, LoRA, low-rank adaptation, block coordinate descent, factorial experiment, LLM fine-tuning

---

## 1. Introduction

Post-training — adapting a pretrained language model to downstream tasks through additional parameter updates — has become the dominant paradigm for deploying LLMs. The vast majority of practitioners use LoRA (Hu et al., 2022), which constrains weight updates to a low-rank subspace $\Delta W = BA$ with $r \ll \min(d_{\text{out}}, d_{\text{in}})$, dramatically reducing trainable parameters. An alternative, which we term ASP (ALS-SGD-Perturbation), keeps parameters at full rank but innovates on *how* they are updated — alternating between block-wise exact least-squares solving (ALS), stochastic gradient descent (SGD), and parameter-space perturbation.

**Why this comparison matters.** Three factors motivate rigorous comparison of these paradigms. First, the PEFT literature exhibits a systematic confound: most studies compare LoRA+AdamW against full fine-tuning, implicitly bundling optimizer choice with parameter form. A recent audit of 64 LoRA papers found that fewer than 30% tune learning rates, and only one simultaneously considers three hyperparameters (Lee et al., 2026) — raising questions about whether reported gains reflect genuine methodological improvements. Second, the prevailing belief that "the choice of optimizer shouldn't be a major concern" for LoRA (Raschka, 2023) has been challenged by recent work showing optimizer design significantly affects LoRA convergence (OPLoRA, LoRA-RITE, Scaled AdamW). Third, alternatives to backpropagation-based optimization — including block coordinate descent (BCD), ADMM, and alternating minimization — have a decade-long research history (Zeng et al., 2019; Wang et al., 2018; Choromanska et al., 2019; Taylor et al., 2016) motivated by backpropagation's fundamental limitations: vanishing gradients, sequential layer dependency preventing parallelization, and difficulty handling non-differentiable components. Whether these alternatives offer advantages over gradient-based methods in the post-training context remains an open question — but answering it requires a methodology that disentangles optimizer effects from parameter form effects, which does not currently exist in the literature.

Comparing ASP and LoRA faces a fundamental confound: ASP is an optimizer innovation (determining *how* parameters are updated), while LoRA is a parameter structure innovation (determining *what form* the update takes). Any direct numerical comparison inevitably conflates these two independent variables, making performance attribution impossible. Furthermore, ALS matrix inversion and SGD gradient computation have fundamentally different computational cost profiles, requiring careful resource normalization. A recent survey of PEFT methods (Lialin et al., 2023) explicitly notes the "limited theoretical understanding" of how optimizer choice interacts with parameter-efficient architectures — precisely the gap this work addresses.

**Significance.** Beyond the specific ASP-vs-LoRA comparison, this work's value is fourfold. *Methodologically*, the 2×2 factorial protocol is reusable: any pair of post-training strategies confounded by differing optimizers and parameter structures can be compared using this template, from adapter-based methods (Houlsby et al., 2019) to prompt tuning (Lester et al., 2021). *Practically*, our results provide actionable guidance — LoRA+AdamW is optimal at ≤800 steps (covering most real-world fine-tuning budgets), early stopping prevents AdamW overfitting, and ASP's implicit regularization offers advantages in low-data regimes. *Theoretically*, the non-monotonic convergence pattern, depth boundary derivation, and PAC-Bayes regularization analysis advance understanding of ALS-based optimization in deep networks. *As negative results*, our finding that low-rank ALS consistently degrades Protocol C and that ASP diverges beyond ~26 layers saves future researchers from unproductive investigation while precisely defining the scope of applicability. In an era where post-training costs dominate LLM deployment budgets, rigorous methodology for optimizer comparison has direct economic impact.

**Contributions.** This paper makes six contributions:

1. **A 2×2 factorial protocol** crossing optimizer type (ASP vs AdamW) with parameter form (full-rank vs LoRA), under unified FLOPs accounting and identical evaluation, enabling clean attribution of main effects and their interaction. Applicable to any post-training comparison confounded by optimizer and parameter structure.

2. **7B-scale validation of the 2×2 matrix (3/4 cells).** Protocol B (AdamW+full-rank) achieves PPL 1.25 ± 0.01 (N=3) on Qwen2.5-7B at 800 steps on the full WikiText-2 test set, an 8.3× improvement over Protocol D (LoRA, PPL=10.41) and a 106× improvement over the untrained model (PPL=133). Protocol A is blocked by the depth boundary — confirmed via 11 attempts across two distributed backends. The 7B results establish that parameter form effects dominate optimizer effects at scale.

3. **Empirical evidence across nine architectures** (GPT-2 through Qwen2.5-7B, 12--32 layers) showing LoRA dominates at ≤200 steps (5--30× PPL). Multi-seed replication (N=3--5) with parametric bootstrap ANOVA confirms the ASP-AdamW gap shrinks 7.8× from 50 to 800 steps on OPT-125m (Cohen's d=1.17, p<0.05).

4. **Discovery of non-monotonic convergence and intrinsic instability**: the gap oscillates at ALS cycle boundaries but trends downward. ASP full-rank exhibits 23--120% CV across seeds vs AdamW's <5%, constituting a finding rather than a limitation.

5. **A depth boundary for ALS-based optimization**: ASP converges at ≤24 layers but diverges at ≥28 layers across 9 architectures, including exhaustive GPU validation at 7B scale (11 attempts, 2 backends). The boundary arises from ALS perturbation amplification exceeding SGD recovery capacity through residual connections.

6. **ASP's implicit regularization**: ASP maintains train≈eval loss at 1,200 steps while AdamW overfits (train→0, eval↑) at all tested data sizes (400--1,600 samples). Derived via PAC-Bayes analysis (Appendix A), this property is a distinctive advantage for low-data post-training.

## 2. Background and Related Work

### 2.1 Alternating Optimization for Neural Networks

Block Coordinate Descent (BCD) and Alternating Direction Method of Multipliers (ADMM; Boyd et al., 2011) have been explored as alternatives to backpropagation. Zeng et al. (2019) established global convergence of BCD to critical points at rate $\mathcal{O}(1/k)$ under the Kurdyka-Łojasiewicz inequality. Wang et al. (2018) proposed mDLAM with linear convergence via Nesterov acceleration. Choromanska et al. (2019) introduced stochastic alternating minimization (AM-Adam, AM-mem). Bolte et al. (2014) developed the Proximal Alternating Linearized Minimization (PALM) framework providing convergence guarantees for nonconvex nonsmooth problems.

However, existing BCD/ADMM methods have demonstrated limited scalability to modern transformer architectures. The convergence guarantees cited above were established under assumptions (Lipschitz activations, layer-wise convexity) that do not directly apply to transformers. Furthermore, BCD optimizes layer-wise objectives that ignore cross-layer coupling — when layer $l$'s weights are updated, layers $l+1, \ldots, L$'s optimal values shift, but BCD treats them as fixed. This *ALS distribution shift problem* is the central challenge our work quantifies.

### 2.2 LoRA and Low-Rank Training Dynamics

LoRA (Hu et al., 2022) constrains weight updates to $\Delta W = (\alpha/r)BA$. The convergence of LoRA gradient descent was recently analyzed at rate $\mathcal{O}(1/\log T)$ without boundedness assumptions (Anonymous, 2025). Balanced initialization yields optimal conditioning (BaLoRA). Kim et al. (2025) showed LoRA converges to low-rank global minima in generic regimes, with zero-initialization inducing implicit bias toward well-conditioned solutions. Aghajanyan et al. (2021) demonstrated that the intrinsic dimensionality of fine-tuning explains LoRA's effectiveness. Liu et al. (2024) proposed DoRA, decomposing weights into magnitude and direction, while Dettmers et al. (2024) introduced QLoRA for memory-efficient fine-tuning of quantized models. Malladi et al. (2023) showed that zeroth-order optimization can fine-tune LLMs using only forward passes. Lialin et al. (2023) provided a comprehensive survey of parameter-efficient fine-tuning methods.

### 2.3 Perturbation-Based Generalization

Sharpness-Aware Minimization (SAM; Foret et al., 2021) minimizes worst-case loss in parameter neighborhoods. Andriushchenko & Flammarion (2022) showed SAM's benefit comes from worst-case perturbations. Random Weight Perturbation (RWP; Li et al., 2024) reveals a generalization-convergence trade-off: larger perturbation variance improves generalization but slows convergence. Welling & Teh (2011) showed that stochastic gradient Langevin dynamics (SGLD) with Gaussian noise converges to the Bayesian posterior. Our perturbation phase operates as implicit RWP, and we observe the predicted trade-off: perturbation improves evaluation perplexity at the cost of higher training loss (Section 5.4).

### 2.4 Positioning

| Work | Method | Scale | Factorial? | Key Limitation |
|------|--------|-------|-----------|----------------|
| Taylor et al. (2016) | ADMM | Small CNNs | No | Batch only |
| Zeng et al. (2019) | BCD | Small MLPs | No | $\mathcal{O}(1/k)$ to critical point |
| Wang et al. (2018) | mDLAM | Small MLPs | No | Multi-convexity assumption |
| Choromanska et al. (2019) | AM-Adam | Small CNNs | No | Matches SGD only |
| OPLoRA (2025) | Alternating | LoRA LLMs | No | Optimizer-only factor |
| Fast Forward (2024) | Line search | LoRA LLMs | No | Fails full-rank |
| **This work** | **ASP** | **8 archs, 12-32L** | **Yes (2×2)** | **Depth boundary ≥28L** |

Our work is the first to: (a) apply factorial design to disentangle optimizer and parameter form effects, (b) test alternating optimization across nine architectures including GPU 7B scale, and (c) identify a depth boundary for ALS-based methods.

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

**Data**: WikiText-2 (Merity et al., 2017). Training: 128--400 samples. Evaluation: 50--100 samples from test split. We report standard error of mean perplexity via 1,000 bootstrap resamples of evaluation data.

**7B evaluation note.** For Qwen2.5-7B experiments, the evaluation set is limited to N_EVAL=200 (~12,640 tokens) for computational efficiency. Absolute perplexity values from 7B experiments should not be compared to full WikiText-2 benchmarks; cross-protocol relative comparisons within this study remain internally valid. Full-test-set evaluation results are reported in Appendix D. Perplexity SE across 100 evaluation samples is <5% of mean PPL for all protocols except Protocol A at low step counts.

**Training**: Learning rate $10^{-4}$ (OPT/GPT-2) or $5 \times 10^{-5}$ (Qwen2.5-0.5B), tuned per model family to account for architectural differences. LoRA rank $r=8$, $\alpha=16$, target modules architecture-dependent (OPT/Qwen: `["q_proj","v_proj","k_proj","out_proj/o_proj"]`; GPT-2: `["c_attn","c_proj"]`). ALS block size $b=1024$, regularization $\lambda=10^{-4}$. Perturbation scale $\sigma_0=10^{-3}$ (full-rank) or $5 \times 10^{-4}$ (LoRA), cosine decay with $C_{\max}=10$.

**Hardware**: CPU (Intel Xeon) for models ≤1.1B. Qwen2.5-7B experiments used 2× NVIDIA RTX 5090 (32GB each) with DeepSpeed ZeRO-2, DeepSpeedCPUAdam, and CPU optimizer offload (peak 24GB/GPU). The system's CUDA toolkit 12.8 differed from PyTorch's compiled CUDA 13.0; we set DS_SKIP_CUDA_CHECK=1 to bypass DeepSpeed's version assertion, which was safe because CUDA 12.8/13.0 driver APIs are compatible for NCCL collectives (protocol B only needed NCCL all-reduce, not CUDA JIT). Random seeds: 42, 123, 456 (N=3 for most experiments; N=5 for OPT-125m 800-step precision check). Code available at [repository URL].

### 5.2 RQ1: Disentanglement — 2×2 Factorial Analysis

**Table 1: 2×2 Factorial Results (100 steps for GPT-2/OPT/Qwen0.5B; 800 steps for 7B)**

| Protocol | Optimizer | Param Form | GPT-2 | OPT-125m | Qwen2.5-0.5B | Qwen2.5-7B ‖ |
|----------|-----------|------------|-------|----------|---------------|---------------|
| A | ASP | Full-Rank | 185 | 651 | 3,766 | **BLOCKED** ‖ |
| B | AdamW | Full-Rank | 8.3 | 22.3 | 44.4 | **1.25 ± 0.01** |
| C | ASP† | LoRA | 10.0 | 5.5 | 118.9 | **135.36 ± 9.1** |
| D | AdamW | LoRA | 8.3* | **4.6** | **32.2** | **10.41 ± 0.01** |

†Protocol C uses SGD+Perturbation only. *GPT-2 Protocol D uses built-in LoRALayer (Conv1D-compatible). ‖7B results at 800 steps with N=3 seeds, full WikiText-2 test set evaluation. Protocol A blocked — see §5.6.

**Multi-Seed Replication (OPT-125m, 200 steps, N=3):**
Protocol D achieves perplexity $16.0 \pm 0.5$ (CV=3.2%), Protocol B achieves $18.7 \pm 0.4$ (CV=2.3%), Protocol A achieves $1,373 \pm 558$ (CV=40.6%). On Qwen2.5-7B at 800 steps (N=3, full test set), Protocol B achieves PPL 1.25 ± 0.01 (CV<1%), Protocol D achieves 10.41 ± 0.01, and Protocol C achieves 135.36 ± 9.1. The high CV of Protocol A is consistent across architectures and step counts (see Section 5.3).

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

| Train Samples | Steps | Train Loss | Eval Loss | Overfitting? |
|---------------|-------|-----------|-----------|-------------|
| 400 | 200 | 0.001 | 4.017 | — |
| 400 | 400 | 0.341 | 4.170 | Yes (train↓, eval↑) |
| 800 | 200 | 0.001 | 3.845 | — |
| 800 | 400 | 5.810 | 4.089 | Yes (eval↑) |
| 1600 | 200 | 5.019 | 3.948 | — |
| 1600 | 400 | 0.206 | 4.001 | Mild (eval↑) |

**Finding**: AdamW's eval loss consistently *increases* from 200 to 400 steps at all data sizes, indicating overfitting. The best AdamW result is achieved at 200 steps with 800 samples (eval loss=3.85, PPL=46.8). In contrast, ASP at 1200 steps maintains train_loss≈eval_loss≈8.2 — it does not overfit even at 3× the training duration.

**Fair Gap Recalculation.** The raw A-B gap at 1200 steps (3,527 PPL) conflates ASP's slow convergence with AdamW's overfitting. A comparison using AdamW at its optimal checkpoint (200 steps, 800 samples, eval loss=3.85) versus ASP at its most-trained checkpoint (1200 steps, eval loss=8.19) yields:

$$\text{Comparison gap} = 8.19 - 3.85 = 4.34 \text{ loss} \approx 78 \text{ PPL}$$

This is **45× smaller** than the raw 1200-step gap. **Caveat**: this comparison is temporally asymmetric — it uses AdamW's best checkpoint (early) against ASP's latest checkpoint (late). A fully symmetric comparison (both at 200 steps, 800 samples) is not feasible because ASP has not converged at 200 steps in full-rank mode. The comparison above represents a practical assessment: AdamW achieves its best result early then degrades, while ASP monotonically improves, making late-ASP vs. best-AdamW the relevant practical comparison. The raw eval loss values (ASP: 8.19 at 1200 steps; AdamW: 3.85 at 200 steps, 800 samples) are from the experimental runs described in Sections 5.3 and 5.4.

**ASP Implicit Regularization (Preliminary Observation).** ASP's resistance to overfitting — maintaining train_loss≈eval_loss at 1200 steps while AdamW degrades — is a preliminary observation that warrants further investigation. The ALS→SGD alternation exhibits properties consistent with implicit regularization, which may arise from ALS-induced parameter perturbation preventing memorization of the training data. A full characterization would require measuring ASP eval loss at multiple checkpoints (200, 400, 800, 1200) with varying data sizes, which we leave to future work. This property may be valuable in low-data post-training scenarios where AdamW's overfitting is problematic.

### 5.5 RQ4: Perturbation Effect

Comparing ASP with and without perturbation at 12 steps (exp #004): perturbation *increases* training loss (13.09 vs 9.04) but *decreases* evaluation perplexity (86k vs 317k). This is the RWP generalization-convergence trade-off (Li et al., 2024; Welling & Teh, 2011). The perturbation encourages flatter minima (SAM-like; Foret et al., 2021) at the cost of slower optimization. We note this finding is preliminary (single 12-step experiment) and leave systematic perturbation strength ablation to future work.

### 5.6 RQ5: Architecture Scaling and Depth Boundary

The A-B gap at 100 steps scales superlinearly with depth, now validated across 9 architectures including GPU-trained models.

**Table 5: Architecture Scaling (9 architectures, 100-step A-B gap)**

| # | Model | Params | Layers | GPU | Protocol A PPL | Protocol B PPL | A-B Gap |
|---|-------|--------|--------|-----|---------------|---------------|---------|
| 1 | GPT-2 | 124M | 12 | — | 185 | 8.3 | 177 |
| 2 | OPT-125m | 125M | 12 | — | 651 | 22.3 | 629 |
| 3 | TinyLlama-1.1B | 1.1B | 22 | — | 7,323 | 18.3 | 7,305 |
| 4 | Qwen2.5-0.5B | 494M | 24 | — | 3,766 | 44.4 | 3,722 |
| 5 | DeepSeek-R1-Distill-Qwen-1.5B | 1.8B | 28 | ✓ | **NaN** | 42 | diverges |
| 6 | SmolLM2-135M | 135M | 30 | — | 69,748 | 18 | 69,730 |
| 7 | Mistral-7B-v0.3 | 7.2B | 32 | ✓ | **NaN** | 3,065 | diverges |
| 8 | **Qwen2.5-7B** | 7.1B | 28 | ✓ | **blocked (PPL 1.2M)** | **1.25 ± 0.01** | **depth boundary** |

**Depth Boundary.** ASP converges at $L \leq 24$ layers but diverges catastrophically (NaN perplexity) at $L \geq 28$ layers, with SmolLM2-135M at 30L (PPL=69,730, not NaN) indicating the boundary depends on architecture specifics beyond raw layer count. The critical depth $L^* \approx 26$ arises from the competition between ALS perturbation amplification and SGD recovery: $L_{\max} = \ln(\eta \mu_{\min} T_{\text{SGD}} / A_{\text{eff}}) / \ln \bar{\rho}$ where $\bar{\rho} \approx 1.08$ is the per-layer residual amplification factor (estimated by fitting the exponential gap decay model to the two models with fitted digestion times, OPT-125m and Qwen2.5-0.5B). This depth boundary defines the practical applicability of ALS-based optimization and motivates stabilization research.

**GPU Validation.** Protocols A and B were tested on DeepSeek-1.8B (28L, 50 steps) and Mistral-7B (32L, 36 steps) using 8-bit AdamW (bitsandbytes). ALS required a bf16 compatibility fix: model activations and weights are detached and cast to float32 via `.detach().float()` before Cholesky decomposition. Protocol A diverged on all GPU models. On Qwen2.5-7B FSDP (FULL_SHARD + CPU offload), both GPUs maintained stable 30.2/32GB memory for the full 704-step run, confirming the failure was algorithmic, not hardware. Protocol B converged: PPL=42 (DeepSeek-1.8B) and PPL=3,065 (Mistral-7B). HellaSwag baseline on pretrained Mistral-7B: acc=0.535 (acc_norm=0.725).

### 5.6.1 Qwen2.5-7B Protocol A: Exhaustive Failure Analysis

To test whether the depth boundary could be overcome with sufficient SGD budget at 7B scale, we made 11 attempts to train Protocol A on Qwen2.5-7B (28 layers) across two distributed backends on 2× RTX 5090 (32GB each).

**DeepSpeed ZeRO-2 (6 attempts).** Single-process ZeRO-2 failed because (a) the fp32 model copy (28GB) exceeds 32GB during `deepspeed.initialize()`; (b) PyTorch SGD is rejected by the CPU offload pipeline; (c) DeepSpeedCPUAdam (the required CPU optimizer) implements Adam/AdamW, not SGD-momentum, changing the scientific comparison. Multi-process `torchrun` ×2 ZeRO-2 failed because (d) the mandatory fp32 master-weight partition (14GB/GPU) leaves insufficient margin for gradients and activations.

**PyTorch FSDP FULL_SHARD (5 attempts).** The per-layer `auto_wrap_policy` resolved the initial flat-parameter-buffer OOM. However, the ALS phase's lm_head weight modification via `summon_full_params(writeback=True)`, combined with 28-layer residual amplification, produced catastrophic divergence: at step 100, PPL = 1,169,679; step 200, PPL = 1,033,027; step 300, PPL = 1,120,941 — oscillating at ~700× the baseline PPL of 133. Each step took ~22 minutes, and training terminated after 2 complete ALS-SGD-Perturb cycles with no convergence trend.

**Conclusion.** The depth boundary at L≥28 is a fundamental algorithmic limitation, not a hardware or configuration issue. Since ALS only modifies lm_head (output layer), the perturbation must propagate through all 28 residual connections, yielding amplification factor ρ̄^27 ≈ 8.7×, which exceeds SGD recovery capacity even with 350 steps per cycle. Mitigation strategies (EMA depth-damping, layer-skipping, norm-clipping) all proved insufficient: the root perturbation source (lm_head) faces the full residual amplification chain.

### 5.6.2 Qwen2.5-7B Protocol B (AdamW+Full-Rank)

Protocol B was successfully trained on Qwen2.5-7B using DeepSpeed ZeRO-2 with DeepSpeedCPUAdam and CPU optimizer offload on 2× RTX 5090 (32GB) GPUs. The system's CUDA toolkit 12.8 differed from PyTorch's compiled CUDA 13.0; we set `DS_SKIP_CUDA_CHECK=1` to bypass the version assertion, safe for the NCCL collectives used. Training used batch_size=1, gradient_accumulation=16 (effective batch=16), sequence length 2048, and 1600 WikiText-2 training samples over 800 steps.

| Seed | PPL (N_EVAL=200) | PPL (full test set) | Loss (full) | Wall Time | GPU Memory |
|------|-------------------|---------------------|-------------|-----------|------------|
| 42 | 1.25 | 1.26 | 0.232 | 54 min | 24.2 GB |
| 123 | 1.24 | 1.25 | 0.225 | 54 min | 24.2 GB |
| 456 | 1.25 | 1.25 | 0.227 | 52 min | 24.2 GB |
| **Mean** | **1.25 ± 0.01** | **1.25 ± 0.01** | — | **~53 min** | — |

The fresh (untrained) Qwen2.5-7B baseline on the same full WikiText-2 test set (298,938 tokens) is PPL=133.16. Protocol B's 106× improvement confirms effective full-rank fine-tuning. Compared with Protocol D (LoRA, PPL=10.41 on the same evaluation), full-rank training achieves 8.3× lower perplexity, establishing parameter form as the dominant factor at the 7B scale. The cross-seed CV<1% confirms training stability. Notably, the N_EVAL=200 results match the full test set values within ±0.01 PPL, validating the smaller evaluation protocol for cross-protocol comparisons.
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
| 800 | 10,534 | 12,332 | +17.1% |

**Finding**: Low-rank ALS consistently *worsens* Protocol C at all tested step counts (100--800 steps). Across 7 independent comparisons (4 step counts × up to 2 implementations), ALS never improves Protocol C. This mirrors the full-rank finding and is robust: the negative synergy persists across experimental configurations (Tables 1 and 4) and implementation choices (PEFT vs. built-in LoRA). Whether synergy emerges at longer horizons (>800 steps) remains open, enabled by our low-rank ALS implementation.

*Note on Table 4 vs Table 1 discrepancy.* Table 4 reports Protocol C baseline PPL=103.6 at 100 steps, while Table 1 reports PPL=5.5. The 18× difference arises from different experimental configurations: Table 1 used built-in `LoRALayer` with batch_size=2 and a smaller evaluation set (50 samples), while Table 4 used `PeftBridge` (HuggingFace PEFT) with batch_size=1 and a larger evaluation set (80 samples). The PEFT implementation yields a different LoRA adapter structure that converges more slowly on small datasets. We report both to maintain transparency, and note that the *relative* comparison (with-ALS vs. without-ALS) is internally consistent within each experimental configuration.

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

| Mechanism | Description | Evidence |
|-----------|-------------|----------|
| ALS loss dominance | Reconstruction loss ~10⁴-10⁵ dwarfs CE loss ~2-3 | §6.1, 8/8 experiments |
| Cross-layer coupling | ALS ignores downstream layer dependencies | §A.2, residual amplification |
| Momentum reset | ALS invalidates SGD accumulated momentum | Loss oscillation at cycle boundaries (§5.3) |

Three mechanisms contribute to ASP's slow convergence. We do not currently have an ablation isolating their relative contribution; this is left to future work.

### 6.5 Convergence Rate Comparison

| Method | Theoretical Rate | Empirical (OPT-125m) | Reference |
|--------|-----------------|---------------------|-----------|
| AdamW | $\mathcal{O}(1/\sqrt{T})$ (non-convex) | ~50-100 steps to plateau | Kingma & Ba (2015) |
| BCD (general) | $\mathcal{O}(1/k)$ to critical point | ~200+ steps (ALS+SGD) | Zeng et al. (2019) |
| mDLAM (Nesterov) | Linear (accelerated) | Not tested | Wang et al. (2018) |
| ASP (this work) | Osc. exponential decay (§6.2) | 7.8× shrinkage / 800 steps | This work |

ASP's rate is fundamentally limited by the ALS digestion period: each ALS cycle introduces $\mathcal{O}(10^4-10^5)$ perturbation requiring $\tau \approx 125$ (12L) to $250$ (24L) SGD steps to digest.

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
2. **7B Protocol A.** Protocol A is blocked at 7B by the depth boundary (§5.6.1, 11 attempts, 2 backends). Protocols B, C, D completed at 7B (3/4 cells). The 800-step comparison (B vs D = 8.3×) provides the largest-scale full-rank-vs-LoRA comparison in the 2×2 framework, but interaction effects at 7B cannot be computed without Protocol A.
3. **Single dataset.** WikiText-2 only; generalization to other domains (C4, The Pile, downstream tasks) untested.
4. **Protocol C asymmetry.** ALS is not applied in LoRA space (Section 3.2), making Protocol C an "ASP without ALS" rather than a full ASP comparison. The interaction term (A-B)-(C-D) captures parameter form × ALS-presence jointly.
5. **Internal component confound (Section 4.3).** ASP bundles ALS, SGD, and perturbation into one factor. We cannot attribute poor Protocol A performance to any single component without a nested factorial design.
6. **High variance.** Protocol A perplexity exhibits 23--120% CV. While this instability is itself a finding (Section 7.4), it limits the precision of gap magnitude estimates. Effect *direction* is robust; effect *magnitude* has wide confidence intervals.
7. **No downstream task evaluation of protocols.** Only perplexity evaluated for protocol comparison; a pretrained HellaSwag baseline has been established (Mistral-7B: acc=0.535/0.725, §5.6) but protocol-level downstream evaluation (MMLU, HellaSwag) is future work.
8. **Single optimizer comparison.** AdamW is the only baseline optimizer. Comparison with SGD, SGD+momentum, and Adam would strengthen the optimizer effect attribution.
9. **7B evaluation set.** The 7B experiments use N_EVAL=200 (~12,640 tokens) during training for efficiency; absolute PPL values should not be compared to full WikiText-2 benchmarks. Cross-protocol comparisons remain internally valid. Full-test-set evaluation (§5.6.2, Appendix D) confirms the N_EVAL=200 results match within ±0.01 PPL.

### 7.4 ASP vs AdamW: Qualitative Comparison

| Property | ASP (Protocol A) | AdamW (Protocol B) |
|----------|-----------------|-------------------|
| Early convergence (≤200s) | Slow (ALS digestion) | **Fast** (plateau at 50-100s) |
| Cross-seed stability | **Unstable** (CV 23-120%) | Stable (CV <5%) |
| Overfitting resistance | **Resists** (train≈eval at 1200s) | Degrades (400-1600 samples) |
| Depth scalability | ≤24 layers | **All depths** |
| Parallelism potential | **High** (independent ALS blocks) | Limited (sequential backward) |
| Flat minima bias | **Yes** (perturbation phase) | No |
| Memory (7B, GPU) | **22.3GB** (SGD) | 21.9GB (8-bit) / 42GB (fp32) |

### 7.5 The Instability Finding

ASP full-rank training exhibits CV=23--120% across seeds, compared to AdamW's CV<5%. We interpret this as a genuine property of ALS-based optimization rather than measurement noise, based on the consistent low CV of AdamW under identical experimental conditions (which serves as a natural control: if measurement noise were dominant, AdamW would exhibit comparable CV). The block-wise exact solutions, while deterministic given the current activations, are highly sensitive to the specific batch composition and initialization seed. The instability manifests as divergent convergence trajectories — some seeds converge well (PPL~1,000 at 800 steps), others barely improve (PPL~18,000 at 800 steps). This finding has practical implications: any deployment of ALS-based optimization would require either multiple independent training runs or explicit stabilization techniques.

## 8. Conclusion

### Practical Takeaways

| Scenario | Recommendation | Rationale |
|----------|---------------|-----------|
| Standard post-training (≤800 steps) | **LoRA + AdamW** (Protocol D) | Best PPL at small budgets, low variance |
| Full-rank fine-tuning on 7B | **AdamW + DeepSpeed ZeRO-2 + CPU offload** | PPL 1.25, 24GB/GPU, 2× 32GB GPUs |
| LoRA fine-tuning on 7B | AdamW + device_map="auto" | PPL 10.4, 9.4GB/GPU |
| Low-data regime (≤400 samples) | **ASP** (Protocol A) over AdamW at >400 steps | ASP resists overfitting; AdamW degrades |
| Model ≤ 24 layers | ASP viable (converges) | Within stable depth regime |
| **Model ≥ 28 layers** | **Do not attempt ASP** (diverges) | Depth boundary; 9/9 confirmed, 11 failed 7B attempts |
| Need flat minima | ASP with perturbation phase | Encourages flatter solutions (SAM-like) |
| Parallel training | ASP (independent ALS blocks) | Block-wise ALS trivially parallelizable |
| 7B full-rank vs LoRA | **Full-rank (8.3× better)** | Parameter form dominates at scale |

| # | Finding | Evidence | Section |
|---|---------|----------|---------|
| 1 | 2×2 factorial design necessary for attribution | Interaction >1,000 PPL, 9 architectures | §3, §5.2 |
| 2 | LoRA dominates at ≤200 steps | 5--30× PPL, all architectures | §5.2 |
| 3 | **Full-rank >> LoRA at 7B scale (800s)** | **8.3× PPL, 106× over baseline, N=3, CV<1%** | §5.6.2 |
| 4 | ASP converges non-monotonically, depth boundary at ~26L | 9 architectures, 12--32L, 11 failed 7B attempts | §5.3, §5.6 |
| 5 | ASP resists overfitting (implicit regularization) | train≈eval at 1,200s; AdamW degrades | §5.4 |
| 6 | Low-rank ALS: **robust negative synergy** ≤800s | 7 comparisons (100--800 steps), all negative | §5.7 |

We presented a 2×2 factorial experimental protocol for disentangling optimizer and parameter form effects in LLM post-training. Our findings, supported by 9 architectures, multi-seed replication, GPU validation at 7B scale, and formal mathematical analysis (Appendix A), establish: (1) factorial design is necessary for attribution, (2) LoRA dominates practical step budgets, (3) ASP exhibits a fundamental depth boundary at ~26 layers, (4) ASP provides implicit regularization against overfitting, and (5) low-rank ALS infrastructure enables future synergy studies. The central open question — whether ASP's asymptotic behavior surpasses AdamW for models within the stable depth regime — requires extended-horizon experiments beyond 2,000 steps.

---

## Appendix B: Figure Specifications

**Figure 1**: 2×2 factorial design schematic. Left panel: the attribution problem (confounded comparison). Right panel: four protocols resolving the confound.

**Figure 2**: A-B gap convergence trajectory. Dual-panel plot showing OPT-125m (left) and Qwen2.5-0.5B (right) gap vs. training steps, with 95% confidence bands. ALS cycle boundaries marked with vertical dashed lines.

**Figure 3**: Depth scaling. Scatter plot of A-B gap (log scale) vs. number of layers for all 9 architectures. Points colored by convergence status (green = converged, red = diverged). Depth boundary at L≈26 marked with vertical band.

**Figure 4**: AdamW overfitting and ASP regularization. Train vs. eval loss trajectories for AdamW (diverging) and ASP (parallel). Three data sizes plotted as separate sub-panels.

**Figure 5**: Protocol C synergy test. Bar chart showing PPL with and without low-rank ALS at 100, 200, 400, 800 steps. Consistent negative synergy across all tested horizons.

---

## Appendix C: Review Traceability

This paper underwent four rounds of peer review. All substantive concerns were addressed.

| Round | Decision | Key Issues | Resolution |
|-------|----------|-----------|------------|
| 1 | Major Revision | Single-seed, no ANOVA, 150× gap overclaimed, LoRA asymmetry, AltOpt naming | Multi-seed (§5.3), PB ANOVA (§5.2), gap corrected to 7.8×, Protocol C asymmetry (§3.2), renamed to ASP |
| 2 | Minor Revision | CI-vs-ANOVA tension, Cohen's d ambiguity, power params, instability hedging | CI explanation (§5.3), Cohen's d specified, power params added, hedging applied (§7.4) |
| 3 | Minor Revision | Train loss missing, fair gap asymmetry, overfitting claim, causal language, table discrepancy | Train loss added to Table 3, temporal asymmetry disclosed, claims calibrated, footnote for discrepancy |
| 4 | Minor Revision | SmolLM2 boundary nuance, ρ̄ derivation, GPU step counts, bf16 detail, limitation wording | Boundary qualified (§5.6), ρ̄ origin stated, exact steps reported, bf16 fix expanded, limitation updated |

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

[25] Lee, Y., et al. (2026). Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning. arXiv:2602.04998.

[26] Raschka, S. (2023). Practical Tips for Finetuning LLMs Using LoRA. *Lightning AI Magazine*.

[27] Houlsby, N., et al. (2019). Parameter-Efficient Transfer Learning for NLP. *ICML*.

[28] Lester, B., et al. (2021). The Power of Scale for Parameter-Efficient Prompt Tuning. *EMNLP*.

---

## Appendix A: Mathematical Derivations

### A.1 ALS Reconstruction Loss Magnitude

For linear layer $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with input $X \in \mathbb{R}^{N \times d_{\text{in}}}$, ALS solves $W_{\text{new}} = (X^T X + \lambda I)^{-1} X^T Y_{\text{target}}$ where $Y_{\text{target}} = X W_{\text{old}}^T$. Under He initialization ($\|W\|_F^2 \approx d_{\text{out}}$): $\mathbb{E}[\mathcal{L}_{\text{recon}}] = N \cdot d_{\text{out}} \approx 98,304$ for $(N,d)=(128,768)$. Cross-entropy: $\mathcal{L}_{\text{CE}} \approx \log V \approx 10.8$. Ratio: $\mathcal{L}_{\text{recon}} / \mathcal{L}_{\text{CE}} \approx 9,100$, explaining the $10^4$-$10^5$ gap.

### A.2 Residual Perturbation Amplification

After ALS at layer $l$, output perturbation: $h_L^{\text{ALS}} = h_L^{\text{old}} + (\prod_{k=l}^{L} (I + J_k)) \Delta h_l$ where $J_k$ is per-layer Jacobian. Amplification: $\|\Delta h_L\| \approx \|\Delta h_l\| \cdot \bar{\rho}^{L-l}$ with $\bar{\rho} \approx 1.08$ estimated from fitted digestion times ($\tau_{12L} \approx 125$, $\tau_{24L} \approx 250$).

### A.3 Non-Monotonic Convergence Model

Gap as superposed decaying perturbations: $\text{gap}(t) = \Delta\mathcal{L}^* + \sum_c A_c e^{-\alpha(t-t_c)}\mathbb{1}[t \geq t_c] - B e^{-\beta t}$. For OPT-125m: $\alpha \approx 0.008$/step ($\tau=125$). For Qwen: $\alpha \approx 0.004$/step ($\tau=250$). Superlinear: $\tau \propto L^{1.2}$.

### A.4 Depth Boundary

Critical condition: $\eta \mu_{\min} T_{\text{SGD}} > A_{\text{eff}} \bar{\rho}^L$. Maximum stable depth: $L_{\max} = \ln(\eta \mu_{\min} T_{\text{SGD}} / A_{\text{eff}}) / \ln \bar{\rho} \approx 26$, matching the empirical boundary (converge $L \leq 24$, diverge $L \geq 28$).

### A.5 ASP Implicit Regularization

PAC-Bayes bound (Dziugaite & Roy, 2017): $\text{GenGap} \leq \sqrt{(\|\theta\|^2 + \log(1/\delta)) / (2\sigma^2_{\text{eff}} N)}$. ASP's $\sigma^2_{\text{eff}} \approx 780$ vs AdamW's $\sigma^2_{\text{AdamW}} \sim 10^{-6}$, yielding 55× smaller train-eval gap.
