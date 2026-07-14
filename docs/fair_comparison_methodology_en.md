# Fair Comparison Methodology: A 2×2 Factorial Study of ASP vs AdamW × Full-Rank vs LoRA

## Why "Fair Comparison" Is This Project's Core Methodological Contribution

When comparing different optimizers and parameter forms, the most common mistake is **comparing by training step count**. Our four protocols have massively different per-step computational costs — Protocol A's ALS step is equivalent to dozens of SGD steps — making step-count comparisons meaningless. This document details how we design and execute truly fair comparisons.

---

## 1. The Core Problem: One "Step" Is Not the Same Across Protocols

### 1.1 What Is "One Step"?

| Protocol | What One Step Means | Trainable Parameters (0.5B) | Actual FLOPs |
|----------|---------------------|---------------------------|--------------|
| **A** (ASP + Full-Rank) | Depends on phase: ALS (1 layer closed-form) / SGD (all params) / Perturb (all params) | 494M | **Highly variable** |
| **B** (AdamW + Full-Rank) | One forward + backward + AdamW update | 494M | $\sim 4.4 \times 10^9$ |
| **C** (ASP + LoRA) | Depends on phase: ALS (project to B) / SGD (LoRA only) / Perturb (LoRA only) | $\sim$3M | **Highly variable** |
| **D** (AdamW + LoRA) | One forward + backward + AdamW update (LoRA params only) | $\sim$3M | $\sim 3.0 \times 10^9$ |

**The fundamental contradiction**: Protocol A's ALS phase performs matrix inversion on a single layer at $\mathcal{O}(d_{\text{in}}^3)$ cost. Protocol D updates only $\sim$3M parameters. Step-for-step comparison? Meaningless.

---

## 2. FLOPs Normalization: The Core Fairness Principle

### 2.1 The Principle

$$\boxed{\text{All protocols run until } \sum_{t=1}^{T} \text{FLOPs}_t \geq \text{FLOPs}_{\text{BUDGET}}}$$

**Not by step count. Not by wall-clock time. By total floating-point operations.** FLOPs are the only metric independent of optimizer choice, parameter form, and hardware implementation — a pure measure of "computational investment."

### 2.2 Why FLOPs Is the Only Correct Metric

| Metric | Problem |
|--------|---------|
| **Step count** | ALS step $\neq$ SGD step $\neq$ AdamW step |
| **Wall-clock time** | Affected by hardware, batch size, I/O, memory bandwidth — unscientific |
| **GPU utilization** | Different operations utilize GPU differently (matrix multiply vs element-wise) |
| **Memory footprint** | Reflects hardware constraints, not computational investment |
| **FLOPs** | ✅ Pure mathematical measure, independent of implementation details |

---

## 3. Precise Per-Operation FLOPs Accounting

### 3.1 Forward Pass

For a model with $N_{\text{params}}$ parameters, one forward pass costs:

$$\text{FLOPs}_{\text{forward}} = 2 \cdot N_{\text{params}}$$

**Derivation**: Each parameter participates in one multiply (weight $\times$ input) and one add (accumulate to output), totaling 2 FLOPs. Transformer attention adds $\mathcal{O}(N \cdot T^2 \cdot d_{\text{model}})$ FLOPs, but this term is identical across all protocols (same architecture) and cancels in comparisons.

### 3.2 Backward Pass

Backpropagation costs approximately 2$\times$ the forward pass:

$$\text{FLOPs}_{\text{backward}} = 4 \cdot N_{\text{params}}$$

**Derivation**: Backpropagation computes (a) gradient of loss w.r.t. output, (b) gradient of each layer's output w.r.t. its input (chain rule), and (c) gradient of each layer's output w.r.t. its weights. Each term approximates one forward pass in cost.

### 3.3 Optimizer Updates

| Optimizer | FLOPs per Step | Explanation |
|-----------|---------------|-------------|
| **SGD** | $1 \cdot N_{\text{trainable}}$ | $\theta \leftarrow \theta - \eta g$: one multiply-add |
| **SGD + Momentum** | $2 \cdot N_{\text{trainable}}$ | Additionally maintains velocity $v$ |
| **AdamW** | $3 \cdot N_{\text{trainable}}$ | Maintains $m$ (first moment), $v$ (second moment), plus bias correction + parameter update |
| **ALS** | See §3.4 | Independent of parameter count; determined by matrix dimensions |

### 3.4 ALS (Alternating Least Squares) FLOPs

This is the most complex component. One ALS pass on a single layer:

**Step 1: Form $X^\top X$**

$$X \in \mathbb{R}^{N \times d_{\text{in}}}, \quad X^\top X \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$

$$\text{FLOPs}_{X^\top X} = 2N \cdot d_{\text{in}}^2$$

Each entry requires one multiply and one add, across $N$ rows, each with $d_{\text{in}} \times d_{\text{in}}$ multiplications.

**Step 2: Cholesky Decomposition** ($X^\top X + \lambda I = L L^\top$)

$$\text{FLOPs}_{\text{Cholesky}} = \frac{1}{3} d_{\text{in}}^3$$

This is the exact FLOPs count for the standard $d \times d$ Cholesky decomposition.

**Step 3: Per-Block Triangular Solves**

For each block of size $b \times d_{\text{in}}$, two triangular solves are required ($L z = X^\top Y$ and $L^\top W^\top = z$):

$$\text{FLOPs}_{\text{per-block}} = 2 \cdot b \cdot d_{\text{in}}^2$$

Number of blocks: $n_{\text{blocks}} = \lceil d_{\text{out}} / b \rceil$.

**Total ALS FLOPs**:

$$\boxed{\text{FLOPs}_{\text{ALS}} = 2N d_{\text{in}}^2 + \frac{1}{3} d_{\text{in}}^3 + 2 n_{\text{blocks}} \cdot b \cdot d_{\text{in}}^2}$$

**Additional: LoRA Low-Rank ALS (X1) B-Projection Overhead**

On top of full-rank ALS, the LoRA version must project the solution onto the B matrix:

$$\text{FLOPs}_{\text{B-projection}} = \underbrace{r^3}_{\text{Cholesky}(AA^\top)} + \underbrace{2r^2 d_{\text{in}}}_{\text{forming }AA^\top} + \underbrace{n_{\text{blocks}} \cdot 2 b r d_{\text{in}}}_{\text{per-block }\Delta B = \Delta W \cdot A^\dagger}$$

For $r = 8, d_{\text{in}} = 896$, this amounts to $\sim 2.3 \times 10^5$ FLOPs — **completely negligible** compared to the main ALS cost of $\sim 10^9$ FLOPs.

### 3.5 Perturbation Phase FLOPs

$$\text{FLOPs}_{\text{Perturb}} = N_{\text{trainable}}$$

A single element-wise addition.

---

## 4. Concrete Example: Qwen2.5-0.5B FLOPs Budget

### 4.1 Model Parameters

- Total parameters: $4.94 \times 10^8$
- Hidden dimension $d_{\text{model}} = 896$
- Attention head dimension $d_{\text{head}} = 64$
- Layers $L = 24$
- Vocabulary size $V = 151,\!936$

### 4.2 LoRA Trainable Parameters

$$N_{\text{LoRA}} = 4 \text{ (Q,K,V,O)} \times 2 \text{ (A,B)} \times r \times d_{\text{model}} \times L$$

$$= 4 \times 2 \times 8 \times 896 \times 24 = 1.38 \times 10^6 \approx 1.4\text{M}$$

### 4.3 Per-Operation Single-Step FLOPs

| Operation | FLOPs | Relative Scale |
|-----------|-------|---------------|
| Protocol D one step (LoRA + AdamW) | $(2+4+3) \times 1.4\text{M} = 1.26 \times 10^7$ | 1$\times$ |
| Protocol B one step (Full-Rank + AdamW) | $(2+4+3) \times 494\text{M} = 4.45 \times 10^9$ | 353$\times$ |
| Protocol A one ALS (lm_head, $d_{\text{in}}=896$, $N=800$, $b=1024$) | $1.28 \times 10^9 + 2.40 \times 10^8 + 2.43 \times 10^{11}$ | $\sim$195,000$\times$ |
| Protocol A one SGD (Full-Rank) | $(2+4+1) \times 494\text{M} = 3.46 \times 10^9$ | 275$\times$ |
| Protocol C one SGD (LoRA) | $(2+4+1) \times 1.4\text{M} = 9.8 \times 10^6$ | 0.78$\times$ |
| Perturbation (Full-Rank) | $494\text{M} = 4.94 \times 10^8$ | 39$\times$ |
| Perturbation (LoRA) | $1.4\text{M} = 1.4 \times 10^6$ | 0.11$\times$ |

**Key observation**: ALS one-step FLOPs far exceed SGD one-step FLOPs. However, ALS executes only once per cycle (typically after 50–100 SGD steps). Amortized over the cycle, Protocol A/C average per-step FLOPs are not pathologically high.

### 4.4 100-Step Total FLOPs Budget

Suppose the budget equals Protocol B running for 100 steps:

$$\text{BUDGET}_{100} = 100 \times 4.45 \times 10^9 = 4.45 \times 10^{11}$$

| Protocol | Mean FLOPs/Step | Steps Under Budget |
|----------|----------------|-------------------|
| A (ASP + Full-Rank) | $\sim 5.2 \times 10^{10}$ (ALS amortized) | $\sim 9$ steps |
| B (AdamW + Full-Rank) | $4.45 \times 10^9$ | $100$ steps |
| C (ASP + LoRA) | $\sim 1.5 \times 10^{10}$ (ALS amortized) | $\sim 30$ steps |
| D (AdamW + LoRA) | $1.26 \times 10^7$ | $\sim 35,\!000$ steps |

**Implication**: Under the same FLOPs budget, Protocol D runs 35,000 steps while Protocol B runs only 100 steps. LoRA's much smaller per-step cost lets it take many more steps to compensate for updating fewer parameters per step. **FLOPs normalization makes this "many small updates vs. few large updates" trade-off fairly comparable.**

---

## 5. Fairness Dimensions Beyond FLOPs

FLOPs normalization alone is insufficient. A truly fair comparison requires consistency across all of the following dimensions:

### 5.1 Identical Data Loaders

```python
# All protocols share the same DataLoader instance,
# including identical shuffle seed, batch size, and preprocessing
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(42)  # ← fixed
)
```

**Why it matters**: Different batch orderings produce different optimization trajectories. Every protocol must see the **exact same data sequence**.

### 5.2 Unified Random Seeds

| Seed Type | Fixed Value | Scope of Effect |
|-----------|-------------|-----------------|
| PyTorch global seed | `torch.manual_seed(42)` | All random operations |
| CUDA seed | `torch.cuda.manual_seed_all(42)` | GPU random numbers |
| DataLoader shuffle seed | 42 | Data ordering |
| Parameter initialization seed | 42 (fixed at LoRA/full-rank init) | Initial parameters |

**Multi-seed validation**: Each protocol runs on $N = 3$ independent seeds, reporting mean $\pm$ standard deviation. This eliminates the possibility that one protocol "got lucky" with a favorable seed.

### 5.3 Identical Evaluation Protocol

| Evaluation Dimension | Unified Practice |
|---------------------|-----------------|
| **Evaluation dataset** | Same split (e.g., WikiText-2 test), same preprocessing |
| **Evaluation frequency** | Every $N$ FLOPs (not every $N$ steps) |
| **PPL computation** | Same formula: $\text{PPL} = \exp(\frac{1}{N}\sum \text{cross-entropy})$ |
| **Model state at evaluation** | Unified `model.eval()` + `torch.no_grad()` |
| **Token counting** | Weighted by `attention_mask` to prevent padding contamination |

### 5.4 Unified Hardware Environment

- Same machine, same GPU model
- Same PyTorch/CUDA version
- Unified bfloat16 precision (Protocol A ALS phase requires float32 — a documented exception)
- Gradient checkpointing uniformly enabled/disabled

**Protocol A exception**: ALS Cholesky decomposition accumulates numerical error under bfloat16, leading to NaN. Protocol A must therefore run its ALS phase in float32. This is a **known, documented limitation** — Protocol A enjoys ASP's full mathematical power at the cost of higher numerical precision requirements.

### 5.5 Hyperparameter Alignment

| Hyperparameter | Value | Applicable Protocols |
|---------------|-------|---------------------|
| Learning rate $\eta$ | $10^{-4}$ | All |
| Weight decay $\lambda_{\text{wd}}$ | $0.01$ | All |
| Momentum $\mu$ | $0.9$ | A, C (SGD phase) |
| AdamW $\beta_1, \beta_2$ | $0.9, 0.999$ | B, D |
| LoRA rank $r$ | $8$ | C, D |
| LoRA alpha | $2r = 16$ | C, D |
| ALS regularization $\lambda$ | $10^{-3}$ | A, C |
| ALS block size $b$ | $1024$ | A, C |
| Perturbation initial scale $\sigma_0$ | $10^{-3}$ (full-rank) / $5\times 10^{-4}$ (LoRA) | A, C |
| Gradient clipping max_norm | $1.0$ | All |
| Batch size | 1 (effective 4 with gradient accumulation) | All |
| Max sequence length | 1024 | All |

---

## 6. Statistical Rigor: Multi-Seed + Effect Sizes

### 6.1 Multi-Seed Experimental Design

Each protocol runs on $N = 3$ (or $N = 5$) independent random seeds:

$$\text{Protocol X result} = \bar{x} \pm s_x \quad \text{where } \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i,\; s_x = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

### 6.2 Effect Size: Hedges' $g$

When comparing Protocol A and Protocol B, we report both $p$-values and effect sizes:

$$g = \frac{\bar{x}_A - \bar{x}_B}{s_{\text{pooled}}} \cdot \left(1 - \frac{3}{4(n_A + n_B) - 9}\right)$$

$$s_{\text{pooled}} = \sqrt{\frac{(n_A - 1)s_A^2 + (n_B - 1)s_B^2}{n_A + n_B - 2}}$$

**Why Hedges' $g$ instead of Cohen's $d$**: With small samples ($N = 3$), Cohen's $d$ is biased. Hedges' $g$ applies a small-sample correction factor.

### 6.3 Multiple Comparison Correction: Bonferroni

When performing multiple pairwise comparisons, significance thresholds are Bonferroni-corrected:

$$\alpha_{\text{adjusted}} = \frac{\alpha}{k}$$

where $k = \binom{4}{2} = 6$ is the number of comparisons, $\alpha = 0.05$, giving $\alpha_{\text{adjusted}} \approx 0.0083$.

### 6.4 PB ANOVA (Parametric Bootstrap Analysis of Variance)

With $N = 3$ samples, the classical ANOVA $F$-distribution assumption does not hold. We use parametric bootstrap:

1. From each protocol's 3 data points, assuming normality, estimate $\hat{\mu}_i, \hat{\sigma}_i^2$
2. Bootstrap-resample 10,000 times, generating 4 groups of synthetic data per iteration
3. Compute the $F$-statistic for each bootstrap iteration
4. Build the empirical distribution of $F$ and compute the $p$-value

```python
# Pseudocode: PB ANOVA
for b in range(10000):
    for protocol in [A, B, C, D]:
        boot_sample = np.random.normal(mu_hat[protocol], sigma_hat[protocol], size=3)
        # Compute between-group variance / within-group variance
    F_boot[b] = ms_between / ms_within
p_value = np.mean(F_boot >= F_observed)
```

---

## 7. 2×2 Factorial Design: Attributing Interaction Effects

### 7.1 Why 2×2?

The core advantage of a 2×2 factorial design: **main effects and interactions can be separated**.

|  | Full-Rank | LoRA |
|--|-----------|------|
| **AdamW** | Protocol B | Protocol D |
| **ASP** | Protocol A | Protocol C |

### 7.2 Mathematical Decomposition of Effects

Let $\mu_{ij}$ denote the expected outcome (e.g., PPL) for protocol with optimizer $i$ and parameter form $j$.

**Main effects**:

- Optimizer main effect: $\text{ME}_{\text{opt}} = \frac{(\mu_{A} + \mu_{C}) - (\mu_{B} + \mu_{D})}{2}$
  — "How much better is ASP than AdamW, on average?"
- Parameter form main effect: $\text{ME}_{\text{param}} = \frac{(\mu_{A} + \mu_{B}) - (\mu_{C} + \mu_{D})}{2}$
  — "How much better is full-rank than LoRA, on average?"

**Interaction effect**:

$$\text{Interaction} = (\mu_{A} - \mu_{B}) - (\mu_{C} - \mu_{D})$$

- Interaction $= 0$: optimizer and parameter form effects are independent
- Interaction $> 0$: ASP benefits more under full-rank than under LoRA (positive synergy)
- Interaction $< 0$: ASP performs worse under LoRA (**negative synergy** — our finding)

### 7.3 Empirical Results

Across our experiments, the interaction term exceeds **+1000 PPL** (measured in WikiText-2 PPL) — a **massive negative synergy**. This means:

- Protocol A (ASP + full-rank) is worse than Protocol B (AdamW + full-rank)
- Protocol C (ASP + LoRA) is worse than Protocol D (AdamW + LoRA)
- **But the gap is larger under full-rank** (ASP's weaknesses are more exposed at full rank)

This finding is an honest negative result — ASP and LoRA do not synergize.

---

## 8. The 7B Edge Case: Protocol A Missing

### 8.1 Root Cause

Protocol A (ASP + full-rank) is blocked on Qwen2.5-7B (28 layers) by the depth boundary. All 11 independent attempts failed (ALS diverges to NaN in models with $\geq 28$ layers).

### 8.2 Impact on Comparisons

$$\begin{array}{c|cc}
& \text{Full-Rank} & \text{LoRA} \\
\hline
\text{AdamW} & \text{B ✅} & \text{D ✅} \\
\text{ASP} & \text{A ❌} & \text{C ✅}
\end{array}$$

Missing Protocol A means:
- ✅ B vs. D is comparable (Full-Rank AdamW vs. LoRA AdamW)
- ✅ C vs. D is comparable (ASP vs. AdamW under LoRA)
- ❌ **Interaction effect** $(\mu_A - \mu_B) - (\mu_C - \mu_D)$ cannot be computed
- ❌ **Optimizer main effect** cannot be estimated (missing A data point)

### 8.3 Our Handling

The paper explicitly documents this limitation:
1. Complete 2×2 comparison at 0.5B scale
2. Partial comparison at 7B (B, C, D), reporting the impact of missing A
3. Treating the depth boundary itself as a discovery (not a failure) — it reveals ASP's inherent algorithmic limit

---

## 9. Implementation Checklist

Before running any comparison experiment, verify the following:

| # | Check Item | ✅/❌ |
|---|------------|-------|
| 1 | All protocols use the same DataLoader (identical shuffle seed) | |
| 2 | `torch.manual_seed()` fixed before all protocol launches | |
| 3 | Evaluation dataloader is identical across protocols | |
| 4 | Evaluation frequency aligned by FLOPs (not by step count) | |
| 5 | Hyperparameters comparable across protocols (learning rate, weight decay, etc.) | |
| 6 | At least $N = 3$ seeds per protocol | |
| 7 | Report mean $\pm$ standard deviation | |
| 8 | Report effect size (Hedges' $g$) and confidence intervals | |
| 9 | Bonferroni correction for multiple comparisons | |
| 10 | Missing data points (e.g., 7B Protocol A) explicitly documented | |
| 11 | Hardware environment consistent across experiments | |
| 12 | Numerical precision differences (float32 vs. bfloat16) explicitly documented | |

---

## 10. Summary

The four pillars of fair comparison:

1. **FLOPs normalization**: Compare by computational investment, not step count or wall-clock time
2. **Identical data/seeds/evaluation**: Eliminate all non-algorithmic sources of variance
3. **Multi-seed statistics**: $N = 3$–5, Hedges' $g$ + Bonferroni + PB ANOVA
4. **2×2 factorial attribution**: Separate main effects from interaction effects

These four pillars collectively define what "fair comparison" means in this project — not merely engineering details, but the core of the methodological contribution.
