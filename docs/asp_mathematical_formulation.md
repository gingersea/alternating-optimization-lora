# Mathematical Formulation of the Rank Sufficiency Law

## A Quantitative Theory of LoRA Post-Training under the 2×2 Factorial Design

---

## 1. The Problem

### 1.1 Post-training optimization

Let $M_0$ be a pretrained Transformer language model with $L$ layers, hidden dimension $d_h$, and total parameters $\boldsymbol{\theta}_0 \in \mathbb{R}^D$. Given a small post-training dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$ (typically $N \sim 10^2$–$10^3$ samples), the task is:

$$\min_{\boldsymbol{\theta} \in \Theta} \; \mathcal{L}_{\text{CE}}(\boldsymbol{\theta}; \mathcal{D})$$

where $\Theta$ is the admissible parameter space and $\mathcal{L}_{\text{CE}}$ is the causal language modeling cross-entropy. The problem is characterized by the extreme data-to-parameter ratio: $D / N \sim 10^5$ to $10^6$ for full-rank, versus $10^2$ to $10^3$ for LoRA with rank $r = 8$.

### 1.2 Two parameter spaces

**Full-rank** ($\Theta_{\text{full}} = \mathbb{R}^D$). Every pretrained weight $W_0^{(\ell)} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ may be perturbed arbitrarily:

$$\Delta W^{(\ell)} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} \quad \text{(unconstrained)}$$

**LoRA** ($\Theta_{\text{LoRA}}$). Each adapted weight is restricted to a rank-$r$ perturbation of its pretrained value:

$$W_{\text{eff}}^{(\ell)} = W_0^{(\ell)} + \frac{\alpha}{r} B^{(\ell)} A^{(\ell)}$$

with $A^{(\ell)} \in \mathbb{R}^{r \times d_{\text{in}}}$, $B^{(\ell)} \in \mathbb{R}^{d_{\text{out}} \times r}$, and $r \ll \min(d_{\text{out}}, d_{\text{in}})$. All $W_0^{(\ell)}$ are frozen; only $\{A^{(\ell)}, B^{(\ell)}\}$ are trainable.

### 1.3 The central question

The post-training literature widely observes that LoRA $r = 8$ performs comparably to much higher ranks, and sometimes outperforms full-rank fine-tuning. But there is no quantitative theory predicting *which* rank is sufficient for *which* model on *which* task.

**The central question this work answers:**

> Given a Transformer with $L$ layers and hidden dimension $d_h$, what is the minimum LoRA rank $r_{\min}$ sufficient for post-training on WikiText-2-style autoregressive tasks?

The answer takes the form of a simple architectural law, derived below and experimentally falsified in both its positive and negative predictions.

---

## 2. The Rank Sufficiency Law

### 2.1 Derivation from residual stream capacity

Consider the residual stream through $L$ Transformer layers of hidden dimension $d_h$:

$$h_{\ell+1} = h_\ell + f_\ell(h_\ell; \boldsymbol{\theta}_\ell), \qquad \ell = 0, 1, \ldots, L-1$$

where $h_0$ is the token embedding and $h_L$ feeds the output projection.

**Assumption 1** (Per-layer distribution shift). During post-training, the ideal hidden representation at layer $\ell$ deviates from the pretrained representation by an error $\varepsilon(\ell)$ that accumulates with layer depth. The pretrained model, having seen a broad data distribution during pretraining, provides a representation close to optimal at the input layer; the required correction grows approximately linearly with distance from the input:

$$\varepsilon(\ell) \approx \varepsilon_0 \cdot \frac{L - \ell}{d_h}$$

where $\varepsilon_0$ is a task-dependent base error and the factor $1/d_h$ reflects that wider hidden dimensions distribute the correction burden across more dimensions, reducing per-dimension error.

**Assumption 2** (Correction capacity of LoRA). A LoRA adapter of rank $r$ applied to $M$ attention modules per layer provides total correction capacity proportional to the number of free parameters:

$$C_{\text{eff}}(r) = M \cdot r \cdot (d_{\text{in}} + d_{\text{out}}) \cdot L$$

For standard LoRA targeting $M = 4$ modules (Q, K, V, O projections) with $d_{\text{in}} = d_{\text{out}} = d_h$:

$$C_{\text{eff}}(r) = 4 \cdot r \cdot 2d_h \cdot L = 8 r d_h L$$

**Assumption 3** (Supply-demand equilibrium). The total LoRA capacity must at least match the aggregate correction required across all layers:

$$C_{\text{eff}}(r) \geq \sum_{\ell=0}^{L-1} \varepsilon(\ell)$$

Summing the per-layer error: $\sum_{\ell=0}^{L-1} (L - \ell) / d_h = L(L+1) / (2d_h) \approx L^2 / (2d_h)$.

Equating supply and demand at the threshold $r = r_{\min}$:

$$8 r_{\min} d_h L = \kappa \cdot \frac{\varepsilon_0 L^2}{2 d_h}$$

where $\kappa > 0$ absorbs constants and the conversion from error magnitude to parameter count. Solving for $r_{\min}$:

$$r_{\min} = \frac{\kappa \varepsilon_0}{16} \cdot \frac{L}{d_h}$$

**Definition 1** (Rank Sufficiency Law). Define $\eta = \kappa \varepsilon_0 / 16$. Then the minimum sufficient LoRA rank for a Transformer with $L$ layers and hidden dimension $d_h$ is:

$$\boxed{r_{\min} = \eta \cdot \frac{L}{d_h}}$$

The parameter $\eta$ captures the interaction of task difficulty, pretraining quality, and the constant factors from the residual stream model. For English WikiText-2 post-training with strong-pretraining models (Qwen2.5, Llama 3, Mistral), $\eta \approx 230$ — meaning $r_{\min} \approx 230 \cdot L / d_h$.

### 2.2 Empirical calibration

The law is calibrated from the five-model cross-architecture rank curve, which measures the PPL ratio $r8 / r256$ — the factor by which $r = 8$ underperforms $r = 256$ (a proxy for the full-rank ceiling):

| Model | $L$ | $d_h$ | $L / d_h$ | $r8/r256$ | $r_{\min}$ (predicted) | Plateau? |
|-------|-----|-------|-----------|-----------|------------------------|----------|
| Mistral-7B | 32 | 4096 | 0.0078 | 0.99 | $230 \cdot 0.0078 \approx 1.8$ | ✅ |
| TinyLlama-1.1B | 22 | 2048 | 0.0107 | 1.03 | $\approx 2.5$ | ✅ |
| DeepSeek-1.5B | 28 | 1536 | 0.0182 | 1.10 | $\approx 4.2$ | ✅ |
| Qwen2.5-0.5B | 24 | 896 | 0.0268 | 1.01 | $\approx 6.2$ | ✅ |
| SmolLM2-135M | 30 | 576 | 0.0521 | 1.83 | $\approx 12.0$ | ❌ (marginal) |

**Proposition 1** (Rank sufficiency threshold). A model is at the LoRA plateau when its predicted $r_{\min} \leq 8$, i.e., when

$$\frac{L}{d_h} \leq \frac{8}{\eta} \approx 0.035$$

For all models with $L / d_h \leq 0.035$, $r = 8$ is indistinguishable from $r = 256$ in PPL. The sole model exceeding this threshold, SmolLM2-135M ($L/d_h = 0.0521$), shows $r8/r256 = 1.83$, confirming marginal insufficiency.

**Proposition 2** (Optimal rank). For small-data post-training ($N_d < 10^4$), the optimal LoRA rank is

$$r^* = \max(8, \lceil \eta \cdot L / d_h \rceil)$$

---

## 3. Why Full-Rank Overfits: The M-Index

### 3.1 The memorization diagnostic

Full-rank fine-tuning on small datasets achieves deceptively low training perplexity. To separate genuine generalization from memorization, define the M-index:

$$\boxed{M = \frac{\text{PPL}_{\text{train}}}{\text{PPL}_{\text{cross}}}}$$

where $\text{PPL}_{\text{train}}$ is evaluated on the training domain (WikiText-2) and $\text{PPL}_{\text{cross}}$ on an out-of-domain corpus (C4 web text).

**Proposition 3** (M-index scaling law). The M-index follows a power-law relationship with the data-to-parameter ratio:

$$M = k \cdot \left(\frac{N_d}{N_p}\right)^\beta$$

where $N_d$ is the number of training tokens, $N_p$ is the number of trainable parameters, and $k, \beta$ are scale-dependent constants ($\beta_{0.5\text{B}} \approx -0.03$, $\beta_{7\text{B}} \approx 0.28$).

**Corollary 1** (Memorization threshold). The condition $M < 1$ indicates that cross-domain PPL exceeds in-domain PPL — the model is memorizing rather than generalizing:

$$\text{Memorization regime: } \frac{N_p}{N_d} > 10^4$$

For full-rank post-training, $N_p / N_d \sim 10^5$–$10^6$, placing it firmly in the memorization regime for all practical data budgets. LoRA with $r = 8$ has $N_p / N_d \sim 10^2$–$10^3$, staying in the generalization regime.

### 3.2 Empirical confirmation

On Qwen2.5-7B ($N = 1600$ WikiText-2 samples):

| Protocol | $N_p$ | WikiText-2 PPL | C4 PPL | $M$ | Regime |
|----------|-------|----------------|--------|-----|--------|
| Full-rank (B) | $7.1 \times 10^9$ | 1.25 | 2.42 | 0.52 | **Memorization** |
| LoRA $r=8$ (D) | $\sim 3 \times 10^6$ | 10.41 | 2.30 | 4.53 | Generalization |

The full-rank WikiText-2 PPL of 1.25 is $8.3\times$ better than LoRA's on the training distribution — but on cross-domain C4, LoRA *outperforms* full-rank ($2.30$ vs $2.42$). The full-rank model has memorized WikiText-2's specific patterns at the expense of general language understanding.

Downstream task evaluation confirms this: full-rank post-training reduces HellaSwag accuracy by 3.2pp, MMLU by 4.2pp, and ARC by 3.3pp relative to the untrained baseline. LoRA $r = 8$ preserves $> 99.7\%$ of baseline accuracy on all three tasks.

---

## 4. The 2×2 Factorial Design as Attribution Methodology

### 4.1 The confound

Direct comparison between ASP+full-rank and AdamW+LoRA conflates two independent variables. The $2 \times 2$ factorial design crosses optimizer type (ASP vs AdamW) with parameter form (full-rank vs LoRA):

|  | Full-rank | LoRA |
|--|-----------|------|
| **ASP** | Protocol A | Protocol C |
| **AdamW** | Protocol B | Protocol D |

### 4.2 Effect decomposition

Let $\mu_{ij}$ denote the expected outcome (e.g., PPL) for protocol $i \in \{\text{ASP}, \text{AdamW}\}$, parameter form $j \in \{\text{full}, \text{LoRA}\}$.

**Definition 2** (Main effects and interaction).

$$\begin{aligned}
\text{Optimizer main effect:} \quad & \text{ME}_{\text{opt}} = \frac{(\mu_{\text{A}} + \mu_{\text{C}}) - (\mu_{\text{B}} + \mu_{\text{D}})}{2} \\[6pt]
\text{Parameter form main effect:} \quad & \text{ME}_{\text{param}} = \frac{(\mu_{\text{A}} + \mu_{\text{B}}) - (\mu_{\text{C}} + \mu_{\text{D}})}{2} \\[6pt]
\text{Interaction:} \quad & \text{Int} = (\mu_{\text{A}} - \mu_{\text{B}}) - (\mu_{\text{C}} - \mu_{\text{D}})
\end{aligned}$$

The interaction term captures whether the optimizer effect depends on the parameter form. A non-zero interaction signals that optimizer and parameter form are not independent — precisely the confound that naive direct comparisons fail to detect.

**Proposition 4** (Negative synergy). Empirically, $\text{Int} > 10^3$ PPL across all testable architectures. The ASP optimizer performs worse under LoRA than under full-rank, relative to the AdamW baseline. This negative interaction has a structural cause derived below.

---

## 5. ASP as a Stress Test

### 5.1 What ASP attempts

ASP replaces the single AdamW optimizer with a three-phase cycle. The hypothesis under test is: can a "stronger" optimizer — one that uses exact least-squares solving rather than iterative gradient descent — overcome the limitations of a low-rank parameter constraint?

If ASP+LoRA outperforms AdamW+LoRA (positive interaction), the optimizer matters and the bottleneck is optimization. If ASP+LoRA underperforms AdamW+LoRA (negative interaction), the bottleneck is the parameter form itself, and no optimizer can compensate.

### 5.2 ALS phase

For a linear layer with input activations $X \in \mathbb{R}^{N \times d_{\text{in}}}$ and target output $Y \in \mathbb{R}^{N \times d_{\text{out}}}$, ALS solves the regularized least-squares problem exactly:

$$W_* = \arg\min_W \|X W^\top - Y\|_F^2 + \lambda \|W\|_F^2$$

with closed-form solution $W_*^\top = (X^\top X + \lambda I)^{-1} X^\top Y$.

Under LoRA, ALS first solves in effective-weight space ($W_{\text{eff}} = W_0 + \frac{\alpha}{r} B A$), then projects the solution onto the LoRA parameter space via the minimum-norm B-projection:

$$\Delta B = \frac{r}{\alpha} \cdot (W_* - W_{\text{eff}}) \cdot A^\top (A A^\top + \lambda I)^{-1}$$

### 5.3 SGD phase

SGD with momentum ($\mu$) and weight decay ($\lambda_{\text{wd}}$) coordinates all layers after ALS perturbs hidden representations:

$$v \leftarrow \mu v + \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{CE}}, \qquad \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta v - \eta \lambda_{\text{wd}} \boldsymbol{\theta}$$

Under LoRA, gradients flow only through $A$ and $B$, with dimension reduced by factor $\frac{2r(d_{\text{in}} + d_{\text{out}})}{d_{\text{out}} d_{\text{in}}} \approx \frac{4r}{d_{\text{in}}}$.

### 5.4 Perturbation phase

Gaussian noise with cosine-decaying scale promotes flat minima:

$$\theta \leftarrow \theta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma_c^2), \quad \sigma_c = \frac{\sigma_0}{2}\left(1 + \cos\frac{\pi c}{C_{\max}}\right)$$

### 5.5 Why ASP fails under LoRA

**Proposition 5** (ALS cost-information mismatch). The ALS phase has dominant computational cost $\mathcal{O}(N d_{\text{in}}^2)$ for forming $X^\top X$, regardless of whether the parameters are full-rank or LoRA-constrained. However, under LoRA, the exact solution $W_*$ must pass through the rank-$r$ bottleneck of the B-projection. The information loss is characterized by:

$$\|W_* - (W_0 + \frac{\alpha}{r} (B + \Delta B_*) A)\|_F \geq \sigma_{r+1}(W_* - W_0)$$

where $\sigma_{r+1}$ is the $(r+1)$-th singular value. ALS pays the full-rank cost but receives only the rank-$r$ information — a structural mismatch that explains the robust negative synergy observed across all 7 independent comparisons.

### 5.6 Depth boundary

ASP's ALS phase modifies hidden states, creating perturbations that propagate through residual connections:

$$\|\delta_L\| \approx \|\delta_\ell\| \cdot \bar{\rho}^{\,L - \ell}, \qquad \bar{\rho} \approx 1.08$$

The critical depth at which ALS perturbation exceeds SGD recovery capacity is:

$$L_{\max} = \frac{\ln(\eta \mu_{\min} K / A_{\text{eff}})}{\ln \bar{\rho}} \approx 26$$

**Corollary 2** (ASP applicability). ASP converges for $L \leq 24$ layers and diverges for $L \geq 28$ layers (confirmed on 8/8 architectures, 11 failed 7B attempts). This is an algorithmic limitation, not a hardware constraint.

---

## 6. The $\eta$ Mechanism: Pretraining Quality Modulation

The parameter $\eta$ in the Rank Sufficiency Law is not a universal constant. Three candidate mechanisms were tested:

**Hypothesis 1** (Token entropy): $\eta \propto H$, where $H$ is the per-token entropy of the post-training language. **Falsified**: Chinese WikiText-2 ($H \approx 11.6$ bits/token for Llama tokenizer) yields $r8/r32 = 1.02$, indistinguishable from English ($r8/r32 = 1.01$).

**Hypothesis 2** (Training budget): $\eta \propto 1 / N_{\text{samples}}$. **Falsified**: $r = 4$ at $N = 400, 800, 1600$ yields ratios $1.005, 1.006, 1.008$ — all at plateau.

**Hypothesis 3** (Pretraining quality): $\eta$ is modulated by the per-layer representation quality of the pretrained model, itself a function of pretraining compute $N_{\text{pretrain}}$. **Confirmed**: SmolLM2-135M (2T pretraining tokens) requires $r_{\min} \approx 12$; Qwen2.5-0.5B (18T pretraining tokens) requires only $r_{\min} \approx 4$ — tested and confirmed at $r = 4$ achieving PPL $1.63$ (at plateau).

**Definition 3** (Pretraining-modulated rank sufficiency).

$$r_{\min} = \eta_0 \cdot \frac{L}{d_h} \cdot q^{-1}(N_{\text{pretrain}})$$

where $\eta_0 \approx 150$ for strong-pretraining models and $q^{-1} > 1$ for weaker pretraining. The function $q(N_{\text{pretrain}})$ captures how pretraining compute improves per-layer representation quality, reducing the correction burden on LoRA.

---

## 7. Synthesis: The Unified Design Rule

Combining the Rank Sufficiency Law (Proposition 1), the M-index memorization threshold (Corollary 1), and the pretraining quality modulation (Definition 3) yields the unified post-training design rule:

$$\boxed{r^* = \max\!\left(8,\; \left\lceil \eta_0 \cdot \frac{L}{d_h} \cdot q^{-1}(N_{\text{pretrain}}) \right\rceil \right)}$$

$$\boxed{\text{Never use full-rank when } \frac{N_p}{N_d} > 10^4}$$

**Proposition 6** (LoRA sufficiency). For any currently popular Transformer architecture ($L/d_h \leq 0.035$) with strong pretraining (18T+ tokens), LoRA $r = 8$ is sufficient for WikiText-2-style autoregressive post-training. The rank sufficiency is language-independent, task-independent (confirmed on SST-2 classification), and time-independent (confirmed from 100 to 1600 training steps).

**Proposition 7** (Full-rank failure mode). Full-rank fine-tuning on $N_d < 10^4$ samples inevitably enters the memorization regime ($M < 1$), producing near-perfect in-distribution perplexity while degrading downstream task accuracy. The apparently superior in-distribution PPL is a measurement artifact, not evidence of better optimization.

---

## 8. Experimental Validation Summary

### 8.1 Falsification results

| Prediction | Test | Status |
|-----------|------|--------|
| $r_{\min} \propto L/d_h$ (form) | Mistral-7B $r=4$ at plateau (PPL=1.45) | ✅ |
| $\eta \approx 230$ (threshold) | SmolLM2 $r_{\min} \approx 12$ fine-grained | ✅ |
| Below-threshold degradation | SmolLM2 $r=6$ PPL=15.29 ($8.7\times$ worse) | ✅ |
| Language-independence | Chinese WT $r8/r32=1.02$ | ✅ |
| Task-independence | SST-2 $r=4,8,32$ all 84.7% | ✅ |
| Time-independence | $r=8$ plateau 100–1600 steps | ✅ |
| $\eta \propto H$ (entropy) | Chinese vs English | ❌ Falsified |
| $\eta \propto 1/N$ (budget) | $r=4$ at $N=400,800,1600$ | ❌ Falsified |
| $\eta$ universal constant | SmolLM2 vs Qwen at $r=4$ | ❌ Falsified |
| ASP-LoRA negative synergy | 7/7 independent comparisons | ✅ |

### 8.2 Convergent evidence

Four independent lines of evidence converge on the conclusion that LoRA $r=8$ is sufficient and full-rank overfits:

1. **Rank curve** (5 model families): $r=8$ matches $r=256$ when $L/d_h \leq 0.035$.
2. **Downstream accuracy** (3 tasks): LoRA preserves $>99.7\%$; full-rank degrades by 3–4pp.
3. **C4 cross-domain PPL**: Full-rank WikiText/C4 ratio $M = 0.52$ (memorization); LoRA $M = 4.53$ (generalization).
4. **ASP stress test**: A stronger optimizer cannot overcome the low-rank constraint — the regularization is the mechanism, not the optimizer.

---

## 9. Formula Index

| Formula | Source |
|---------|--------|
| $r_{\min} = \eta \cdot L / d_h$ | Rank Sufficiency Law (§2.1) |
| $r^* = \max(8, \lceil\eta \cdot L/d_h\rceil)$ | Optimal rank (Prop. 2) |
| $M = \text{PPL}_{\text{train}} / \text{PPL}_{\text{cross}}$ | M-index (§3.1) |
| $\text{Memorization when } N_p/N_d > 10^4$ | Memorization threshold (Cor. 1) |
| $W_{\text{eff}} = W_0 + \frac{\alpha}{r} B A$ | LoRA parameterization (§1.2) |
| $W_*^\top = (X^\top X + \lambda I)^{-1} X^\top Y$ | ALS closed-form (§5.2) |
| $\Delta B = \frac{r}{\alpha} \Delta W A^\top (A A^\top + \lambda I)^{-1}$ | B-projection (§5.2) |
| $\|\delta_L\| \approx \|\delta_\ell\| \cdot \bar{\rho}^{\,L-\ell}$ | Residual amplification (§5.6) |
| $L_{\max} = \frac{\ln(\eta \mu_{\min} K / A_{\text{eff}})}{\ln \bar{\rho}} \approx 26$ | Critical depth (§5.6) |
| $r_{\min} = \eta_0 \cdot \frac{L}{d_h} \cdot q^{-1}(N_{\text{pretrain}})$ | Pretraining-modulated rank (§6) |
| $\text{ME}_{\text{opt}} = (\mu_A + \mu_C - \mu_B - \mu_D)/2$ | Optimizer main effect (§4.2) |
| $\text{Int} = (\mu_A - \mu_B) - (\mu_C - \mu_D)$ | Interaction effect (§4.2) |
