# ASP: A Mathematical Formulation

---

## 1. The Problem

### 1.1 Post-training optimization

Given a pretrained Transformer language model with parameters $\boldsymbol{\theta}_0 \in \mathbb{R}^D$ and a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$, post-training solves

$$\min_{\boldsymbol{\theta} \in \Theta} \; \mathcal{L}(\boldsymbol{\theta})$$

where $\mathcal{L}$ is the causal language modeling cross-entropy:

$$\mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T_i} \log P_{\boldsymbol{\theta}}(y_{i,t} \mid y_{i,\lt t}, x_i)$$

and $\Theta \subseteq \mathbb{R}^D$ is the admissible parameter space.

Two choices of $\Theta$ are studied:

- **Full-rank** ($\Theta_{\text{full}} = \mathbb{R}^D$). All $D$ pretrained parameters are trainable.
- **LoRA** ($\Theta_{\text{LoRA}} \subset \mathbb{R}^D$). For each adapted linear layer with pretrained weight $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, the perturbation is constrained to rank $r \ll \min(d_{\text{out}}, d_{\text{in}})$:

  $$\Delta W = \frac{\alpha}{r} B A, \qquad A \in \mathbb{R}^{r \times d_{\text{in}}},\; B \in \mathbb{R}^{d_{\text{out}} \times r}$$

  All pretrained weights are frozen; only $\{A, B\}$ are free variables. The effective weight is $W_{\text{eff}} = W_0 + \frac{\alpha}{r} B A$.

### 1.2 The standard approach and its limitations

The standard approach uses **AdamW** — an adaptive first-order method — to solve the post-training problem. AdamW maintains per-parameter momentum and second-moment estimates, updating via

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda_{\text{wd}} \boldsymbol{\theta}_t$$

This works well but suffers from two limitations:

1. **No structural knowledge.** AdamW treats the Transformer as a black box. It does not exploit the fact that the output layer (lm_head) is a linear classifier — a problem with a closed-form least-squares solution given fixed hidden representations.

2. **Narrow minima.** First-order methods with decaying learning rates can become trapped in sharp local minima. The loss landscape of overparameterized Transformers contains many such minima, and which one is found depends sensitively on initialization and data order.

### 1.3 What ASP proposes

ASP replaces the single AdamW optimizer with a three-phase alternating procedure. Each phase addresses a distinct aspect of the optimization:

| Phase | Mechanism | What it solves |
|-------|-----------|----------------|
| **ALS** | Closed-form least squares | Solves the output layer exactly given fixed hidden states |
| **SGD** | First-order gradient descent | Coordinates all layers after ALS perturbs hidden representations |
| **Perturbation** | Gaussian parameter noise with cosine decay | Escapes narrow local minima, promotes flat basins |

These three phases alternate in cycles of $K$ SGD steps punctuated by one ALS step and one Perturbation step. The same cycle structure applies to both $\Theta_{\text{full}}$ and $\Theta_{\text{LoRA}}$; the differences lie in how each phase operates under the two parameterizations.

A **2×2 factorial design** crosses optimizer (ASP vs AdamW) with parameter form (full-rank vs LoRA), yielding four protocols. This document gives the mathematical formulation of the ASP branch (protocols A and C).

---

## 2. Phase I: Alternating Least Squares

### 2.1 The ALS subproblem

Consider a single linear layer with weight $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. During the ALS phase, all other layers are held fixed. Let $X \in \mathbb{R}^{N \times d_{\text{in}}}$ be the activations arriving at this layer across $N$ tokens and $Y \in \mathbb{R}^{N \times d_{\text{out}}}$ be the target output. ALS solves

$$\min_{W} \; \|X W^\top - Y\|_F^2 + \lambda \|W\|_F^2$$

with $\lambda > 0$.

**Proposition 1** (Closed-form solution). The objective is strictly convex in $W$ with unique global minimizer

$$W_*^\top = (X^\top X + \lambda I_{d_{\text{in}}})^{-1} X^\top Y$$

*Proof.* The objective $\phi(W) = \operatorname{tr}((XW^\top \!-\! Y)^\top (XW^\top \!-\! Y)) + \lambda \operatorname{tr}(W^\top W)$ is quadratic. Setting $\nabla_W \phi = 0$ gives the normal equations $X^\top X W^\top + \lambda W^\top = X^\top Y$, yielding the stated formula. Since $X^\top X + \lambda I \succ 0$ for $\lambda > 0$, the Hessian is everywhere positive-definite. $\square$

### 2.2 Block-wise computation

When $d_{\text{out}}$ is large (e.g., 151,936 for the lm_head vocabulary), forming $X^\top Y \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ at once is prohibitive. Partition the rows of $W$ into $m$ contiguous blocks $W_{[1]}, \ldots, W_{[m]}$ with $W_{[i]} \in \mathbb{R}^{b_i \times d_{\text{in}}}$, and partition $Y$ column-wise accordingly.

**Proposition 2** (Block separability). The ALS solution factorizes block-wise:

$$W_{[i],*}^\top = (X^\top X + \lambda I)^{-1} X^\top Y_{[i]} \qquad \forall i$$

*Proof.* The squared Frobenius norm separates over column partitions of $Y$ and row partitions of $W$:

$$\|XW^\top - Y\|_F^2 = \sum_{i=1}^{m} \|X W_{[i]}^\top - Y_{[i]}\|_F^2$$

The regularizer likewise separates: $\|W\|_F^2 = \sum_i \|W_{[i]}\|_F^2$. No cross-block coupling exists. $\square$

**Corollary 1.** The matrix $(X^\top X + \lambda I)^{-1}$ is computed once and reused across all blocks. Total arithmetic is unchanged ($\sum b_i = d_{\text{out}}$) but peak memory drops from $\mathcal{O}(d_{\text{out}} d_{\text{in}})$ to $\mathcal{O}(b_{\max} d_{\text{in}})$.

### 2.3 Target specification

The target matrix $Y$ depends on layer type:

- **Output layer (lm_head).** $Y_{[i]}$ is a one-hot encoding of ground-truth tokens whose label falls in vocabulary block $[s_i, e_i)$. Tokens with labels outside the block are excluded from that block's data. This gives ALS access to true supervised signal.

- **Intermediate layers.** No ground-truth target exists. The current weight's own output serves as the reconstruction target:

  $$Y_{[i]} = X W_{[i],\text{old}}^\top$$

  In the limit $\lambda \to 0$, this yields $W_{[i],*}^\top = W_{[i],\text{old}}^\top$ — ALS preserves the weight under its own input distribution. Change occurs only indirectly: when ALS modifies another layer, the activations $X$ arriving at this layer change, and the reconstruction under the new $X$ differs from the old weight.

### 2.4 Damping

Directly writing $W_{[i],*}$ to the weight would replace it in one step. In a Transformer, this is destabilizing because the new weight changes activations flowing to downstream layers, whose weights have not been updated. ALS instead applies an exponential moving average:

$$W_{[i]} \leftarrow (1 - \alpha_\ell) \cdot W_{[i],\text{old}} + \alpha_\ell \cdot W_{[i],*}$$

**Definition 1** (Depth-aware mixing coefficient). Let layers be ordered from input (index 0) to output (index $T-1$). For a layer at position $\ell$, let $d_\ell = (T - 1 - \ell) / (T - 1)$ be its normalized distance from the output. Then

$$\alpha_\ell = \max\!\left(\alpha_0 \cdot e^{-\beta(1 - d_\ell)},\; \alpha_{\min}\right)$$

with $\alpha_0 = 0.01$, $\beta = 2.0$, $\alpha_{\min} = 0.005$. Layers near the output receive $\alpha_\ell \approx \alpha_0$; shallow layers are exponentially suppressed.

### 2.5 ALS under LoRA

Under $\Theta_{\text{LoRA}}$, the effective weight is $W_{\text{eff}} = W_0 + \frac{\alpha}{r} B A$. ALS first solves the full-rank problem in effective-weight space:

$$W_{\text{new}}^\top = (X^\top X + \lambda I)^{-1} X^\top (X W_{\text{eff}}^\top)$$

Define $\Delta W = W_{\text{new}} - W_{\text{eff}}$. Since only $B$ is updated during ALS ($A$ remains fixed, and $W_0$ is frozen), we must solve

$$\frac{\alpha}{r} \cdot \Delta B \cdot A = \Delta W$$

for $\Delta B \in \mathbb{R}^{d_{\text{out}} \times r}$.

**Proposition 3** (Minimum-norm B-projection). The minimum-Frobenius-norm solution is

$$\Delta B_* = \frac{r}{\alpha} \cdot \Delta W \cdot A^\top (A A^\top + \lambda I_r)^{-1}$$

*Proof.* Set $C = \frac{r}{\alpha} \Delta W$. For the underdetermined system $\Delta B \cdot A = C$ ($r \ll d_{\text{in}}$), the general solution is $\Delta B = C A^\dagger + Z(I - A A^\dagger)$ with pseudoinverse $A^\dagger = A^\top (A A^\top)^{-1}$ and arbitrary $Z$. The nullspace term $Z(I - A A^\dagger)$ only increases the norm, so $Z = 0$ gives the minimum. Regularizing $A A^\top \to A A^\top + \lambda I$ stabilizes the pseudoinverse. $\square$

---

## 3. Phase II: Stochastic Gradient Descent

ALS solves each layer independently, breaking cross-layer consistency. SGD restores it.

**Definition 2** (SGD with momentum and weight decay).

$$\begin{aligned}
g_t &= \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t) \\
v_{t+1} &= \mu \cdot v_t + g_t \\
\boldsymbol{\theta}_{t+1} &= \boldsymbol{\theta}_t - \eta \cdot v_{t+1} - \eta \lambda_{\text{wd}} \cdot \boldsymbol{\theta}_t
\end{aligned}$$

with learning rate $\eta$, momentum $\mu \in [0,1)$, weight decay $\lambda_{\text{wd}}$, and gradient clipping $\|g_t\|_2 \leq \gamma$.

Under $\Theta_{\text{full}}$, the gradient $g_t$ spans all $D$ parameters. Under $\Theta_{\text{LoRA}}$, gradients flow only through $A$ and $B$:

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} \cdot B^\top \cdot \frac{\partial \mathcal{L}}{\partial h_{\text{out}}} \cdot h_{\text{in}}^\top, \qquad \frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \cdot \frac{\partial \mathcal{L}}{\partial h_{\text{out}}} \cdot (A h_{\text{in}})^\top$$

The gradient dimension ratio is $\frac{\dim(\nabla_{\text{LoRA}})}{\dim(\nabla_{\text{full}})} = \frac{2r(d_{\text{in}} + d_{\text{out}})}{d_{\text{out}} d_{\text{in}}} \approx \frac{4r}{d_{\text{in}}}$ when $d_{\text{out}} \approx d_{\text{in}}$.

---

## 4. Phase III: Parameter Perturbation

**Definition 3** (Stochastic perturbation). Each trainable parameter receives independent Gaussian noise:

$$\theta_j \leftarrow \theta_j + \varepsilon_j, \qquad \varepsilon_j \sim \mathcal{N}(0, \sigma_c^2 \cdot s_j^2)$$

where $s_j \in (0, 1]$ is a per-layer-type sensitivity multiplier.

**Definition 4** (Cosine decay schedule). The noise scale $\sigma_c$ decays with cycle index $c$:

$$\sigma_c = \frac{\sigma_0}{2}\left(1 + \cos\frac{\pi c}{C_{\max}}\right)$$

with initial scale $\sigma_0$ and annealing horizon $C_{\max}$, clamped below by $\sigma_{\min}$. Early cycles use large perturbations for exploration; late cycles reduce noise for exploitation.

Under $\Theta_{\text{LoRA}}$, noise applied to $A$ and $B$ produces effective-weight perturbation

$$(B + \varepsilon_B)(A + \varepsilon_A) - BA = \varepsilon_B A + B \varepsilon_A + \varepsilon_B \varepsilon_A$$

which has rank at most $2r$. The perturbation cannot explore the full $d_{\text{out}} \times d_{\text{in}}$ weight manifold — the LoRA constraint itself acts as implicit regularization against narrow minima.

---

## 5. Depth Boundary: When ALS Destabilizes

ALS modifies a layer's weights, perturbing its output hidden state. This perturbation propagates forward through residual connections.

Let $h_\ell^{\text{ALS}}$ be the hidden state after ALS modifies layer $\ell$, and $\delta_\ell = h_\ell^{\text{ALS}} - h_\ell$ the perturbation. The Transformer residual recursion is $h_{k+1} = h_k + f_k(h_k; \boldsymbol{\theta}_k)$.

**Proposition 4** (Linearized propagation). To first order,

$$\delta_{k+1} \approx (I + J_{f_k}) \cdot \delta_k, \qquad J_{f_k} = \frac{\partial f_k}{\partial h}\bigg|_{h_k}$$

Iterating from layer $\ell$ to the final layer $L$:

$$\|\delta_L\| \approx \|\delta_\ell\| \cdot \bar{\rho}^{\,L - \ell}, \qquad \bar{\rho} = \left(\prod_{k=\ell}^{L-1} \|I + J_{f_k}\|\right)^{1/(L-\ell)}$$

**Proposition 5** (Critical depth). SGD recovery capacity over $K$ steps is $C_{\text{SGD}} = \eta \cdot \mu_{\min} \cdot K$. ALS divergence occurs when $\|\delta_L\| > C_{\text{SGD}}$, giving the critical layer count

$$L_{\max} = \frac{\ln(\eta \mu_{\min} K / A_{\text{eff}})}{\ln \bar{\rho}} \approx 26$$

where $\bar{\rho} \approx 1.08$ is calibrated from two model families. This predicts convergence for $\leq 24$ layers and divergence for $\geq 28$ layers, consistent with all 8/8 empirical measurements.

**Protective measures.** Three constraints are derived from this analysis: (i) skip layers in the first 50% of depth (longest amplification chains), (ii) the depth-decay damping of Definition 1, and (iii) per-layer norm clipping $\frac{\|\Delta W\|_F}{\|W_{\text{old}}\|_F} \leq \tau$.

---

## 6. The Two ASP Protocols

**Definition 5** (ASP-full). For $\Theta_{\text{full}}$: in each of $C$ cycles,

1. **ALS** (1 step). Solve Proposition 1 for the output layer only, with depth-aware damping (Definition 1).
2. **SGD** ($K$ steps). Apply Definition 2 to all $D$ parameters.
3. **Perturb** (1 step). Apply Definition 3 with cosine schedule (Definition 4), $\sigma_0 = 10^{-3}$.

**Definition 6** (ASP-LoRA). For $\Theta_{\text{LoRA}}$: same cycle structure, but

1. **Low-rank ALS** (1 step). Solve Proposition 1 in effective-weight space, then project to $B$ via Proposition 3.
2. **SGD** ($K$ steps). Apply Definition 2 restricted to $\{A, B\}$.
3. **Perturb** (1 step). Apply Definition 3 to $\{A, B\}$ only, $\sigma_0 = 5 \times 10^{-4}$.

---

## 7. Computational Asymmetry and the Negative Synergy

**Proposition 6** (ALS cost invariance). The dominant ALS cost term, $\mathcal{O}(N d_{\text{in}}^2)$ for forming $X^\top X$, is identical under both parameterizations. The LoRA B-projection adds $\mathcal{O}(b_i r d_{\text{in}})$ per block — a lower-order term.

**Proposition 7** (SGD cost ratio). Under LoRA, per-step SGD cost drops by factor $\approx d_{\text{in}} / (4r) \approx 28\times$ for $d_{\text{in}} = 896$, $r = 8$.

**Corollary 2** (Negative synergy). ALS pays full-rank cost regardless of parameterization (Proposition 6), yet under LoRA its solution passes through a rank-$r$ bottleneck (Proposition 3), discarding information. SGD enjoys LoRA's efficiency (Proposition 7) but operates on parameters suboptimally updated by ALS. This structural mismatch is the mathematical root of the observed negative synergy between ASP and LoRA.

---

## 8. Formula Index

| Formula | Source |
|---------|--------|
| $\min_{\boldsymbol{\theta} \in \Theta} \mathcal{L}(\boldsymbol{\theta})$ | Post-training problem (§1.1) |
| $W_{\text{eff}} = W_0 + \frac{\alpha}{r} B A$ | LoRA parameterization (§1.1) |
| $W_*^\top = (X^\top X + \lambda I)^{-1} X^\top Y$ | ALS closed-form (Prop. 1) |
| $W_{[i],*}^\top = (X^\top X + \lambda I)^{-1} X^\top Y_{[i]}$ | Block separability (Prop. 2) |
| $\alpha_\ell = \alpha_0 e^{-\beta(1 - d_\ell)}$ | Depth-aware damping (Def. 1) |
| $\Delta B_* = \frac{r}{\alpha} \Delta W A^\top (A A^\top + \lambda I)^{-1}$ | B-projection (Prop. 3) |
| $\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^\top (\partial \mathcal{L} / \partial h_{\text{out}}) h_{\text{in}}^\top$ | LoRA gradient |
| $\sigma_c = \frac{\sigma_0}{2}(1 + \cos(\pi c / C_{\max}))$ | Cosine schedule (Def. 4) |
| $\|\delta_L\| \approx \|\delta_\ell\| \cdot \bar{\rho}^{\,L-\ell}$ | Propagation (Prop. 4) |
| $L_{\max} = \frac{\ln(\eta \mu_{\min} K / A_{\text{eff}})}{\ln \bar{\rho}}$ | Critical depth (Prop. 5) |
