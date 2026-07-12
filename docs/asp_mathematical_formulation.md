# ASP: A Mathematical Formulation

## Alternating Least Squares, Stochastic Gradient Descent, and Parameter Perturbation under Full-Rank and Low-Rank Parameterizations

---

## 1. Preliminaries

### 1.1 Notation

Let $(\mathcal{X}, \mathcal{Y})$ denote the input–label space of a causal language modeling task. A pretrained autoregressive model defines a parameterized conditional distribution $P_{\boldsymbol{\theta}}(y \mid x)$ over $\mathcal{Y}$ given $\mathcal{X}$. Post-training seeks parameters $\boldsymbol{\theta}^*$ minimizing the empirical cross-entropy risk over a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$:

$$\mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T_i} \log P_{\boldsymbol{\theta}}(y_{i,t} \mid y_{i,\lt t}, x_i)$$

The model is a Transformer with $L$ layers. Each layer $\ell$ contains a multi-head self-attention sublayer and a feed-forward sublayer, each wrapped by a residual connection:

$$h_{\ell+1} = h_\ell + f_\ell(h_\ell; \boldsymbol{\theta}_\ell)$$

where $h_0$ is the token embedding, $h_L$ feeds the output projection (lm_head), and $\boldsymbol{\theta}_\ell$ collects all parameters in layer $\ell$.

### 1.2 Parameter spaces

**Definition 1** (Full-rank parameter space). $\Theta_{\text{full}} = \mathbb{R}^D$, where $D = \sum_{\ell} \dim(\boldsymbol{\theta}_\ell)$ is the total parameter count. All parameters are trainable.

**Definition 2** (LoRA-constrained parameter space). For each targeted linear layer with pretrained weight $W_0^{(\ell)} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, the admissible weight perturbation is restricted to

$$\Delta W^{(\ell)} \in \left\{ \frac{\alpha}{r} B A \;\middle|\; A \in \mathbb{R}^{r \times d_{\text{in}}},\; B \in \mathbb{R}^{d_{\text{out}} \times r} \right\}$$

with $r \ll \min(d_{\text{out}}, d_{\text{in}})$ and scaling factor $\alpha/r$ (typically $\alpha = 2r$). The effective weight is $W_{\text{eff}}^{(\ell)} = W_0^{(\ell)} + \frac{\alpha}{r} B A$. The LoRA-constrained parameter space $\Theta_{\text{LoRA}}$ is the subset of $\mathbb{R}^D$ formed by freezing all pretrained weights and admitting only the $A, B$ matrices as free variables.

---

## 2. Alternating Least Squares (ALS)

### 2.1 Full-rank ALS problem

Consider a single linear layer with weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. Let $X \in \mathbb{R}^{N \times d_{\text{in}}}$ be the matrix of activations arriving at this layer across $N$ tokens (reshaped from the batch $\times$ sequence tensor), and let $Y \in \mathbb{R}^{N \times d_{\text{out}}}$ be the target output.

**Definition 3** (ALS subproblem). Holding all other layers fixed, the ALS phase solves the regularized least-squares problem

$$\min_{W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}} \; \|X W^\top - Y\|_F^2 + \lambda \|W\|_F^2$$

with regularization parameter $\lambda > 0$.

**Proposition 1** (Closed-form solution). The ALS subproblem is strictly convex in $W$ and admits the unique global minimizer

$$W_*^\top = (X^\top X + \lambda I_{d_{\text{in}}})^{-1} X^\top Y$$

*Proof.* The objective $\phi(W) = \operatorname{tr}((XW^\top - Y)^\top (XW^\top - Y)) + \lambda \operatorname{tr}(W^\top W)$ is quadratic in the entries of $W$. Setting $\nabla_W \phi = 0$ yields the normal equations $X^\top X W^\top + \lambda W^\top = X^\top Y$, hence $W^\top = (X^\top X + \lambda I)^{-1} X^\top Y$. Since $X^\top X + \lambda I \succ 0$ for any $\lambda > 0$, the Hessian is everywhere positive-definite, so $W_*$ is the unique global minimum. $\square$

**Proposition 2** (Block separability). Partition the rows of $W$ into $m$ contiguous blocks, $W = [W_{[1]}^\top \;|\; \cdots \;|\; W_{[m]}^\top]^\top$ with $W_{[i]} \in \mathbb{R}^{b_i \times d_{\text{in}}}$, and partition $Y$ column-wise accordingly, $Y = [Y_{[1]} \;|\; \cdots \;|\; Y_{[m]}]$. Then the ALS solution factorizes block-wise:

$$W_{[i],*}^\top = (X^\top X + \lambda I)^{-1} X^\top Y_{[i]} \qquad \forall i \in \{1, \ldots, m\}$$

*Proof.* The objective separates over row-blocks: $\|XW^\top - Y\|_F^2 = \sum_{i=1}^{m} \|X W_{[i]}^\top - Y_{[i]}\|_F^2$, and the regularizer likewise: $\|W\|_F^2 = \sum_i \|W_{[i]}\|_F^2$. There is no cross-block coupling term, so the minimizer of the sum is the concatenation of the per-block minimizers. Each per-block minimizer satisfies the same normal equations with the shared matrix $(X^\top X + \lambda I)$. $\square$

**Corollary 1** (Computational amortization). The matrix $(X^\top X + \lambda I)^{-1}$ is computed once per ALS phase and reused across all $m$ blocks. The per-block cost is $\mathcal{O}(b_i \cdot d_{\text{in}}^2)$ for forming $X^\top Y_{[i]}$ and solving, versus $\mathcal{O}(d_{\text{out}} \cdot d_{\text{in}}^2)$ for the unpartitioned system. Since $\sum_i b_i = d_{\text{out}}$, the total arithmetic is unchanged; the partition reduces peak memory from $\mathcal{O}(d_{\text{out}} d_{\text{in}})$ to $\mathcal{O}(\max_i b_i \cdot d_{\text{in}})$.

**Definition 4** (Target specification). The target matrix $Y$ is defined according to layer type:

- **Output layer (lm_head)**. $Y_{[i]}$ is a one-hot encoding over the vocabulary block $[s_i, e_i)$:

  $$Y_{[i]}[j, k] = \mathbb{1}[y_j \in [s_i, e_i) \land y_j - s_i = k]$$

  where $y_j$ is the ground-truth token at position $j$. Rows of $X$ corresponding to tokens whose label falls outside $[s_i, e_i)$ are excluded from the block's data.

- **Intermediate layer**. No ground-truth target exists. The current weight's output serves as the reconstruction target:

  $$Y_{[i]} = X \, W_{[i],\text{old}}^\top$$

  yielding $W_{[i],*}^\top = (X^\top X + \lambda I)^{-1} X^\top X W_{[i],\text{old}}^\top = W_{[i],\text{old}}^\top$ when $X^\top X + \lambda I$ is invertible and $\lambda \to 0$. Thus intermediate-layer ALS preserves the weight under its own input distribution, only producing change insofar as $X$ has been altered by ALS on other layers.

### 2.2 Depth-aware damping

**Definition 5** (Depth-aware mixing coefficient). Let $\mathcal{T}$ be the set of all linear layers ordered from input (index 0) to output (index $T-1$). For a layer at ordinal position $\ell_{\text{idx}}$, the EMA mixing coefficient is

$$\alpha(\ell_{\text{idx}}) = \max\left(\alpha_0 \cdot \exp\!\left(-\beta\left(1 - \frac{T - 1 - \ell_{\text{idx}}}{T - 1}\right)\right),\; \alpha_{\min}\right)$$

with base step size $\alpha_0$, depth decay $\beta > 0$, and floor $\alpha_{\min}$.

The update rule after solving the ALS subproblem for block $i$ is

$$W_{[i]} \leftarrow (1 - \alpha(\ell_{\text{idx}})) \cdot W_{[i],\text{old}} + \alpha(\ell_{\text{idx}}) \cdot W_{[i],*}$$

This replaces the direct assignment $W_{[i]} \leftarrow W_{[i],*}$ with an exponential moving average whose mixing rate decays with distance from the output.

### 2.3 Low-rank ALS (LoRA parameterization)

**Definition 6** (Low-rank ALS problem). Under LoRA parameterization, the effective weight is $W_{\text{eff}} = W_0 + \frac{\alpha}{r} B A$. ALS first solves the full-rank problem in effective-weight space:

$$W_{\text{new}}^\top = (X^\top X + \lambda I)^{-1} X^\top (X W_{\text{eff}}^\top)$$

Define the discrepancy $\Delta W = W_{\text{new}} - W_{\text{eff}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. The LoRA constraint requires updating only $B$ (keeping $A$ fixed during ALS), so we solve

$$\frac{\alpha}{r} \cdot \Delta B \cdot A = \Delta W$$

for $\Delta B \in \mathbb{R}^{d_{\text{out}} \times r}$.

**Proposition 3** (Minimum-norm B-projection). The minimum Frobenius-norm solution to $\frac{\alpha}{r} \Delta B \cdot A = \Delta W$ is

$$\Delta B_* = \frac{r}{\alpha} \cdot \Delta W \cdot A^\top (A A^\top + \lambda I_r)^{-1}$$

*Proof.* Let $C = \frac{r}{\alpha} \Delta W$. The equation $\Delta B \cdot A = C$ is an underdetermined linear system over the rows of $\Delta B$ (since $r \ll d_{\text{in}}$). For each row $j$, the general solution is $\Delta B_{j,:} = C_{j,:} A^\dagger + z_j^\top (I - A A^\dagger)$ where $A^\dagger$ is the Moore-Penrose pseudoinverse of $A$ and $z_j \in \mathbb{R}^{d_{\text{in}}}$ is arbitrary. The term $z_j^\top (I - A A^\dagger)$ lies in the nullspace of $A$ and contributes only to the norm. Setting $z_j = 0$ for all $j$ minimizes $\|\Delta B\|_F$. With Tikhonov regularization $A^\dagger_\lambda = A^\top (A A^\top + \lambda I)^{-1}$, the minimum-norm solution is $\Delta B = C A^\dagger_\lambda$, yielding the stated formula. $\square$

**Remark 1** (Efficiency of the projection). The matrix $A A^\top \in \mathbb{R}^{r \times r}$ has size independent of $d_{\text{in}}$. For $r = 8$, its Cholesky factorization costs $\mathcal{O}(8^3) = 512$ floating-point operations — negligible relative to the $\mathcal{O}(d_{\text{in}}^3)$ cost of the full-rank ALS solve. The per-block projection $\Delta W_{[i]} \cdot A^\top (A A^\top + \lambda I)^{-1}$ adds $\mathcal{O}(b_i \cdot r \cdot d_{\text{in}})$ operations per block.

---

## 3. Stochastic Gradient Descent (SGD)

### 3.1 Full-rank SGD

**Definition 7** (SGD with momentum and weight decay). Let $\boldsymbol{\theta} \in \Theta_{\text{full}}$ collect all trainable parameters. The SGD phase iterates for $K$ steps, each comprising:

$$g_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$$

$$v_{t+1} = \mu \cdot v_t + g_t$$

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot v_{t+1} - \eta \lambda_{\text{wd}} \cdot \boldsymbol{\theta}_t$$

with learning rate $\eta$, momentum coefficient $\mu \in [0, 1)$, weight decay $\lambda_{\text{wd}}$, and gradient clipping $\|g_t\|_2 \leq \gamma$.

The gradient norm $\|g_t\|_2$ is monitored to track convergence across cycles.

**Remark 2** (Role of SGD in ASP). The ALS phase solves each layer independently under the assumption that activations $X$ are fixed. After ALS modifies a layer's weights, the activations flowing into downstream layers change, yet those layers' weights have not been updated to reflect the new input distribution. SGD corrects this: by jointly optimizing all layers via backpropagation, it restores cross-layer consistency. The number of SGD steps $K$ must be sufficient for this "digestion" — empirically $K \propto L^{1.2}$ for models up to 24 layers.

### 3.2 LoRA-constrained SGD

Under $\Theta_{\text{LoRA}}$, gradients flow only through $A$ and $B$:

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} \cdot B^\top \cdot \frac{\partial \mathcal{L}}{\partial h_{\text{out}}} \cdot h_{\text{in}}^\top$$

$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \cdot \frac{\partial \mathcal{L}}{\partial h_{\text{out}}} \cdot (A h_{\text{in}})^\top$$

where $h_{\text{in}} \in \mathbb{R}^{d_{\text{in}}}$ is the layer input and $h_{\text{out}} \in \mathbb{R}^{d_{\text{out}}}$ the layer output. The Jacobian of the loss with respect to the LoRA parameters has dimension $r(d_{\text{in}} + d_{\text{out}})$, versus $d_{\text{out}} d_{\text{in}}$ for the full-rank Jacobian — a reduction factor of $\frac{d_{\text{out}} d_{\text{in}}}{r(d_{\text{in}} + d_{\text{out}})} \approx \frac{d_{\text{in}}}{2r}$ when $d_{\text{out}} \approx d_{\text{in}}$.

---

## 4. Parameter Perturbation

### 4.1 Perturbation mechanism

**Definition 8** (Stochastic perturbation). During the perturbation phase, independent Gaussian noise is added to every trainable parameter:

$$\theta_j \leftarrow \theta_j + \varepsilon_j, \qquad \varepsilon_j \sim \mathcal{N}(0, \sigma_c^2 \cdot s^2(\theta_j))$$

where $\sigma_c$ is the cycle-dependent noise scale and $s(\theta_j) \in (0, 1]$ is a layer-type sensitivity multiplier.

### 4.2 Noise schedule

**Definition 9** (Cosine decay schedule). The noise scale follows a cosine annealing trajectory over cycles $c = 0, 1, \ldots, C-1$:

$$\sigma_c = \frac{\sigma_0}{2}\left(1 + \cos\frac{\pi c}{C_{\max}}\right)$$

clamped below by $\sigma_{\min}$, with $\sigma_0$ the initial scale and $C_{\max}$ the annealing horizon.

This schedule transitions the optimization from an exploration-dominant regime (large $\sigma_c$ at early cycles, encouraging escape from narrow local minima) to an exploitation-dominant regime (small $\sigma_c$ at late cycles, enabling fine convergence).

### 4.3 Perturbation in LoRA space

When noise is applied directly to $A$ and $B$, the effective-weight perturbation is

$$(B + \varepsilon_B)(A + \varepsilon_A) - BA = \varepsilon_B A + B \varepsilon_A + \varepsilon_B \varepsilon_A$$

which has rank at most $2r$ (the sum of two rank-$r$ matrices plus a rank-$r$ cross-term). Consequently, LoRA-space perturbations cannot explore the full $d_{\text{out}} \times d_{\text{in}}$-dimensional weight manifold. The low-rank constraint acts as an implicit regularizer: perturbations are confined to a subspace of dimension $\mathcal{O}(r(d_{\text{in}} + d_{\text{out}}))$ rather than the full $d_{\text{out}} d_{\text{in}}$, making it structurally harder to overfit to narrow basins.

---

## 5. Depth Boundary Theory

### 5.1 Residual perturbation propagation

**Definition 10** (ALS perturbation). When ALS modifies the weights of layer $\ell$, the hidden state at that layer changes from $h_\ell$ to $h_\ell^{\text{ALS}}$. Define the perturbation vector $\delta_\ell = h_\ell^{\text{ALS}} - h_\ell$.

**Proposition 4** (Linearized propagation). Under a first-order Taylor expansion of the layer functions $f_k$, the perturbation $\delta_\ell$ propagates forward through subsequent layers as

$$\delta_{k+1} \approx (I + J_{f_k}) \cdot \delta_k, \qquad J_{f_k} = \frac{\partial f_k}{\partial h}\bigg|_{h = h_k}$$

for $k = \ell, \ell+1, \ldots, L-1$.

**Proposition 5** (Cumulative amplification). The perturbation magnitude at the final layer satisfies

$$\|\delta_L\| \approx \|\delta_\ell\| \cdot \prod_{k=\ell}^{L-1} \|I + J_{f_k}\|$$

Let $\bar{\rho}$ be the geometric mean of the amplification factors:

$$\bar{\rho} = \left(\prod_{k=\ell}^{L-1} \|I + J_{f_k}\|\right)^{1/(L-\ell)}$$

Then $\|\delta_L\| \approx \|\delta_\ell\| \cdot \bar{\rho}^{\,L - \ell}$.

### 5.2 Critical depth

**Definition 11** (SGD recovery capacity). Over $K$ SGD steps with learning rate $\eta$, the effective recovery capacity of SGD against a perturbation $\delta$ is

$$C_{\text{SGD}} = \eta \cdot \mu_{\min} \cdot K$$

where $\mu_{\min}$ is a lower bound on the gradient norm during the recovery phase.

**Proposition 6** (Depth boundary). ALS divergence occurs when the propagated perturbation exceeds SGD's per-cycle recovery capacity. The critical layer count $L_{\max}$ satisfies

$$L_{\max} = \frac{\ln(\eta\, \mu_{\min}\, K / A_{\text{eff}})}{\ln \bar{\rho}}$$

where $A_{\text{eff}}$ is the effective perturbation amplitude at the ALS-modified layer.

*Derivation.* Setting $\|\delta_L\| = C_{\text{SGD}}$ and substituting $\|\delta_L\| \approx A_{\text{eff}} \cdot \bar{\rho}^{\,L - \ell}$ with $\ell$ indexing the ALS-modified layer yields $A_{\text{eff}} \cdot \bar{\rho}^{\,L - \ell} \approx \eta \mu_{\min} K$. Solving for $L$ gives the expression above. With empirically calibrated $\bar{\rho} \approx 1.08$ (fitted from digestion-time measurements on OPT-125M and Qwen2.5-0.5B), this predicts $L_{\max} \approx 26$, consistent with the observed boundary: convergence for $\leq 24$ layers, divergence for $\geq 28$ layers (confirmed on 8/8 architectures). $\square$

### 5.3 Protective constraints

The depth boundary theory motivates three constraints applied during ALS:

1. **Layer exclusion.** Layers with $\ell_{\text{idx}} < \tau_{\text{skip}} \cdot T$ (typically $\tau_{\text{skip}} = 0.5$) are skipped, as they create the longest residual amplification chains.
2. **Depth-decay damping.** $\alpha(\ell_{\text{idx}})$ given in Definition 5, which exponentially suppresses updates to shallow layers.
3. **Norm clipping.** A per-layer relative-change bound:

   $$\frac{\|W_{\text{new}} - W_{\text{old}}\|_F}{\|W_{\text{old}}\|_F} \leq \tau_{\text{clip}}$$

   with a higher catastrophic threshold $\tau_{\text{catastrophic}}$ that triggers full rollback of the ALS cycle.

---

## 6. Unified Formulation of the ASP Family

### 6.1 Full-rank ASP

**Definition 12** (ASP-full). Let $C$ be the number of cycles and $K$ the number of SGD steps per cycle. Define the alternating optimization sequence:

$$\boldsymbol{\theta}^{(c,0)} = \boldsymbol{\theta}^{(c-1, \text{final})}, \qquad \boldsymbol{\theta}^{(0,\text{final})} = \boldsymbol{\theta}_0$$

1. **ALS** (1 step). For the output projection layer only:

   $$W_{\text{head}} \leftarrow \arg\min_{W} \|X W^\top - Y_{\text{target}}\|_F^2 + \lambda \|W\|_F^2$$

   with $Y_{\text{target}}$ as in Definition 4 and EMA damping per Definition 5.

2. **SGD** ($K$ steps). For $t = 1, \ldots, K$:

   $$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \cdot \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}) - \eta \lambda_{\text{wd}} \boldsymbol{\theta}$$

   with momentum and gradient clipping as in Definition 7.

3. **Perturb** (1 step). Per Definition 8 with cosine schedule (Definition 9).

### 6.2 LoRA-constrained ASP

**Definition 13** (ASP-LoRA). Same cycle structure as ASP-full, but parameterized within $\Theta_{\text{LoRA}}$:

1. **Low-rank ALS** (1 step). For each adapted layer:

   $$W_{\text{eff}} = W_0 + \frac{\alpha}{r} B A$$

   $$W_{\text{new}} = \arg\min_{W} \|X W^\top - X W_{\text{eff}}^\top\|_F^2 + \lambda \|W\|_F^2$$

   $$B \leftarrow B + \frac{r}{\alpha} \cdot (W_{\text{new}} - W_{\text{eff}}) \cdot A^\top (A A^\top + \lambda I)^{-1}$$

   where the B-projection is per Proposition 3.

2. **SGD** ($K$ steps). Gradient updates restricted to $\{A^{(\ell)}, B^{(\ell)}\}_{\ell}$ per §3.2.

3. **Perturb** (1 step). Noise applied to $A, B$ only, with $\sigma_0$ scaled down (typically $5 \times 10^{-4}$) relative to full-rank.

---

## 7. Computational Asymmetry

**Proposition 7** (ALS cost invariance). The dominant term in ALS cost, $\mathcal{O}(N d_{\text{in}}^2)$ for forming $X^\top X$, is identical under both full-rank and LoRA parameterizations. The LoRA-specific B-projection adds $\mathcal{O}(b_i r d_{\text{in}})$ per block, which is a lower-order term.

**Proposition 8** (SGD cost reduction under LoRA). The per-step SGD cost ratio is

$$\frac{\text{FLOPs}_{\text{SGD-LoRA}}}{\text{FLOPs}_{\text{SGD-full}}} \approx \frac{2r(d_{\text{in}} + d_{\text{out}})}{d_{\text{out}} d_{\text{in}}}$$

For typical configurations ($d_{\text{out}} \approx d_{\text{in}}$, $r=8$), this ratio is $\approx 16 / d_{\text{in}} \approx 1.8\%$ for $d_{\text{in}} = 896$.

**Corollary 2** (Negative synergy source). The ALS phase incurs full-rank computational cost regardless of parameterization (Proposition 7), yet under LoRA its solution is projected through a rank-$r$ bottleneck (Proposition 3), discarding information. The SGD phase benefits from LoRA's reduced parameter count (Proposition 8) but operates on parameters that have been suboptimally updated by the preceding ALS. This structural mismatch — full-rank ALS cost $\times$ low-rank ALS information throughput — is the mathematical root of the observed negative synergy between ASP and LoRA.

---

## 8. Key Formulae

| Formula | Description |
|---------|-------------|
| $W_*^\top = (X^\top X + \lambda I)^{-1} X^\top Y$ | ALS closed-form solution (Proposition 1) |
| $W_{[i],*}^\top = (X^\top X + \lambda I)^{-1} X^\top Y_{[i]}$ | Block-separable ALS (Proposition 2) |
| $\alpha(\ell) = \alpha_0 \exp(-\beta(1 - \frac{T-1-\ell_{\text{idx}}}{T-1}))$ | Depth-aware mixing coefficient (Definition 5) |
| $\Delta B_* = \frac{r}{\alpha} \Delta W \cdot A^\top (A A^\top + \lambda I)^{-1}$ | Low-rank B-projection (Proposition 3) |
| $\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^\top (\partial \mathcal{L} / \partial h_{\text{out}}) h_{\text{in}}^\top$ | LoRA gradient through $A$ |
| $\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} (\partial \mathcal{L} / \partial h_{\text{out}}) (A h_{\text{in}})^\top$ | LoRA gradient through $B$ |
| $\sigma_c = \frac{\sigma_0}{2}(1 + \cos(\pi c / C_{\max}))$ | Perturbation cosine schedule (Definition 9) |
| $\|\delta_L\| \approx \|\delta_\ell\| \cdot \bar{\rho}^{\,L-\ell}$ | Residual perturbation amplification (Proposition 5) |
| $L_{\max} = \frac{\ln(\eta \mu_{\min} K / A_{\text{eff}})}{\ln \bar{\rho}}$ | Critical depth (Proposition 6) |
| $W_{\text{eff}} = W_0 + \frac{\alpha}{r} B A$ | LoRA effective weight (Definition 2) |
| $\frac{\text{FLOPs}_{\text{SGD-LoRA}}}{\text{FLOPs}_{\text{SGD-full}}} \approx \frac{16}{d_{\text{in}}}$ | SGD cost ratio (Proposition 8) |
