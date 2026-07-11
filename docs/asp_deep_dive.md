# ASP 算法在全秩与 LoRA 下的应用：数学推导、代码实现与公平比较

本文聚焦于 **ASP (ALS+SGD+Perturbation)** 这一种优化算法，详细展示它在两种参数形态（全秩和 LoRA）下的数学原理和代码实现，以及如何确保与 AdamW 的公平比较。

---

## 一、ASP 算法的整体结构

ASP 不是一个单一的优化器，而是一个**三阶段交替执行的调度器**。无论应用于全秩还是 LoRA，循环结构完全相同：

```
┌─────────────────────────────────────────────┐
│  for cycle in range(n_cycles):              │
│      Phase I:  ALS         (1 step)          │
│      Phase II: SGD         (k steps)          │
│      Phase III: Perturb    (1 step)          │
└─────────────────────────────────────────────┘
```

其中 k 是每周期 SGD 步数（典型值 50），n_cycles 是总周期数（100步对应约2个周期，800步对应约15个周期）。

这个循环由 `altopt/framework.py` 中的 `AltOptFramework.step()` 驱动。

---

## 二、ASP 在全秩参数下的应用 (Protocol A)

### 2.1 适用场景

全秩意味着模型的**每一个参数都可以被更新**。对于 Qwen2.5-0.5B，就是全部 4.94 亿个参数。

ASP 在全秩下展示了其完整的数学威力——ALS 可以直接在满秩权重矩阵上进行操作。

### 2.2 Phase I: ALS — 闭式求解最优权重

**数学问题**：对于模型中的某一层（例如 lm_head 输出层），给定输入激活值 $X$ 和目标输出 $Y$，直接求解最优的权重矩阵 $W$：

$$\min_{W} \|X W^T - Y\|_F^2 + \lambda \|W\|_F^2$$

这是标准的**岭回归** (ridge regression)，存在闭式解：

$$\boxed{W_{\text{new}}^T = (X^T X + \lambda I)^{-1} X^T Y}$$

**为什么存在闭式解？** 因为这个优化问题是**凸的**——目标函数是 $W$ 的二次型（两个二次项相加），二次型的唯一最小值就是梯度为零的点。令梯度为零可以直接解出上述公式，不需要迭代。

**代码实现** (altopt/als.py, 全秩版本)：

```python
# Phase I: ALS — 在全秩 nn.Linear 层上直接操作

def solve_block_for_linear_layer(X, Y, W_current, block_size=1024, reg_lambda=1e-4):
    """
    X:  [N, d_in]    该层的输入激活值
    Y:  [N, d_out]   该层的目标输出
    W:  [d_out, d_in] 当前权重矩阵

    返回: 更新后的 W（直接修改 W.data）
    """
    d_out, d_in = W_current.shape

    # Step 1: 构建正规方程矩阵 (X^T X + λI)
    # 这是一个 d_in × d_in 的对称正定矩阵
    XtX = X.T @ X                          # [d_in, d_in]  — 花费 O(N·d_in²)
    I_reg = reg_lambda * torch.eye(d_in)    # λI
    XtX_reg = XtX + I_reg                  # (X^T X + λI)   — 保证可逆

    # Step 2: 按块求解（把 d_out 行分成 b=1024 的小块）
    n_blocks = (d_out + block_size - 1) // block_size

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, d_out)

        # 当前块的"目标"
        Y_block = X @ W_current[start:end, :].T   # [N, block_size]

        # 右侧项: X^T Y_block
        XtY = X.T @ Y_block                        # [d_in, block_size]

        # 求解: (X^T X + λI) · W_new^T = X^T Y
        # 因为 XtX_reg 是正定矩阵，torch.linalg.solve 自动选择
        # Cholesky 分解进行高效求解
        W_new_T = torch.linalg.solve(XtX_reg, XtY) # [d_in, block_size]

        # 写回权重矩阵
        W_current[start:end, :] = W_new_T.T        # [block_size, d_in]

    return W_current
```

**复杂度分析**：

| 步骤 | 复杂度 | 说明 |
|------|--------|------|
| 计算 $X^T X$ | $\mathcal{O}(N \cdot d_{\text{in}}^2)$ | 一次, 对所有块共享 |
| Cholesky 分解 | $\mathcal{O}(d_{\text{in}}^3)$ | 一次, 对所有块共享 |
| 每块的三角求解 | $\mathcal{O}(\text{block\_size} \cdot d_{\text{in}}^2)$ | 对每块重复 |

**数值精度要求**：ALS 的矩阵运算（尤其是 $X^T X$ 的形成和 Cholesky 分解）在 float16 下会累积较大数值误差。因此 Protocol A 必须使用 **float32 精度**，这是它比 Protocol B/D 慢的一个原因。

### 2.3 Phase II: SGD — 梯度下降协调各层

**数学问题**：ALS 只改了一层，但 Transformer 的各层之间高度耦合。SGD 的任务是让**所有层的参数互相适应**。

SGD 执行标准的梯度下降更新：

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \nabla_{\boldsymbol{\theta}}\mathcal{L}_{\text{CE}}(\boldsymbol{\theta}_t)$$

带动量版本：

$$v_{t+1} = \beta v_t + \eta \cdot \nabla_{\boldsymbol{\theta}}\mathcal{L}_{\text{CE}}(\boldsymbol{\theta}_t)$$

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - v_{t+1} - \eta\lambda \cdot \boldsymbol{\theta}_t$$

其中 $\beta = 0.9$, $\eta = 10^{-4}$, $\lambda = 0.01$。

**注意这里用的是 SGD 而不是 AdamW**。这是 ASP 的另一个关键选择——不使用自适应学习率。原因有两个：
1. ALS 已经提供了"好"的权重更新方向，SGD 只需要微调
2. SGD 比 AdamW 更轻量（没有动量状态数组），节省显存

```python
# Phase II: SGD — 全参数梯度下降

optimizer = torch.optim.SGD(
    model.parameters(),          # 所有 494M 参数
    lr=1e-4,                    # 学习率
    momentum=0.9,                # 动量
    weight_decay=0.01           # 权重衰减
)

model.train()
for sgd_step in range(sgd_steps_per_cycle):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(**batch).loss   # 前向传播
        loss.backward()              # 反向传播 — 计算所有参数的梯度

        # 梯度裁剪：防止 ALS 导致某步梯度过大
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()             # 沿梯度更新参数
```

### 2.4 Phase III: Perturb — 随机扰动促进泛化

**数学操作**：在参数空间注入高斯噪声。

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

噪声强度随时间衰减（余弦退火）：

$$\sigma_c = \sigma_0 \cdot \frac{1}{2}\left(1 + \cos\frac{\pi c}{C_{\max}}\right)$$

其中 $\sigma_0 = 10^{-3}$, $C_{\max} = 10$。

```python
# Phase III: Perturb — 加高斯噪声

sigma = 1e-3  # 初始扰动强度

with torch.no_grad():
    for param in model.parameters():
        noise = sigma * torch.randn_like(param)  # N(0, σ²I)
        param.add_(noise)                         # θ ← θ + ε
```

### 2.5 Protocol A 的完整数学描述

$$
\boxed{
\begin{aligned}
&\textbf{Protocol A: ASP + Full-Rank} \\[6pt]
&\text{输入: } \boldsymbol{\theta}_0, \mathcal{D}, C, K, \sigma_0 \\
&\text{for } c = 1, \ldots, C: \\
&\quad \text{① ALS: } W_{\text{head}} \leftarrow \arg\min_W \|X W^T - Y\|_F^2  \\
&\qquad\qquad = (X^T X + \lambda I)^{-1} X^T Y \quad \text{(闭式解, 只改输出层)} \\[4pt]
&\quad \text{② SGD: for } k = 1, \ldots, K: \\
&\qquad \boldsymbol{g} = \nabla_{\boldsymbol{\theta}}\mathcal{L}_{\text{CE}}(\boldsymbol{\theta}) \quad \text{(所有层)} \\
&\qquad \boldsymbol{v} \leftarrow \beta\boldsymbol{v} + \eta\boldsymbol{g} \\
&\qquad \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \boldsymbol{v} - \eta\lambda\boldsymbol{\theta} \\[4pt]
&\quad \text{③ Perturb: } \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \sigma_c \cdot \boldsymbol{\varepsilon}, \; \boldsymbol{\varepsilon} \sim \mathcal{N}(0, I)
\end{aligned}}
$$

---

## 三、ASP 在 LoRA 参数下的应用 (Protocol C)

### 3.1 核心挑战

LoRA 的参数形态与全秩完全不同。对于某层 $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$：

- **全秩**: 直接读写 $W_0.data$（一个 $d_{\text{out}} \times d_{\text{in}}$ 的矩阵）
- **LoRA**: 有效权重 $W_{\text{eff}} = W_0 + \frac{\alpha}{r}BA$，其中 $W_0$ 冻结，$A \in \mathbb{R}^{r \times d_{\text{in}}}$ 和 $B \in \mathbb{R}^{d_{\text{out}} \times r}$ 可训练

**核心问题**: ALS 求得的是满秩的 $W_{\text{new}}$（大小 $d_{\text{out}} \times d_{\text{in}}$），但 LoRA 只接受对 $A$ 和 $B$ 的更新。如何把满秩的解"投影"到低秩空间？

### 3.2 Phase I: 低秩 ALS（X1 扩展，2026-06-23 实现）

**数学推导**：

ALS 在满秩空间求解得到 $W_{\text{new}}$。定义差异：

$$\Delta W = W_{\text{new}} - W_{\text{eff}} = W_{\text{new}} - \left(W_0 + \frac{\alpha}{r}BA\right)$$

我们希望更新 $B \leftarrow B + \Delta B$，使得：

$$\frac{\alpha}{r}(B + \Delta B)A \approx \frac{\alpha}{r}BA + \Delta W$$

化简得：

$$\frac{\alpha}{r} \cdot \Delta B \cdot A = \Delta W$$

即：

$$\Delta B \cdot A = \frac{r}{\alpha} \cdot \Delta W$$

这是一个关于 $\Delta B$ 的**欠定线性系统**。$A \in \mathbb{R}^{r \times d_{\text{in}}}$ 是一个"宽"矩阵（$r \ll d_{\text{in}}$），因此有无限多解。我们选择**最小 Frobenius 范数解**：

$$\boxed{\Delta B = \frac{r}{\alpha} \cdot \Delta W \cdot A^T \cdot (AA^T + \lambda I)^{-1}}$$

简化（将 $r/\alpha$ 吸收进缩放因子）：

$$\boxed{\Delta B = \Delta W \cdot A^T \cdot (AA^T + \lambda I)^{-1} / \alpha}$$

其中 $\lambda$ 是正则化参数（$10^{-4}$），防止 $AA^T$ 不可逆。

**推导细节——为什么这是最小范数解？**

对于欠定系统 $\Delta B \cdot A = C$（令 $C = \frac{r}{\alpha}\Delta W$）：

1. 通解：$\Delta B = C \cdot A^{\dagger} + Z(I - AA^{\dagger})$，其中 $A^{\dagger} = A^T(AA^T)^{-1}$ 是 $A$ 的 Moore-Penrose 伪逆，$Z$ 是任意矩阵。
2. 当 $Z = 0$ 时，解 $\Delta B = C \cdot A^T \cdot (AA^T)^{-1}$ 是所有可能解中 Frobenius 范数最小的。

加上正则化项 $\lambda I$ 后即得上式。

**代码实现** (altopt/als.py, X1 版本)：

```python
# Phase I: 低秩 ALS — 在 LoRA 参数空间求解

def solve_low_rank_block(X, lora_A, lora_B, base_W, scaling, block_size=128):
    """
    X:       [N, d_in]     当前层的输入激活值
    lora_A:  [r, d_in]     LoRA 的 A 矩阵
    lora_B:  [d_out, r]    LoRA 的 B 矩阵
    base_W:  [d_out, d_in] 冻结的原始权重
    scaling: float          α / r

    返回: 更新后的 lora_B（直接修改）
    """
    d_out, d_in = base_W.shape
    r = lora_A.shape[0]

    # Step 1: 计算"有效权重"
    effective_W = base_W + scaling * (lora_B @ lora_A)
    #              ↑ 冻结的        ↑ LoRA 修正

    # Step 2: 建正规方程 (X^T X + λI)
    XtX = X.T @ X                          # [d_in, d_in]
    reg = 1e-4 * torch.eye(d_in)
    XtX_reg = XtX + reg

    # Step 3: 预计算 A 的伪逆 A⁺ = A^T (AA^T + λI)^{-1}
    # 这只需要算一次，对所有块复用
    AAT = lora_A @ lora_A.T               # [r, r] — 很小 (8×8)!
    reg_r = 1e-4 * torch.eye(r)
    try:
        L = torch.linalg.cholesky(AAT + reg_r)     # Cholesky 分解 (r×r)
        A_pinv = torch.cholesky_solve(lora_A, L)    # A⁺ via Cholesky
    except RuntimeError:
        A_pinv = torch.linalg.lstsq(AAT + reg_r, lora_A).solution

    # Step 4: 按块求解，每块投影回 B 空间
    n_blocks = (d_out + block_size - 1) // block_size

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, d_out)

        # 满秩求解
        Y_block = X @ effective_W[start:end, :].T   # [N, block_size]
        XtY = X.T @ Y_block                         # [d_in, block_size]

        W_new_T = torch.linalg.solve(XtX_reg, XtY)  # [d_in, block_size]
        W_new_block = W_new_T.T                     # [block_size, d_in]

        # 计算差异
        delta_W = W_new_block - effective_W[start:end, :]

        # ★ 核心投影步骤 ★
        # ΔB = ΔW · A^T · (AA^T + λI)^{-1} / α
        delta_B = delta_W @ A_pinv.T / scaling

        # 直接在 B 矩阵上做 in-place 更新
        lora_B[start:end, :] += delta_B.to(lora_B.dtype)

    return lora_B
```

**为什么 $AA^T$ 只有 $8 \times 8$ 大小？** 这是低秩 ALS 最优雅的地方：

- $A \in \mathbb{R}^{r \times d_{\text{in}}}$ → $AA^T \in \mathbb{R}^{r \times r}$
- 对于 $r=8$, $AA^T$ 只有 $8 \times 8$ — Cholesky 分解在这个大小上几乎是免费的
- **无论 $d_{\text{in}}$ 多大（可以是 4096 甚至更大），$AA^T$ 始终只有 $r \times r$**

这意味着低秩 ALS 的额外开销（相对于全秩 ALS）可以忽略不计！

### 3.3 Phase II: SGD — 只更新 LoRA 参数

```python
# Protocol C 的 SGD 阶段: 只优化 LoRA 参数

optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    #  ↑ 只取 LoRA 的 A 和 B 矩阵（~3M 参数）
    lr=1e-4,
    momentum=0.9,
    weight_decay=0.01
)

# 训练逻辑和全秩版本完全相同 —— 但 optimizer.step()
# 只修改 A 和 B，基模型权重 W_base 保持冻结
```

### 3.4 Phase III: Perturb — 只扰动 LoRA 参数

```python
# Protocol C 的扰动阶段: 更小的噪声，只作用在 LoRA 参数上

sigma = 5e-4  # 比全秩 (1e-3) 小一半

with torch.no_grad():
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            param.add_(sigma * torch.randn_like(param))
```

### 3.5 Protocol C 的完整数学描述

$$
\boxed{
\begin{aligned}
&\textbf{Protocol C: ASP + LoRA} \\[6pt]
&\text{for } c = 1, \ldots, C: \\[4pt]
&\quad \text{① 低秩 ALS: } \\
&\qquad W_{\text{new}} = (X^T X + \lambda I)^{-1} X^T Y_{\text{eff}} \quad \text{(满秩空间求解)} \\
&\qquad \Delta W = W_{\text{new}} - (W_0 + \frac{\alpha}{r}BA) \\
&\qquad \Delta B = \Delta W \cdot A^T \cdot (AA^T + \lambda I)^{-1} / \alpha \quad \text{(投影到低秩空间)} \\
&\qquad B \leftarrow B + \Delta B \\[4pt]
&\quad \text{② SGD: for } k = 1, \ldots, K: \\
&\qquad \boldsymbol{g}_A = \nabla_A \mathcal{L}_{\text{CE}}, \; \boldsymbol{g}_B = \nabla_B \mathcal{L}_{\text{CE}} \\
&\qquad A \leftarrow A - \eta\boldsymbol{g}_A, \; B \leftarrow B - \eta\boldsymbol{g}_B \\[4pt]
&\quad \text{③ Perturb: } A \leftarrow A + \sigma_c \cdot \boldsymbol{\varepsilon}_A, \; B \leftarrow B + \sigma_c \cdot \boldsymbol{\varepsilon}_B
\end{aligned}}
$$

---

## 四、全秩 ASP vs LoRA ASP：核心差异对比

| 维度 | Protocol A (全秩 ASP) | Protocol C (LoRA ASP) |
|------|----------------------|----------------------|
| **ALS 操作对象** | `nn.Linear.weight.data` (直接) | $W_{\text{eff}}$, 然后投影到 $B$ |
| **ALS 计算量** | $N \cdot d_{\text{in}}^2 + d_{\text{in}}^3$ (同 Protocol A) | $N \cdot d_{\text{in}}^2 + d_{\text{in}}^3 + \cancel{r^3}$ |
| **额外投影开销** | 无 | $b \cdot r \cdot r$ (每块) — 可以忽略 |
| **可训练参数量** | 494M | ~3M |
| **数值精度要求** | float32 | bfloat16 (除 ALS 阶段外) |
| **SGD 更新对象** | 所有参数 | 只有 $A$, $B$ |
| **扰动强度 $\sigma_0$** | $10^{-3}$ | $5 \times 10^{-4}$ |
| **深度限制** | ≤24 层 | ≤24 层 (相同边界) |
| **7B 可行性** | ❌ 被深度边界阻断 | ✅ X1 修复后可运行 |

---

## 五、如何确保公平比较：FLOPs 归一化

### 5.1 为什么不按"步数"比较？

两种优化器的每步计算量差异很大：

| 操作 | 相对成本 (FLOPs) |
|------|-----------------|
| Protocol B/D 一步 (所有参数) | $2 + 4 + 3 = 9 \cdot N_{\text{params}}$ |
| Protocol B/D 一步 (LoRA only) | $2 + 4 + 3 \cdot \frac{N_{\text{LoRA}}}{N_{\text{total}}} \cdot N_{\text{total}}$ |
| Protocol A ALS (全秩) | $N \cdot d_{\text{in}}^2 + d_{\text{in}}^3$ (一次, 每约 52 步) |
| Protocol C ALS (LoRA) | $N \cdot d_{\text{in}}^2 + d_{\text{in}}^3 + \cancel{r^3}$ (一次, 每约 52 步) |

如果按步数比较，ALS 一步的 FLOPs 相当于 SGD 的几十步——这显然不公平。

### 5.2 公平比较协议

$$\text{Protocol 运行至 } \sum_{t=1}^{T} \text{FLOPs}_t \geq \text{FLOPs}_{\text{BUDGET}}$$

**所有 Protocol 达到相同的总 FLOPs 预算，而非相同的总步数。**

对于 ALS，每步 FLOPs 的精确计数包括：

$$\text{FLOPs}_{\text{ALS}} = \underbrace{2N \cdot d_{\text{in}}^2}_{\text{计算 }X^T X} + \underbrace{\frac{1}{3}d_{\text{in}}^3}_{\text{Cholesky}} + \underbrace{2 \cdot n_{\text{blocks}} \cdot b \cdot d_{\text{in}}^2}_{\text{每块三角求解}}$$

### 5.3 评估标准的统一化

除 FLOPs 外, 所有 Protocol 共享：

1. **完全相同的数据加载器** — 相同的 batch, 相同的 shuffle seed
2. **完全相同的评估协议** — 相同的 eval dataloader, tokenizer, 评估频率
3. **完全相同的随机种子** — N=3 多种子验证
4. **完全相同的硬件环境** — 同一台机器, 同一种精度 (除 Protocol A 需要 float32)

### 5.4 Protocol A 在 7B 的特殊情况

Protocol A 在 Qwen2.5-7B (28 层) 上被深度边界阻断（11 次独立尝试均失败）。这是 ASP 的**内在算法限制**，而非硬件或配置问题。

对于一个真正的 2×2 7B 比较，Protocol A 缺失意味着：
- B vs D：可以比较（AdamW 下全秩 vs LoRA）✅
- C vs D：可以比较（LoRA 下 ASP vs AdamW）✅
- **A vs B 和交互效应 (A-B)-(C-D) 无法计算** ❌

这是论文明确记录的已知限制，也是 X1+X2 扩展试图解决的核心挑战。

### 5.5 代码层面：如何保证公平性

以下是 `altopt/trainer.py` 中保证公平性的关键代码：

```python
# ── 评估公平性保障 ──
class AltOptTrainer:
    def __init__(self, model, config, eval_dataloader=None, tokenizer=None):
        # ① 在 init 时固定所有随机种子
        if config.seed is not None:
            torch.manual_seed(config.seed)

        # ② 评估使用相同的 dataloader（在 init 时传入）
        self.eval_dataloader = eval_dataloader

    def _execute_step(self, batch):
        """
        ③ 所有 Protocol 使用相同的 batch，在同一设备上执行
        """
        device = self.device
        batch = {k: v.to(device) for k, v in batch.items()}

        if self.altopt is not None:
            return self.altopt.step(batch)       # ASP 路径
        elif self.lora_baseline is not None:
            return self.lora_baseline.step(batch) # LoRA+AdamW 路径
        elif self.peft_bridge is not None:
            return self._peft_altopt_step(batch)  # ASP+LoRA 路径
        else:
            return self._adamw_step(batch)        # 全秩 AdamW 路径

    def train(self, dataloader):
        """
        ④ 所有 Protocol 使用相同的训练数据 (相同 dataloader)
           在相同的 FLOPs 预算下停止 (通过 max_steps × FLOPs/step)
        """
        for batch in dataloader:
            loss = self._execute_step(batch)     # 转发到具体 Protocol
            self.state.global_step += 1

            # ⑤ 在相同的步数间隔进行评估
            if self.state.global_step % self.config.eval_every == 0:
                self._evaluate()

            if self.state.global_step >= self.config.max_steps:
                break
```

---

## 六、总结：ASP 的全秩与 LoRA 应用

| 方面 | 全秩 ASP (Protocol A) | LoRA ASP (Protocol C) |
|------|----------------------|----------------------|
| **ALS 操作** | 直接修改 `W.data` | ALS→$\Delta W$→$\Delta B$ 投影 |
| **核心投影公式** | 不需要 | $\Delta B = \Delta W \cdot A^T(AA^T+\lambda I)^{-1}/\alpha$ |
| **SGD 步骤** | 更新所有 494M 参数 | 只更新 ~3M LoRA 参数 |
| **扰动** | $\sigma_0=10^{-3}$, 全参数 | $\sigma_0=5\times 10^{-4}$, 仅 LoRA |
| **数值精度** | float32 | bfloat16 (ALS 阶段除外) |
| **7B 可行性** | ❌ 深度边界阻断 | ✅ X1 修复后可行 |
| **优势** | 完整的 ASP 机制 | 参数效率 + 不易过拟合 |
| **劣势** | 慢, 深度受限, 高 CV | ALS 相位需额外投影步骤 |
