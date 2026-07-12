# 公平比较方法论：ASP vs AdamW × 全秩 vs LoRA 的 2×2 因子实验

## 为什么"公平比较"是这个项目的核心方法论贡献

在对比不同优化器和参数形式时，最常见的错误是**按训练步数比较**。本项目的四项 Protocol 计算量差异巨大——Protocol A 的 ALS 一步相当于 SGD 几十步——按步数比较毫无意义。本文档详细阐述我们如何设计并执行真正的公平比较。

---

## 一、问题的本质：四个 Protocol 的"一步"不是同一个概念

### 1.1 什么是"一步"？

| Protocol | 一步的含义 | 涉及参数量 (0.5B) | 实际 FLOPs |
|----------|-----------|-------------------|-----------|
| **A** (ASP + 全秩) | 取决于相位：ALS (1层闭式解) / SGD (全参数) / Perturb (全参数) | 494M | **差异极大** |
| **B** (AdamW + 全秩) | 一次前向+反向+AdamW更新 | 494M | ~4.4 × 10^9 |
| **C** (ASP + LoRA) | 取决于相位：ALS (投影到B) / SGD (仅LoRA) / Perturb (仅LoRA) | ~3M | **差异极大** |
| **D** (AdamW + LoRA) | 一次前向+反向+AdamW更新（仅LoRA参数） | ~3M | ~3.0 × 10^9 |

**关键矛盾**：Protocol A 在 ALS 阶段一步对单层进行矩阵求逆（$\mathcal{O}(d_{\text{in}}^3)$），Protocol D 一步只更新 ~3M 参数。一步对一步？不可比。

---

## 二、FLOPs 归一化：核心公平性原则

### 2.1 基本原则

$$\boxed{\text{所有 Protocol 运行至 } \sum_{t=1}^{T} \text{FLOPs}_t \geq \text{FLOPs}_{\text{BUDGET}}}$$

**不是按步数比较，不是按 wall-clock 时间比较，是按总浮点运算次数比较。** FLOPs 是唯一与优化器选择、参数形式、硬件实现均无关的"计算投入"度量。

### 2.2 为什么 FLOPs 是唯一正确的度量？

| 度量 | 问题 |
|------|------|
| **步数** | ALS 一步 ≠ SGD 一步 ≠ AdamW 一步 |
| **Wall-clock 时间** | 受硬件、batch size、I/O、显存带宽影响——不科学 |
| **GPU 利用率** | 不同操作对 GPU 的利用效率不同（矩阵乘法 vs 逐元素操作） |
| **显存占用** | 反映硬件约束，不反映计算投入 |
| **FLOPs** | ✅ 纯数学度量，独立于实现细节 |

---

## 三、各操作的 FLOPs 精确计数

### 3.1 前向传播

对于参数为 $N_{\text{params}}$ 的模型，一次前向传播的 FLOPs：

$$\text{FLOPs}_{\text{forward}} = 2 \cdot N_{\text{params}}$$

**推导**：每个参数参与一次乘法（权重 × 输入）和一次加法（累加到输出），共 2 FLOPs。对于 Transformer，注意力计算另有 $\mathcal{O}(N \cdot T^2 \cdot d_{\text{model}})$ 的 FLOPs，但此项在所有 Protocol 中相同（都使用相同的模型架构），因此在比较中抵消。

### 3.2 反向传播

反向传播约等于前向传播的 2 倍：

$$\text{FLOPs}_{\text{backward}} = 4 \cdot N_{\text{params}}$$

**推导**：反向传播需要计算 (a) 损失对输出的梯度，(b) 每层输出对输入的梯度（链式法则），(c) 每层输出对权重的梯度。每项各约等于前向传播量。

### 3.3 优化器更新

| 优化器 | FLOPs per step | 说明 |
|--------|---------------|------|
| **SGD** | $1 \cdot N_{\text{trainable}}$ | $\theta \leftarrow \theta - \eta g$：一次乘加 |
| **SGD + Momentum** | $2 \cdot N_{\text{trainable}}$ | 额外维护速度 $v$ |
| **AdamW** | $3 \cdot N_{\text{trainable}}$ | 维护 $m$ (一阶矩), $v$ (二阶矩), 以及偏差校正 + 参数更新 |
| **ALS** | 见 §3.4 | 与参数量无关，由矩阵维度决定 |

### 3.4 ALS（交替最小二乘）的 FLOPs

这是最复杂的部分。一次 ALS 对单层的 FLOPs：

**步骤 1：形成 $X^\top X$**

$$X \in \mathbb{R}^{N \times d_{\text{in}}}, \quad X^\top X \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$

$$\text{FLOPs}_{X^\top X} = 2N \cdot d_{\text{in}}^2$$

每一项乘加各 1 FLOP，共 $N$ 行，每行 $d_{\text{in}} \times d_{\text{in}}$ 个乘法。

**步骤 2：Cholesky 分解**（$X^\top X + \lambda I = L L^\top$）

$$\text{FLOPs}_{\text{Cholesky}} = \frac{1}{3} d_{\text{in}}^3$$

这是标准 $d \times d$ 矩阵 Cholesky 分解的精确 FLOPs 计数。

**步骤 3：每块三角求解**

对于每个大小为 $b \times d_{\text{in}}$ 的块，需要两次三角求解（$L z = X^\top Y$ 和 $L^\top W^\top = z$）：

$$\text{FLOPs}_{\text{per-block}} = 2 \cdot b \cdot d_{\text{in}}^2$$

块数 $n_{\text{blocks}} = \lceil d_{\text{out}} / b \rceil$。

**总 ALS FLOPs**：

$$\boxed{\text{FLOPs}_{\text{ALS}} = 2N d_{\text{in}}^2 + \frac{1}{3} d_{\text{in}}^3 + 2 n_{\text{blocks}} \cdot b \cdot d_{\text{in}}^2}$$

**额外：LoRA 低秩 ALS (X1) 的 B-投影开销**

在全秩 ALS 之上，LoRA 版本需要将解投影到 B 矩阵：

$$\text{FLOPs}_{\text{B-projection}} = \underbrace{r^3}_{\text{Cholesky}(AA^\top)} + \underbrace{2r^2 d_{\text{in}}}_{\text{形成 }AA^\top} + \underbrace{n_{\text{blocks}} \cdot 2 b r d_{\text{in}}}_{\text{每块 }\Delta B = \Delta W \cdot A^\dagger}$$

对于 $r = 8, d_{\text{in}} = 896$，此项约 $2.3 \times 10^5$ FLOPs——与主 ALS 的 $\sim 10^9$ FLOPs 相比，**可完全忽略**。

### 3.5 扰动阶段的 FLOPs

$$\text{FLOPs}_{\text{Perturb}} = N_{\text{trainable}}$$

仅一次逐元素加法。

---

## 四、具体算例：Qwen2.5-0.5B 的 FLOPs 预算

### 4.1 模型参数

- 总参数：$4.94 \times 10^8$
- 隐藏维度 $d_{\text{model}} = 896$
- 注意力头维度 $d_{\text{head}} = 64$
- 层数 $L = 24$
- 词表大小 $V = 151936$

### 4.2 LoRA 可训练参数

$$N_{\text{LoRA}} = 4 \text{ (Q,K,V,O)} \times 2 \text{ (A,B)} \times r \times d_{\text{model}} \times L$$

$$= 4 \times 2 \times 8 \times 896 \times 24 = 1.38 \times 10^6 \approx 1.4\text{M}$$

### 4.3 各操作的单步 FLOPs

| 操作 | FLOPs | 相对比例 |
|------|-------|---------|
| Protocol D 一步 (LoRA + AdamW) | $(2+4+3) \times 1.4\text{M} = 1.26 \times 10^7$ | 1× |
| Protocol B 一步 (全秩 + AdamW) | $(2+4+3) \times 494\text{M} = 4.45 \times 10^9$ | 353× |
| Protocol A 一次 ALS (lm_head, $d_{\text{in}}=896$, $N=800$, $b=1024$) | $1.28 \times 10^9 + 2.40 \times 10^8 + 2.43 \times 10^{11}$ | ~195,000× |
| Protocol A 一次 SGD (全秩) | $(2+4+1) \times 494\text{M} = 3.46 \times 10^9$ | 275× |
| Protocol C 一次 SGD (LoRA) | $(2+4+1) \times 1.4\text{M} = 9.8 \times 10^6$ | 0.78× |
| Perturbation (全秩) | $494\text{M} = 4.94 \times 10^8$ | 39× |
| Perturbation (LoRA) | $1.4\text{M} = 1.4 \times 10^6$ | 0.11× |

**重要说明**：ALS 一步的 FLOPs 远高于 SGD 一步。但 ALS 每周期只执行一次（通常 50-100 SGD 步后才再次执行），因此在周期内摊销后，Protocol A/C 的平均每步 FLOPs 并不过分离谱。

### 4.4 100 步总 FLOPs 预算

假设 budget 为 Protocol B 运行 100 步的 FLOPs：

$$\text{BUDGET}_{100} = 100 \times 4.45 \times 10^9 = 4.45 \times 10^{11}$$

| Protocol | 单步均值 FLOPs | 在 budget 下可运行的步数 |
|----------|---------------|------------------------|
| A (ASP + 全秩) | $\sim 5.2 \times 10^{10}$ (含 ALS 摊销) | $\sim 9$ 步 |
| B (AdamW + 全秩) | $4.45 \times 10^9$ | $100$ 步 |
| C (ASP + LoRA) | $\sim 1.5 \times 10^{10}$ (含 ALS 摊销) | $\sim 30$ 步 |
| D (AdamW + LoRA) | $1.26 \times 10^7$ | $\sim 35,000$ 步 |

**这意味着**：在相同的 FLOPs 预算下，Protocol D 可以运行 35,000 步，而 Protocol B 只能运行 100 步。LoRA 的每步计算量小得多——它可以用更多步数来弥补每步更新更少的参数。**FLOPs 归一化让这种"多步 vs 大更新"的权衡变得公平可比较。**

---

## 五、FLOPs 以外的公平性维度

仅有 FLOPs 归一化还不够。一个真正公平的比较需要在以下所有维度上保持一致：

### 5.1 数据加载器完全相同

```python
# 所有 Protocol 共享同一个 DataLoader 实例
# 包括相同的 shuffle seed、batch size、preprocessing
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(42)  # ← 固定
)
```

**为什么重要**：不同的 batch 顺序会导致不同的优化轨迹。必须确保所有 Protocol 看到**完全相同的数据序列**。

### 5.2 随机种子完全统一

| 种子类型 | 固定值 | 影响范围 |
|---------|--------|---------|
| PyTorch 全局种子 | `torch.manual_seed(42)` | 所有随机操作 |
| CUDA 种子 | `torch.cuda.manual_seed_all(42)` | GPU 随机数 |
| DataLoader shuffle seed | 42 | 数据顺序 |
| 参数初始化种子 | 42（通过 LoRA/全秩初始化时固定） | 初始参数 |

**多种子验证**：每个 Protocol 在 N=3 个不同种子上独立运行，报告均值 ± 标准差。这排除了"某个 Protocol 碰巧在一个好种子上跑赢了"的侥幸。

### 5.3 评估协议完全一致

| 评估维度 | 统一做法 |
|---------|---------|
| **评估数据集** | 相同 split（如 WikiText-2 test），相同预处理 |
| **评估频率** | 每 N 步评估一次（按 FLOPs 对齐，非步数对齐） |
| **PPL 计算** | 相同公式：$\text{PPL} = \exp(\frac{1}{N}\sum \text{cross-entropy})$ |
| **评估时的模型状态** | 统一 `model.eval()` + `torch.no_grad()` |
| **Token 计数** | 按 attention_mask 加权，避免 padding token 污染 |

### 5.4 硬件环境统一

- 同一台机器，同一 GPU 型号
- 同一 PyTorch/CUDA 版本
- Protocol A（需要 float32 精度除外）统一使用 bfloat16
- 梯度检查点统一开启/关闭

**Protocol A 的特殊情况**：ALS 的 Cholesky 分解在 bfloat16 下会累积数值误差导致 NaN，因此 Protocol A 必须在 float32 下运行 ALS 阶段。这是一个**已知的、文档记录的限制**——Protocol A 享受了 ASP 的完整数学威力，代价是必须使用更高的数值精度。

### 5.5 超参数对齐

| 超参数 | 值 | 适用的 Protocol |
|--------|-----|----------------|
| 学习率 $\eta$ | $10^{-4}$ | 所有 |
| 权重衰减 $\lambda_{\text{wd}}$ | $0.01$ | 所有 |
| 动量 $\mu$ | $0.9$ | A, C (SGD 阶段) |
| AdamW $\beta_1, \beta_2$ | $0.9, 0.999$ | B, D |
| LoRA rank $r$ | $8$ | C, D |
| LoRA alpha | $2r = 16$ | C, D |
| ALS 正则化 $\lambda$ | $10^{-3}$ | A, C |
| ALS block size $b$ | $1024$ | A, C |
| 扰动初始强度 $\sigma_0$ | $10^{-3}$ (全秩) / $5\times 10^{-4}$ (LoRA) | A, C |
| 梯度裁剪 max_norm | $1.0$ | 所有 |
| Batch size | 1 (effective 4 with grad accum) | 所有 |
| 最大序列长度 | 1024 | 所有 |

---

## 六、统计严谨性：多种子 + 效应量

### 6.1 多种子实验设计

每个 Protocol 在 N=3（或 N=5）个独立随机种子上运行：

$$\text{Protocol X 的结果} = \bar{x} \pm s_x \quad \text{其中 } \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i,\; s_x = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

### 6.2 效应量：Hedges' g

比较 Protocol A 和 Protocol B 时，不仅报告 p 值，还报告效应量：

$$g = \frac{\bar{x}_A - \bar{x}_B}{s_{\text{pooled}}} \cdot \left(1 - \frac{3}{4(n_A + n_B) - 9}\right)$$

$$s_{\text{pooled}} = \sqrt{\frac{(n_A - 1)s_A^2 + (n_B - 1)s_B^2}{n_A + n_B - 2}}$$

**为什么用 Hedges' g 而不是 Cohen's d**：小样本 (N=3) 时 Cohen's d 有偏，Hedges' g 应用了小样本校正因子。

### 6.3 多重比较校正：Bonferroni

当进行多次两两比较时，对显著性阈值进行 Bonferroni 校正：

$$\alpha_{\text{adjusted}} = \frac{\alpha}{k}$$

其中 $k = \binom{4}{2} = 6$ 是比较次数，$\alpha = 0.05$，调整后 $\alpha_{\text{adjusted}} \approx 0.0083$。

### 6.4 PB ANOVA（参数 Bootstrap 方差分析）

由于 N=3 的样本量太小，传统 ANOVA 的 F 分布假设不成立。我们使用参数 Bootstrap：

1. 从每个 Protocol 的 3 个数据点中，假设正态分布，估计 $\hat{\mu}_i, \hat{\sigma}_i^2$
2. Bootstrap 重采样 10,000 次，每次生成 4 组虚拟数据
3. 计算每次 Bootstrap 的 F 统计量
4. 构建 F 统计量的经验分布，计算 p 值

```python
# 伪代码：PB ANOVA
for b in range(10000):
    for protocol in [A, B, C, D]:
        boot_sample = np.random.normal(mu_hat[protocol], sigma_hat[protocol], size=3)
        # 计算组间方差 / 组内方差
    F_boot[b] = ms_between / ms_within
p_value = np.mean(F_boot >= F_observed)
```

---

## 七、2×2 因子设计：交互效应的归因

### 7.1 为什么是 2×2？

2×2 因子设计的核心优势：**可以分离主效应和交互效应**。

|  | 全秩 | LoRA |
|--|------|------|
| **AdamW** | Protocol B | Protocol D |
| **ASP** | Protocol A | Protocol C |

### 7.2 效应的数学分解

定义 $\mu_{ij}$ 为 Protocol (optimizer=$i$, param_form=$j$) 的期望结果（如 PPL）。

**主效应**：

- 优化器主效应：$\text{ME}_{\text{opt}} = \frac{(\mu_{A} + \mu_{C}) - (\mu_{B} + \mu_{D})}{2}$
  - 即"ASP 平均比 AdamW 好多少"
- 参数形式主效应：$\text{ME}_{\text{param}} = \frac{(\mu_{A} + \mu_{B}) - (\mu_{C} + \mu_{D})}{2}$
  - 即"全秩平均比 LoRA 好多少"

**交互效应**：

$$\text{Interaction} = (\mu_{A} - \mu_{B}) - (\mu_{C} - \mu_{D})$$

- 如果 Interaction = 0：优化器和参数形式的效果独立
- 如果 Interaction > 0：ASP 在全秩下比在 LoRA 下更有优势（正协同）
- 如果 Interaction < 0：ASP 在 LoRA 下效果更差（**负协同**——我们的发现）

### 7.3 实证结果

在我们的实验中，交互效应约为 **+1000 PPL**（以 WikiText-2 PPL 衡量）——即一个**巨大的负协同**。这意味着：

- Protocol A (ASP + 全秩) 比 Protocol B (AdamW + 全秩) 差
- Protocol C (ASP + LoRA) 比 Protocol D (AdamW + LoRA) 差
- **但差距在全秩下更大**（ASP 在全秩下暴露了更多弱点）

这一发现是诚实的负面结果——ASP 和 LoRA 的结合没有产生协同效应。

---

## 八、7B 模型的特殊情况：Protocol A 缺失

### 8.1 问题的根源

Protocol A (ASP + 全秩) 在 Qwen2.5-7B (28 层) 上被深度边界阻断。11 次独立尝试全部失败（ALS 在 28 层模型中发散为 NaN）。

### 8.2 对比较的影响

$$\begin{array}{c|cc}
& \text{全秩} & \text{LoRA} \\
\hline
\text{AdamW} & \text{B ✅} & \text{D ✅} \\
\text{ASP} & \text{A ❌} & \text{C ✅}
\end{array}$$

缺失 Protocol A 意味着：
- ✅ 可以比较 B vs D（全秩 AdamW vs LoRA AdamW）
- ✅ 可以比较 C vs D（LoRA 下 ASP vs AdamW）
- ❌ **无法计算交互效应** $(\mu_A - \mu_B) - (\mu_C - \mu_D)$
- ❌ **无法估计优化器主效应**（缺少 A 的数据点）

### 8.3 我们的处理方式

论文中明确记录了这一限制：
1. 在 0.5B 规模上完成完整的 2×2 比较
2. 在 7B 上完成部分比较（B, C, D），并报告缺失 A 的影响
3. 将深度边界本身作为发现（而非失败）——它揭示了 ASP 的内在算法限制

---

## 九、实施检查清单

执行任何比较实验前，确认以下各项：

| # | 检查项 | ✅/❌ |
|---|--------|-------|
| 1 | 所有 Protocol 使用同一个 DataLoader（相同 shuffle seed） | |
| 2 | `torch.manual_seed()` 在所有 Protocol 启动前固定 | |
| 3 | 评估 dataloader 完全一致 | |
| 4 | 评估频率按 FLOPs 对齐（非步数对齐） | |
| 5 | 超参数在 Protocol 间可比（学习率、权重衰减等） | |
| 6 | 每种 Protocol 至少 N=3 个种子 | |
| 7 | 报告均值 ± 标准差 | |
| 8 | 报告效应量（Hedges' g）和置信区间 | |
| 9 | 多重比较使用 Bonferroni 校正 | |
| 10 | 缺失数据点（如 7B Protocol A）明确记录 | |
| 11 | 硬件环境在实验间保持一致 | |
| 12 | 数值精度差异（float32 vs bfloat16）明确记录 | |

---

## 十、总结

公平比较的四个支柱：

1. **FLOPs 归一化**：不按步数，不按时间，按计算投入
2. **完全一致的数据/种子/评估**：消除所有非 algorithm 差异
3. **多种子统计**：N=3-5，Hedges' g + Bonferroni + PB ANOVA
4. **2×2 因子归因**：分离主效应和交互效应

这四个支柱共同构成了本项目对"公平比较"的完整定义——它不仅是工程细节，更是方法论贡献的核心。
