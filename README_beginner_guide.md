# 从零理解 Alternating Optimization vs LoRA

> **一份面向初学者的完整项目讲解**
>
> 阅读本文后，你将理解：ASP (ALS+SGD+Perturbation) 的完整工作过程、四种策略 (Protocol A/B/C/D) 的比较标准、训练过程中的关键代码逻辑、以及 17 个实验背后的核心发现。

---

## 一、核心概念：两种不同的"优化思路"

### 1.1 什么是后训练 (Post-Training)？

假设你已经有了一个训练好的大语言模型，比如 **Qwen2.5-7B**。这个模型已经看过海量文本（约 18 万亿个 token），能够理解语言。

现在你想让这个模型**适应某个特定任务**——比如更好地理解维基百科的文风，或者学会做情感分类。这个"适应"的过程就叫**后训练 (Post-Training)**。

后训练与预训练的区别：
- **预训练**：从零开始，用海量数据教会模型语言的基本规则——耗时数月，需要数千张 GPU
- **后训练**：在一个已经训练好的模型基础上，用少量数据做微调——耗时几小时，通常只需要 1-2 张 GPU

### 1.2 后训练的本质：调整模型的参数

一个 7B 参数的大模型有大约 70 亿个参数（可以理解为 70 亿个旋钮）。后训练就是**微调这些旋钮**，让模型在新任务上表现更好。

但问题来了：**你是调整所有的 70 亿个旋钮，还是只调整其中很小的一部分？**

---

### 1.3 思路一 (LoRA)：只调整很少的参数

**LoRA** (Low-Rank Adaptation) 的核心思想：不需要调整全部 70 亿个参数，只需要调整**几百万个**就够了。

具体的数学形式：

$$\Delta W = \frac{\alpha}{r} \cdot B \cdot A$$

其中：
- $W$ 是原始的大矩阵（例如 4096 × 4096，约 1600 万个参数）
- $A$ 是一个很小的矩阵（r × 4096，r=8 时约 3 万个参数）
- $B$ 是另一个很小的矩阵（4096 × r，同样约 3 万个参数）
- $r$ 就是 **秩 (rank)**，通常默认为 r=8

**举例说明**：原始矩阵有 4096×4096 ≈ 1600 万个参数，LoRA (r=8) 只需要训练 2×(8×4096) ≈ 6.5 万个参数——减少了 99.6%。

**LoRA 背后的直觉**：语言模型在学习新任务时，其实不需要重新学习所有东西。比如，让一个已经会写英文的模型去写维基百科文章，它只需要在**少数几个方向上**调整参数就够了。这就像你学会了骑自行车，现在想学会骑摩托车——你不需要从头学平衡，只需要微调几个关键动作。

---

### 1.4 思路二 (ASP)：改变参数的更新方式

**ASP** (ALS+SGD+Perturbation) 的核心思想：保留更新所有参数的能力，但在**"如何更新"参数**上做创新。

标准的方法是**梯度下降**：每次看一小批数据，计算"往哪个方向走能让损失函数降低"，然后沿着那个方向走一小步。这就像在黑暗中摸着一座山往下走——每次只走一小步，沿着当前感觉最陡的方向。

ASP 提出了一个不同的方法：不是只用梯度下降，而是**混合使用三种更新方式**——交替进行。

| 方法 | 核心思想 | 类比 |
|------|---------|------|
| **LoRA** | 改变参数的存在形式（低秩 vs 全秩） | 换个更轻便的自行车 |
| **ASP** | 改变参数的更新方式（ALS+SGD+Perturbation） | 换一种骑法（同时用手和脚） |

---

## 二、ASP 的三阶段工作过程

ASP 不是一种单一的优化算法，而是**三个阶段的交替循环**：

```
ALS (1 步) → SGD (50 步) → Perturb (1 步) → 重复...
```

### Phase I: ALS (Alternating Least Squares)

**直观理解**：把模型的某一层看作一个大矩阵 W。ALS 的做法是：

1. 把 W 按行分成很多小块（每块 1024 行）
2. 对每一块，用数学公式**精确求解**：
   - 给定当前的输入 X，什么样的 W 能最好地预测目标输出 Y？
   - 答案：$W_{\text{new}} = (X^T X + \lambda I)^{-1} X^T Y$
3. 这是一次性给出**闭式解** (closed-form solution)——不需要迭代

**ALS 代码**（来自 `altopt/als.py`，第 525-570 行，简化版）：

```python
# 对于 LoRA 参数化的层，先计算"有效权重"
# W_eff = W_base + (α/r) * B * A
effective_W = base_W + scaling * (lora_B @ lora_A)

# 构建正则化矩阵 (X^T X + λI)
XtX = X.T @ X                                      # [d_in, d_in]
reg = self.reg_lambda * torch.eye(d_in)            # λI
XtX_reg = XtX + reg                                # 正定矩阵

# 按块求解：把输出维度分成 b=128 的小块
for i in range(n_blocks):
    start = i * block_size
    end = min(start + block_size, d_out)
    
    # Y_block: 当前块的"目标输出" [N, b]
    Y_block = X @ effective_W[start:end, :].T
    
    # 解线性方程：(X^TX + λI) · W_new^T = X^T Y
    # torch.linalg.solve 利用矩阵是正定的这一性质，
    # 内部使用高效的 Cholesky 分解
    W_new_T = torch.linalg.solve(XtX_reg, X.T @ Y_block)
    W_new_block = W_new_T.T                         # [b, d_in]
    
    # 计算新旧权重的差异
    delta_W = W_new_block - effective_W[start:end, :]
    
    # 关键步骤：把差异投影回 LoRA 的 B 矩阵
    # delta_B = ΔW · A^T · (AA^T + λI)^{-1} / α
    delta_B = delta_W @ A_pinv.T / scaling
    
    # in-place 更新 B 矩阵（只改 LoRA 参数！）
    lora_B[start:end, :] += delta_B
```

**类比**：梯度下降像是在黑暗中摸索着下山（每次一小步），ALS 像是**直接看地图找到最低点**（一次到达）。ALS 更精确，但计算量也更大（需要矩阵求逆）。

### Phase II: SGD (Stochastic Gradient Descent)

ALS 虽然精确，但它有一个致命缺陷：它**只看当前这一层，不考虑其他层的变化**。

想象一个 28 层的 Transformer：
- 你改了第 5 层的参数
- 但第 6-28 层的参数还是旧的
- 第 6 层本来学到的东西是基于"第 5 层输出应该是这样的"的假设
- 现在第 5 层变了，但第 6 层还不知道！

这就像你搬家了，但快递员还在往旧地址送货。SGD 的作用就是**通知所有层**："嘿，上游变了，大家赶紧适应一下！"

```python
# SGD 步骤的核心代码
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for step in range(50):              # 50 步 SGD
    loss = model(batch).loss        # 前向传播
    loss.backward()                 # 反向传播（计算梯度）
    optimizer.step()                # 沿梯度方向更新参数
    optimizer.zero_grad()           # 清空梯度，准备下一步
```

### Phase III: Perturbation (扰动)

这是最反直觉的部分：**故意给参数加一点随机噪声**。

为什么要这么做？研究 (SAM, RWP) 表明：
- 标准 SGD 容易陷入**窄的局部最优**（一个很小的坑，在训练数据上表现不错，但泛化能力差）
- 加适量噪声可以把模型**"推"出窄坑**，让它滚到旁边更宽、更平的盆地
- 更平坦的最优点 → 更好的泛化能力（在未见过的数据上表现更好）

```python
# 扰动代码
with torch.no_grad():
    for param in model.parameters():
        noise = sigma * torch.randn_like(param)    # 高斯噪声
        param.add_(noise)                          # 加到参数上
```

**参数设置**：
- 扰动尺度 $\sigma_0 = 10^{-3}$（全秩）或 $5 \times 10^{-4}$（LoRA）
- 余弦衰减：$\sigma_c = \sigma_0 \cdot 0.5(1 + \cos(\pi c / C_{\max}))$

---

## 三、2×2 因子设计：四种策略的公平比较

### 3.1 为什么需要 2×2 设计？

假设你直接比较两种方法：
- **方法 1**：ASP 优化器 + 全秩参数 (Protocol A)
- **方法 2**：AdamW 优化器 + LoRA 参数 (Protocol D)

如果方法 2 更好，你无法判断是**因为用了 AdamW**，还是**因为用了 LoRA**——两个变量同时变了！

这就是 **2×2 因子设计**（2×2 factorial design）要解决的问题。它把两个独立变量**交叉**，形成四种实验条件：

| | 全秩参数 (Full-Rank ΔW) | LoRA 参数 (ΔW = BA, r ≪ d) |
|---|----------------------|-------------------------|
| **ASP 优化器** | Protocol A | Protocol C |
| **AdamW 优化器** | Protocol B | Protocol D |

### 3.2 每种比较的含义

| 比较 | 改变的变量 | 控制的变量 | 能回答什么问题？ |
|------|-----------|-----------|----------------|
| **A vs B** | 优化器类型 | 参数形态 = 全秩 | "在全秩条件下，ASP 比 AdamW 好吗？" |
| **C vs D** | 优化器类型 | 参数形态 = LoRA | "在 LoRA 条件下，ASP 比 AdamW 好吗？" |
| **A vs C** | 参数形态 | 优化器 = ASP | "在 ASP 下，全秩比 LoRA 好吗？" |
| **B vs D** | 参数形态 | 优化器 = AdamW | "在 AdamW 下，全秩比 LoRA 好吗？" |
| **(A-B)-(C-D)** | 交互效应 | — | "优化器的效果是否依赖于参数形态？" |

### 3.3 公平比较的关键规则

不同优化器的计算成本不同：
- ALS：一次矩阵求逆 $\mathcal{O}(b^3)$ ——昂贵但精确
- SGD：一次前向+反向传播 $\mathcal{O}(d^2)$ ——适中
- AdamW：一次前向+反向传播+两次动量状态更新 ——比 SGD 稍贵

所以不能按**步数**比较，必须按**总计算量 (FLOPs)** 比较。

```
最终规则：所有 Protocol 都运行到相同的总 FLOPs 预算
```

---

## 四、训练过程详解

### 4.1 基本设置

| 参数 | 值 | 为什么？ |
|------|-----|---------|
| 学习率 | 1e-4 | 标准微调学习率，恒定不变（无需预热或衰减） |
| LoRA 秩 r | 8 | 默认值；本文证明 r=8 在 95%+ 模型上已足够 |
| ALS 块大小 b | 1024 | 平衡矩阵求逆开销 O(b³) 和块独立性 |
| ALS 正则化 λ | 1e-4 | 防止矩阵求逆时出现奇异矩阵 |
| 扰动尺度 σ₀ | 1e-3 (全秩) | 足够大到跳出局部最优，足够小到不破坏训练 |
| 序列长度 | 1024 | 平衡上下文需求和显存占用 |
| 批次大小 | 1 (梯度累积=4) | 有效批次=4，节省显存 |
| 硬件 (7B) | 2× RTX 5090, DeepSpeed ZeRO-2 | 把大模型分到两张 GPU，每张用 24GB |
| 随机种子 | 42, 123, 456 | N=3 多 seed 验证 |

### 4.2 Protocol B 训练流程（AdamW + 全秩，7B 规模）

```
1. 加载 Qwen2.5-7B 模型 (7B 参数, float16, ~14GB)
2. DeepSpeed ZeRO-2：分到 2 张 RTX 5090，每张用 ~24GB
3. 加载 1600 个 WikiText-2 训练样本
4. 设置 AdamW 优化器：
   - lr = 1e-4
   - weight_decay = 0.01
   - betas = (0.9, 0.999)
5. 训练循环 800 步：
   for batch in dataloader:
       outputs = model(batch)          # 前向传播
       loss = outputs.loss             # 计算交叉熵损失
       loss.backward()                 # 反向传播
       clip_grad_norm(1.0)            # 梯度裁剪
       optimizer.step()               # AdamW 更新参数
       optimizer.zero_grad()          # 清空梯度
6. 每 100 步评估困惑度
7. 每 200 步保存检查点
8. 最终 PPL = 1.25 ± 0.01 (N=3 seeds, full test set)
```

### 4.3 Protocol A 训练流程（ASP + 全秩）

与 Protocol B 相似，但用 ASP 循环替代单纯的 AdamW：

```python
n_cycles = max_steps // (sgd_per_cycle + 2)   # ex: 1200 // 52 ≈ 23

for cycle in range(n_cycles):
    # Phase I: ALS (1 step)
    als_solver.solve_block(batch, block_size=1024)
    
    # Phase II: SGD (50 steps)
    optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9)
    for i in range(50):
        loss = model(batch).loss
        loss.backward()
        clip_grad_norm(1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    # Phase III: Perturb (1 step)
    add_gaussian_noise(model.parameters(), sigma=1e-3)
```

**训练时长对比 (Qwen2.5-0.5B, 100 steps)**：
- Protocol A (ASP+full-rank)：ALS ~2s + SGD ~30s + Perturb ~1s = **~33s**
- Protocol B (AdamW+full-rank)：纯 AdamW ~30s = **~30s**
- Protocol D (LoRA r=8)：纯 AdamW ~25s = **~25s**

### 4.4 关键评估指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **困惑度 (PPL)** | $\text{PPL} = e^{\text{avg\_loss}}$ | 越低越好；衡量模型对文本的"惊讶程度" |
| **下游准确率** | HellaSwag / MMLU / ARC | 评估真实的语言理解能力 |
| **M-index** | $M = \text{PPL}_{\text{train}} / \text{PPL}_{\text{cross}}$ | M < 1 = 过拟合（死记硬背），M > 2 = 泛化 |
| **效率比** | $\frac{\text{PPL}_A}{\text{PPL}_B} \cdot \frac{N_B}{N_A}$ | 衡量每参数的"性价比" |

---

## 五、秩曲线实验代码讲解

这是产生"r=8 通用平坦区"核心发现的实验 (`experiments/_xval.py`)。

### 5.1 整体实验流程

```python
def run_one_model(name, model_path, targets):
    results = []
    
    # ── Step 1: 基线评估 ──
    model = AutoModelForCausalLM.from_pretrained(model_path)
    baseline_ppl = compute_perplexity(model, eval_dataloader)
    # 这告诉我们：模型在训练前有多差？
    results.append({"run": "baseline", "ppl": baseline_ppl})
    
    # ── Step 2: Protocol B (AdamW + 全秩) ──
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.gradient_checkpointing_enable()  # 节省显存
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    for step in range(100):
        loss = model(batch).loss
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    b_ppl = compute_perplexity(model, eval_dataloader)
    results.append({"run": "B_full", "ppl": b_ppl})
    
    # ── Step 3: Protocol D (AdamW + LoRA, 不同秩) ──
    for rank in [8, 32, 256]:
        base = AutoModelForCausalLM.from_pretrained(model_path)
        
        # 用 PEFT 库给模型加上 LoRA 适配器
        model = get_peft_model(base, LoraConfig(
            r=rank,                     # 秩
            lora_alpha=rank * 2,       # α = 2r
            lora_dropout=0.05,         # LoRA dropout
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        ))
        
        # 只优化 LoRA 参数
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=1e-4)
        
        for step in range(100):
            loss = model(batch).loss
            loss.backward()
            clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        ppl = compute_perplexity(model, eval_dataloader)
        results.append({"run": f"D_r{rank}", "ppl": ppl})
    
    return results
```

### 5.2 评估函数

```python
def compute_perplexity(model, dataloader, device):
    """计算困惑度 (Perplexity)"""
    model.eval()  # 切换到评估模式（关 dropout/batch norm）
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():  # 不计算梯度，节省显存
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            n_tokens = batch["attention_mask"].sum().item()
            
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    model.train()  # 切回训练模式
    return perplexity.item()
```

### 5.3 下游评估代码 (`experiments/_finalize3.py`)

```python
# 加载已训练的检查点
model = load_model_from_checkpoint(checkpoint_path)

# 如果是 LoRA 模型 (Protocol C/D)，先合并 LoRA 参数到基模型中
if is_peft_model:
    model = model.merge_and_unload()  # 把 BA 合并进 W，变成标准 HF 模型

# 保存为 HuggingFace 格式
model.save_pretrained(temp_dir)
tokenizer.save_pretrained(temp_dir)

# 用 lm-eval-harness (EleutherAI 标准评估工具) 跑评估
from lm_eval import simple_evaluate

results = simple_evaluate(
    model="hf",
    model_args={"pretrained": temp_dir},
    tasks=["hellaswag", "mmlu", "arc_challenge"],
    num_fewshot=0,  # 0-shot: 不给例子直接问
    limit=None,     # 使用全部评估数据
)
```

---

## 六、评估标准 → [详见评估指南](docs/evaluation_guide.md)\n\n## 七、核心实验发现总结

### 6.1 秩充足律 (Rank Sufficiency Law)

$$r_{\min} = \eta \cdot \frac{L}{d_h}$$

其中 $\eta$ 由预训练质量调制：
- 强预训练模型 (Qwen, 18T tokens)：$\eta \approx 150$，$r_{\min} \approx 4$
- 紧凑模型 (SmolLM2, 2T tokens)：$\eta \approx 230$，$r_{\min} \approx 12$

**关键数字**：
- r=8 在 **12/13 测试模型**上已处于平坦区
- SmolLM2-135M 是**唯一例外**（需要 r≥12）
- r=4 在 Qwen2.5-0.5B 上有效（PPL=1.63 vs r=8 PPL=1.62）
- r=4 在 SmolLM2-135M 上**灾难性失败**（PPL=88.22 vs PPL=1.76）

### 6.2 PPL ≠ 泛化

全秩微调在 WikiText-2 上能达到近乎完美的困惑度 (PPL=1.25)，但这是**死记硬背**的结果：

| 任务 | 未训练的基线 | LoRA r=8 | 全秩 | 谁赢？ |
|------|-----------|----------|------|--------|
| **WikiText-2 PPL** | 133.16 | 10.41 | **1.25** | 全秩 (but...) |
| **C4 PPL** (跨域) | 79.44 | **2.30** | 2.42 | **LoRA 胜** |
| **HellaSwag** | **59.9%** | **59.7%** | 56.7% | LoRA = 基线 |
| **MMLU** | — | **76.3%** | 72.2% | **LoRA 胜 +4.1pp** |
| **ARC** | — | **50.4%** | 47.2% | **LoRA 胜 +3.2pp** |
| **SST-2** | — | 84.7% | — | r=4/8/32 完全相同 |

### 6.3 ASP 的深度边界

| 层数 | 模型示例 | Protocol A 结果 |
|------|---------|----------------|
| ≤ 24 | GPT-2, OPT-125m, Qwen-0.5B | ✅ 收敛（慢但稳定） |
| = 12 | OPT-125m (真 Cholesky ALS) | ✅ 非单调收敛，400-600 步最优 |
| ≥ 28 | Qwen2.5-7B, Mistral-7B, SmolLM2 | ❌ 灾难性发散 (PPL → NaN/1.2M) |

**原因**：ALS 在某一层的扰动通过残差连接逐层放大。$\bar{\rho} \approx 1.08$ 的放大因子在 28 层累积后导致输出完全崩溃。

### 6.4 M-index：轻量级过拟合诊断

$$M = \frac{\text{PPL}_{\text{训练域}}}{\text{PPL}_{\text{跨域}}}$$

- $M < 1$ → **过拟合**（模型在训练数据上表现得比泛化数据上好）→ 全秩微调总是落在这个区域
- $M > 2$ → **泛化**（模型学到的东西可以迁移到其他数据）→ 所有 LoRA 配置都在这个区域
- $M \approx 1.73$ → 自然域偏置（维基百科文本天生比网页文本简单）

**只需要两次 PPL 评估**就能诊断模型是否过拟合，不需要跑下游任务！

---

## 七、环境配置与运行

### 硬件要求

| 规模 | 硬件 | 显存 | 时间 |
|------|------|------|------|
| ≤ 1.1B 参数 | CPU (Intel Xeon) | — | 2-5 分钟 |
| 7B 参数 (LoRA) | 1× RTX 5090 (32GB) | ~10GB | 10-15 分钟 |
| 7B 参数 (全秩) | 2× RTX 5090 + DeepSpeed ZeRO-2 | ~24GB/GPU | 50-60 分钟 |

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/gingersea/alternating-optimization-lora.git
cd alternating-optimization-lora

# 安装依赖
pip install -r requirements.txt

# 运行单模型秩曲线
python experiments/_xval.py

# 运行下游评估
python experiments/_finalize3.py --protocols B,D --tasks hellaswag

# 运行证伪实验
python experiments/_falsify.py

# 生成论文 PDF
cd paper && pdflatex paper_v3.3.tex && pdflatex paper_v3.3.tex
```

---

## 八、项目文件结构

```
alternating-optimization-lora/
│
├── paper/                           # 📄 论文文件
│   ├── paper_v3.3.tex               #   LaTeX 主稿 (v3.4, 16页, 6张图)
│   ├── paper_v3.3.pdf               #   编译后的 PDF
│   ├── paper_draft_v0.2.md          #   Markdown 参考版
│   ├── figures/                     #   6张期刊级矢量图
│   │   ├── fig1_factorial.pdf
│   │   ├── fig2_convergence.pdf
│   │   ├── fig3_depth.pdf
│   │   ├── fig4_overfitting.pdf
│   │   ├── fig5_als_synergy.pdf
│   │   └── fig6_nomogram.pdf
│   └── review_round{1-6}.md         #   六轮审稿记录
│
├── altopt/                          # 🧠 核心框架库
│   ├── framework.py                 #   三阶段交替循环调度器
│   ├── als.py                       #   ALS 块求解器 (Cholesky → linalg.solve)
│   ├── sgd.py                       #   SGD 优化器封装
│   ├── perturbation.py              #   扰动调度器 (余弦衰减)
│   ├── lora.py                      #   LoRA 基线实现
│   ├── trainer.py                   #   统一训练器 (含 DeepSpeed/FSDP)
│   ├── checkpoint.py                #   检查点管理
│   ├── evaluation.py                #   统一评估协议
│   └── profiling/                   #   FLOPs 计数和显存追踪
│
├── experiments/                     # 🧪 实验脚本
│   ├── _xval.py                     #   跨架构秩曲线验证
│   ├── _falsify.py                  #   证伪实验 (Mistral r=4, SmolLM2 r=6/16)
│   ├── _finalize3.py               #   多 seed 下游评估 (HellaSwag/MMLU/ARC)
│   ├── _p0_chinese_wt.py            #   P0: 中文 WikiText 实验
│   ├── _p1_crossover.py             #   P1: ASP 收敛交叉点
│   ├── _f1_eta_mechanism.py         #   F1: η 机制归因
│   ├── _f2_full_asp.py              #   F2: 真实 Cholesky ALS 实验
│   ├── _a_sst2.py                   #   A: SST-2 分类秩曲线
│   ├── _e4_ffn_lora.py              #   E4: FFN LoRA 实验
│   ├── _x3_gpt2_opt.py              #   X3: GPT-2/OPT-125m 填充实验
│   └── configs/                     #   实验配置文件
│
├── docs/                            # 📚 文档
│   ├── causal_depth_boundary.md     #   X2: 因果深度边界理论
│   └── experiment-registry.md       #   实验注册表
│
├── runs/                            # 📊 实验结果
│   ├── cross_arch/                  #   跨架构秩曲线数据
│   ├── falsify/                     #   证伪实验数据
│   ├── qwen25_7b_800s/              #   7B 实验数据
│   ├── param_matched_baseline/      #   参数匹配基线数据
│   └── x3_nomogram/                 #   η 列线图数据
│
├── README.md                        # 项目主页
├── README_beginner_guide.md         # 本文档
└── todo.md                          # 项目状态追踪
```

---

## 九、17 个实验清单

| ID | 实验名称 | 核心发现 |
|----|---------|---------|
| **P0** | 中文 WikiText | r=8 语言无关；η∝H 被推翻 |
| **P1** | ASP 收敛交叉点 | SGD+Perturb 领先 AdamW 28% (800 步, GPT-2) |
| **P2** | T5 编码器-解码器 | 边界条件确认——LM PPL 不兼容 |
| **P3** | M-index 跨规模校准 | β 规模依赖相变 (β₀.₅B≈−0.03 vs β₇B≈0.28) |
| **P4** | SmolLM2 细粒度 r_min | r_min≈12±1 确认 (10 个秩点) |
| **P5** | 多 seed 秩曲线 | SE<0.002; max\|Δ\|=0.0055 |
| **F1** | η 机制归因 | 任务稳定——H 和 N_samples 候选机制均被推翻 |
| **F2** | 真实 Cholesky ALS | 非单调——PPL 最优在 400-600 步 |
| **A** | SST-2 分类 | r=4/8/32 准确率完全相同 (84.7%, 739/872) |
| **E4** | FFN LoRA | attn+FFN r=4 优于 attn-only r=8 |
| **Critical** | SmolLM2 r=4 | PPL=88.22 → η 是模型特异性的 |
| **E2** | 长时间秩稳定性 | r=8 在 1600 步最优；r=256 过拟合 |
| **X1** | 低秩 ALS 求解器 | linalg.solve 替代 Cholesky — 7B 可运行 |
| **X2** | 因果深度边界 | SCM 框架, 5 个可证伪预测 |
| **X3** | 通用 η 列线图 | 12 模型查表, R²=0.88 |
| **X3+** | OPT-125m 秩曲线 | r4/r8=1.28, η≈200 |
| **E1** | η 与训练预算无关 | r4/r8=1.006 常数 (N=400/800/1600) |

---

## 十、总结与建议

### 如果你是 LoRA 的实践者

1. **默认使用 r=8**。在 WikiText-2 式后训练中，r=8 和 r=256 的 PPL 差异不超过 0.02。
2. **永远不要在数据量少于 1 万条时使用全秩微调**——它会严重过拟合。
3. **用 M-index 诊断过拟合**：计算训练域和 C4 的 PPL 比值。如果 M < 1，你的模型在死记硬背。
4. **用 η 列线图选择秩**：$r_{\min} = \max(8, \lceil\eta \cdot L/d_h\rceil)$。

### 如果你是 ALS 的研究者

1. ALS 的深度边界是真实的——不要试图在 ≥28 层的模型上使用 ALS。
2. 在 12 层模型上，ASP 在 800 步时开始超越 AdamW（GPT-2: +28% PPL 改善）。
3. ALS 提供隐式正则化——train≈eval 在 1200 步时仍然保持。
4. 我们的 CG→linalg.solve 修复让 LoRA-ALS 在 7B 上可用（`altopt/als.py`）。

---

## 引用

论文 v3.4 全文：[/paper/paper_v3.3.pdf](paper/paper_v3.3.pdf)

GitHub 仓库：[https://github.com/gingersea/alternating-optimization-lora](https://github.com/gingersea/alternating-optimization-lora)

**17 个实验全部完成。v3.4 FINAL。**
