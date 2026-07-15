# Alternating Optimization Framework (ASP) vs LoRA

> **统一评分体系下的后训练方法对比研究：交替最小二乘（ALS）+ 随机梯度下降（SGD）+ 随机扰动 vs 低秩适配（LoRA）**
>
> A unified evaluation framework for comparing post-training strategies. Core question: can the ALS-SGD-Perturbation (ASP) alternating protocol be a superior post-training optimizer compared to the dominant LoRA+AdamW paradigm?
>
> **状态**: 论文 v0.7 — Round 6 对抗评审结论为 **Major Revision**
> **Phase B 完成**: Qwen2.5-7B full-rank 3/3 seeds, PPL 1.25 ± 0.01  
> **实验注册表**: [docs/experiment-registry.md](docs/experiment-registry.md) — 全 5 架构, 8 模型, 50-800 步  
> **当前路线图**: [todo.md](todo.md) — 证据审计、参数量匹配与下游评估优先

---

## 📑 文档导航

### 核心文档

| 文档 | 说明 |
|------|------|
| **[综合报告](docs/synthesis-report.md)** | 全部实验结论 + 论文思路 + 论证链 |
| **[实验注册表](docs/experiment-registry.md)** | 🆕 全 5 架构 × 4 协议 × 50-800 步矩阵 |
| **[当前路线图](todo.md)** | 最新差距评估、证据边界与优先级（当前状态入口） |
| **[最终评估](docs/final_assessment.md)** | 2026-06-13 历史快照，已被后续评审取代 |
| **[切合度审计](docs/alignment_audit.md)** | 2026-06-20 Phase B 历史快照 |
| **[数学分析](docs/math-analysis.md)** | ALS loss 量级、收敛理论、扰动正则化、文献引用 |
| **[AltOpt 形式化](docs/framework.md)** | 框架的数学定义与推导 |
| **[比较难题分析](docs/comparison-challenges.md)** | 公平比较的方法论分析 |
| **[相关工作](docs/literature.md)** | 文献综述 |

### 论文 + 评审

| 版本 | 日期 | 评审决策 |
|------|------|---------|
| **[v0.7](paper/paper_draft_v0.7.md)** (canonical draft) | 06-20 | Round 6: Major Revision |
| [v0.4](paper/paper_draft_v0.7.md) | 06-13 | +Round 8-9 findings (superseded in-place) |
| [v0.3](paper/paper_draft_v0.7.md) | 06-12 | Round 2 Minor Revision fixes (superseded in-place) |
| [v0.1](paper/paper_draft_v0.1.md) | 06-12 | Round 1 Major Revision |

| 评审轮次 | 决策 | 文档 |
|----------|------|------|
| Round 1 | Major Revision (7 Required) | [`review_round1.md`](paper/review_round1.md) |
| Round 2 | Minor Revision (5 MINOR) | [`review_round2.md`](paper/review_round2.md) |
| Round 3 | **ACCEPT** (5 text fixes) | [`review_round3.md`](paper/review_round3.md) |
| Round 5 | Minor Revision | [`review_round5.md`](paper/review_round5.md) |
| **Round 6** | **Major Revision** | [`review_round6.md`](paper/review_round6.md) |

### 实验报告 (9+ 轮)

| # | 报告 | 模型 | 核心发现 |
|---|------|------|----------|
| 1 | [report-001](docs/experiment-report-001.md) | GPT-2 | 2×2 框架可行；Protocol C FLOPs 效率 |
| 2 | [report-002](docs/experiment-report-002.md) | OPT-125m | LoRA 主导；ALS:SGD=1:20 最优 |
| 3 | [report-003](docs/experiment-report-003.md) | — | 7B 基础设施 + RQ 消融框架 |
| 4 | [report-004](docs/experiment-report-004.md) | GPT-2 | 可复现性差；扰动正则化 |
| 5 | [report-005](docs/experiment-report-005.md) | OPT-125m | 3-seed 统计 2×2；交互效应 1197 |
| 6 | [matrix experiment] | OPT+Qwen | 50-800步 gap 矩阵 |
| 7 | — | GPT-2+OPT | 交叉点未达成；overfitting 发现 |
| 8 | [round8](docs/round8_results.md) | OPT-125m | 1200步 crossover；Protocol C 协同 |
| 9 | [round9](docs/round9_overfitting.md) | OPT-125m | AdamW 过拟合确认 |
| **10** | **[Phase B](docs/experiment-registry.md)** 🆕 | **Qwen2.5-7B** | **Full-rank PPL 1.25, 深度边界证实** |

### 缺陷 + 修订

| 文档 | 说明 |
|------|------|
| [flaw-analysis-001](docs/flaw-analysis-001.md) | GPT-2 Conv1D 架构缺陷 |
| [revision_plan](paper/revision_plan.md) | Round 1 修订方案 |
| [multi_seed_results](paper/multi_seed_results.json) | 5 step × 2 model 多 seed 汇总 |

### 实验脚本

| 脚本 | 说明 |
|------|------|
| `experiments/runner.py` | CLI 实验执行器 |
| `experiments/matrix_runner.py` | 50-800步矩阵实验 |
| `experiments/round5_runner.py` | Round 5 (3 seeds) |
| `experiments/round6_runner.py` | Round 6 (长 SGD 周期) |
| `experiments/ablation.py` | RQ1-RQ6 消融 |
| `experiments/statistical_analysis.py` | PB ANOVA + Fieller CI |
| `experiments/run_experiment_004.py` | 实验 #004 |
| `experiments/visualization.py` | 6 种图表类型 |

| # | 文档 | 内容 |
|---|------|------|
| 1 | [`flaw-analysis-001`](docs/flaw-analysis-001.md) | GPT-2 Conv1D 架构与 LoRA 不兼容性分析 |

---

## 项目目标 (Project Objectives)

### 总体目标

在一个**统一的评分与资源核算体系**下，系统比较两类后训练方法——交替优化框架（ALS + SGD + 随机扰动）与低秩适配（LoRA）——在大语言模型后训练场景中的性能、效率与泛化能力。

### 具体目标

1. **建立公平比较协议**：设计 2×2 析因实验，将「优化器/更新策略」与「参数形态（全秩 vs 低秩）」两类独立变量解耦，使性能差异可归因。

2. **统一评分体系**：在相同 FLOPs 预算、相同显存约束、相同 wall-clock 时间三种尺度下，用统一的 evaluation protocol（perplexity + downstream task accuracy）评估所有方法。

3. **量化 ALS 的计算成本是否值得**：ALS 的矩阵求逆开销（O(b³)）何时被其全局拟合优势所抵消？找出 ALS:SGD 最优调度比。

4. **回答 LoRA 低秩流形是否改变交替优化的逃逸局部最优能力**：LoRA 的低秩投影是否削弱了随机扰动（Phase III）的效果？

5. **探索 ALS-SGD-扰动框架与 LoRA 的协同可能性**：Protocol C（LoRA 参数结构 + AltOpt 优化器）是否能同时获得两者的优势？

---

## 名词解释 (Glossary)

| 术语 | 英文 | 解释 |
|------|------|------|
| **后训练** | Post-training | 在预训练完成的大模型基础上，用特定任务数据进一步调整参数的过程。区别于从随机初始化开始的预训练（pre-training）。 |
| **交替最小二乘** | ALS (Alternating Least Squares) | 将参数矩阵按行分块，每次固定其他块，对当前块做闭式最小二乘求解，需要矩阵求逆。求解方式是精确的（exact），但计算开销大。 |
| **随机梯度下降** | SGD (Stochastic Gradient Descent) | 每次取一个小批量数据，沿负梯度方向更新参数。细粒度、计算开销适中，但容易陷入局部最优。 |
| **随机扰动** | Stochastic Perturbation | 在参数空间中注入受控噪声，帮助优化器跳出窄的局部极小值。类似 simulated annealing 的思想，但作用于参数空间。 |
| **低秩适配** | LoRA (Low-Rank Adaptation) | 将参数更新限制在低秩子空间内：ΔW = (α/r)·BA，其中 B ∈ ℝ^{d×r}，A ∈ ℝ^{r×k}，r ≪ min(d,k)。大幅减少可训练参数数量，默认绑定 AdamW 优化器。 |
| **交替优化框架** | AltOpt (Alternating Optimization) | 将 ALS、SGD、扰动三种机制按阶段调度，交替执行的参数更新策略。本质上是**优化器创新**（决定参数「怎么更新」），而非参数结构创新。 |
| **全秩更新** | Full-Rank ΔW | 直接更新完整的权重矩阵 W，不施加低秩约束。AltOpt 的默认工作模式。 |
| **AdamW** | AdamW | 带权重衰减的 Adam 优化器，LoRA 的默认优化器。属于自适应矩估计（adaptive moment estimation）类方法。 |
| **FLOPs 预算** | FLOPs Budget | 浮点运算总次数上限。ALS 单步 FLOPs 远高于 SGD 单步，因此不能按「步数」比较，必须按「总计算量」比较。 |
| **析因实验** | Factorial Experiment | 同时变化两个因子（优化器类型 × 参数形态），形成 2×2=4 种协议，以分离主效应和交互效应。 |
| **参数形态** | Parameter Form | 指参数更新以何种数学结构存在：全秩矩阵（full-rank）或低秩分解（low-rank BA）。这是独立于优化器选择的另一个维度。 |
| **评分体系** | Evaluation Protocol | 统一的评估标准：包括训练损失、验证困惑度（perplexity）、下游任务准确率、以及资源消耗（FLOPs、显存、时间）的归一化指标。 |

---

## 问题背景

给定一个预训练好的大语言模型，其参数为 θ₀ ∈ ℝᵈ。后训练的目标是在任务数据集 D = {(xᵢ, yᵢ)} 上找到更新后的参数 θ*，最小化经验风险。

两种主流的后训练思路：

- **思路一（优化器路线）**：保持参数的全秩形态，但在「如何更新参数」上创新——例如交替使用 ALS（块状精确求解）、SGD（细粒度梯度收敛）、随机扰动（跳出局部最优）。
- **思路二（参数结构路线）**：将参数更新约束为低秩形式 ΔW = BA，然后使用标准优化器（如 AdamW）进行训练。本质是在低维流形上进行优化。

**核心矛盾**：思路一决定的是参数「怎样被更新」（优化策略），思路二决定的是参数「以何种形态存在」（参数结构）。任何直接数值对比都不可避免地将两类独立变量混杂在一起。

---

## 交替优化框架 (AltOpt)

### 三种机制

| 阶段 | 机制 | 粒度 | 计算特征 |
|------|------|------|----------|
| **Phase I** | ALS（交替最小二乘） | 块状精确求解 | 矩阵求逆 O(b³) / 块 |
| **Phase II** | SGD（随机梯度下降） | 逐样本细粒度 | 梯度反传 O(d²) |
| **Phase III** | 随机扰动 | 全局参数噪声 | 随机注入 O(d) |

### 互补性

- **ALS** → 在当前激活值下给出块状全局最优解，但忽略跨块耦合
- **SGD** → 通过梯度捕获跨块交互，但容易陷入局部最优
- **扰动** → 通过参数空间噪声逃离窄局部极小值，帮助探索更优的 loss basin

### 调度策略

默认调度：ALS (1 步) → SGD (100 步) → 扰动 (1 步)，重复 3 个周期。

ALS 计算昂贵但运行稀少（1:100 的步数比），SGD 计算适中但运行频繁。

---

## 低秩适配 (LoRA)

LoRA 将参数更新约束为低秩分解：

$$\Delta W = \frac{\alpha}{r} B A, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}, \; A \in \mathbb{R}^{r \times d_{\text{in}}}$$

其中秩 r ≪ min(d_out, d_in)（典型值 r=8）。

**关键特性**：
- 可训练参数量从 d_out × d_in 降至 r × (d_out + d_in)，减少 ~100-1000×
- 默认使用 AdamW 作为优化器
- 低秩流形可能改变损失地形，平滑或阻碍优化路径

---

## 关键研究问题 (Key Research Questions)

### RQ1: 归因分离 (Disentanglement)

能否设计实验协议，将交替优化机制的效应与低秩参数化的效应**独立分离**？

**方法**：2×2 析因设计（优化器 × 参数形态），通过比较 A vs B（全秩下优化器效应）和 C vs D（低秩下优化器效应）来分离主效应，通过 (A-B)-(C-D) 检验交互效应。

### RQ2: 效率边界 (Efficiency Frontier)

在什么计算/显存预算下，ALS 矩阵求逆的额外开销（O(b³)）值得付出？

**方法**：在多个 FLOPs 预算档次（10¹², 10¹³, 10¹⁴, 10¹⁵）下重复实验，绘制「FLOPs → 最终损失」的 Pareto 曲线。

### RQ3: 损失地形交互 (Loss Landscape Interaction)

LoRA 的低秩流形是否**改变了交替优化宣称的跳出局部最优的能力**？

**假设**：低秩约束减少了参数空间的自由度 → 减少了可逃离方向 → 可能削弱随机扰动的效果。但也可能低秩流形本身已经足够平滑，使扰动不再必要。

**方法**：比较 Protocol A（全秩 AltOpt）和 Protocol C（低秩 AltOpt）中扰动阶段前后的 loss drop 幅度。

### RQ4: 泛化能力 (Generalization)

ALS 全局拟合 + SGD 局部精化 + 扰动探索的组合，是否比纯梯度优化（AdamW）在相同参数形态下具有更好的泛化能力？

**方法**：比较训练损失相同时的验证困惑度和下游任务（如 MMLU, HellaSwag）的表现。

### RQ5: 协同可能 (Synergy)

ALS-SGD-扰动优化器 + LoRA 参数结构（Protocol C）是否能同时获得**低秩的效率**和**交替优化的收敛优势**？

**方法**：Protocol C vs Protocol D（LoRA-AdamW），在相同 FLOPs 预算下比较最终损失和下游任务表现。

### RQ6: ALS:SGD 最优比

如何确定 ALS 步数与 SGD 步数的最优比例？

**方法**：固定总 FLOPs 预算，扫描 ALS:SGD 比例（1:10, 1:50, 1:100, 1:500, 1:1000），找出使最终损失最小的比例。

---

## 公平比较协议 (2×2 Factorial Design)

我们将两类独立变量交叉，形成四种实验条件：

| | **全秩更新 (Full-Rank ΔW)** | **低秩更新 (LoRA ΔW = BA)** |
|---|---|---|
| **AltOpt 优化器** | Protocol A | Protocol C |
| **AdamW 优化器** | Protocol B | Protocol D |

### 各比较的含义

| 比较 | 测试的变量 | 控制的条件 |
|------|-----------|-----------|
| A vs B | 优化器效应（全秩条件下） | 参数形态 |
| C vs D | 优化器效应（低秩条件下） | 参数形态 |
| A vs C | 参数形态效应（AltOpt 下） | 优化器 |
| B vs D | 参数形态效应（AdamW 下） | 优化器 |
| **(A-B)-(C-D)** | 交互效应（优化器效应是否依赖参数形态） | — |

### 资源归一化

由于 ALS 和 SGD 的单步计算开销不同，我们按**等总 FLOPs**（而非等步数）进行比较：

$$\text{Protocol 运行至 } \sum_{t=1}^T \text{FLOPs}_t \geq \text{BUDGET}$$

三种预算维度：
1. **FLOPs 预算**（主维度）：总浮点运算次数
2. **显存预算**（次维度）：峰值 GPU 显存占用
3. **时间预算**（第三维度）：墙面时钟时间

---

## 仓库结构

```
alternating-optimization-lora/
├── README.md                     # 本文件
├── docs/
│   ├── framework.md              # AltOpt 形式化定义（含数学推导）
│   ├── comparison-challenges.md  # 比较难题的详细分析
│   ├── literature.md             # 相关工作综述
│   ├── experiment-report-001.md  # 实验报告 #001: GPT-2 2×2
│   ├── experiment-report-002.md  # 实验报告 #002: OPT-125m 消融
│   ├── experiment-report-003.md  # 实验报告 #003: 规模化+消融框架
│   └── flaw-analysis-001.md      # GPT-2 Conv1D 缺陷分析
├── altopt/
│   ├── __init__.py
│   ├── framework.py              # 核心 AltOpt 协调器
│   ├── als.py                    # ALS 块求解器
│   ├── sgd.py                    # SGD 优化器
│   ├── perturbation.py           # 随机扰动调度器
│   ├── lora.py                   # LoRA 基线实现
│   ├── model_utils.py            # 7B+ 模型加载工具
│   ├── deepspeed_engine.py       # DeepSpeed ZeRO 集成
│   ├── peft_bridge.py            # HuggingFace PEFT 桥接
│   ├── trainer.py                # 统一训练器 (含 DeepSpeed)
│   ├── checkpoint.py             # 检查点管理
│   ├── evaluation.py             # 统一评分协议
│   └── profiling/
│       ├── flops.py              # FLOPs 精确计数
│       └── memory.py             # 显存追踪
├── experiments/
│   ├── runner.py                 # 实验执行器
│   ├── metrics.py                # 资源感知指标
│   ├── analysis.py               # 2×2 析因分析
│   ├── ablation.py               # RQ1-RQ6 消融实验
│   ├── visualization.py          # 可视化工具包
│   └── configs/
│       ├── base.yaml             # GPT-2/OPT 配置
│       └── llama2_7b.yaml        # Llama-2-7B + DeepSpeed
├── tests/
│   ├── test_framework.py         # AltOpt 单元测试
│   ├── test_lora.py              # LoRA 单元测试
│   ├── test_trainer.py           # Trainer 集成测试
│   ├── test_profiling.py         # FLOPs/Memory 测试
│   └── test_checkpoint.py        # Checkpoint 测试
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

## 快速开始

```bash
# 安装
pip install -e ".[dev]"

# 运行小规模实验（GPT-2 级别）
python experiments/runner.py experiments/configs/base.yaml

# 分析结果
python experiments/analysis.py logs/
```

---

## 当前状态

| 维度 | 状态 |
|------|------|
| **论文** | v0.7 — Round 6 对抗评审 → **Major Revision** |
| **实验** | 9 轮, 5 架构, 8 模型, 50-800 步, 100+ runs |
| **测试** | **115/115 passing** |
| **代码** | ~6,500 LOC, ALS 深度边界修复已应用 |
| **文档** | 21 Markdown docs, 9 报告, 3 评审 |
| **主要阻塞** | 结果追溯、参数量混杂、协议级下游评估 |
| **研究定位** | 严谨负结果 + 深度相关失稳 + 可复用比较协议 |

- [x] 2×2 析因框架 (评审公认核心贡献)
- [x] 注册表记录 8 个实测架构（4 个 ≤24L 收敛，4 个 ≥28L 失稳）
- [x] 多 seed 统计 (N=3-5, PB ANOVA, Fieller CI)
- [x] 非单调收敛 + ASP 隐式正则化
- [x] 深度边界发现 (4/4 架构 ≥28L 发散, 数学建模)
- [x] 低秩 ALS 求解器实现
- [x] 三轮同行评审 (Major → Minor → Accept)
- [x] **Qwen2.5-7B full-rank 训练 (Protocol B, 3/3 seeds)**
- [x] **Qwen2.5-7B LoRA 训练 (Protocol C+D, 3/3 seeds each)**
- [ ] 审计并补齐主张到机器可读结果的证据链
- [ ] Qwen2.5-7B Protocol B/D 的同协议 HellaSwag 评估
- [ ] 参数量匹配的 full-rank/LoRA 对照
- [ ] 将论文重构为准析因设计并收缩因果主张

详细任务与完成门槛仅在 [`todo.md`](todo.md) 维护。

---

## License

MIT
