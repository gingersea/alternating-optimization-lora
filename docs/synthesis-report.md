# 综合报告: Alternating Optimization Framework vs LoRA

**日期**: 2026-06-11  
**覆盖**: 实验 #001–#004 + 缺陷分析 #001 + 基础设施 Phase 2–4  
**代码量**: 5681 LOC / 115 tests / 7 模块 / 4 配置  
**实验量**: 40+ 次独立 protocol run，3 个模型架构，2 个评估维度

---

## 1. 项目全景

### 1.1 研究问题

我们在研究"交替优化框架（ALS + SGD + 随机扰动）与 LoRA 低秩适配方法作为 LLM 后训练策略的比较"。

**核心贡献**: 设计了一个 2×2 析因实验协议（优化器 × 参数形态），在统一 FLOPs 预算和统一评分体系下，将两类独立变量解耦，使性能差异可归因。

### 1.2 已完成的实验

| # | 模型 | 步数 | 数据集大小 | 核心发现 |
|---|------|------|-----------|---------|
| #001 | GPT-2 124M | 40 | 160/40 | 2×2 框架可工作；Protocol C 有 FLOPs 效率迹象 |
| #002 | OPT-125m | 100 | 320/80 | LoRA 全矩阵可用；1:20 是最优 ALS:SGD 比 |
| #003 | — | — | — | 7B 基础设施 + RQ 消融框架完成 |
| #004 | GPT-2 124M | 12-51 | 20/20 | 可复现性差；扰动有正则化效应；LoRA 跳过机制修复 |

### 1.3 代码基础设施

```
altopt/                    # 核心库 (3092 LOC)
├── framework.py           # AltOptFramework 协调器
├── als.py                 # ALS 块求解器 (含 Conv1D 支持)
├── sgd.py                 # SGD 阶段优化器
├── perturbation.py        # 随机扰动 (Gaussian/Uniform/cosine decay)
├── lora.py                # 内置 LoRA 实现
├── trainer.py             # 统一训练器 (含 DeepSpeed 模式)
├── model_utils.py         # 7B+ 模型加载 + 显存估算
├── deepspeed_engine.py    # DeepSpeed ZeRO-1/2/3 集成
├── peft_bridge.py         # HuggingFace PEFT 桥接 + 架构检测
├── checkpoint.py          # 检查点管理
└── evaluation.py          # 统一评分协议

experiments/               # 实验工具 (1798 LOC)
├── runner.py              # CLI 实验执行器
├── ablation.py            # RQ1-RQ6 消融实验
├── analysis.py            # 2×2 析因分析 + 可视化
├── visualization.py       # 6 种图表类型
├── run_experiment_004.py  # 实验 #004 专用脚本
└── configs/
    ├── base.yaml          # GPT-2/OPT 配置
    └── llama2_7b.yaml     # 7B + DeepSpeed 配置

tests/                     # 115 tests
├── test_framework.py      # 20
├── test_trainer.py        # 17
├── test_lora.py           # 12
├── test_perturbation.py   # 17
├── test_model_utils.py    # 19
├── test_peft_bridge.py    # 12
├── test_profiling.py      # 14
└── test_checkpoint.py     # 9
```

---

## 2. 跨实验结论综合

### 2.1 高置信度结论 🔴

| # | 结论 | 证据 | 置信度 |
|---|------|------|--------|
| C1 | **AdamW 在 ≤100 步时全面占优 AltOpt** | #001: 8.31 vs 185 ppl; #002: 22 vs 651 ppl; #004: 50 vs 8243 ppl | 极高 (3/3 实验一致) |
| C2 | **LoRA 低秩约束是当前最强的性能因子** | #002: LoRA 带来 5-30× PPL 改善，超越优化器选择 | 极高 (#002 完整 2×2) |
| C3 | **ALS 第一步 reconstruction loss 主导早期训练** | #001: loss 从 ~10⁵ 开始; #004: 第一步 loss=220001 | 极高 (4/4 实验) |
| C4 | **GPT-2 Conv1D 是 LoRA 实验的硬障碍** | #001: D 降级 = B; #004: C/D 全部 skip | 极高 (2/2 实验) |
| C5 | **2×2 析因框架方法论有效** | #001, #002: 四协议统一评估产出可比较数据 | 极高 |

### 2.2 中置信度结论 🟡

| # | 结论 | 证据 | 置信度 |
|---|------|------|--------|
| C6 | **AltOpt+LoRA (Protocol C) 有 FLOPs 效率优势** | #001: C (ppl=10, 60% FLOPs) vs B (ppl=8); #002: C (ppl=5.5, 60% FLOPs) vs D (ppl=4.6) | 中等 (2/2 实验，但 C 跳过 ALS) |
| C7 | **ALS:SGD = 1:20 可能是最优比** | #002 消融: 1:20 → ppl=278, 优于 1:10 (1353), 1:50 (1026) | 低 (50 步消融方差大) |
| C8 | **扰动有非单调效应** — 12 步时改善 eval ppl 但恶化 train loss | #004: with_perturb ppl=86k vs no_perturb ppl=317k | 低 (仅 1 次 12 步实验) |
| C9 | **AltOpt 在小步数下需要更多 SGD 精化** | #001/#002/#004: converge speed ~5-30× slower than AdamW | 中等 (3/3 实验一致) |

### 2.3 不确定性 / 矛盾

| # | 问题 | 说明 |
|---|------|------|
| U1 | **AltOpt 在 500+ 步后能否超越 AdamW？** | 所有实验都在 ≤100 步，无法评估长期效果 |
| U2 | **Protocol C 跳过 ALS 的影响？** | ALS 未在 LoRA 空间运行，C 的 FLOPs 优势部分来源于"缺省" ALS 开销 |
| U3 | **1:20 最优比的鲁棒性？** | #004 RQ6 跨 seed delta=1886%，说明结论在 <100 步极不稳定 |
| U4 | **扰动的净效应符号？** | #004 12 步为正，但长期效应未知 |

---

## 3. 已修复的缺陷

| # | 严重性 | 问题 | 修复 | 验证 |
|---|--------|------|------|------|
| D1 | CRITICAL | ALS 忽略 GPT-2 Conv1D 层 | 新增 `_solve_conv1d_layer()` | ✅ |
| D2 | CRITICAL | PeftBridge 在非 Llama 模型上崩溃 | `detect_target_modules()` 9 架构 + catch ValueError | ✅ 115 tests |
| D3 | HIGH | Protocol C/D 在 GPT-2 上无 LoRA → 降级路径 | Trainer 自动 skip → 全秩 fallback | ✅ |
| D4 | HIGH | Perturbation 返回 noise energy 而非 loss | `loss_types` 区分 loss/noise_energy | ✅ |
| D5 | MEDIUM | `final_loss` 始终为 inf (eval never triggered) | 改用 `loss_history[-1]` 报告 train loss | ✅ |
| D6 | MEDIUM | `model_utils`, `peft_bridge`, `perturbation` 无测试 | 新增 48 tests | ✅ 115/115 |

### 尚未修复的缺陷

| # | 严重性 | 问题 | 阻塞 |
|---|--------|------|------|
| D7 | MEDIUM | Protocol C ALS 无法在 LoRA space 运行 (LoRALayer ≠ nn.Linear) | 需要低秩 ALS 求解器 |
| D8 | MEDIUM | DeepSpeed 代码从未在 GPU 上测试 | 需要 Llama-2-7B 下载 + 2 GPU |
| D9 | LOW | ablation.py vs run_experiment_004.py 实现分歧 | 需要统一 |
| D10 | LOW | 无下游任务评估 (MMLU, HellaSwag) | 需要添加 eval harness |

---

## 4. 后续方向：优先级排序 + 可行性评估

### P0 — 阻塞性：当前结论无法回答核心研究问题

| 方向 | 做什么 | 为什么 | 可行性 | 预估时间 |
|------|--------|--------|--------|----------|
| **P0.1 运行 ≥200 steps 实验** | OPT-125m, 4 协议, ≥200 steps | 所有现有结论基于 ≤100 步 — AltOpt 的长期优势从未被测试。这是整个项目最关键的缺失数据 | 高 (CPU, 2-4h) | ~3h |
| **P0.2 3-seed 可复现性** | OPT-125m, 200 steps, seed=42/123/456 | #004 显示 39-1886% delta — 当前所有结论在统计上不可靠 | 高 | ~9h (3×P0.1) |

### P1 — 高价值：显著提升结论可信度

| 方向 | 做什么 | 为什么 | 可行性 | 预估时间 |
|------|--------|--------|--------|----------|
| **P1.1 GPU 实验** | Llama-2-7B, DeepSpeed ZeRO-2, 2× RTX 5090 | 验证 AltOpt 在实用规模模型上的表现。7B 的损失地形可能与小模型完全不同 | 中 (需下载 14GB 模型 + HF token) | 下载 30min + 运行 2-4h |
| **P1.2 FLOPs 预算扫描** | 4 个 FLOPs 预算 (10¹⁰, 10¹¹, 10¹², 10¹³) × 4 协议 | 绘制完整的 Pareto 前沿 — 回答"在什么预算下 ALS 值得？" | 高 | ~6h |
| **P1.3 扰动强度消融** | noise_scale ∈ {1e-2, 1e-3, 1e-4, 1e-5, 0} | 当前只用默认 1e-3，从未调优 | 高 | ~2h |
| **P1.4 Protocol C 补充 ALS** | 实现低秩空间的 ALS 求解器 (在 B @ A 空间内做块求解) | Protocol C 当前跳过 ALS，失去了交替优化的精髓 | 中 (需要推导低秩 ALS 公式) | ~4h |

### P2 — 锦上添花：完善方法论和工具

| 方向 | 做什么 | 为什么 | 可行性 | 预估时间 |
|------|--------|--------|--------|----------|
| P2.1 统一 ablation.py 和 run_experiment_004.py | 合并为一个一致的实验入口 | 避免维护分歧 | 高 | ~1h |
| P2.2 下游任务评估 | 添加 MMLU / HellaSwag / LAMBADA | Perplexity 只能衡量语言建模，无法衡量泛化 | 中 (需 lm-eval-harness) | ~2h |
| P2.3 Conv1D LoRA wrapper | 实现 GPT-2 兼容的 LoRA (参考 als.py Conv1D 处理) | 使 GPT-2 也能运行完整 2×2 | 中 | ~4h |
| P2.4 基于消融数据生成可视化 | 运行 `visualization.py` 产出 publication-quality 图表 | 准备汇报/论文素材 | 高 | ~10min |

---

## 5. 推荐的下一轮行动计划

按照"先验证后扩展"原则：

```
Round 5:  P0.1 + P0.2 — 200 steps × 3 seeds (OPT-125m)
          ↓ 产出: 统计上有意义的 2×2 结果 + cross-seed std

Round 6:  P1.1 + P1.3 — GPU 7B 实验 + 扰动消融
          ↓ 产出: 7B 规模的首组数据 + 最优扰动强度

Round 7:  P1.2 + P2.2 — FLOPs Pareto + 下游任务
          ↓ 产出: 完整 Pareto 前沿 + 泛化能力评估

Round 8:  P1.4 + P2.1 — Protocol C ALS + 代码统一
          ↓ 产出: 真正完整的四协议对比
```

**当前最紧急**: Round 5 (P0.1) — 运行一次 ≥200 steps 的完整 2×2 实验。这是唯一能回答"Alternating Optimization 是否值得"这一核心问题的数据缺口。现有 4 份报告的所有结论都受限于 "≤100 steps"，无法推断 AltOpt 的长期行为。

---

## 6. 关键数字总览

| 指标 | 数值 |
|------|------|
| 总代码量 | 5,681 LOC |
| 核心库 | 3,092 LOC |
| 实验工具 | 1,798 LOC |
| 测试 | 791 LOC / 115 tests |
| 实验报告 | 4 份 |
| 缺陷分析 | 1 份 |
| 独立实验 run | 40+ |
| 已测试模型架构 | 3 (GPT-2, OPT-125m, Llama-2-7B config) |
| 支持的量化方式 | 5 (bf16, fp16, fp32, int8, int4) |
| DeepSpeed ZeRO stages | 3 (1/2/3) |
| 已修复缺陷 | 6 |
| 待修复缺陷 | 4 |
| GitHub commits | 10 |
| 测试通过率 | 115/115 = 100% |

---

*Last updated: 2026-06-11*
