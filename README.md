# ASP vs LoRA: A Quasi-Factorial Comparison for LLM Post-Training

> **状态**: 论文 v0.7.1 — Round 6 **Major Revision**
> **核心**: 8 架构实测，深度 24–28 层失稳转变，准析因 2×2 比较框架
> **路线图**: [todo.md](todo.md) — 当前差距、已完成事项、下一步方向

---

## 文档导航

| 文档 | 说明 |
|------|------|
| **[论文 v0.7.1](paper/paper_draft_v0.7.md)** | Canonical draft（925 行，14 项主张，证据收缩至 Major Revision 范围） |
| **[实验注册表](docs/experiment-registry.md)** | 8 架构 × 4 协议 × 50–800 步矩阵，含证据标签 |
| **[主张→证据映射](docs/claims-audit.md)** | 14 项核心主张，含 observed/transcribed/inferred/predicted 标签 |
| **[当前路线图](todo.md)** | P0 已完成 / P1 需 GPU / P2 长期方向 |
| **[公平比较方法论](docs/fair_comparison_methodology.md)** | 准析因设计的核心方法论 |
| **[数学分析](docs/math-analysis.md)** | ALS 重建损失、收敛理论、PAC-Bayes 分析 |
| **[深度失稳因果理论](docs/causal_depth_boundary.md)** | SCM 框架下的残差干预传播模型 |
| **[机制笔记](docs/mechanism-notes.md)** | 组件归因设计、隐式正则化验证缺口 |
| **[P0.2 可行性报告](docs/p0.2-feasibility.md)** | HellaSwag + 参数量匹配实验评估 |

### 子目录

| 目录 | 内容 | 文件数 |
|------|------|--------|
| [`docs/archive/`](docs/archive/) | 历史评分、早期实验报告、已被取代的评估 | 24 |
| [`docs/reference/`](docs/reference/) | 算法详解、协议实现、评估标准（教育性文档） | 9 |
| [`paper/reviews/`](paper/reviews/) | Round 1–6 同行评审记录 | 6 |

---

## 核心发现

| 发现 | 证据强度 | 来源 |
|------|---------|------|
| ASP 在 ≤24 层收敛，≥28 层失稳 | 8/8 架构确认，11 次 7B 尝试 | [§5.6](paper/paper_draft_v0.7.md) |
| ASP 提供隐式正则化（train≈eval @1200 步） | 单数据集，单模型 | [§5.4](paper/paper_draft_v0.7.md) |
| 7B B/D 8.3× PPL 差异主要由参数量驱动 | 0.5B 参数量匹配基线确认 | [§5.7](paper/paper_draft_v0.7.md) |
| 低秩 ALS 始终负收益（7/7 比较） | 3 模型，100–800 步 | [§5.8](paper/paper_draft_v0.7.md) |
| LoRA 在 ≤800 步主导（5–30× PPL） | 5/5 架构 | [§5.2](paper/paper_draft_v0.7.md) |

---

## 快速开始

```bash
pip install -e ".[dev]"
python experiments/runner.py experiments/configs/base.yaml
python experiments/analysis.py logs/
pytest tests/  # 122 passed, 2 pre-existing failures (bitsandbytes GPU dependency)
```

---

## 仓库结构

```
├── paper/paper_draft_v0.7.md    # 唯一论文草稿
├── docs/
│   ├── claims-audit.md          # 主张→产物可追溯性
│   ├── experiment-registry.md   # 全实验矩阵
│   ├── archive/                 # 历史快照（24 文件）
│   └── reference/               # 教育性文档（9 文件）
├── altopt/                      # 核心框架（ALS, SGD, Perturb, LoRA, trainer）
├── experiments/                 # 实验脚本（55 文件）
├── tests/                       # 122 测试
└── runs/                        # 数据产物（git-ignored, 部分已入库）
```

## License

MIT
