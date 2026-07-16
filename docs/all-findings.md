# All Findings — Alternating Optimization vs LoRA

**Date**: 2026-07-16 | **Paper**: v0.7.1 (Major Revision)

---

## 一、核心发现（17 实验，8 架构）

### 方法论

1. **Quasi-factorial 2×2 框架** — 优化器(ASP vs AdamW) × 参数形态(full-rank vs LoRA)，统一 FLOPs 核算。Protocol C 缺 ALS，故为 quasi-factorial（已知限制）。

### 性能对比

2. **ASP 始终弱于 AdamW** — 在所有已测步数预算内（50–1200 步），ASP 的 eval PPL 始终比 AdamW 差。这是项目最核心的负结果。

3. **LoRA 在 ≤200 步主导** — 5–30× PPL 优势，5/5 架构确认。

4. **7B scale: full-rank 在 WikiText-2 上大幅领先 LoRA** — AdamW+full-rank PPL=1.25±0.01 vs LoRA r=8 PPL=10.41±0.01。但 8.3× 差异主要来自参数量（2300×），非秩结构本身（0.5B 参数匹配基线确认）。

5. **ASP-AdamW gap 随步数缩小** — OPT-125m 上 7.8×（50→800 步），但 AdamW 始终领先。

### 组件归因（P1.1，OPT-125m, 3 seeds, 200 steps）

6. **ALS 单独损害 SGD** — ALS+SGD PPL=62.5 vs SGD-only PPL=59.4（+3.1，基线 231.4）

7. **Perturbation 单独损害 SGD** — SGD+Perturb PPL=62.2 vs SGD-only PPL=59.4（+2.8）

8. **ALS × Perturb 拮抗交互** — Full ASP PPL=69.0，比期望加性效应差 3.6 PPL

9. **结论：在 200 步短预算上，ASP 的所有组件各自损害性能，组合更差**（注意：长预算可能不同）

### 深度失稳（P1.2，4 模型跨家族，100 步）

10. **深度边界：≤24L 收敛，≥28L 发散** — 8/8 架构确认。跨 OPT(12L)/TinyLlama(22L)/Qwen0.5B(24L)/Qwen7B(28L) + 历史 GPT-2/SmolLM2/DeepSeek/Mistral。

11. **失稳是连续恶化而非突然跃迁** — 12L 已有非单调收敛（P1.1 证实），24L 仍在稳定区，28L 灾难性发散。

12. **7B 11 次尝试全部失败** — DeepSpeed ZeRO-2 和 PyTorch FSDP 两个后端。PPL 振荡在 ~1.2M。

### 隐式正则化（P1.3，OPT-125m, WT2+C4）

13. **ASP 在跨域上比 AdamW 好 1.9×** — ASP@800 C4 PPL=48.1 vs AdamW@200 C4 PPL=92.4。即使 ASP 在 WT2 上更差（75.1 vs 18.5）。

14. **ASP WT2/C4=1.56（泛化），AdamW WT2/C4=0.20（记忆）** — ASP 在未见过的 C4 数据上比训练域 WT2 表现更好。

15. **AdamW 在 1600 WT2 样本上严重过拟合** — 产生接近完美的 WT2 PPL(1.25) 但降低下游准确率：HellaSwag −3.2pp, MMLU −4.2pp, ARC −3.3pp。

### 秩充分律

16. **r=8 对 WT2 后训练普遍充分** — 当 L/dh≤0.035 时，r=8 匹配 r=256 在 ±0.02 PPL 内，5 模型族 × 中/英文 × 100-1600 步一致。

17. **r_min 与模型架构有关** — SmolLM2-135M r_min≈12（唯一验证的例外），解释为其高 L/dh(0.052) + 中等预训练(2T tokens)。

18. **η ≈ 230（用于 r_min = η·L/dh）被证伪为通用常数** — η 是模型特异的（受预训练质量调节），而非固定值。

### 低秩 ALS

19. **低秩 ALS 始终负收益** — 7/7 比较（100–800 步，3 架构），ALS 从未改善 Protocol C。

### 参数量匹配（0.5B 尺度）

20. **8.3× PPL 差异是参数量效应，非秩结构效应** — 高秩 LoRA(r=256, 34.6M) 达到 PPL=1.61，比 full-rank(494M) 的 PPL=44.4 好 27×。

### 跨域评估

21. **WikiText-2 PPL 是不可靠的后训练质量指标** — 7B full-rank 的 WT2 PPL=1.25 但 HellaSwag 下降 3.2pp 且 C4 比 LoRA 差。8.3× WT2 差距在 C4 上坍塌为 1.05×。

---

## 二、证据强度总览

| 发现 | 证据强度 | 状态 |
|------|---------|------|
| ASP 弱于 AdamW | 🔴 High — 5 架构, multi-seed | 已发表（论文 v0.7.1） |
| 深度边界 ≤24/≥28 | 🔴 High — 8/8 架构 | 已发表 |
| 组件归因（ALS/SGD/Perturb） | 🟡 Medium — 1 模型, 3 seeds, 短预算 | P1.1 新结果，未入论文 |
| 隐式正则化（C4 泛化） | 🟡 Medium — 1 模型, WT2+C4, 收敛匹配 | P1.3 新结果，未入论文 |
| r=8 秩充分 | 🔴 High — 5 模型族, 多语言 | 已发表 |
| AdamW 过拟合 | 🔴 High — WT2+C4+3 下游 | 已发表 |
| 低秩 ALS 负收益 | 🟡 Medium — 7/7 比较 | 已发表 |
| 参数匹配基线 | 🟡 Medium — 0.5B 单模型 | 已发表 |

---

## 三、论文 v0.7.1 与 P1 新结果的关系

论文 v0.7.1 已包含发现 #1–5、#10–12、#15–21。
**P1 新结果（#6–9、#13–14）尚未写入论文**，记录于 [`docs/p2-synthesis.md`](docs/p2-synthesis.md)，计划纳入 v0.8。
