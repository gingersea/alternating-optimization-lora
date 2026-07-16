# 高价值研究路线图

**更新日期**: 2026-07-15
**当前判定**: **Major Revision**（Round 6 对抗评审结论）
**上次重组织**: 2026-07-15（docs/ → 11 活跃 + 24 archive + 9 reference；paper/ → 单稿 + 6 评审）

> 本文件是项目当前状态与优先级的唯一入口。所有历史评分、自评指标、v3.4 "Accept" 声明均已归档至 `docs/archive/`。当前 v0.7.1 草稿按证据边界收缩。

---

## 1. 现状评估

### 已有贡献（可保留部分）

- **可复用的准析因比较框架**：优化器 × 参数化形式 2×2 组织对照，统一 FLOPs/显存/时间核算。Protocol C 因缺 ALS 为 quasi-factorial（已知限制）。
- **深度相关失稳现象**：8 个实测架构中，4 个 ≤24L 收敛，4 个 ≥28L 失稳，11 次 7B 尝试记录。稳定区域内 ASP 提供隐式正则化。
- **负结果记录**：ASP 在已测预算内始终弱于 AdamW；低秩 ALS 始终负收益（7/7 比较）。
- **参数量匹配基线**（Qwen2.5-0.5B）：证明 8.3× PPL 差异主要来自参数量而非秩结构。
- **HellaSwag + C4 交叉域评估**：Protocol B/D 均完成 3-seed 下游评估，数据在仓库。
- **工程资产**：122 测试（2 预存在失败：bitsandbytes GPU 依赖），多 seed、DeepSpeed/FSDP 记录。

### 已知差距

| 优先级 | 差距 | 阻塞什么 |
|---|---|---|
| **P0** | protobuf/checkpoint 侧证据薄弱 | 7B Protocol B 训练 checkpoint 不在仓库，复算需 GPU |
| **P1** | 交互效应不可严格分离 | Protocol C 缺 ALS，交互项 (A-B)-(C-D) 混入 ALS 存在性 |
| **P1** | 深度机制未因果验证 | 8 架构样本来自不同模型族，ρ̄=1.08 是两点拟合 |
| **P1** | 隐式正则化未健壮复现 | 单数据集（WikiText-2），无 early-stopped AdamW 对照 |
| **P2** | 外部有效性不足 | 仅 WikiText-2，长预算交叉点未验证 |

---

## 2. P0：已完成 ✓

### P0.1 数据审计与可复现包 ✓

- [x] 主张→产物映射：[`docs/claims-audit.md`](docs/claims-audit.md)（14 项主张，含证据标签）
- [x] 7B 完整测试集恢复：`runs/qwen25_7b_800s/full_test_eval.json`（已恢复）
- [x] Protocol B 原始结果：`runs/qwen25_7b_800s/protocol_b_full_rank_results.json`（已创建）
- [x] Protocol B 数据进入 `combined_results.json`（已补充）
- [x] 架构计数统一为 **8 个实测**（Llama-2-7B = predicted）
- [x] 论文文件已重命名为 `paper_draft_v0.7.md`
- [x] 评估协议统一：N_EVAL=200（~12,640 tokens）为主协议，完整测试集仅 B 协议可用
- [x] 证据状态标签已建立：`observed` / `replicated` / `transcribed` / `inferred` / `predicted`

### P0.2 最高信息增益实验 ✓

- [x] **协议级 HellaSwag**：PB（56.74% ± 0.98%）和 PD（59.74% ± 0.07%）均为 3-seed（同一 lm-eval harness），基线 59.91%
- [x] **参数量匹配对照**：Qwen2.5-0.5B 已完成（r=256/512 vs full-rank），7B 受 GPU 限制不可行
- [x] 可行性报告：[`docs/p0.2-feasibility.md`](docs/p0.2-feasibility.md)

### P0.3 文稿主张收缩 ✓

- [x] v3.4 "Final Draft — Accept" → v0.7.1 "Major Revision"
- [x] "2×2 factorial" → "quasi-factorial comparison"
- [x] "depth boundary at ~26" → "instability transition between 24 and 28 layers"
- [x] "full test set validated" 全部移除，统一 N_EVAL=200 基线 105.56
- [x] 架构计数 9→8，缺失 Appendix D 修复

---

## 3. P1：需 GPU 实验（当前不可执行）

### P1.1 组件归因（ALS/SGD/Perturb 嵌套消融）
- 设计已记录：[`docs/mechanism-notes.md`](docs/mechanism-notes.md)
- 需要：OPT-125m 上 4-way 消融（SGD-only / ALS+SGD / SGD+Perturb / Full ASP）
- 预计：~15 min GPU

### P1.2 深度机制因果验证
- 设计已记录：[`docs/causal_depth_boundary.md`](docs/causal_depth_boundary.md) + [`docs/mechanism-notes.md`](docs/mechanism-notes.md)
- 需要：同模型族深度扫描 + 每层谱范数/激活漂移测量
- 预计：~2h GPU

### P1.3 隐式正则化复现
- 设计已记录：[`docs/mechanism-notes.md`](docs/mechanism-notes.md)
- 需要：C4 重复 + early-stopped AdamW 对照 + weight decay/dropout 匹配基线
- 预计：~1h GPU

---

## 4. P2：长期方向（仅在 P1 完成后）

1. **安全 ALS 控制器**：基于实时激活漂移的自适应阻尼/跳过/回滚
2. **预算条件化方法选择器**：根据模型深度、数据量和算力预测最优策略
3. **负结果基准套件**：整理 8 架构失稳矩阵 + 11 次 7B 尝试为 stress test
4. **三因素研究**（参数量 × 秩 × 优化器）：扩展当前 quasi-factorial 为可识别设计

---

## 5. 暂缓事项

- [x] 不再新增自评分、评审轮次或 "Accept" 声明
- [x] 不再以 8.3× PPL 宣称 low-rank/full-rank 因果结论（已承认参数量混杂）
- [x] 不再用异构架构堆叠"深度阈值"样本
- [x] 暂缓 7B Protocol A 暴力重跑
- [x] 暂缓 >2000 步 crossover 搜索

---

## 6. 下一里程碑

**Milestone: Evidence-Complete Major Revision**（部分达成）

- [x] P0.1 数据审计（claims-audit + 数据产物 + 证据标签）
- [x] B/D HellaSwag 协议级结果（3 seeds 各，数据在库）
- [x] 参数量匹配实验（0.5B 完成，7B GPU 不可行已注明）
- [x] 论文重新定位为：严谨的负结果 + 深度相关失稳 + 可复用评估协议
- [ ] 由独立复核者从空环境复算所有主表

**当前状态**：P0 全部完成，P0.3 主张已收缩至证据支持范围。P1 方向清晰但需 GPU 资源。论文 v0.7.1 可提交为 **Major Revision 回应稿**，核心主张有证据支撑。
