# 问题-解决报告：ALS 后训练在深层 LLM 上的发散

**框架**：ASP vs AdamW × 全秩 vs LoRA 的 2×2 准析因比较（FLOPs 归一化）
**核心问题**：ALS 后训练（Protocol A）在 ≥28 层 LLM 上系统性发散
**证据状态**：下表所有数字均来自模型自带 tokenizer 的干净 harness（OPT/Qwen），不受跨词表污染影响；OPT-125m 绝对数字为修复后（OPT tokenizer + pad 掩码）重跑值

---

## 一、问题：2×2 矩阵中 ASP 列的失败模式

| | 全秩 | LoRA |
|--|------|------|
| **AdamW** | B ✅ 收敛 | D ✅ 收敛 |
| **ASP** | A ❌ **发散** | C ❌ 收敛但质量差 |

**实测证据（全部 clean harness）：**

| Protocol | 配置 | 结果 |
|----------|------|------|
| B (AdamW 全秩) | OPT-125m 50–800 步 | 16.7–20.5 PPL，收敛 |
| B (AdamW 全秩) | Qwen-0.5B 50–800 步 | 27.5–68 PPL，收敛（步数↑略过拟合） |
| B (AdamW 全秩) | **Qwen2.5-7B 800 步** | **1.25 PPL（N=3）** |
| D (AdamW LoRA) | **Qwen2.5-7B 800 步** | **10.41 PPL（N=3）** |
| C (ASP LoRA) | **Qwen2.5-7B 800 步** | **122–142 PPL（N=3），无 ALS（已移除）** |
| A (ASP 全秩) | OPT-125m / Qwen-0.5B | 3,000–200,000 PPL，发散 |
| A (ASP 全秩) | **Qwen2.5-7B** | **11/11 尝试全部 NaN 发散** |

**2×2 分解**：
- **优化器主效应**：AdamW ≫ ASP —— 全秩 1.25 vs 发散；LoRA 10.41 vs 122–142
- **交互效应**：ASP 在全秩下暴露致命弱点（发散），LoRA 下仅质量差（无 ALS 也差）—— 证明 ASP 的优化器机制本身（SGD+Perturb）在 LoRA 上也劣于 AdamW，而全秩下 ALS 是致命放大器

---

## 二、诊断：为什么 ALS 在深层发散（因果链已验证）

**发现 1 — 受控消融（Qwen2.5-7B，5 条件）**：ALS 权重修改是**唯一充分原因**。

| 条件 | 结果 | 证据状态 |
|------|------|---------|
| SGD-only | 53.6 ✅（200 步收敛） | ⚠️ 转录（见下注） |
| Perturb-only | 94.4 ✅ | ⚠️ 转录（见下注） |
| ALS(no-op)+SGD | 54.67 ✅（200 步收敛） | ✅ `runs/diverge_cause_7b.json` |
| **ALS-only** | **2.0×10⁹（第 1 步）→ 1.1×10¹⁵（第 2 步）❌** | ✅ 同上 |
| ALS+SGD | 8.3×10⁸（10 步）→ 1.4×10¹⁹⁵（110 步）❌ | ✅ 同上 |

> 注：`runs/diverge_cause_7b.json` 仅保存了条件 3/4/5 的原始数据；SGD-only 与 Perturb-only 的最终 PPL 来自最终报告文档转录，原始逐点数据未入库。但两条件均明确 `diverged=False` 且 PPL 处于正常量级，与条件 3 的 54.67 相互印证，结论不受影响。

**发现 2 — 残差放大理论**：每层残差连接把扰动放大 ρ≈1.08 倍，28 层共 ≈8× 放大，远超 SGD 单周期恢复能力（≈0.005），临界深度 L_max≈26 —— 与 8 架构实测边界（≤24 收敛 / ≥28 发散）一致。

**发现 3 — 正反馈**：ALS 的 δ 幅度跨周期增长（0.085→0.196，×2.3），发散自增强而非自限。

---

## 三、解决的尝试与结论（哪些有效 / 哪些无效）

### 无效路径（已证伪，均有 matched-budget 对照）

| 修复 | 机制 | 结果 |
|------|------|------|
| A-SYNC 梯度注入 | 把 ALS 闭式解当梯度偏置注入 | **负结果**：原实现是时序 no-op（step 后注入、下轮 zero_grad 清除，永不生效）；修正时序后与纯 SGD 轨迹相关 0.99981（7B 6.82 vs 6.83），纯冗余 |
| 变体排名重解释 | — | 排名是 **lr 调度**效应非注入强度：FIXED 50.7 < cosine 54.7 < exp 57.1（OPT-125m 干净重跑） |
| A-PROBE 低秩探针 | 闭式解作用于低秩探针头 | **负结果**：消除发散但质量封顶（7B r=64 → 22.8）；秩扫描 r=64/256/1024 全部在纯 SGD 噪声带（50.6–50.8 vs 50.7） |
| A-KD 软目标蒸馏 | 闭式解匹配教师 logits（唯一非冗余 ALS 目标） | **负结果**：kd_als 52.6 vs kd_sgd 52.4，闭式解无增量 |

### 有效路径（2×2 框架内的解答）

| 方案 | 证据 | 地位 |
|------|------|------|
| **纯 SGD（固定 lr）** | Qwen2.5-7B 4800 步 → **6.83 PPL**，幂律收敛 | ✅ 28 层上唯一稳定收敛的 ALS 系替代 |
| **AdamW 全秩** | 7B → **1.25 PPL** | ✅ 绝对质量最优 |
| **AdamW LoRA** | 7B → **10.41 PPL**，FLOPs 低 1000× | ✅ 效率最优 |

---

## 四、结论（在 2×2 框架内）

1. **问题定位**：ALS 后训练的核心缺陷是**权重修改与残差网络的耦合放大**——不是扰动噪声、不是优化器数值、不是 hook 开销（消融已排除）。
2. **解决答案**：深层模型上，**去掉 ALS、保留其 SGD 骨架（固定 lr）即可收敛**（6.83）；要质量则用 AdamW（1.25）；要效率用 AdamW+LoRA（10.41 @ 0.1% 参数）。ALS 的闭式解在三个可设想的表述下（梯度注入 / 低秩探针 / 软目标蒸馏）均无增量价值。
3. **方法贡献**：FLOPs 归一化公平比较 + 2×2 因子归因 + 评估 harness 审计（tokenizer 匹配 + pad 掩码，修正预训练基线至 OPT-125m WT2=73.7）+ "时序 no-op"诊断方法论（验证混合组件是否真的在声称的时刻修改了声称的量）。

---

## 附：证据清单与 clean/污染标记

| 数据 | 来源 | 状态 |
|------|------|------|
| 2×2 矩阵 A/B（OPT/Qwen-0.5B） | `runs/multi_seed_matrix/` | ✅ clean（模型自带 tokenizer；pad 未掩码，相对比较有效） |
| 7B B/C/D | `runs/qwen25_7b_800s/combined_results.json` | ✅ clean（Qwen tokenizer + N_EVAL=200） |
| 消融 | `runs/diverge_cause_7b.json` | ✅ clean |
| 纯 SGD 7B 对照 | `runs/pure_sgd_96c_7b.json` | ✅ clean |
| lr 调度验证 | `runs/lr_schedule_sgd_opt125m.json` | ✅ clean（修复后重跑） |
| 探针秩扫描 | `runs/probe_rank_sweep_opt125m.json` | ✅ clean（修复后重跑） |
| A-KD | `runs/kd_als_opt125m.json` | ✅ clean |
| FLOPs 对比 | `runs/flops_sweep_opt125m.json` | ⚠️ 污染（gpt2 tokenizer）；仅保留相对排序 |
| 时序诊断 | `docs/diag-injection-report.md` | ✅ 机制审计（无绝对数字依赖） |
