# Claims Audit — claim → run → config → result → table 映射

**Date**: 2026-07-15
**Scope**: 论文 v0.7（`paper/paper_draft_v0.7.md`）中的全部核心数字主张。
**目的**: 使每个核心主张可从仓库内机器可读产物追溯；无法追溯的主张显式降级。

## 证据状态标签（全仓库统一）

| 标签 | 定义 |
|------|------|
| `observed` | 单次（或单实验室）运行得到，原始机器可读产物在仓库内 |
| `replicated` | 多 seed / 多次独立运行一致，产物在仓库内 |
| `transcribed` | 数字来自文档转录，原始产物已丢失，无法从仓库复算 |
| `inferred` | 从 observed 数据经模型拟合 / 推导得出 |
| `predicted` | 纯外推，无实验数据支持 |

## 核心主张映射表

| # | 主张（论文位置） | Run/产物 | 配置来源 | 状态 | 审计结论 |
|---|------------------|----------|----------|------|----------|
| C1 | LoRA (D) 在 ≤200 步主导，5–30× PPL（Abstract, §5.2 Table 1） | `paper/multi_seed_results.json`（OPT/Qwen A、B）；GPT-2/TinyLlama 数字仅存在于 `docs/experiment-report-00x.md` 文本 | `experiments/multi_seed_matrix.py`, `experiments/round5_runner.py` | OPT/Qwen: `replicated`；GPT-2/TinyLlama: `transcribed` | 部分可复算。GPT-2 与 TinyLlama 的原始 JSON 未入库 |
| C2 | Qwen2.5-7B Protocol B PPL 1.25 ± 0.01（N=3）（Abstract, §5.6.2） | `runs/qwen25_7b_800s/protocol_b_full_rank_results.json`（2026-07-15 由文档转录入库） | §5.6.2 + registry 硬件配置；checkpoint 未入库 | `transcribed` | 数字已入库但不可复算；训练 checkpoint 与训练日志缺失 |
| C3 | 7B Protocol C 135.36 ± 9.05 / D 10.41 ± 0.01（§5.2 Table 1） | `runs/qwen25_7b_800s/combined_results.json` | 同上 | `observed`（3 seeds 一致 → 接近 `replicated`，但仅单机单实验室） | 可复核原始 JSON；评估协议 N_EVAL=200 |
| C4 | "full test set" 验证（原 Abstract、§5.6.2、Appendix D 引用） | `runs/qwen25_7b_800s/full_test_eval.json` — **不存在，从未提交** | `experiments/_eval_full_test.py`（脚本在库，输出丢失） | 不可追溯 | **已从论文删除全部 "full test set" 表述**（2026-07-15）。论文现仅报告 N_EVAL=200 数字 |
| C5 | 未训练 7B 基线 PPL（Abstract 原为 133；registry 为 105.56 @ N_EVAL=200） | registry + `protocol_b_full_rank_results.json` baseline 字段 | — | `transcribed` | 133.16 是丢失的 full-test 评估的产物；已统一改用 N_EVAL=200 基线 105.56（B 相对基线 ≈ 84×，非 106×） |
| C6 | A–B gap 随步数 7.8× 收缩，Cohen's d=1.17（Abstract, §5.3 Table 2） | `paper/multi_seed_results.json` | `experiments/multi_seed_matrix.py`, `experiments/statistical_analysis.py` | `replicated`（N=3–5） | 可复算；800 步 bootstrap CI 跨零的张力已在 §5.3 披露 |
| C7 | 深度失稳：≤24 层收敛，≥28 层失稳（Abstract, §5.6 Table 5） | 8 个架构中，OPT/Qwen0.5B 有 `multi_seed_results.json`；GPT-2/TinyLlama/SmolLM2/DeepSeek/Mistral/Qwen7B 数字仅在文档文本 | `docs/gpu_7b_validation.md`, `docs/round10_results.md`, registry | 混合：2 架构 `replicated`，6 架构 `transcribed` | **实测架构数 = 8**。Llama-2-7B 仅出现于外推表（§6.3），标 `predicted`，不计入验证数量 |
| C8 | 7B Protocol A 11 次尝试全部失败，PPL ~1.2M（§5.6.1） | 无机器可读产物；叙述见 `docs/final-analysis-phase-b-and-depth-boundary.md` | `experiments/run_7b_gpu.py`, `experiments/run_7b_fsdp.py` | `transcribed` | 失败记录有工程价值，但逐次日志未入库 |
| C9 | AdamW 过拟合（400–1600 样本）（§5.4 Table 3） | 无机器可读产物；数字仅在论文与 `docs/round9_overfitting.md` | `experiments/round6_runner.py` | `transcribed` | 原始 JSON 未入库 |
| C10 | 低秩 ALS 一致负收益 +10.6%–64.8%（§5.7 Table 4） | 无机器可读产物；数字在论文文本 | `experiments/round5_runner.py`（低秩 ALS 路径） | `transcribed` | 原始 JSON 未入库 |
| C11 | 深度边界 L* ≈ 26 推导（§6.2–6.4, Appendix A） | 基于 C6/C7 的拟合（仅 2 个模型深度的 τ 拟合） | `docs/math-analysis.md` | `inferred` | 论文表述已收缩为"24–28 层之间的失稳转变"；ρ̄=1.08 为两点拟合，机制未做因果验证 |
| C12 | 交叉点外推 800–5000 步（§6.3） | 无 | — | `predicted` | 论文已标注 speculative；Llama-2-7B 行为纯外推 |
| C13 | HellaSwag 基线 acc=0.535（§5.6） | 无机器可读产物 | lm-eval harness（未入库配置） | `transcribed` | 仅预训练 Mistral 基线；**不存在任何协议级下游评估**（见 P0.2 预注册） |
| C14 | ASP 隐式正则化（train≈eval @1200s）（§5.4, §7.2） | 无机器可读产物 | `docs/round9_overfitting.md` | `transcribed` | 单数据集、未与 early-stopping/weight-decay 基线对照（P1.3） |

## 数字冲突登记（禁止跨评估集直接计算比率）

三套评估数字，禁止混用：

1. **N_EVAL=200（~12,640 tokens）** — 7B 全部入库结果使用此协议。基线 105.56；B=1.25；C=135.36；D=10.41。合法比率：B/D=8.3×，基线/B≈84×。
2. **完整测试集（~298,938 tokens）** — 产物丢失（C4），全部数字（1.26/1.25/1.25、基线 133.16）视为不可验证，已从论文正文移除。
3. **小模型评估（50–100 samples，逐实验不同）** — Table 1 与 Table 4 的 Protocol C 基线相差 18×（103.6 vs 5.5）源于评估集与实现差异，论文 §5.7 已披露；不得将小模型 PPL 与 7B PPL 直接比较。

## 审计总结

- **可从仓库复算**：C3、C6，及 C1/C7 的 OPT/Qwen-0.5B 部分。
- **仅可核对转录**：C2、C5、C8、C9、C10、C13、C14（原始产物丢失或从未入库）。
- **推断/预测**：C11、C12。
- 净结论：**论文的最强主张（7B B vs D、深度失稳矩阵）目前有一半证据只处于 `transcribed` 状态**。在原始日志/评估产物补齐或重跑之前，任何"validated"措辞都不成立；论文文本已按此收缩（见 P0.3 变更记录，`paper/paper_draft_v0.7.md` 顶部 changelog）。
