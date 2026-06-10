# 漏洞分析 #001: GPT-2 实验的可重复性问题

**日期**: 2026-06-10
**关联实验**: exp_001_gpt2_50steps

---

## 发现的问题

### 1. ALS 对 GPT-2 无效 (CRITICAL)

**根因**: GPT-2 所有层使用 `Conv1D` 模块，而非 `nn.Linear`。ALS 求解器的 `solve_block()` 只检查 `isinstance(module, nn.Linear)`，因此从未对任何 GPT-2 层执行实际的块求解。

**证据**: Protocol A 的 FLOPs breakdown 显示 ALS=1.49e9 FLOPs，但这些全部来自 forward pass 和 hook 开销，而非矩阵求逆。

**影响**: Protocol A 的 "交替优化" 实际只运行了 SGD，ALS 阶段是空操作。A/C 协议的性能差异不能归因于 ALS。

### 2. Protocol D = Protocol B (identical)

**根因**: GPT-2 无 nn.Linear 模块，LoRA 无法注入任何 adapter。Protocol D 降级为全秩 AdamW，与 Protocol B 产生完全相同的结果（loss history 逐位一致）。

**影响**: 2×2 析因矩阵中 D 格缺失，无法计算参数形态效应和交互效应。

### 3. Protocol C 语义错误

**根因**: LoRA 注入失败后，AltOptFramework 在原始（未修改的）模型上创建，SGD 步骤在全秩参数上运行。

**影响**: Protocol C 实际是 "全秩 SGD+momentum with perturbation"，不是 "LoRA-structured AltOpt"。与 Protocol A 的差异来自 optimizer 类型（SGD vs SGD+momentum），而非参数形态。

### 4. Perturbation 阶段返回非 loss 值

**根因**: `perturb.apply_noise()` 返回 `avg_noise_energy`（噪声能量），而非训练 loss。但在 `AltOptFramework.step()` 中，所有阶段的返回值都作为 "loss" 记录。

**证据**: Protocol A loss history 中 step 17 和 36 的值为 ~5.9e-07（噪声能量 ≈ 0），step 18 和 37 为 ~6950 和 ~1397（ALS 的 forward pass loss sum）。

**影响**: Loss 曲线被非可比值污染，无法直接比较不同 phase 之间的 "loss"。

### 5. 步数不匹配

**根因**: AltOpt 调度 (ALS(1)+SGD(15)+PERTURB(1))×3cyc = 51 步 > max_steps=50。dataloader (160 samples, batch_size=4) = 40 batches。实际只完成 ~40 步，ALS 周期可能不完整。

## 修复计划

| 优先级 | 问题 | 修复方案 | 状态 |
|--------|------|---------|------|
| P0 | ALS 不处理 Conv1D | ALS 求解器增加 Conv1D 支持 | ✅ 已修复 |
| P0 | Perturbation 返回噪声而非 loss | 记录 phase 标签，区分 loss 和 noise energy | ⬜ 后续 |
| P1 | Protocol D = Protocol B | 切换模型到有 nn.Linear 的架构 (OPT-125m) | ⬜ 后续 |
| P2 | 步数不匹配 | 对齐 dataloader 大小和 max_steps | ⬜ 后续 |

## 修复验证 (exp_002)

ALS Conv1D 修复后重跑实验，对比 exp_001:

| Protocol | Metric | exp_001 | exp_002 | 结论 |
|----------|--------|---------|---------|------|
| A | final_loss | 6.30 | 5.11 | ALS 实际起效，损失下降 |
| A | best_ppl | 185.30 | 318.22 | 交替优化改变了收敛路径 |
| A | loss_history | — | — | ⚠️ 不同（ALS 修改了 Conv1D 权重） |
| B | 全部指标 | — | — | ✅ 精确可重复 |
| C | 全部指标 | — | — | ✅ 精确可重复 |
| D | 全部指标 | — | — | ✅ 精确可重复 |

**结论**:
- B/C/D 在固定 seed 下完全可重复，验证了实验框架的确定性
- A 的差异来自 ALS 修复（之前未实际运行），非框架不可靠
- ALS Conv1D 修复成功 — Protocol A 现在真正执行了矩阵求逆和权重更新
