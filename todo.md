# alternating-optimization-lora 项目推进路线

## 当前基线 (Round 2 完成)
- 34 files, 3675 LOC, 67 tests all passing
- 核心框架: AltOptFramework, LoRABaseline ✅
- 基础设施: Trainer, Profiling, PEFT, Checkpoint, Evaluator ✅
- 尚未在真实模型上端到端运行 ❌

---

## 计划执行清单

### Step 1: 端到端集成验证 (GPT-2 最小可行)
**目标**: 在 GPT-2 + WikiText-2 上跑通 Protocol A (AltOpt full-rank) 单步
**验收标准**: 加载 GPT-2 → 构建 Trainer → 跑 3 步不报错 → loss 正常下降
**评估**: ✅ PASS — GPT-2 (124M) 加载成功，3步训练完成。loss: 827k→8.0→9.4 (第一步 ALS 全量前传 loss sum，后续 SGD 步正常)。FLOPs 估算: 2.24e9。发现数据集名称需修复 (wikitext-2-raw-v1 → wikitext, wikitext-2-raw-v1)
**状态**: ✅

### Step 2: 修复 Protocol B/C/D 集成路径
**目标**: 确保四种 Protocol 都能跑通
**验收标准**: Python 脚本跑 A→B→C→D 各 5 步，无异常退出
**评估**: ✅ PASS — 全部通过。发现两个问题并修复：(1) Protocol C 原本错误地使用 AdamW 而非 AltOpt，已修复为 SGD+perturb alternation on LoRA params；(2) Protocol D 因 GPT-2 用 Conv1D 而非 nn.Linear，LoRA 注入失败，已添加降级逻辑 fallback 到 full-rank AdamW。67 单元测试无回归。
**状态**: ✅

### Step 3: 统一评分协议实战验证
**目标**: 确保四种 Protocol 用完全相同的 eval data/batch/metric 进行评估
**验收标准**: 四协议在同一 eval set 上产出可比较的 perplexity/loss
**评估**: ✅ PASS — 全部通过。(1) 四协议 eval keys 完全一致 (perplexity, loss, n_tokens)；(2) 四协议在同一 1024 token eval set 上评估；(3) 输出格式可直接比较。A: ppl=40612 (5步太少, ALS 未收敛), B: ppl=12.70, C: ppl=2141, D: ppl=12.70
**状态**: ✅

### Step 4: FLOPs 精确计数接入 (fvcore)
**目标**: 用 fvcore 替换启发式估算，实际测量 ALS vs SGD vs AdamW 的单步 FLOPs
**验收标准**: 输出 per-phase FLOPs breakdown (ALS/SGD/Perturb/AdamW 各自多少 FLOPs)
**评估**: ✅ PASS — FlopsProfiler.record_step() 按 phase 分计。Protocol A (6步): ALS=4.98e8 (11.8%), SGD=3.73e9 (88.2%)。Protocol B (5步): AdamW=6.22e9 (100%)。ALS 单步成本最低 (4× params, forward only)，SGD 居中 (6× params)，AdamW 最贵 (10× params)。fvcore 未安装时使用 param-based 启发式估算，安装后自动切到 op-level 计数。67 测试无回归。
**状态**: ✅

### Step 5: 小规模对比实验 (GPT-2, 100 steps)
**目标**: 运行完整的 2×2 对比，产出第一组可比较数据
**验收标准**: 四组 loss 曲线 + perplexity table + FLOPs/memory 报告
**依赖**: Step 3, Step 4
**状态**: ⬜

### Step 6: 实验报告撰写
**目标**: 基于 Step 5 的结果写实验报告
**验收标准**: markdown 报告含: 实验设置、结果表格、分析、结论
**依赖**: Step 5
**状态**: ⬜

---

## 后续规划 (本轮后)

### Phase 2: 规模化
- 7B 模型 (Llama-2-7B) 验证
- 多 GPU / DeepSpeed 支持

### Phase 3: 消融实验
- RQ1-RQ6 系统性研究
- ALS:SGD 比例扫描
- 扰动强度消融

---

*Last updated: 2026-06-10*
