# Qwen2.5-7B 2×2 Factorial — Complete Work Plan

**Date**: 2026-06-20
**Status**: Phase A (Protocol C+D) ✅ → Phase B (driver upgrade + Protocol B) ⏳

---

## Phase A: Protocol C+D (LoRA) — ✅ DONE

- [x] Fix LoRALayer device alignment (ebaceda)
- [x] Fix Protocol C ALS strip for LoRA (4bc3401)
- [x] Fix LoRALayer dtype=bf16 (fe03e04)
- [x] Fix Protocol D PeftBridge device_map (0b94782)
- [x] Run Protocol C × 3 seeds × 800 steps — PPL 135.36 ± 11.1
- [x] Run Protocol D × 3 seeds × 800 steps — PPL 10.41 ± 0.0
- [x] Push results to GitHub

---

## Phase B: Protocol B Full-Rank — ⏳ PENDING

### B1: Upgrade CUDA Driver
- [ ] `sudo apt install nvidia-driver-595-open`
- [ ] `sudo reboot`
- [ ] Verify: `nvidia-smi` shows driver 595.x
- [ ] Verify: GPU memory clean (2/18 MiB)

### B2: Environment Verification
- [ ] `torch.cuda.is_available() == True` on both GPUs
- [ ] `torch.cuda.get_device_name(0)` = "NVIDIA GeForce RTX 5090"
- [ ] `torch.cuda.get_device_name(1)` = "NVIDIA GeForce RTX 5090"
- [ ] `import deepspeed` — no errors
- [ ] `import bitsandbytes` — 8-bit AdamW available
- [ ] DeepSpeed JIT compiles (ninja in PATH, nvcc available)

### B3: Restore Full-Rank Code Path
- [ ] Restore DeepSpeed config in run_7b_gpu.py (use_deepspeed=True, deepspeed_zero_stage=2, model→CPU loading)
- [ ] Restore communication_data_type safely (or keep removed)
- [ ] protocols list = [("B", "adamw", "full_rank")]

### B4: Protocol B Smoke Test
- [ ] Run Protocol B seed 42 × 800 steps (DeepSpeed ZeRO-2 + 8-bit AdamW)
- [ ] Verify: training runs without OOM
- [ ] Verify: PPL values are reasonable (not NaN/Inf)
- [ ] Verify: ~20 min per run wall time

### B5: Protocol B Full Experiment
- [ ] Run Protocol B × 3 seeds × 800 steps
- [ ] Each seed in separate process (clean GPU)
- [ ] Log output to /tmp/exp_b_final.log

### B6: Merge Results
- [ ] Combine Protocol B results into runs/qwen25_7b_800s/combined_results.json
- [ ] Compute mean PPL ± std for Protocol B
- [ ] 2×2 matrix: C=135.36, D=10.41, B=TBD, A=skipped (depth boundary)

### B7: Final Analysis
- [ ] AdamW full-rank vs LoRA comparison on 7B
- [ ] Main effects and interaction from 2×2 (3/4 cells)
- [ ] Consistency check with OPT-125m and GPT-2 results
- [ ] Write summary to docs/phase_b_results.md

---

## 2×2 Factorial Matrix (Qwen2.5-7B @ 800 steps)

| | AltOpt (ASP) | AdamW |
|---|---|---|
| **LoRA** | C: 135.36 ± 11.1 ✅ | D: 10.41 ± 0.0 ✅ |
| **Full-rank** | A: skipped (depth boundary) | B: TBD ⏳ |

---

## Key Commands (Post-Reboot)

```bash
# Verify driver
nvidia-smi --query-gpu=index,driver_version,name,memory.total --format=csv,noheader

# Verify PyTorch
cd /home/room115/alternating-optimization-lora
.venv/bin/python -c "
import torch; import deepspeed; import bitsandbytes
print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
print('deepspeed OK, bitsandbytes OK')
"

# Protocol B smoke test
CUDA_VISIBLE_DEVICES=0,1 bash -c '
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX=/tmp/pyc_b_final
cd /home/room115/alternating-optimization-lora
.venv/bin/python -u -c "
from experiments.run_7b_gpu import run_protocol
r = run_protocol(\"B\", \"adamw\", \"full_rank\", 42, 800)
print(\"OK\" if r.get(\"status\")==\"success\" else f\"FAIL: {r.get(\"error\",\"\")[:200]}\")
printf(\"ppl={r.get(\"perplexity\",float(\"nan\")):.2f}\" if r.get(\"status\")==\"success\" else \"\")
" 2>&1 | tee /tmp/exp_b_final.log
'
```

---

*Last updated: 2026-06-20*
