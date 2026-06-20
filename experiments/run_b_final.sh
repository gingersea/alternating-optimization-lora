#!/usr/bin/env bash
# Phase B: AdamW + Full-Rank (Protocol B) × 3 seeds × 800 steps
# Qwen2.5-7B with DeepSpeed ZeRO-2 on 2 GPUs + 8-bit AdamW
# CUDA driver 595.x, PyTorch 2.12+cu130

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJ_DIR/.venv"

# ── NVIDIA CUDA 13 Library Path (for bitsandbytes) ──
NVIDIA_LIBS=$(find "$VENV_DIR/lib/python3.12/site-packages/nvidia" -name "lib" -type d 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"

# ── DeepSpeed JIT Compilation (ninja + nvcc) ──
export PATH="$VENV_DIR/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export DS_SKIP_CUDA_CHECK=1

# ── HF Offline Mode ──
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── Performance ──
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=/tmp/pyc_b_ds

LOGFILE="/tmp/phase_b_ds_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================" | tee "$LOGFILE"
echo "Phase B: AdamW Full-Rank (DeepSpeed ZeRO-2) — Qwen2.5-7B, 800 steps, 3 seeds" | tee -a "$LOGFILE"
echo "Started at: $(date)" | tee -a "$LOGFILE"
echo "GPUs:" | tee -a "$LOGFILE"
nvidia-smi --query-gpu=index,driver_version,name,memory.total,memory.used --format=csv,noheader | tee -a "$LOGFILE"
echo "=================================================================" | tee -a "$LOGFILE"

cd "$PROJ_DIR"

for seed in 42 123 456; do
    echo "" | tee -a "$LOGFILE"
    echo "--- Seed $seed @ $(date) ---" | tee -a "$LOGFILE"

    # DeepSpeed needs both GPUs for ZeRO-2 optimizer sharding
    CUDA_VISIBLE_DEVICES=0,1 "$VENV_DIR/bin/python" -u -c "
from experiments.run_7b_gpu import run_protocol
result = run_protocol('B', 'adamw', 'full_rank', $seed, 800)
if result.get('status') == 'success':
    print(f'SEED $seed OK: ppl={result[\"perplexity\"]:.2f}, time={result[\"wall_time_s\"]:.0f}s')
else:
    print(f'SEED $seed FAIL: {result.get(\"error\", \"unknown\")[:300]}')
" 2>&1 | tee -a "$LOGFILE"
    echo "--- Seed $seed done @ $(date) ---" | tee -a "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
echo "Phase B complete @ $(date)" | tee -a "$LOGFILE"
echo "Log: $LOGFILE" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "--- Results Summary ---" | tee -a "$LOGFILE"
"$VENV_DIR/bin/python" -c "
import json, numpy as np
from pathlib import Path
out_dir = Path('runs/qwen25_7b_800s')
success = []
for f in sorted(out_dir.glob('Qwen25-7B_PB_*.json')):
    d = json.loads(f.read_text())
    if d.get('status') == 'success':
        success.append(d)
        print(f'{f.stem}: ppl={d[\"perplexity\"]:.2f}, time={d[\"wall_time_s\"]:.0f}s')
if success:
    ppls = [r['perplexity'] for r in success]
    print(f'Mean PPL: {np.mean(ppls):.2f} ± {np.std(ppls):.2f} (N={len(ppls)})')
else:
    print('No successful results found.')
" 2>&1 | tee -a "$LOGFILE"
