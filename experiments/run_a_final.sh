#!/usr/bin/env bash
# Protocol A v3: AltOpt + Full-Rank + DeepSpeed ZeRO-2 + CPU offload
# Qwen2.5-7B, 704 steps (2 cycles), 3 seeds
# ALS lm_head-only, reg_lambda=1e-3, step_size=0.01, block_size=512
# SGD 350 steps per cycle (digestion τ≈350 for 28L)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJ_DIR/.venv"

# ── NVIDIA CUDA 13 Library Path ──
NVIDIA_LIBS=$(find "$VENV_DIR/lib/python3.12/site-packages/nvidia" -name "lib" -type d 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"

# ── DeepSpeed JIT + CUDA ──
export PATH="$VENV_DIR/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export DS_SKIP_CUDA_CHECK=1

# ── HF Offline ──
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── Performance ──
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=/tmp/pyc_a_v3

LOGFILE="/tmp/phase_a_v3_$(date +%Y%m%d_%H%M%S).log"

echo "==================================================================" | tee "$LOGFILE"
echo "Protocol A v3: AltOpt+Full-Rank (DeepSpeed ZeRO-2 + CPU offload)" | tee -a "$LOGFILE"
echo "ALS: lm_head only, step_size=0.01, reg_lambda=1e-3, block=512" | tee -a "$LOGFILE"
echo "SGD: 350 steps/cycle, lr=5e-5" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"
echo "GPUs:" | tee -a "$LOGFILE"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee -a "$LOGFILE"
echo "==================================================================" | tee -a "$LOGFILE"

cd "$PROJ_DIR"

for seed in 42 123 456; do
    echo "" | tee -a "$LOGFILE"
    echo "--- Seed $seed @ $(date) ---" | tee -a "$LOGFILE"
    CUDA_VISIBLE_DEVICES=0 "$VENV_DIR/bin/python" -u -c "
from experiments.run_7b_gpu import run_protocol
result = run_protocol('A', 'altopt', 'full_rank', $seed, 800)
if result.get('status') == 'success':
    print(f'SEED $seed OK: ppl={result[\"perplexity\"]:.2f}, time={result[\"wall_time_s\"]:.0f}s')
else:
    print(f'SEED $seed FAIL: {result.get(\"error\", \"unknown\")[:300]}')
" 2>&1 | tee -a "$LOGFILE"
    echo "--- Seed $seed done @ $(date) ---" | tee -a "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"
echo "==================================================================" | tee -a "$LOGFILE"
echo "Protocol A complete @ $(date)" | tee -a "$LOGFILE"
echo "Log: $LOGFILE" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "--- Results ---" | tee -a "$LOGFILE"
"$VENV_DIR/bin/python" -c "
import json, numpy as np
from pathlib import Path
out_dir = Path('runs/qwen25_7b_800s')
sfx = '_800s_'
for proto in ['A', 'B']:
    for f in sorted(out_dir.glob(f'Qwen25-7B_P{proto}{sfx}*.json')):
        d = json.loads(f.read_text())
        if d.get('status') == 'success':
            print(f'{f.stem}: ppl={d[\"perplexity\"]:.2f}, time={d[\"wall_time_s\"]:.0f}s')
" 2>&1 | tee -a "$LOGFILE"
