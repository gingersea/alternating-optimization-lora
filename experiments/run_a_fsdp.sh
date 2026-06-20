#!/usr/bin/env bash
# Protocol A: AltOpt + Full-Rank, FSDP FULL_SHARD + CPU offload
# torchrun × 2, Qwen2.5-7B, 704 steps (2 cycles), 3 seeds
# ALS: lm_head only, SGD: 350 steps/cycle, Perturb: 1 step/cycle
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJ_DIR/.venv"

# ── NVIDIA CUDA 13 Library Path ──
NVIDIA_LIBS=$(find "$VENV_DIR/lib/python3.12/site-packages/nvidia" -name "lib" -type d 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"

# ── HF Offline ──
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── Performance ──
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=/tmp/pyc_a_fsdp
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL

LOGFILE="/tmp/phase_a_fsdp_$(date +%Y%m%d_%H%M%S).log"

echo "==================================================================" | tee "$LOGFILE"
echo "Protocol A: FSDP FULL_SHARD — Qwen2.5-7B, 704 steps, 3 seeds" | tee -a "$LOGFILE"
echo "ALS: lm_head only, SGD: 350s/cycle, Perturb: 1s/cycle, 2 cycles" | tee -a "$LOGFILE"
echo "Memory: FSDP + CPU offload, peak ~23GB/GPU" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee -a "$LOGFILE"
echo "==================================================================" | tee -a "$LOGFILE"

cd "$PROJ_DIR"

for seed in 42 123 456; do
    echo "" | tee -a "$LOGFILE"
    echo "--- Seed $seed @ $(date) ---" | tee -a "$LOGFILE"

    torchrun \
        --nproc_per_node=2 \
        --master_port=$((29500 + seed)) \
        "$VENV_DIR/bin/python" -u "$PROJ_DIR/experiments/run_7b_fsdp.py" \
        "$seed" 800 \
        2>&1 | tee -a "$LOGFILE"

    echo "--- Seed $seed done @ $(date) ---" | tee -a "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"
echo "==================================================================" | tee -a "$LOGFILE"
echo "Protocol A (FSDP) complete @ $(date)" | tee -a "$LOGFILE"
echo "Log: $LOGFILE" | tee -a "$LOGFILE"

# Summary
echo "" | tee -a "$LOGFILE"
echo "--- Results ---" | tee -a "$LOGFILE"
"$VENV_DIR/bin/python" -c "
import json, numpy as np
from pathlib import Path
out_dir = Path('runs/qwen25_7b_800s')
for proto in ['A', 'B']:
    for f in sorted(out_dir.glob(f'Qwen25-7B_P{proto}_*.json')):
        d = json.loads(f.read_text())
        if d.get('status') == 'success':
            print(f'{f.stem}: ppl={d[\"perplexity\"]:.2f}, time={d[\"wall_time_s\"]:.0f}s')
" 2>&1 | tee -a "$LOGFILE"
