#!/bin/bash
# Run Protocol D seeds in separate processes for clean GPU state
set -e
cd /home/room115/alternating-optimization-lora
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX=/tmp/pyc_altopt
export CUDA_VISIBLE_DEVICES=0,1

RESULTS_DIR=runs/qwen25_7b_800s
mkdir -p "$RESULTS_DIR"

for seed in 42 123 456; do
    echo "=== Protocol D seed $seed ==="
    .venv/bin/python -u -c "
import json, torch, gc, time
from experiments.run_7b_gpu import run_protocol

print(f\"Start s$seed at {time.strftime('%H:%M:%S')}\")
r = run_protocol('D', 'adamw', 'lora', $seed, 800)

outfile = '$RESULTS_DIR/Qwen25-7B_PD_800s_s${seed}_v2.json'
with open(outfile, 'w') as f:
    json.dump(r, f, indent=2)

if r.get('status') == 'success':
    print(f\"OK s$seed ppl={r['perplexity']:.2f}\")
else:
    print(f\"FAIL s$seed: {r.get('error','')[:200]}\")
    exit(1)
" 2>&1
    echo ""
done

echo "=== Protocol D: all 3 seeds complete ==="
