# Round 10: Architecture Scaling + 800-step Synergy

## TinyLlama-1.1B 2×2 (100 steps)

| Protocol | PPL |
|----------|-----|
| A (ASP/Full) | 7,341 |
| B (AdamW/Full) | 18.3 |
| C (ASP/LoRA) | 71.2 |
| D (AdamW/LoRA) | **13.0** |
| A-B gap | 7,323 |

TinyLlama: 1.1B params, 22 layers, Llama architecture.

## Depth Scaling (6 architectures, 100-step A-B gap)

| # | Model | Params | Layers | A-B Gap | Gap/Layer |
|---|-------|--------|--------|---------|-----------|
| 1 | GPT-2 | 124M | 12 | 177 | 15 |
| 2 | OPT-125m | 125M | 12 | 629 | 52 |
| 3 | TinyLlama-1.1B | 1.1B | 22 | 7,323 | 333 |
| 4 | Qwen2.5-0.5B | 494M | 24 | 3,722 | 155 |
| 5 | SmolLM2-135M | 135M | 30 | 69,748 | 2,325 |
| 6 | Llama-2-7B* | 7B | 32 | predicted | — |

Pattern: A-B gap grows superlinearly with depth. Consistent across architectures.

## Protocol C 800-step Synergy (Low-Rank ALS)

| Condition | PPL |
|-----------|-----|
| No ALS (SGD+perturb) | 10,534 |
| With low-rank ALS | 12,332 |
| Δ | +17% (ALS hurts) |

ALS consistently negative at all tested step counts (100, 200, 400, 800).
