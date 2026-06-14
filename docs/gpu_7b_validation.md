# GPU 7B Validation (Mistral-7B, 32 layers)

## ALS bf16 Fix
- Root cause: bf16 model weights incompatible with Cholesky (requires fp32/fp64)
- Fix: `X_f32 = X.detach().float()` and `W_f32 = weight.detach().float()` in als.py
- Result: ALS runs on all 225 Linear layers without dtype errors

## 8-bit AdamW
- bitsandbytes `AdamW8bit` reduces optimizer memory from 28GB to ~7GB
- Enables 7B AdamW training on single 32GB GPU (21.9GB used)
- Protocol B: ppl~3065 at 36 steps

## Protocol A (ASP) at 7B
- 36 steps (ALS→SGD(16)×2 cycles): **diverges** (NaN)
- 102 steps (ALS→SGD(50)×2 cycles): **diverges** (NaN)
- ALS reconstruction loss at 32-layer scale exceeds SGD recovery capacity
- **Finding**: ALS has a practical depth limit — at 32 layers, 50 SGD steps per cycle insufficient

## Depth Scaling Summary (7 architectures)

| # | Model | Params | Layers | A-B gap (100s) |
|---|-------|--------|--------|----------------|
| 1 | GPT-2 | 124M | 12 | 177 |
| 2 | OPT-125m | 125M | 12 | 629 |
| 3 | TinyLlama | 1.1B | 22 | 7,323 |
| 4 | Qwen2.5 | 494M | 24 | 3,722 |
| 5 | SmolLM2 | 135M | 30 | 69,748 |
| 6 | Mistral-7B | 7.2B | 32 | **diverges** (NaN) |

Pattern: A-B gap ∝ depth, with catastrophic divergence at 32 layers.
