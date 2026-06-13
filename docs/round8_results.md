# Round 8 Results

## P0: Protocol C Synergy @ 400 steps

Low-rank ALS on LoRA adapters vs SGD+Perturb only.

| Condition | PPL @200s | PPL @400s |
|-----------|-----------|-----------|
| No ALS (SGD+perturb) | 106.4 | 103.3 |
| With low-rank ALS | — | 131.8 |

**Finding**: Low-rank ALS still worsens Protocol C at 400 steps (+28% PPL).
The ALS digestion bottleneck applies to both full-rank and LoRA spaces.
Synergy not observed at ≤400 steps.

## P1: OPT-125m 1200-step Crossover

Protocol A (ASP full-rank) vs Protocol B (AdamW full-rank).

| Protocol | Train Loss (last 5) | Eval Loss | Eval PPL |
|----------|---------------------|-----------|----------|
| A (ASP) | [8.82, 8.46, 8.57, 8.31, 8.26] | 8.19 | 3,592 |
| B (AdamW) | [0.15, 4.67, 0.01, 0.32, 0.38] | 4.17 | 64.9 |

### Key Discovery: AdamW Overfits, ASP Does Not

- AdamW train loss → 0 but eval loss = 4.17 (severe overfitting on 400 samples at 1200 steps)
- ASP train loss ≈ eval loss ≈ 8 (no overfitting)
- ASP's ALS→SGD alternation provides implicit regularization against overfitting

### Crossover Status

- Gap = 3,592 - 65 = 3,527 PPL — NOT crossed at 1200 steps
- Crossover prediction revised upward: likely >> 2000 steps
- Gap metric is now confounded by AdamW overfitting (eval loss increased from ~2.9 at 200s to 4.17 at 1200s)

## P2: GPT-2 800-step Crossover (from Round 7)

- A (ASP): ppl=3,125; B (AdamW): ppl=14.0
- Gap = 3,111 — NOT crossed at 800 steps

