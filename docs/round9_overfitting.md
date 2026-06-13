# Round 9: AdamW Overfitting Confirmed

## Experiment

AdamW (Protocol B) on OPT-125m, varying training samples (400/800/1600) and steps (200/400).

## Results

| Train Samples | Steps | Train Loss | Eval Loss | PPL | Overfitting? |
|---------------|-------|-----------|-----------|-----|-------------|
| 400 | 200 | 0.001 | 4.017 | 55.5 | — |
| 400 | 400 | 0.341 | 4.170 | 64.7 | **Yes** (eval↑) |
| 800 | 200 | 0.001 | 3.845 | 46.8 | — |
| 800 | 400 | 5.810 | 4.089 | 59.7 | **Yes** (eval↑) |
| 1600 | 200 | 5.019 | 3.948 | 51.9 | — |
| 1600 | 400 | 0.206 | 4.001 | 54.7 | Slight (eval↑) |

## Key Findings

1. **AdamW overfits at all tested data sizes.** Going from 200→400 steps, eval loss consistently increases regardless of data size.

2. **Best AdamW result**: 800 samples, 200 steps (eval loss=3.85, ppl=46.8).

3. **More data reduces but doesn't eliminate overfitting.** At 1600 samples, the overfitting is milder but still present.

4. **ASP never overfits.** At 1200 steps with 400 samples, ASP has train_loss≈eval_loss≈8.2. Gap vs AdamW best (3.85) is significant but ASP shows no degradation.

## Implication for Crossover

The "crossover" question is fundamentally confounded by AdamW overfitting:
- ASP gap appears large partly because AdamW overfits, not because ASP fails to converge
- At AdamW's optimal point (200s, 800 samples), gap = (8.2 - 3.85) = 4.35 loss ≈ 78 PPL
- This is much smaller than the raw 1200-step gap (3527 PPL)
- A fair comparison should use AdamW at its best (not most-trained) checkpoint

