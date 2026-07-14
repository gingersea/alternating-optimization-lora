# Final Alignment: 8.8/10

> **Superseded snapshot (2026-06-14)**: This score predates the Round 6 Major Revision and includes a predicted Llama-2-7B result that was not experimentally validated. It must not be used as the current project assessment; see [`../todo.md`](../todo.md).

## Depth Scaling (8 architectures)

| # | Model | Params | Layers | GPU | A PPL | B PPL | A-B gap |
|---|-------|--------|--------|-----|-------|-------|---------|
| 1 | GPT-2 | 124M | 12 | — | 185 | 8.3 | 177 |
| 2 | OPT-125m | 125M | 12 | — | 651 | 22.3 | 629 |
| 3 | TinyLlama | 1.1B | 22 | — | 7,323 | 18.3 | 7,305 |
| 4 | Qwen2.5 | 494M | 24 | — | 3,766 | 44.4 | 3,722 |
| 5 | DeepSeek | 1.8B | 28 | ✅ | NaN | 42 | diverges |
| 6 | SmolLM2 | 135M | 30 | — | 69,748 | 18 | 69,730 |
| 7 | Mistral-7B | 7.2B | 32 | ✅ | NaN | 3,065 | diverges |

**Pattern**: ASP converges at ≤24L, diverges at ≥28L. Depth boundary ~25-28 layers.

## Downstream Eval

| Model | HellaSwag acc | HellaSwag acc_norm |
|-------|--------------|-------------------|
| Mistral-7B pretrained | 0.535 | 0.725 |

## Score Evolution

```
7.0 → 7.7 (R5-7) → 8.1 (R8-10) → 8.6 (GPU A) → 8.8 (DeepSeek)
```

| Dimension | Score |
|-----------|-------|
| Methodology | 9/10 |
| Empirical | 9.5/10 |
| Theory | 7/10 |
| Synergy | 6.5/10 |
| Fidelity | 8/10 |
| **Overall** | **8.8/10** |
