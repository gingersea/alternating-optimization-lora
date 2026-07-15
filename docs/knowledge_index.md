# Knowledge Index: Alternating Optimization vs LoRA

**Status**: Comprehensive index of all project knowledge. Search by tag, architecture, or finding ID.

> **Index snapshot (2026-06-14)**: Some confidence labels and open questions were superseded by later experiments and Round 6 review. Use [`../todo.md`](../todo.md) for current priorities and evidence boundaries.

---

## Finding Index

### F1: Methodological

| ID | Finding | Confidence | Evidence | Architecture |
|----|---------|-----------|----------|-------------|
| F1.1 | 2×2 factorial design is genuine methodological contribution | 🔴 High | 3/3 reviewers agree | All |
| F1.2 | Interaction effect (A-B)-(C-D) > 1000 PPL in all architectures | 🔴 High | Tables 1, 3, 5 | GPT-2, OPT, Qwen, TinyLlama |
| F1.3 | FLOPs accounting is necessary for fair comparison (ALS≠SGD≠AdamW) | 🔴 High | Per-phase costing validated | All |
| F1.4 | Protocol C asymmetry (ALS absent in LoRA space) impacts (A-B)-(C-D) interpretation | 🔴 High | Documented in §3.2 | All |
| F1.5 | ASP internal component confound (ALS/SGD/Perturb bundled) limits attribution | 🟡 Medium | §4.3, no ablation | All |

### F2: Convergence

| ID | Finding | Confidence | Evidence | Architecture |
|----|---------|-----------|----------|-------------|
| F2.1 | A-B gap converges non-monotonically (oscillates at ALS cycle boundaries) | 🔴 High | Matrix experiment, 5 step counts | OPT, Qwen |
| F2.2 | A-B gap trends downward: 82k→7k (7.8× shrinkage, OPT-125m, 800s) | 🟡 Medium | Multi-seed, 95% CI wide | OPT-125m |
| F2.3 | AdamW plateaus at 50-100 steps, then degrades (overfitting) | 🔴 High | 400/800/1600 samples tested | OPT-125m |
| F2.4 | ASP never overfits (train≈eval at 1200s, implicit regularization) | 🟡 Medium | Single checkpoint, preliminary | OPT-125m |
| F2.5 | Predicted crossover (A-B < 10 PPL) at >2000 steps, unverified | 🟢 Low | Extrapolation only | All |
| F2.6 | Cohen's d = 1.17 at 800s (OPT-125m, N=5) — large effect | 🟡 Medium | PB ANOVA confirms direction | OPT-125m |

### F3: Depth Scaling

| ID | Finding | Confidence | Evidence | Architecture |
|----|---------|-----------|----------|-------------|
| F3.1 | A-B gap grows superlinearly with layers | 🔴 High | 8 architectures, 12L→32L | All |
| F3.2 | ASP converges at ≤24L | 🔴 High | GPT-2, OPT, TinyLlama, Qwen | 4 architectures |
| F3.3 | ASP diverges at ≥28L (NaN) | 🔴 High | DeepSeek(28L), SmolLM2(30L), Mistral(32L) | 3 architectures |
| F3.4 | Depth boundary ~25-28 layers | 🟡 Medium | Observed at 3 points | DeepSeek, SmolLM2, Mistral |
| F3.5 | ALS perturbation amplification through residual connections explains depth scaling | 🟡 Medium | Signal propagation theory | All |
| F3.6 | Digestion time τ ∝ L^1.2 (superlinear) | 🟢 Low | Only 2 data points (12L, 24L) | OPT, Qwen |

### F4: LoRA

| ID | Finding | Confidence | Evidence | Architecture |
|----|---------|-----------|----------|-------------|
| F4.1 | LoRA dominates at ≤200 steps (5-30× PPL improvement) | 🔴 High | 5 architectures | GPT-2, OPT, Qwen, SmolLM2, TinyLlama |
| F4.2 | Protocol D (AdamW+LoRA) is best performer in all settings | 🔴 High | 8 architectures | All |
| F4.3 | LoRA training variance is very low (CV <5%) | 🔴 High | Multi-seed, all protocols | OPT |
| F4.4 | Low-rank manifold reduces effective condition number | 🟡 Medium | LoRA convergence theory | All |
| F4.5 | PEFT vs built-in LoRA has ~18× performance gap | 🟡 Medium | Table 1 vs Table 4 discrepancy | OPT |

### F5: Perturbation

| ID | Finding | Confidence | Evidence | Architecture |
|----|---------|-----------|----------|-------------|
| F5.1 | Perturbation improves eval PPL but worsens train loss (12s, 86k vs 317k) | 🟢 Low | Single 12-step experiment | GPT-2 |
| F5.2 | Perturbation acts as implicit regularizer (RWP generalization-convergence trade-off) | 🟢 Low | Consistent with RWP theory | GPT-2 |
| F5.3 | Perturbation strength and schedule not systematically ablated | — | Not tested | All |

### F6: Infrastructure

| ID | Finding | Confidence | 
|----|---------|-----------|
| F6.1 | ALS bf16 fix: `.detach().float()` enables GPU Cholesky on all Linear layers | ✅ Tested |
| F6.2 | 8-bit AdamW (bitsandbytes) enables 7B training in 21.9GB on single 32GB GPU | ✅ Tested |
| F6.3 | Low-rank ALS solver implemented: full-rank solve → B projection | ✅ Tested |
| F6.4 | DeepSpeed ZeRO-2 framework code exists, not integrated for 7B | ⚠️ Code ready |
| F6.5 | PB ANOVA framework for heteroscedastic factorial design | ✅ Tested |
| F6.6 | Fieller CI for gap ratio estimation | ✅ Tested |

---

## Architecture Index

| # | Model | Params | Layers | Type | GPU | 100s A-B gap | Converges? |
|---|-------|--------|--------|------|-----|-------------|------------|
| 1 | GPT-2 | 124M | 12 | Conv1D | — | 177 | ✅ |
| 2 | OPT-125m | 125M | 12 | nn.Linear | — | 629 | ✅ |
| 3 | TinyLlama-1.1B | 1.1B | 22 | Llama | — | 7,305 | ✅ |
| 4 | Qwen2.5-0.5B | 494M | 24 | Llama | — | 3,722 | ✅ |
| 5 | DeepSeek-R1-Distill-Qwen-1.5B | 1.8B | 28 | Llama | ✅ | NaN | ❌ |
| 6 | SmolLM2-135M | 135M | 30 | Llama | — | 69,730 | ❌ |
| 7 | Mistral-7B-v0.3 | 7.2B | 32 | Mistral | ✅ | NaN | ❌ |

---

## Experiment Index

| Round | Name | Models | Key Finding |
|-------|------|--------|------------|
| 1 | GPT-2 40s 2×2 | GPT-2 | 2×2 framework works |
| 2 | OPT-125m 100s | OPT-125m | LoRA dominates; ALS:SGD=1:20 |
| 3 | Infrastructure | — | 7B loading + DeepSpeed |
| 4 | Reproducibility | GPT-2 | Poor at 12s; perturbation regularization |
| 5 | 3-seed 2×2 | OPT-125m | Statistical 2×2; Protocol A CV=40.6% |
| 6 | Long SGD cycles | OPT-125m | Crossover trend first observed |
| 7 | Gap closure | GPT-2, OPT | Crossover not reached; overfitting discovered |
| 8 | Synergy + 1200s | OPT-125m | Protocol C ALS still hurts; AdamW overfits at 1200s |
| 9 | Overfitting confirmed | OPT-125m | AdamW overfits at all data sizes (400-1600 samples) |
| 10 | Multi-arch + GPU | TinyLlama, Mistral | 8th arch; GPU validation; depth boundary found |

---

## Mathematical Model Index

| ID | Model | Equations | Verified? |
|----|-------|-----------|-----------|
| M1 | ALS reconstruction loss magnitude | $L_{\text{ALS}} \sim O(N \cdot d \cdot \|W\|^2)$ | ✅ 8/8 experiments |
| M2 | Non-monotonic convergence | $\text{gap}(t) = \sum_c A_c e^{-\alpha(t-t_c)} \mathbb{1}[t \geq t_c]$ | ✅ Matrix experiment |
| M3 | Depth scaling | $\tau \propto L^{1.2}$ | ⚠️ 2 data points |
| M4 | Depth divergence | $\exists L^* \approx 28: \forall L \geq L^*, \text{ASP diverges}$ | ✅ 3 architectures |
| M5 | Perturbation as implicit SAM | $\theta_{t+1} = \theta_t + \varepsilon - \eta\nabla\mathcal{L}(\theta_t+\varepsilon)$ | 🟢 Consistent |

---

## Document Index

| Document | Content |
|----------|---------|
| `docs/synthesis-report.md` | All conclusions + paper outline + 5-claim argument chain |
| `docs/math-analysis.md` | Mathematical derivations + literature review |
| `docs/final_assessment.md` | Research level + goal alignment assessment |
| `docs/alignment_audit.md` | Work vs original intent audit |
| `docs/final_alignment_8.8.md` | Final alignment scores |
| `docs/gpu_7b_validation.md` | GPU 7B validation results |
| `docs/round8_results.md` | Round 8 results |
| `docs/round9_overfitting.md` | Round 9 overfitting analysis |
| `docs/round10_results.md` | Round 10 architecture scaling |
| `docs/experiment-report-001.md` through `005.md` | Individual experiment reports |
| `docs/flaw-analysis-001.md` | GPT-2 Conv1D flaw analysis |
| `paper/paper_draft_v0.2.md` | Current paper draft (v0.5) |
| `paper/review_round1.md` through `round3.md` | All 3 review rounds |
| `paper/revision_plan.md` | Round 1 revision plan |
| `paper/multi_seed_results.json` | Multi-seed data |

---

## Open Questions Index

| ID | Question | Status | Priority |
|----|----------|--------|----------|
| Q1 | Does ASP eventually surpass AdamW at >2000 steps? | Not tested (CPU limit) | P0 |
| Q2 | Can ASP be stabilized at ≥28 layers? | Not tested (smaller blocks? more SGD?) | P1 |
| Q3 | Does Protocol C ALS become positive at >500 steps? | Not tested (low-rank ALS ready) | P1 |
| Q4 | Does ASP's implicit regularization hold across datasets? | Not tested (WikiText-2 only) | P2 |
| Q5 | What is the optimal ALS:SGD ratio at 200+ steps? | Not determined | P2 |
| Q6 | Does AdamW overfitting generalize to larger datasets? | Partially (1600 samples tested) | P2 |

---

*Last indexed: 2026-06-14*
