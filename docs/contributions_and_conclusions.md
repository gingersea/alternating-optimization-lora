# Contributions and Conclusions — "Disentangling Optimizer and Parameter Form" (v3.4)

---

## 7 Contributions

### C1. 2×2 Factorial Methodology
The first application of factorial experimental design to post-training optimization, crossing optimizer type (ASP vs AdamW) with parameter form (full-rank vs LoRA). Enables clean attribution of main effects and their interaction — reusable across any pair of confounded post-training strategies.

### C2. Rank Sufficiency Law
A closed-form architectural law $r_{\min} = \eta \cdot L/d_h$ ($\eta \approx 230$) predicting the minimum LoRA rank from layer count $L$ and hidden dimension $d_h$. Derived from residual stream capacity equilibrium, calibrated on 5 model families, confirmed by 3/3 falsification tests. Three alternative mechanisms (token entropy scaling, training budget scaling, universal constant) experimentally eliminated; $\eta$ shown to be modulated by pretraining quality.

### C3. Full-Rank Overfitting + M-Index Diagnostic
Demonstration that full-rank fine-tuning on $N_d < 10^4$ samples catastrophically overfits: near-perfect in-distribution PPL (1.25 on WikiText-2) masks downstream degradation (HellaSwag −3.2pp, MMLU −4.2pp, ARC −3.3pp). The M-index ($M = \text{PPL}_{\text{train}}/\text{PPL}_{\text{cross}}$) provides a lightweight diagnostic: $M < 1$ when $N_p/N_d > 10^4$ reliably flags memorization. LoRA $r=8$ preserves $>99.7\%$ baseline accuracy on all three downstream tasks.

### C4. ASP Depth Boundary
Discovery of a fundamental depth limit at $L_{\max} \approx 26$ layers for ALS-based optimization. ALS perturbation amplifies through residual connections as $\|\delta_L\| \approx \|\delta_\ell\| \cdot \bar{\rho}^{L-\ell}$ ($\bar{\rho} \approx 1.08$). Converges for $\leq 24$ layers, diverges catastrophically for $\geq 28$ layers. Confirmed on 8 architectures with 11 failed 7B attempts across two distributed backends (DeepSpeed ZeRO-2, PyTorch FSDP). A causal reinterpretation frames the boundary through structural causal model intervention propagation.

### C5. ASP-LoRA Negative Synergy
Robust negative interaction ($\text{Int} > 10^3$ PPL) showing ASP degrades LoRA performance across all 7 independent comparisons. Root cause: ALS pays full-rank computational cost ($\mathcal{O}(N d_{\text{in}}^2 + d_{\text{in}}^3)$) but its solution must pass through a rank-$r$ B-projection bottleneck, producing cost-information mismatch. The low-rank ALS solver (X1) closes Protocol C's factorial symmetry at all scales but confirms the negative synergy.

### C6. $\eta$ Nomogram
A practical lookup tool mapping any Transformer's $(L, d_h, N_{\text{pretrain}})$ to its recommended LoRA rank. Regression $\eta(L/d_h, N_{\text{pretrain}}) = 269 + 2386 \cdot (L/d_h) - 47 \cdot \log_{10}(N_{\text{pretrain}})$, $R^2 = 0.88$ across 7 calibration architectures. Across 14 popular architectures, only SmolLM2-135M requires $r > 8$.

### C7. Systematic Falsification of Alternative Hypotheses
Three competing explanations for LoRA rank sufficiency — token entropy scaling ($\eta \propto H$), training budget scaling ($\eta \propto 1/N$), and universal constant ($\eta$ identical across models) — each tested and experimentally falsified. The sole surviving mechanism is pretraining quality modulation, confirmed by a 50× performance gap between identically-sized models with different pretraining budgets.

---

## 5 Conclusions

### Conclusion 1: The Rank Sufficiency Law
$r_{\min} = \eta \cdot L/d_h$ ($\eta \approx 230$, modulated by pretraining quality) predicts $r=8$ is sufficient for all currently popular architectures ($L/d_h \leq 0.035$). The plateau is confirmed across languages (Chinese/English), tasks (SST-2 classification), and training horizons (100–1600 steps). Three alternative mechanisms falsified.

### Conclusion 2: Full-Rank Catastrophically Overfits
Full-rank fine-tuning on small data produces near-perfect in-distribution PPL while reducing downstream accuracy. LoRA $r=8$ preserves $>99\%$ baseline accuracy. The M-index ($M < 1$ when $N_p/N_d > 10^4$) reliably flags the memorization regime. Near-perfect perplexity on small domains reflects memorization, not generalization — a caution for post-training evaluation practice.

### Conclusion 3: ASP Depth Continuum
ASP exhibits non-monotonic convergence at 12 layers with real Cholesky ALS, catastrophic divergence at 28+ layers. Within the stable regime ($L \leq 24$), ASP provides implicit regularization against AdamW overfitting, maintaining train–eval loss parity at 1,200 steps while AdamW degrades. The depth boundary is a continuum endpoint, not an isolated threshold — confirmed by 8/8 architecture measurements.

### Conclusion 4: The Optimal LoRA Rank
$r^* = \max(8, \lceil \eta \cdot L/d_h \rceil)$ — never full-rank when $N_d < 10^4$. At long horizons, $r=8$ is not merely sufficient but optimal ($r=256$ overfits). The $r=8$ plateau is independent of total model scale (0.5B to 7B). Full-rank should never be used for small-data post-training, regardless of model scale.

### Conclusion 5: FFN-Adapted LoRA Lowers $r_{\min}$
Applying LoRA to FFN layers in addition to attention modules reduces the minimum sufficient rank (e.g., attn+FFN $r=4$ outperforms attn-only $r=8$). Confirms the per-layer correction capacity model and enables 2× more parameter-efficient fine-tuning by distributing the correction burden across more modules.

---

## Unified Design Rule

$$\boxed{r^* = \max\!\left(8,\; \left\lceil \eta_0 \cdot \frac{L}{d_h} \cdot q^{-1}(N_{\text{pretrain}}) \right\rceil \right)}$$

$$\boxed{\text{Never use full-rank when } \frac{N_p}{N_d} > 10^4}$$
