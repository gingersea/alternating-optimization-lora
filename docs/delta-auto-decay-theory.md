# Why Does A-SYNC CONSTANT Converge Without External Decay?

> **⚠️ SUPERSEDED — READ THIS FIRST**
>
> **2026-07-24 update**: The "gradient injection" this theory explains was subsequently proven to be a **timing no-op** in the implementation (injected after `optimizer.step()`, cleared by the next `zero_grad()`), and a matched-budget pure-SGD control on Qwen7B replicated the "A-SYNC" trajectory exactly (correlation 0.99981; 6.83 vs 6.82 PPL). **The auto-decay mechanism described here is the convergence dynamics of plain SGD post-training, not of any ALS injection.** The power-law convergence, the lr-schedule explanation of the CONSTANT-vs-Cosine ranking, and the fixed-point argument survive only re-assigned to SGD dynamics. See [`docs/diag-injection-report.md`](diag-injection-report.md) and the honestly rewritten [`docs/final-report.md`](final-report.md).

**A mathematical analysis of the auto-decaying ALS delta**
**Date**: 2026-07-24
**Status**: Theory note — grounding the empirical observation (C44 convergence at 48c) in a formal argument. **Partially superseded** — see banner above.

---

## 1. The Empirical Fact

A-SYNC CONSTANT keeps both `sync=0.05` and `lr=2e-4` **constant** across all cycles. There is no decay schedule of any kind. Yet:

- 24c run: converges monotonically to PPL 9.0 (still improving at C24)
- 48c run: converges at C44 to PPL 7.6, then plateaus
- Convergence rate decays naturally without external intervention:

| Cycle window | dPPL/dcycle |
|:------------:|:-----------:|
| C1–6    | −7.19 |
| C6–11   | −2.00 |
| C12–17  | −0.38 |
| C18–23  | −0.19 |
| C24–29  | −0.078 |
| C30–35  | −0.061 |
| C36–41  | −0.046 |
| C42–47  | −0.045 |

The learning rate and injection strength stay constant, yet the *effective progress* decays roughly geometrically. This is the **auto-decay phenomenon**. Why?

---

## 2. The Core Mechanism: ALS δ ∝ Residual Error

Recall the A-SYNC cycle:

```
① ALS solves:   W_als = argmin_W ‖X W^T − Y‖²   (X = final_hidden)
② δ = W_als − W_before
③ W ← W_before (REVERTED — forward pass never sees ALS weights)
④ SGD 50 steps: grad += sync × δ
```

The key quantity is **δ — the ALS's desired change to lm_head**. Its magnitude is determined by how far the *current* lm_head is from the least-squares optimum *given the current body representations*.

Consider a single ALS solve. The least-squares residual is:

$$r(X) = \min_W \|X W^T - Y\|^2$$

The optimal $W^* = (X^T X + \lambda I)^{-1} X^T Y$. The delta is:

$$\delta = W^* - W_{\text{before}}$$

**Claim**: $\|\delta\|$ is proportional to the *model's output error on the training batch*, which decreases as the body improves.

### 2.1 Sketch of the argument

Let $Y = f_\theta(X)$ be the target (one-hot labels or the current forward output). The least-squares solution fits $X W^T \approx Y$. If $W_{\text{before}}$ were already optimal, then $\delta = 0$ and the cycle does nothing. If the model's outputs are poor (large CE loss), then the least-squares solve finds a substantially different $W$, and $\|\delta\|$ is large.

Formally, for a fixed $X$, the residual norm of the current head $W_b$:

$$\|X W_b^T - Y\|_F^2 = \|X(W_b - W^*)^T\|_F^2 + \underbrace{\min_W \|X W^T - Y\|^2}_{\text{irreducible}}$$

The first term is what ALS can remove. Its magnitude is bounded by the singular values of $X$ times $\|W_b - W^*\|_F$, and $W_b - W^* = -\delta$ (up to damping). So:

$$\|X \delta^T\|_F^2 \lesssim \|X W_b^T - Y\|_F^2$$

The model's output error $\|X W_b^T - Y\|_F^2$ is *directly related to the CE loss* — which we observe monotonically decreasing. **As the body improves and the model's predictions get better, the achievable ALS improvement shrinks, and so does ‖δ‖.**

### 2.2 Why the body improves monotonically under CONSTANT

The body receives gradient signal from two sources each SGD step:

$$\nabla L_{\text{eff}} = \nabla L_{\text{CE}} + \text{sync} \times \delta$$

Both terms push the model toward better predictions:
- $\nabla L_{\text{CE}}$: standard CE gradient, always points toward lower loss
- $\text{sync} \times \delta$: a bias term pointing toward the current least-squares optimum. As long as the body hasn't fully adapted, this term provides additional signal; once the body adapts, $\delta \to 0$ and the injection naturally fades.

### 2.3 The fixed-point argument

Let $P(\theta)$ denote the model's prediction quality (lower CE = better). The per-cycle dynamics:

$$P_{t+1} = \Phi(P_t, \delta(P_t))$$

where $\delta(P_t)$ is the ALS delta when the body has quality $P_t$. Two forces:

1. **Gradient descent**: $P_{t+1} > P_t$ (improving predictions) via CE gradient
2. **ALS injection**: accelerates improvement toward the current optimum, but $\|\delta\|$ shrinks as $P_t$ improves

The system converges to a fixed point $P^*$ where $\delta(P^*) \approx 0$ — i.e., the body has adapted so well that the least-squares solution *coincides with the current weights*. At that point:

$$\delta \to 0 \quad \Rightarrow \quad \text{injection} \to 0 \quad \Rightarrow \quad \text{convergence} \to \text{natural plateau}$$

This is exactly what we observe at C44 (PPL 7.6) in the 48c run: the model stops improving *not because* the algorithm decays the signal, but because **the signal has nothing left to say**.

---

## 3. Contrast: Why External Decay Hurts

External decay schedules (cosine, exponential) assume the signal should weaken over time. But the signal's *required* strength is self-regulating:

- **Early cycles**: body is far from optimal → δ is large → strong injection is exactly what's needed for fast progress
- **Late cycles**: body is close to optimal → δ is small → weak injection is automatically what's needed

A decay schedule **kills the signal exactly when it's still useful**:

- Cosine 32c: sync and lr → 0 by C20, yet the model was still improving. Result: plateau at PPL 13.2 — 74% worse than CONSTANT's 7.6.
- Exponential (Vanilla): sync × 0.8/cycle → by C6 the injection is 0.8⁵ ≈ 33% strength. Signal dies early, progress stalls.

External decay is not just redundant — it's *harmful* because it imposes an artificial time-dependence on a quantity whose correct behavior is *state-dependent* (δ should decay based on body quality, not on cycle count).

---

## 4. Testable Predictions

1. **‖δ‖ measured per cycle should decay monotonically** even with constant sync — tracking ‖δ‖/‖W‖ across cycles in the 96c run would confirm this directly.
2. **The plateau PPL should be the same regardless of sync** (as long as sync > 0) — because the plateau is set by the body's capacity, not the injection strength.
3. **Higher sync → faster early convergence but same asymptote** — sync only controls *how fast* the body approaches the fixed point, not *where* it lands.
4. **The 96c run should plateau near PPL 7.6** (matching 48c) OR continue improving if C44 was not the true fixed point — the C44–48 data (7.79 → 7.64, still dropping slowly) suggests the asymptote may be slightly below 7.6.

---

## 5. Open Questions

1. **Formal convergence guarantee**: Can we prove that the A-SYNC fixed point exists and is stable? This would require showing Φ is a contraction in a suitable metric.
2. **Is the plateau PPL the *model's* capacity limit or an A-SYNC-specific limit?** On Qwen0.5B, all A-SYNC+ variants hit PPL 5.5 (the model floor). On 7B, CONSTANT hits 7.6 while AdamW reaches 1.25 — suggesting the fixed point is *not* the model floor but an A-SYNC-specific attractor (likely because ALS only guides lm_head, leaving body layers to pure SGD).
3. **Can we derive ‖δ(P)‖ explicitly?** A closed form would let us *predict* the plateau from the body's initial quality and the ALS solve's properties.

---

*This analysis is grounded in the 24c/48c run data (`runs/a_sync_constant_7b.json`, `runs/a_sync_48cycle_7b.json`). The 96c run (`runs/a_sync_96cycle_7b.json`) is in progress to test prediction 4.*
