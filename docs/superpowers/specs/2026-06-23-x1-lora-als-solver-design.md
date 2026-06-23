# X1: Iterative Low-Rank ALS Solver — Design Spec

**Date**: 2026-06-23
**Status**: Approved — proceeding to implementation

---

## Problem

The existing `solve_low_rank_block` method (`altopt/als.py:459-579`) attempts Cholesky decomposition on $(X^TX + \lambda I)$ for LoRA-parameterized layers. For the composite weight $W_{\text{eff}} = W_{\text{base}} + (\alpha/r)BA$, the Cholesky factorization frequently fails (RuntimeError: matrix not positive definite), especially at 7B scale with bf16 precision.

This prevents Protocol C from including the ALS phase, making the 2×2 factorial asymmetric — the paper's most significant methodological limitation.

## Solution

Replace Cholesky decomposition with **conjugate gradient (CG) on normal equations**.

For each output block of size $b$:

$$(X^TX + \lambda I) \cdot W_{\text{block}}^T = X^T Y_{\text{block}}$$

Solve iteratively via `torch.linalg.solve` with CG backend instead of $\mathcal{O}(b^3)$ Cholesky.

## Implementation

### Single function replacement

New function `_solve_block_cg()` replaces the inner Cholesky → `cholesky_solve` → B-projection loop inside `solve_low_rank_block()`.

```python
def _solve_block_cg(self, XtX_reg, X, Y_block, device, max_iter=50, tol=1e-6):
    """Solve (X^TX + λI)W^T = X^T Y via conjugate gradient."""
    rhs = X.T @ Y_block  # [d_in, b]
    
    # Use torch.linalg.solve with CG iteration
    # Precondition with diagonal of XtX_reg
    diag = torch.diag(XtX_reg)
    M = torch.diag(1.0 / diag)  # Jacobi preconditioner
    
    # CG: solve (X^TX + λI) @ W^T = X^T @ Y
    # We solve column by column to keep memory bounded
    W_new_T = torch.zeros_like(rhs)
    for j in range(rhs.shape[1]):
        w, info = self._cg_solve(XtX_reg, rhs[:, j], M, max_iter, tol)
        W_new_T[:, j] = w
    
    return W_new_T.T  # [b, d_in]
```

### Conservative fallback

If CG fails to converge (residual > 0.1 after max_iter), fall back to torch.linalg.lstsq.

### Unchanged components

- PEFT adapter discovery (`all_adapter_info()`)
- Activation hooks for layer interrogation
- Block partitioning logic (start/end stride)
- B-update projection: `delta_B = Delta_W @ A_pinv.T / scaling`

## Success Criteria

1. On **Qwen2.5-0.5B**: CG result matches Cholesky result within 1% PPL
2. On **Qwen2.5-7B**: CG succeeds where Cholesky fails (no NaN, no RuntimeError)
3. Full Protocol C cycle (ALS→SGD→Perturb) produces valid (non-NaN, non-M) PPL

## Scope

- One file: `altopt/als.py`
- Two functions modified: `solve_low_rank_block()` (inner solver swap) + new `_cg_solve()` helper
- Tested on one 7B model (Qwen2.5-7B, Protocol C)
