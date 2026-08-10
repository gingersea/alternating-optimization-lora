"""Analyze 96-cycle A-SYNC CONSTANT run — merge 24c/48c/96c trajectories.

Questions answered:
  1. Does the C44 plateau (PPL 7.6) continue improving with more cycles?
  2. What is the projected asymptote? (power-law fit)
  3. How does the gap to AdamW (PPL 1.25) shrink?
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

FIG = "docs/figures"
os.makedirs(FIG, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

def load(name):
    try:
        with open(f"runs/{name}") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

d24 = load("a_sync_constant_7b.json")
d48 = load("a_sync_48cycle_7b.json")
d96 = load("a_sync_96cycle_7b.json")

# ── 1. Merge trajectories ──────────────────────────────────────────
trajectories = {}
if d24: trajectories["CONSTANT 24c"] = d24["ppls"]
if d48: trajectories["CONSTANT 48c"] = d48["ppls"]
if d96: trajectories["CONSTANT 96c"] = d96["ppls"]

# ── 2. Fit power law: PPL(c) = PPL_inf + A * c^(-alpha) ───────────
def powerlaw(c, ppl_inf, A, alpha):
    return ppl_inf + A * np.power(c, -alpha)

def fit_asymptote(ppls, n_use=None):
    """Fit PPL(c) = ppl_inf + A*c^-alpha on the tail. Returns (ppl_inf, ...)"""
    if n_use is None:
        n_use = len(ppls)
    c = np.arange(1, n_use + 1, dtype=float)
    y = np.array(ppls[:n_use], dtype=float)
    try:
        popt, _ = curve_fit(powerlaw, c, y, p0=[5.0, 50.0, 0.5], maxfev=20000)
        return popt if popt is not None else None
    except Exception as e:
        print(f"  fit failed: {e}")
        return None

# ── Print summary ──────────────────────────────────────────────────
print("=" * 70)
print("96-CYCLE A-SYNC CONSTANT ANALYSIS — Qwen2.5-7B")
print("=" * 70)
for label, ppls in trajectories.items():
    print(f"{label}: C1={ppls[0]:.2f}, best={min(ppls):.2f}, final={ppls[-1]:.2f}, cycles={len(ppls)}")
    if len(ppls) >= 8:
        print(f"  Last 8: {' -> '.join(f'{x:.2f}' for x in ppls[-8:])}")

if d96:
    ppls = d96["ppls"]
    print(f"\n96c elapsed: {d96['elapsed']/3600:.1f}h")
    # Tail convergence rate
    print("\nPer-8-cycle convergence rate (dPPL/dcycle):")
    for i in range(0, len(ppls) - 8, 8):
        rate = (ppls[i+8] - ppls[i]) / 8
        print(f"  C{i+1}-{i+8}: {rate:+.4f} PPL/cycle")

# ── 3. Asymptote fit ───────────────────────────────────────────────
print("\nAsymptote fits (PPL(c) = ppl_inf + A*c^-alpha):")
for label, ppls in trajectories.items():
    if len(ppls) < 12:
        continue
    popt = fit_asymptote(ppls)
    if popt is not None:
        print(f"  {label}: ppl_inf={popt[0]:.3f}, A={popt[1]:.1f}, alpha={popt[2]:.3f}")
        # Project at C200
        proj = powerlaw(200, *popt)
        print(f"    projected PPL at C200: {proj:.2f}")

# ── 4. AdamW comparison ────────────────────────────────────────────
ADAMW_7B_PPL = 1.25  # from prior runs, 800 steps
print(f"\nAdamW 7B reference: PPL={ADAMW_7B_PPL}")
if d96:
    gap = d96["ppls"][-1] / ADAMW_7B_PPL
    print(f"96c final gap to AdamW: {gap:.2f}x")

# ── 5. Plot ────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

colors = {"CONSTANT 24c": "#3498db", "CONSTANT 48c": "#e74c3c", "CONSTANT 96c": "#8e44ad"}
for label, ppls in trajectories.items():
    x = np.arange(1, len(ppls) + 1)
    ax1.plot(x, ppls, "o-", color=colors.get(label, "gray"), label=label, lw=2.5, markersize=6, alpha=0.85)
    ax1.scatter([x[-1]], [ppls[-1]], s=120, color=colors.get(label, "gray"), zorder=10, edgecolors="white")

ax1.axhline(y=1.25, color="green", ls="--", alpha=0.7, lw=2, label="AdamW 7B (800 steps)")
ax1.axhline(y=7.6, color="gray", ls=":", alpha=0.5, label="48c plateau (7.6)")
ax1.set_xlabel("Cycle", fontsize=14)
ax1.set_ylabel("Perplexity", fontsize=14)
ax1.set_title("A-SYNC CONSTANT: 24c/48c/96c Trajectories (Qwen2.5-7B)", fontsize=15, fontweight="bold")
ax1.set_yscale("log")
ax1.legend(fontsize=11, loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 100)

# Right: zoom on tail (C20-C96)
for label, ppls in trajectories.items():
    if len(ppls) < 20: continue
    x = np.arange(1, len(ppls) + 1)
    ax2.plot(x, ppls, "o-", color=colors.get(label, "gray"), label=label, lw=2, markersize=4, alpha=0.85)

# Fit lines on tail
for label, ppls in trajectories.items():
    if len(ppls) < 24: continue
    popt = fit_asymptote(ppls)
    if popt is not None:
        x_fit = np.linspace(max(2, len(ppls)-40), len(ppls) + 40, 200)
        y_fit = powerlaw(x_fit, *popt)
        ax2.plot(x_fit, y_fit, ls="--", color=colors.get(label, "gray"), alpha=0.5, lw=1.5,
                 label=f"{label} fit (asym={popt[0]:.2f})")

ax2.axhline(y=1.25, color="green", ls="--", alpha=0.7, lw=2)
ax2.set_xlabel("Cycle", fontsize=14)
ax2.set_ylabel("Perplexity", fontsize=14)
ax2.set_title("Tail Convergence (zoom C20+) — Does C44 plateau break?", fontsize=15, fontweight="bold")
ax2.set_yscale("log")
ax2.legend(fontsize=10, loc="upper right")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
out = os.path.join(FIG, "a_sync_96cycle_analysis.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")
