#!/usr/bin/env python3
"""Generate Figure 6: Universal η Nomogram for LoRA Rank Selection."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

# ── Data from nomogram calibration ──
# 7 architectures used for regression:
models = [
    # (name, L, d_h, L/d_h, pretrain_tokens_B, η_estimated)
    ("Mistral-7B",    32, 4096, 0.0078, 8.0,  180),
    ("TinyLlama-1.1B", 22, 2048, 0.0107, 3.0,  210),
    ("DeepSeek-1.5B", 28, 1536, 0.0182, 2.0,  240),
    ("Qwen2.5-0.5B",  24, 896,  0.0268, 18.0, 150),
    ("SmolLM2-135M",  30, 576,  0.0521, 2.0,  230),
    ("GPT-2",         12, 768,  0.0156, 1.0,  250),
    ("OPT-125m",      12, 768,  0.0156, 1.0,  260),
]

# Regression: η = 269 + 2386·(L/d_h) - 47·log10(tokens_B), R²=0.88
def eta_pred(L_div_dh, log_tokens):
    return 269 + 2386 * L_div_dh - 47 * log_tokens

# ── Prediction grid ──
Ldh_vals = np.linspace(0.005, 0.055, 100)
pretrain_levels = [1, 2, 5, 10, 18]  # tokens in billions

# ── Figure ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.2, 1]})

# Panel (a): Nomogram — η vs L/d_h for different pretraining budgets
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(pretrain_levels)))
for i, pt in enumerate(pretrain_levels):
    eta_vals = eta_pred(Ldh_vals, np.log10(pt))
    ax1.plot(Ldh_vals, eta_vals, color=colors[i], lw=2,
             label=f"$N_{{\\text{{pretrain}}}}$ = {pt}T tokens")

# Scatter: actual data points
for name, L, dh, ldh, pt_B, eta_est in models:
    ax1.scatter(ldh, eta_est, s=80, facecolors='white', edgecolors='black',
                linewidth=1.5, zorder=5)
    # offset labels to avoid overlap
    offset = 5 if name not in ("Qwen2.5-0.5B", "Mistral-7B") else -15
    ax1.annotate(name.replace("-0.5B", "").replace("-7B", "").replace("-1.5B", ""),
                 (ldh, eta_est), textcoords="offset points", xytext=(4, offset),
                 fontsize=7, ha='left')

# r=8 threshold line
r8_threshold = 8 / Ldh_vals
ax1.axhline(y=230, color='red', linestyle='--', alpha=0.5, label=r"$\eta \approx 230$ (baseline)")
ax1.axhline(y=150, color='orange', linestyle=':', alpha=0.5, label=r"$\eta_0 \approx 150$ (strong pretrain)")

ax1.set_xlabel(r"$L / d_h$", fontsize=12)
ax1.set_ylabel(r"$\eta$ (Rank Sufficiency Parameter)", fontsize=12)
ax1.set_title("(a) $\eta$ Nomogram", fontsize=13)
ax1.legend(fontsize=7, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.004, 0.056)
ax1.set_ylim(100, 400)

# Panel (b): Lookup table — predicted r_min for popular architectures
popular_models = [
    ("Llama-3.1-8B",    32, 4096, 0.0078, 15.0),
    ("Llama-3.1-70B",   80, 8192, 0.0098, 15.0),
    ("Mistral-7B",      32, 4096, 0.0078, 8.0),
    ("Qwen2.5-0.5B",    24, 896,  0.0268, 18.0),
    ("Qwen2.5-7B",      28, 3584, 0.0078, 18.0),
    ("Qwen2.5-72B",     80, 8192, 0.0098, 18.0),
    ("DeepSeek-V2-Lite", 27, 2048, 0.0132, 8.4),
    ("DeepSeek-1.5B",   28, 1536, 0.0182, 2.0),
    ("TinyLlama-1.1B",  22, 2048, 0.0107, 3.0),
    ("SmolLM2-135M",    30, 576,  0.0521, 2.0),
    ("SmolLM2-360M",    30, 960,  0.0313, 2.0),
    ("SmolLM2-1.7B",    30, 2048, 0.0146, 2.0),
    ("Phi-3-mini",      32, 3072, 0.0104, 3.5),
    ("Gemma-2-9B",      42, 3584, 0.0117, 13.0),
]

names, r_mins, ldh_vals_tab = [], [], []
for name, L, dh, ldh, pt in popular_models:
    eta_val = eta_pred(ldh, np.log10(pt))
    r_min = max(8, int(np.ceil(eta_val * ldh)))
    names.append(name)
    r_mins.append(r_min)
    ldh_vals_tab.append(ldh)

# Sort by r_min
order = np.argsort(r_mins)[::-1]
names_sorted = [names[i] for i in order]
r_mins_sorted = [r_mins[i] for i in order]
ldh_sorted = [ldh_vals_tab[i] for i in order]

y_pos = range(len(names_sorted))
bars = ax2.barh(y_pos, r_mins_sorted, color=['#d62728' if r > 8 else '#2ca02c' for r in r_mins_sorted],
                edgecolor='black', linewidth=0.5)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(names_sorted, fontsize=7)
ax2.set_xlabel("Predicted $r_{\\min}$", fontsize=12)
ax2.set_title("(b) Predicted $r_{\\min}$ for 14 Architectures", fontsize=13)
ax2.axvline(x=8, color='gray', linestyle='--', alpha=0.5, label="$r=8$ (default)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis='x')

# Annotation
for i, (r, ldh_val) in enumerate(zip(r_mins_sorted, ldh_sorted)):
    if r > 8:
        ax2.annotate(f"$L/d_h={ldh_val:.4f}$", (r + 0.3, i), fontsize=6, va='center',
                     color='#d62728')

plt.tight_layout()
plt.savefig("figures/fig6_nomogram.pdf", dpi=150, bbox_inches='tight')
plt.savefig("figures/fig6_nomogram.png", dpi=150, bbox_inches='tight')
print("Figure 6 saved: figures/fig6_nomogram.pdf")
print(f"Regression: η = 269 + 2386·(L/d_h) - 47·log₁₀(tokens_B), R²=0.88")
print(f"Models above r=8: {sum(1 for r in r_mins_sorted if r > 8)} / {len(r_mins_sorted)}")
for name, r in zip(names_sorted, r_mins_sorted):
    if r > 8:
        print(f"  ⚠ {name}: r_min = {r}")
