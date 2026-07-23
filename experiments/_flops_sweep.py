"""FLOPs-normalized comparison: A-SYNC CONSTANT vs AdamW full-rank vs LoRA AdamW.

Runs 3 protocols on OPT-125m (12L, ~125M params) with FLOPs tracking.
Evaluates PPL every K steps. Generates FLOPs-vs-PPL plot.

Protocols:
  A-SYNC CONSTANT: ALS delta → gradient bias injection, sync=0.05, no decay, no perturb
  AdamW full-rank: Standard AdamW on all params
  LoRA AdamW: AdamW on LoRA adapters (r=8, α=16)
"""
import json, math, time, os
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer

MODEL = "facebook/opt-125m"
EVAL_MODEL = "gpt2"  # OPT tokenizer is GPT2-based
DTYPE = torch.float32
device = torch.device("cuda:0")
SEED = 42

# ── FLOPs accounting ─────────────────────────────────────────────────
# Per-step FLOPs estimates (multiplier × trainable_params)
# Based on altopt/profiling/flops.py heuristic:
#   ALS:   4 × params  (matrix solve per block)
#   SGD:   6 × params  (fwd + bwd for gradient)
#   AdamW: 10 × params (fwd + bwd + 2 moment buffers)
#   LoRA SGD:  6 × lora_params (fwd+bwd on small adapter)
#   LoRA AdamW: 10 × lora_params
# Plus forward-only eval: 3 × params (fwd pass, no bwd)

def flops_als(n_params): return n_params * 4.0
def flops_sgd(n_params): return n_params * 6.0
def flops_adamw(n_params): return n_params * 10.0
def flops_eval(n_params): return n_params * 3.0

# ── Data loading ─────────────────────────────────────────────────────

torch.manual_seed(SEED)
tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL)
tokenizer.pad_token = tokenizer.eos_token
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
def tok(x): return tokenizer(x["text"], truncation=True, max_length=128, padding="max_length")
train_ds = ds["train"].map(tok, batched=True, remove_columns=["text"])
eval_ds = ds["test"].map(tok, batched=True, remove_columns=["text"])
train_ds.set_format("torch", columns=["input_ids", "attention_mask"])
eval_ds.set_format("torch", columns=["input_ids", "attention_mask"])
def collate(b):
    r = {k: torch.stack([x[k] for x in b]) for k in b[0]}
    r["labels"] = r["input_ids"].clone()
    return r
train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)
eval_dl = DataLoader(eval_ds, batch_size=8, collate_fn=collate)

def evaluate(m):
    m.eval(); total_l, total_n = 0.0, 0
    with torch.no_grad():
        for b in eval_dl:
            b = {k: v.to(device) for k, v in b.items()}
            try: out = m(**b)
            except: return float("inf")
            if out.loss is None or torch.isnan(out.loss) or torch.isinf(out.loss):
                return float("inf")
            total_l += out.loss.item() * b["attention_mask"].sum().item()
            total_n += b["attention_mask"].sum().item()
    m.train()
    avg = total_l / max(total_n, 1)
    return math.exp(avg) if avg < 700 else float("inf")

# ── 1. A-SYNC CONSTANT ───────────────────────────────────────────────

print("="*60)
print("PROTOCOL 1: A-SYNC CONSTANT (gradient-injection)")
print("="*60)
torch.cuda.empty_cache()
m1 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(device)
_lm1 = m1.lm_head
n_params_full = sum(p.numel() for p in m1.parameters() if p.requires_grad)
print(f"Trainable params: {n_params_full:,}")

als = ALSBlockSolver(m1, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
sgd = SGDPhaseOptimizer(m1, lr=1e-4, momentum=0.0, weight_decay=0.01)
it1 = iter(train_dl)

async_ppls, async_flops = [], []
cumulative_flops = 0.0
sync_strength = 0.05
# Note: use many cycles, eval every 400 SGD steps (8 cycles * 50 sgd/cycle)
N_CYCLES = 24
EVAL_EVERY_CYCLES = 4
step_cnt = 0
t0 = time.time()

# Eval at step 0
ppl0 = evaluate(m1); async_ppls.append({"step": 0, "ppl": ppl0, "flops": 0.0})
print(f"  Step {0:>4d}: PPL={ppl0:.2f}, FLOPs=0")

for cyc in range(N_CYCLES):
    # ALS step
    w_before = _lm1.weight.data.cpu().clone()
    try: b = next(it1)
    except StopIteration: it1 = iter(train_dl); b = next(it1)
    b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
    step_cnt += 1
    try: als.solve_block(b_dev, block_size=512)
    except: pass
    cumulative_flops += flops_als(n_params_full)
    delta = _lm1.weight.data.cpu() - w_before
    _lm1.weight.data.copy_(w_before.to(_lm1.weight.data.device))

    # SGD steps with gradient injection
    for j in range(50):
        step_cnt += 1
        try: b2 = next(it1)
        except StopIteration: it1 = iter(train_dl); b2 = next(it1)
        sgd.step(b2)
        cumulative_flops += flops_sgd(n_params_full)
        if _lm1.weight.grad is not None:
            g = sync_strength * delta.to(device=_lm1.weight.grad.device, dtype=_lm1.weight.grad.dtype)
            _lm1.weight.grad.add_(g)

    ppl = evaluate(m1); cumulative_flops += flops_eval(n_params_full)
    async_ppls.append({"step": step_cnt, "ppl": ppl, "flops": cumulative_flops})
    ppl_str = f"{ppl:.1f}" if ppl < 1e10 else "inf"
    delta_ppl = ppl - async_ppls[-2]["ppl"] if len(async_ppls) > 1 else 0
    print(f"  C{cyc+1:2d} step={step_cnt:>4d}: PPL={ppl_str} (Δ={delta_ppl:+.1f}), FLOPs={cumulative_flops/1e12:.4f}T")

elapsed1 = time.time() - t0
result1 = {"protocol": "A-SYNC CONSTANT", "ppls": async_ppls, "elapsed": elapsed1, "cycles": N_CYCLES}
del m1; torch.cuda.empty_cache()
print(f"A-SYNC CONSTANT done: {elapsed1:.0f}s, final PPL={async_ppls[-1]['ppl']:.1f}, FLOPs={cumulative_flops/1e12:.3f}T")

# ── 2. AdamW Full-Rank (Protocol B) ──────────────────────────────────

print("\n" + "="*60)
print("PROTOCOL 2: AdamW Full-Rank")
print("="*60)
torch.cuda.empty_cache()
m2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(device)
opt_adamw = torch.optim.AdamW(m2.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
it2 = iter(train_dl)

# Match total FLOPs budget to A-SYNC: 24 * (ALS_flops + 50*SGD_flops)
target_flops = N_CYCLES * (flops_als(n_params_full) + 50 * flops_sgd(n_params_full))
adamw_ppls, cumulative_flops2 = [], 0.0
step_cnt2 = 0
t0 = time.time()

ppl0 = evaluate(m2); adamw_ppls.append({"step": 0, "ppl": ppl0, "flops": 0.0})
print(f"  Step {0:>4d}: PPL={ppl0:.2f}, FLOPs=0")

# Match: ALS uses 4×params, SGD uses 6×params per step. AdamW uses 10×params.
# So 1 A-SYNC cycle = 4 + 50*6 = 304 units. AdamW per step = 10 units.
# AdamW steps per eval = 304 * EVAL_EVERY_CYCLES / 10 = 121.6 → 122 steps
# Total AdamW steps matching full 24 cycles: 24*304/10 = 729.6 → 730
ad_steps_per_cycle = int((flops_als(n_params_full) + 50 * flops_sgd(n_params_full)) / flops_adamw(n_params_full))
ad_eval_interval = ad_steps_per_cycle  # eval every ~1 cycle equivalent
total_adamw_steps = N_CYCLES * ad_steps_per_cycle

for s in range(total_adamw_steps):
    try: b = next(it2)
    except StopIteration: it2 = iter(train_dl); b = next(it2)
    b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
    opt_adamw.zero_grad()
    out = m2(**b_dev)
    out.loss.backward()
    opt_adamw.step()
    cumulative_flops2 += flops_adamw(n_params_full)
    step_cnt2 += 1

    if (s + 1) % ad_eval_interval == 0 or s == total_adamw_steps - 1:
        ppl = evaluate(m2); cumulative_flops2 += flops_eval(n_params_full)
        adamw_ppls.append({"step": step_cnt2, "ppl": ppl, "flops": cumulative_flops2})
        ppl_str = f"{ppl:.1f}" if ppl < 1e10 else "inf"
        print(f"  Step {step_cnt2:>4d}: PPL={ppl_str}, FLOPs={cumulative_flops2/1e12:.4f}T")

elapsed2 = time.time() - t0
result2 = {"protocol": "AdamW Full-Rank", "ppls": adamw_ppls, "elapsed": elapsed2}
del m2; torch.cuda.empty_cache()
print(f"AdamW done: {elapsed2:.0f}s, final PPL={adamw_ppls[-1]['ppl']:.1f}, FLOPs={cumulative_flops2/1e12:.3f}T")

# ── 3. LoRA AdamW (Protocol C/D style) ───────────────────────────────

print("\n" + "="*60)
print("PROTOCOL 3: LoRA AdamW (r=8, α=16)")
print("="*60)
torch.cuda.empty_cache()

# Use PEFT if available, else manual LoRA
try:
    from peft import LoraConfig, TaskType, get_peft_model
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    )
    m3 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(device)
    m3 = get_peft_model(m3, lora_config)
    n_lora = sum(p.numel() for p in m3.parameters() if p.requires_grad)
    use_peft = True
except ImportError:
    print("  PEFT not available, using manual LoRA")
    use_peft = False
    LORA_R, LORA_ALPHA = 8, 16
    scaling = LORA_ALPHA / LORA_R
    m3 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(device)
    # Freeze all
    for p in m3.parameters(): p.requires_grad = False
    lora_params = []
    for n, mod in m3.named_modules():
        if isinstance(mod, nn.Linear) and any(t in n for t in ["q_proj", "v_proj", "k_proj", "out_proj"]):
            d_in, d_out = mod.in_features, mod.out_features
            A = nn.Parameter(torch.randn(d_in, LORA_R) * 0.02)
            B = nn.Parameter(torch.zeros(LORA_R, d_out))
            setattr(mod, "lora_A", A); setattr(mod, "lora_B", B)
            lora_params.extend([A, B])
            mod.register_parameter("lora_A", A); mod.register_parameter("lora_B", B)
            # Patch forward
            old_fwd = mod.forward
            def make_fwd(old, a, b, sc):
                def fwd(x): return old(x) + (x @ a @ b) * sc
                return fwd
            mod.forward = make_fwd(old_fwd, A, B, scaling)
    n_lora = sum(p.numel() for p in lora_params)

print(f"LoRA trainable params: {n_lora:,}")

opt_lora = torch.optim.AdamW([p for p in m3.parameters() if p.requires_grad], lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
it3 = iter(train_dl)

# Match FLOPs: LoRA AdamW uses 10×lora_params per step (much less FLOPs/step)
# We run the same number of AdamW steps as Protocol B for fair wall-time comparison
# But FLOPs tracking will show the real advantage
lora_ppls, cumulative_flops3 = [], 0.0
step_cnt3 = 0
t0 = time.time()

ppl0 = evaluate(m3); lora_ppls.append({"step": 0, "ppl": ppl0, "flops": 0.0})
print(f"  Step {0:>4d}: PPL={ppl0:.2f}, FLOPs=0")

for s in range(total_adamw_steps):
    try: b = next(it3)
    except StopIteration: it3 = iter(train_dl); b = next(it3)
    b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
    opt_lora.zero_grad()
    out = m3(**b_dev)
    out.loss.backward()
    opt_lora.step()
    cumulative_flops3 += flops_adamw(n_lora)
    step_cnt3 += 1

    if (s + 1) % ad_eval_interval == 0 or s == total_adamw_steps - 1:
        ppl = evaluate(m3); cumulative_flops3 += flops_eval(n_params_full)  # eval uses full model
        lora_ppls.append({"step": step_cnt3, "ppl": ppl, "flops": cumulative_flops3})
        ppl_str = f"{ppl:.1f}" if ppl < 1e10 else "inf"
        print(f"  Step {step_cnt3:>4d}: PPL={ppl_str}, FLOPs={cumulative_flops3/1e12:.6f}T")

elapsed3 = time.time() - t0
result3 = {"protocol": "LoRA AdamW (r=8)", "ppls": lora_ppls, "elapsed": elapsed3, "lora_params": n_lora}
del m3; torch.cuda.empty_cache()
print(f"LoRA done: {elapsed3:.0f}s, final PPL={lora_ppls[-1]['ppl']:.1f}, FLOPs={cumulative_flops3/1e12:.6f}T")

# ── Combine & Save ───────────────────────────────────────────────────

results = {
    "model": MODEL,
    "model_params": n_params_full,
    "lora_params": n_lora,
    "protocols": {
        "a_sync_constant": result1,
        "adamw_full_rank": result2,
        "lora_adamw": result3,
    },
    "flops_model": {
        "als_per_step": f"{n_params_full * 4 / 1e6:.1f}M FLOPs",
        "sgd_per_step": f"{n_params_full * 6 / 1e6:.1f}M FLOPs",
        "adamw_per_step": f"{n_params_full * 10 / 1e6:.1f}M FLOPs",
        "adamw_lora_per_step": f"{n_lora * 10 / 1e6:.1f}M FLOPs",
        "eval_per_call": f"{n_params_full * 3 / 1e6:.1f}M FLOPs",
    },
}

os.makedirs("runs", exist_ok=True)
with open("runs/flops_sweep_opt125m.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# ── Summary ──────────────────────────────────────────────────────────

print("\n" + "="*60)
print("FLOPs-NORMALIZED COMPARISON SUMMARY")
print("="*60)
print(f"Model: {MODEL} ({n_params_full:,} params)")
print(f"{'Protocol':<25} {'Final PPL':>8} {'FLOPs (T)':>12} {'Wall Time':>10}")
print("-"*60)
for key, r in results["protocols"].items():
    p = r["ppls"][-1]
    f = p["flops"] / 1e12 if isinstance(p, dict) else 0
    label = {"a_sync_constant": "A-SYNC CONSTANT 24c",
             "adamw_full_rank": "AdamW Full-Rank",
             "lora_adamw": "LoRA AdamW (r=8)"}[key]
    print(f"{label:<25} {p['ppl']:>8.1f} {f:>12.4f} {r['elapsed']:>8.0f}s")

print(f"\nResults saved to runs/flops_sweep_opt125m.json")

# ── Generate plot ────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    colors = {"a_sync_constant": "#e74c3c", "adamw_full_rank": "#3498db", "lora_adamw": "#2ecc71"}
    labels = {"a_sync_constant": "A-SYNC CONSTANT",
              "adamw_full_rank": "AdamW Full-Rank",
              "lora_adamw": "LoRA AdamW (r=8)"}

    for key, r in results["protocols"].items():
        ppls = r["ppls"]
        x_flops = [p["flops"] / 1e12 for p in ppls]
        y_ppls = [p["ppl"] for p in ppls]
        ax1.plot(x_flops, y_ppls, "o-", color=colors[key], label=labels[key],
                linewidth=2.5, markersize=8, alpha=0.85)

    ax1.set_xlabel("Cumulative FLOPs (TFLOPs)", fontsize=13)
    ax1.set_ylabel("Perplexity", fontsize=13)
    ax1.set_title("PPL vs FLOPs — OPT-125m", fontsize=15)
    ax1.set_yscale("log")
    ax1.legend(fontsize=11, frameon=True)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Wall time
    for key, r in results["protocols"].items():
        ppls = r["ppls"]
        x_time = [i * r["elapsed"] / (len(ppls)-1) if len(ppls)>1 else 0 for i in range(len(ppls))]
        y_ppls = [p["ppl"] for p in ppls]
        ax2.plot(x_time, y_ppls, "o-", color=colors[key], label=labels[key],
                linewidth=2.5, markersize=8, alpha=0.85)

    ax2.set_xlabel("Wall Time (seconds)", fontsize=13)
    ax2.set_ylabel("Perplexity", fontsize=13)
    ax2.set_title("PPL vs Wall Time — OPT-125m", fontsize=15)
    ax2.set_yscale("log")
    ax2.legend(fontsize=11, frameon=True)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs("docs/figures", exist_ok=True)
    fig.savefig("docs/figures/flops_sweep_opt125m.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Plot saved to docs/figures/flops_sweep_opt125m.png")
except ImportError:
    print("matplotlib not available, skipping plot")

print("\nDONE.")
