"""Controlled divergence experiment on Qwen2.5-7B (28L).

Ablation design — 6 conditions testing which component causes divergence:
  1. SGD-only (control)       — pure standard SGD, no ALS, no perturb
  2. Perturb-only             — only noise injection, no ALS, no SGD
  3. ALS (no-op) + SGD        — ALS forward hook runs but step_size=0, then SGD
  4. ALS-only                 — only ALS solves, no SGD, no perturb
  5. ALS + SGD (no perturb)   — Protocol A minus perturb
  6. Full Protocol A (ASP)    — ALS + SGD + Perturb (original recipe)

Each: 4 cycles (or equivalent steps), same eval schedule, FLOPs-tracked.
Hypothesis: divergence is caused by ALS weight modification, not by SGD or perturb alone.
"""
import json, math, time, os
import torch, torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer
from altopt.perturbation import PerturbationScheduler

MODEL = "Qwen/Qwen2.5-7B"
N_CYCLES = 4  # Short: test divergence quickly
SGD_STEPS_PER_CYCLE = 50
DTYPE = torch.bfloat16
device = torch.device("cuda:0")
print(f"Device: {device}")

# ── Data ────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
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
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate)
eval_dl  = DataLoader(eval_ds, batch_size=2, collate_fn=collate)

# ── Eval ────────────────────────────────────────────────────────────  
def evaluate(m):
    m.eval(); tl, tn = 0.0, 0
    with torch.no_grad():
        for b in eval_dl:
            b = {k: v.to(device) for k, v in b.items()}
            try: out = m(**b)
            except: return float("inf")
            if out.loss is None: return float("inf")
            l = out.loss.item()
            if math.isnan(l) or math.isinf(l): return float("inf")
            tl += l * b["attention_mask"].sum().item()
            tn += b["attention_mask"].sum().item()
    m.train()
    avg = tl / max(tn, 1)
    return math.exp(avg) if avg < 700 else float("inf")

# ── FLOPs accounting ────────────────────────────────────────────────  
n_full = 7_615_000_000
FLOP_ALS     = 4 * n_full
FLOP_SGD     = 6 * n_full
FLOP_ADAMW   = 10 * n_full
FLOP_EVAL    = 3 * n_full
FLOP_PERTURB = n_full  # one add per param

STRIDE = 10  # eval every 10 steps for fine-grained tracking

def run_condition(label, config_desc):
    """Run one condition. Returns dict with step-by-step PPL+FLOPs data."""
    print(f"\n{'='*60}")
    print(f"CONDITION: {label}")
    print(f"{'='*60}")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="auto")
    
    # Find lm_head
    _lm = None
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and ("lm_head" in n or "score" in n):
            _lm = mod; break
    
    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    sgd = SGDPhaseOptimizer(m, lr=1e-4, momentum=0.0, weight_decay=0.01)
    perturb = PerturbationScheduler(m, initial_scale=1e-3)
    it = iter(train_dl)
    
    data = []  # list of (step, ppl, cumulative_flops, label)
    cumulative_flops = 0.0
    step_cnt = 0
    t0 = time.time()
    
    # Baseline eval
    ppl = evaluate(m); cumulative_flops += FLOP_EVAL
    data.append({"step": 0, "ppl": ppl, "flops": cumulative_flops})
    if ppl > 1e10: ppl_str = "DIVERGED"
    else: ppl_str = f"{ppl:.1f}"
    print(f"  step={0:>4d}: PPL={ppl_str}, FLOPs={cumulative_flops/1e12:.4f}T")
    
    try:
        for cyc in range(N_CYCLES):
            # ── ALS phase ──
            if "ALS" in config_desc:
                if "no-op" in config_desc:
                    # Run forward hook but don't modify weights (step_size=0 forces identity)
                    # Actually: need a different approach. Just capture and discard.
                    w_before = _lm.weight.data.cpu().clone()
                    try: b = next(it)
                    except StopIteration: it = iter(train_dl); b = next(it)
                    b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
                    step_cnt += 1
                    try: als.solve_block(b_dev, block_size=512)
                    except: pass
                    # Discard ALS result — restore
                    _lm.weight.data.copy_(w_before.to(_lm.weight.data.device))
                    cumulative_flops += FLOP_ALS * 0.5  # half: fwd only, no solve
                else:
                    # Real ALS: modifies weights
                    try: b = next(it)
                    except StopIteration: it = iter(train_dl); b = next(it)
                    b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
                    step_cnt += 1
                    try: als.solve_block(b_dev, block_size=512)
                    except Exception as e:
                        if "OOM" not in str(e): print(f"  ALS: {e}")
                    cumulative_flops += FLOP_ALS
            
            # ── SGD phase ──
            if "SGD" in config_desc:
                for j in range(SGD_STEPS_PER_CYCLE):
                    step_cnt += 1
                    try: b2 = next(it)
                    except StopIteration: it = iter(train_dl); b2 = next(it)
                    loss = sgd.step(b2)
                    cumulative_flops += FLOP_SGD
                    
                    if step_cnt % STRIDE == 0:
                        ppl = evaluate(m); cumulative_flops += FLOP_EVAL
                        data.append({"step": step_cnt, "ppl": ppl, "flops": cumulative_flops})
                        if ppl > 1e10: ppl_str = "DIVERGED"
                        else: ppl_str = f"{ppl:.1f}"
                        print(f"  step={step_cnt:>4d}: PPL={ppl_str}, FLOPs={cumulative_flops/1e12:.4f}T")
                        if ppl > 1e10: raise StopIteration("diverged")
            
            # ── Perturb phase ──
            if "Perturb" in config_desc:
                step_cnt += 1
                try: perturb.apply_noise(scale=1e-3)
                except: pass
                cumulative_flops += FLOP_PERTURB
                ppl = evaluate(m); cumulative_flops += FLOP_EVAL
                data.append({"step": step_cnt, "ppl": ppl, "flops": cumulative_flops})
                if ppl > 1e10: ppl_str = "DIVERGED"
                else: ppl_str = f"{ppl:.1f}"
                print(f"  step={step_cnt:>4d}: PPL={ppl_str}, FLOPs={cumulative_flops/1e12:.4f}T (after perturb)")
                if ppl > 1e10: break
            
            # ── "ALS-only" or "Perturb-only": debug print ──
            if config_desc in ("ALS", "Perturb"):
                if step_cnt % max(STRIDE, 1) == 0 or cyc == N_CYCLES-1:
                    ppl = evaluate(m); cumulative_flops += FLOP_EVAL
                    data.append({"step": step_cnt, "ppl": ppl, "flops": cumulative_flops})
                    if ppl > 1e10: ppl_str = "DIVERGED"
                    else: ppl_str = f"{ppl:.1f}"
                    print(f"  step={step_cnt:>4d}: PPL={ppl_str}, FLOPs={cumulative_flops/1e12:.4f}T")
                    if ppl > 1e10: break
    
    except StopIteration:
        pass
    
    elapsed = time.time() - t0
    
    # Final eval
    if data[-1]["ppl"] < 1e10:
        ppl = evaluate(m)
        data.append({"step": step_cnt+1, "ppl": ppl, "flops": cumulative_flops + FLOP_EVAL})
    
    del m; torch.cuda.empty_cache()
    
    # Summarize
    final_ppl = data[-1]["ppl"]
    diverged = final_ppl > 1e10
    status = "DIVERGED" if diverged else f"CONVERGED (PPL={final_ppl:.1f})"
    print(f"\n{label}: {elapsed:.0f}s, {len(data)} eval points, {status}")
    
    return {
        "label": label, "config": config_desc,
        "data": data, "elapsed": elapsed,
        "diverged": diverged, "final_ppl": final_ppl,
        "n_evals": len(data),
    }

# ═══════════════════════════════════════════════════════════════════
# RUN ALL CONDITIONS
# ═══════════════════════════════════════════════════════════════════

CONDITIONS = [
    ("1. SGD-only",           "SGD"),                    # Control
    ("2. Perturb-only",       "Perturb"),                # Noise alone
    ("3. ALS(no-op)+SGD",     "ALS+no-op+SGD"),          # Hook overhead
    ("4. ALS-only",           "ALS"),                    # ALS alone
    ("5. ALS+SGD",            "ALS+SGD"),                # No perturb
    ("6. Full ASP (A+P+S)",   "ALS+SGD+Perturb"),        # Original
]

results = {}
for label, config in CONDITIONS:
    results[label] = run_condition(label, config)

# ═══════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════
os.makedirs("runs", exist_ok=True)
with open("runs/diverge_cause_7b.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Summary
print("\n" + "="*70)
print("DIVERGENCE CAUSE EXPERIMENT — Qwen2.5-7B (28L)")
print("="*70)
print(f"{'Condition':<30} {'Final PPL':>10} {'Diverged?':>10} {'Evals':>6} {'Time':>8}")
print("-"*70)
for label in [c[0] for c in CONDITIONS]:
    r = results[label]
    ppl_str = f"{r['final_ppl']:.1f}" if r['final_ppl'] < 1e10 else "DIVERGED"
    div_str = "YES" if r['diverged'] else "NO"
    print(f"{label:<30} {ppl_str:>10} {div_str:>10} {r['n_evals']:>6} {r['elapsed']:>7.0f}s")
print("="*70)
print("Saved runs/diverge_cause_7b.json")
