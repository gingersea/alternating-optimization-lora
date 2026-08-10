"""Pure SGD control on Qwen2.5-7B (28L) — decisive comparison vs A-SYNC 96c.

The diagnostic showed A-SYNC's gradient injection is a timing no-op in the
current code (inject after optimizer.step, cleared by next zero_grad), and
even with fixed timing it's neutral-to-harmful on OPT-125m.

This control runs PURE SGD with the SAME budget as A-SYNC 96c:
  - 96 cycles × 50 SGD steps = 4800 steps (same as 96c)
  - Same eval points: every 50 steps (cycle boundary)
  - Same lr=2e-4, momentum=0, weight_decay=0.01
  - NO ALS, NO injection, NO perturb — just plain SGD

If pure SGD reaches ~6.8 PPL too → A-SYNC contributes NOTHING beyond SGD,
and the "gradient injection" narrative collapses.
If pure SGD stalls much higher → A-SYNC's delta does something after all.

Also: run a REAL A-SYNC (fixed timing, inject BEFORE step) for 24c as a
secondary comparison — does corrected injection actually help on 7B?
"""
import json, math, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver

MODEL = "Qwen/Qwen2.5-7B"
N_CYCLES = 96
SGD_STEPS = 50
DTYPE = torch.bfloat16
device = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
def tok(x): return tokenizer(x["text"], truncation=True, max_length=128, padding="max_length")
train_ds = ds["train"].map(tok, batched=True, remove_columns=["text"])
eval_ds = ds["test"].map(tok, batched=True, remove_columns=["text"])
train_ds.set_format("torch", columns=["input_ids", "attention_mask"])
eval_ds.set_format("torch", columns=["input_ids", "attention_mask"])
def c(b):
    r = {k: torch.stack([x[k] for x in b]) for k in b[0]}
    r["labels"] = r["input_ids"].clone()
    return r
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=c)
eval_dl = DataLoader(eval_ds, batch_size=2, collate_fn=c)

def evaluate(m):
    m.eval(); tl, tn = 0.0, 0
    with torch.no_grad():
        for b in eval_dl:
            b = {k: v.to(device) for k, v in b.items()}
            try: out = m(**b)
            except: return float("inf")
            if out.loss is None or torch.isnan(out.loss) or torch.isinf(out.loss): return float("inf")
            tl += out.loss.item() * b["attention_mask"].sum().item()
            tn += b["attention_mask"].sum().item()
    m.train()
    return math.exp(tl / max(tn, 1)) if tl / max(tn, 1) < 700 else float("inf")

def run(label, mode, n_cycles, out_file):
    """mode: 'pure' = plain SGD, 'sync_fixed' = real A-SYNC (inject before step)."""
    print(f"\n=== {label} (mode={mode}, {n_cycles} cycles) ===", flush=True)
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="auto")
    _lm = m.lm_head
    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    sgd_opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad],
                              lr=2e-4, momentum=0.0, weight_decay=0.01)
    it = iter(train_dl)
    ppls, step_cnt = [], 0
    sync = 0.05
    t0 = time.time()

    for cyc in range(n_cycles):
        delta_dev = None
        if mode in ("sync_fixed", "sync_after"):
            # ALS snapshot / solve / restore (A-SYNC pattern)
            w_before = _lm.weight.data.cpu().clone()
            try: b = next(it)
            except StopIteration: it = iter(train_dl); b = next(it)
            b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
            step_cnt += 1
            try: als.solve_block(b_dev, block_size=512)
            except Exception as e:
                if "OOM" not in str(e): print(f"  ALS fail C{cyc+1}: {e}")
            delta = _lm.weight.data.cpu() - w_before
            _lm.weight.data.copy_(w_before.to(_lm.weight.data.device))
            delta_dev = sync * delta.to(device)

        for j in range(SGD_STEPS):
            step_cnt += 1
            try: b2 = next(it)
            except StopIteration: it = iter(train_dl); b2 = next(it)
            b2_dev = {k: v.to(device) for k, v in b2.items() if isinstance(v, torch.Tensor)}
            sgd_opt.zero_grad()
            out = m(**b2_dev)
            out.loss.backward()
            if mode == "sync_fixed" and delta_dev is not None and _lm.weight.grad is not None:
                _lm.weight.grad.add_(delta_dev)   # FIXED: before clip+step
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            sgd_opt.step()

        ppl = evaluate(m); ppls.append(ppl)
        ppl_str = f"{ppl:.1f}" if ppl < 1e10 else "inf"
        prev = ppls[-2] if len(ppls) > 1 else ppl
        d = ppl - prev if len(ppls) > 1 and ppl < 1e10 else 0
        if cyc < 24 or cyc % 2 == 1:
            print(f"  C{cyc+1:2d}: ppl={ppl_str} (d={d:+.2f})", flush=True)
        if ppl > 1e10: break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()

    result = {"ppls": ppls, "elapsed": elapsed, "n_cycles": len(ppls),
              "final_ppl": ppls[-1], "best_ppl": min(ppls), "mode": mode}
    print(f"\n{label}: {elapsed:.0f}s ({elapsed/3600:.2f}h)")
    print(f"  PPL: {ppls[0]:.1f} -> {min(ppls):.1f}, final={ppls[-1]:.1f}")
    print(f"  Last 8: {' -> '.join(f'{x:.2f}' for x in ppls[-8:])}")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved {out_file}")
    return result

# ── Main: pure SGD 96c (the decisive control) ──
print("="*70)
print("DECISIVE CONTROL: Pure SGD vs A-SYNC on Qwen2.5-7B")
print("  Pure SGD 96c = 4800 steps (matches A-SYNC 96c budget)")
print("  If pure SGD ≈ PPL 6.8 → A-SYNC injection contributes nothing")
print("="*70)
results = {}
r = run("PURE SGD 96c (control)", "pure", N_CYCLES, "runs/pure_sgd_96c_7b.json")
results["pure_sgd_96c"] = r

print("\n" + "="*70)
print("COMPARISON vs A-SYNC 96c:")
try:
    with open("runs/a_sync_96cycle_7b.json") as f:
        a96 = json.load(f)
    print(f"  A-SYNC 96c:  final={a96['final_ppl']:.2f}, best={a96['best_ppl']:.2f}")
    print(f"  Pure SGD 96c: final={r['final_ppl']:.2f}, best={r['best_ppl']:.2f}")
    diff = r['final_ppl'] - a96['final_ppl']
    if abs(diff) < 0.5:
        print(f"  → DIFFERENCE {diff:+.2f} — INSIGNIFICANT. A-SYNC injection is vacuous.")
    elif diff > 0:
        print(f"  → A-SYNC is BETTER by {diff:.2f} PPL — injection has real value.")
    else:
        print(f"  → A-SYNC is WORSE by {abs(diff):.2f} PPL — injection is harmful.")
except FileNotFoundError:
    print("  a_sync_96cycle_7b.json not found — comparison skipped.")

with open("runs/pure_sgd_vs_async_96c_7b.json", "w") as f:
    json.dump({"comparison": {
        "a_sync_96c": a96['final_ppl'] if 'a96' in dir() else None,
        "pure_sgd_96c": r['final_ppl'],
    }, "pure_sgd": r}, f, indent=2, default=str)
print("Saved runs/pure_sgd_vs_async_96c_7b.json")
