"""A-SYNC CONSTANT 96-cycle on Qwen2.5-7B (28L) — extend to find the floor.

48c converged at C44 (PPL 7.6). 96c tests whether the C44 plateau is a
true asymptote or whether extended training continues improving — the key
question for narrowing the 6.1x gap to AdamW (PPL 1.25).

Config: sync=0.05 CONSTANT, lr=2e-4 CONSTANT, no decay, no perturb.
Same as 48c but N_CYCLES=96.
"""
import json, math, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer

MODEL = "Qwen/Qwen2.5-7B"
N_CYCLES = 96
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

print(f"\nA-SYNC CONSTANT 96-CYCLE on Qwen2.5-7B (28L)")
print(f"sync=0.05 (const), lr=2e-4 (const), no perturb, no decay")
torch.cuda.empty_cache()

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="auto")
_lm = m.lm_head
als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
sgd = SGDPhaseOptimizer(m, lr=2e-4, momentum=0.0, weight_decay=0.01)
it = iter(train_dl)
ppls, step_cnt = [], 0
sync = 0.05
t0 = time.time()

for cyc in range(N_CYCLES):
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

    for j in range(50):
        step_cnt += 1
        try: b2 = next(it)
        except StopIteration: it = iter(train_dl); b2 = next(it)
        sgd.step(b2)
        if _lm.weight.grad is not None:
            g = sync * delta.to(device=_lm.weight.grad.device, dtype=_lm.weight.grad.dtype)
            _lm.weight.grad.add_(g)

    ppl = evaluate(m); ppls.append(ppl)
    ppl_str = f"{ppl:.1f}" if ppl < 1e10 else "inf"
    prev = ppls[-2] if len(ppls) > 1 else ppl
    d = ppl - prev if len(ppls) > 1 and ppl < 1e10 else 0
    # Print every cycle for first 48, every 2nd for 48-96
    if cyc < 48 or cyc % 2 == 1:
        print(f"  C{cyc+1:2d}: ppl={ppl_str} (d={d:+.2f})", flush=True)
    if ppl > 1e10: break

elapsed = time.time() - t0
del m; torch.cuda.empty_cache()

result = {"ppls": ppls, "elapsed": elapsed, "n_cycles": N_CYCLES,
          "sync": sync, "decay": "none", "final_ppl": ppls[-1], "best_ppl": min(ppls)}
print(f"\nA-SYNC 96-cycle: {elapsed:.0f}s ({elapsed/3600:.2f}h)")
print(f"  PPL: {ppls[0]:.1f} -> {min(ppls):.1f}")
print(f"  Last 8: {' -> '.join(f'{x:.2f}' for x in ppls[-8:])}")
print(f"  Final delta: {ppls[-1]-ppls[-2]:+.2f}")

with open("runs/a_sync_96cycle_7b.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print("Saved runs/a_sync_96cycle_7b.json")
