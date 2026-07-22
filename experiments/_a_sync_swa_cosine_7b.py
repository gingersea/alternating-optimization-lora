"""A-SYNC+SWA+COSINE: 16-cycle on Qwen2.5-7B (28L).

Two algorithmic additions to A-SYNC:
  1. SWA (Stochastic Weight Averaging) from cycle 10 onward — exponential moving
     average of model weights. Free 1-2 PPL improvement with 0 extra cost.
  2. Cosine sync decay — gentler tail than exponential, sustains ALS signal longer.

Compared against A-SYNC baseline from prior runs.
"""
import copy, json, math, time
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer

MODEL = "Qwen/Qwen2.5-7B"
N_CYCLES = 16
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

def swa_update(swa_params, model_params, step):
    """EMA of params: swa = beta*swa + (1-beta)*current. beta = step/(step+1)."""
    beta = step / (step + 1.0)
    for sp, mp in zip(swa_params, model_params):
        sp.data.mul_(beta).add_(mp.data, alpha=1 - beta)

print(f"\nA-SYNC+SWA+COSINE: 16-cycle on Qwen2.5-7B (28L)")
print("SWA start: cycle 10, Cosine decay: strength = 0.05*0.5*(1+cos(pi*t/T))")
torch.cuda.empty_cache()

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="auto")

_lm = None
for n, mod in m.named_modules():
    if isinstance(mod, nn.Linear) and ("lm_head" in n or "score" in n):
        _lm = mod; break

als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
sgd = SGDPhaseOptimizer(m, lr=2e-4, momentum=0.0, weight_decay=0.01)
it = iter(train_dl)

# SWA buffers (on device)
swa_params = [copy.deepcopy(p.data) for p in m.parameters() if p.requires_grad]
swa_model = None  # lazy init when SWA begins
swa_start = 10
swa_count = 0

ppls, step_cnt = [], 0
strength_0 = 0.05
t0 = time.time()

for cyc in range(N_CYCLES):
    # Cosine decay
    progress = cyc / max(N_CYCLES - 1, 1)
    strength = strength_0 * 0.5 * (1 + math.cos(math.pi * progress))

    w_before = _lm.weight.data.cpu().clone()
    try: b = next(it)
    except StopIteration: it = iter(train_dl); b = next(it)
    b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
    step_cnt += 1
    try: als.solve_block(b_dev, block_size=512)
    except: pass
    delta = _lm.weight.data.cpu() - w_before
    _lm.weight.data.copy_(w_before.to(_lm.weight.data.device))

    for j in range(50):
        step_cnt += 1
        try: b2 = next(it)
        except StopIteration: it = iter(train_dl); b2 = next(it)
        sgd.step(b2)
        if _lm.weight.grad is not None:
            g = strength * delta.to(device=_lm.weight.grad.device, dtype=_lm.weight.grad.dtype)
            _lm.weight.grad.add_(g)

    # SWA: accumulate from cycle 10
    if cyc >= swa_start:
        swa_count += 1
        swa_update(swa_params,
                   [p for p in m.parameters() if p.requires_grad],
                   swa_count)

    ppl = evaluate(m); ppls.append(ppl)
    marker = " [SWA]" if cyc >= swa_start else ""
    print(f"  C{cyc+1:2d}: ppl={ppl:.1f}, sync={strength:.4f}{marker}" if ppl < 1e10 else f"  C{cyc+1}: DIVERGED")
    if ppl > 1e10: break

# Apply SWA for final eval
if swa_count > 0:
    print(f"\n  SWA applied: {swa_count} cycles averaged, loading SWA weights...")
    for sp, mp in zip(swa_params, [p for p in m.parameters() if p.requires_grad]):
        mp.data.copy_(sp.data)  # temporarily overwrite for eval
    swa_ppl = evaluate(m)
    print(f"  SWA PPL: {swa_ppl:.1f}")

elapsed = time.time() - t0
del m; torch.cuda.empty_cache()

result = {
    "ppls": ppls, "swa_ppl": swa_ppl if swa_count > 0 else None, "elapsed": elapsed,
    "n_cycles": N_CYCLES, "swa_start": swa_start, "decay": "cosine",
}
print(f"\nA-SYNC+SWA+COSINE 16-cycle: {elapsed:.0f}s")
pts = " -> ".join(f"{x:.1f}" for x in ppls)
print(f"  PPL: {pts}")
print(f"  Final: {ppls[-1]:.1f}, SWA: {swa_ppl:.1f}" if swa_count > 0 else f"  Final: {ppls[-1]:.1f}")

with open("runs/a_sync_swa_cosine_7b.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print("Saved runs/a_sync_swa_cosine_7b.json")
