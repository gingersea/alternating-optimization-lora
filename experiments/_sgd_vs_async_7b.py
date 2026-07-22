"""Pure SGD ablation vs A-SYNC on Qwen2.5-7B (28L).

Question: Does A-SYNC (ALS gradient injection) outperform pure SGD with
identical cosine schedule and step count? Or is the PPL improvement
just from longer training?

Design:
  A-SYNC: ALS(lm_head) -> SGD+sync 50 steps (16 cycles, cosine decay)
  SGD:    pure SGD 52 steps (16 cycles, cosine lr decay)
  Match total step count: 16 * 50 = 800 SGD steps for each
"""
import json, math, time
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

def run_pure_sgd():
    """16 cycles of pure SGD with cosine lr schedule. 52 steps per cycle."""
    print(f"\n{'='*50}\nPURE SGD (no ALS, no sync), 16 cycles, cosine lr\n{'='*50}")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="auto")
    it = iter(train_dl)
    lr_0 = 2e-4

    # Manually manage parameters with cosine schedule
    optimizer = torch.optim.SGD(m.parameters(), lr=lr_0, momentum=0.0,
                                weight_decay=0.01, foreach=False)
    ppls, step_cnt = [], 0
    t0 = time.time()

    for cyc in range(N_CYCLES):
        progress = cyc / max(N_CYCLES - 1, 1)
        lr = lr_0 * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        for j in range(52):
            step_cnt += 1
            try: b = next(it)
            except StopIteration: it = iter(train_dl); b = next(it)
            b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
            optimizer.zero_grad()
            out = m(**b_dev)
            loss = out.loss if hasattr(out, "loss") else out[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            optimizer.step()

        ppl = evaluate(m); ppls.append(ppl)
        print(f"  C{cyc+1:2d}: ppl={ppl:.1f}, lr={lr:.6f}" if ppl < 1e10 else f"  C{cyc+1}: DIVERGED")

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed, "label": "pure SGD"}


def run_a_sync_cosine():
    """16 cycles A-SYNC with cosine decay. 1 ALS + 50 sync SGD per cycle."""
    print(f"\n{'='*50}\nA-SYNC COSINE, 16 cycles\n{'='*50}")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="auto")
    _lm = None
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and ("lm_head" in n or "score" in n):
            _lm = mod; break

    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    sgd = SGDPhaseOptimizer(m, lr=2e-4, momentum=0.0, weight_decay=0.01)
    it = iter(train_dl)
    ppls, step_cnt = [], 0
    t0 = time.time()

    for cyc in range(N_CYCLES):
        progress = cyc / max(N_CYCLES - 1, 1)
        sync_strength = 0.05 * 0.5 * (1 + math.cos(math.pi * progress))
        lr = 0.0002 * 0.5 * (1 + math.cos(math.pi * progress))
        sgd.set_lr(lr)

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
                g = sync_strength * delta.to(device=_lm.weight.grad.device, dtype=_lm.weight.grad.dtype)
                _lm.weight.grad.add_(g)

        ppl = evaluate(m); ppls.append(ppl)
        print(f"  C{cyc+1:2d}: ppl={ppl:.1f}, sync={sync_strength:.4f}" if ppl < 1e10 else f"  C{cyc+1}: DIVERGED")

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed, "label": "A-SYNC cosine"}


# ── Main ──
results = {}
results["pure_sgd"] = run_pure_sgd()
results["a_sync"] = run_a_sync_cosine()

print(f"\n{'='*60}")
print("ABLATION: Pure SGD vs A-SYNC on Qwen2.5-7B (28L), 16 cycles")
for label, r in results.items():
    p = r["ppls"]
    pts = " -> ".join(f"{x:.1f}" for x in p)
    print(f"  {r['label']:15s}: {pts}")
    if p: print(f"  {'':15s}  Final={p[-1]:.1f}  Δ={p[0]-p[-1]:.1f}")

with open("runs/sgd_vs_async_7b.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Saved runs/sgd_vs_async_7b.json")
