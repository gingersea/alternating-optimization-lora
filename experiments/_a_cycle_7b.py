"""A-CYCLE: Warm restart A-SYNC on Qwen2.5-7B (28L).

A-SYNC 32-cycle plateaued at C16 because cosine decay killed sync+lr.
A-CYCLE resets every 8 cycles: 3 blocks of 8-cycle cosine, each starting
fresh. 24 total cycles, but sync+lr never drop to zero.

Hypothesis: beats A-SYNC 16-cycle (PPL 10.5) because the restarts
re-energize the ALS delta injection each block.
"""
import json, math, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer

MODEL = "Qwen/Qwen2.5-7B"
N_BLOCKS = 3
N_PER_BLOCK = 8
DTYPE = torch.bfloat16
device = torch.device("cuda:0")
print(f"Device: {device}")

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

print(f"\nA-CYCLE: {N_BLOCKS} blocks × {N_PER_BLOCK} cycles on Qwen2.5-7B (28L)")
print("Warm restart: sync+lr reset to initial at each block boundary")
torch.cuda.empty_cache()

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="auto")
_lm = m.lm_head
als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
sgd = SGDPhaseOptimizer(m, lr=2e-4, momentum=0.0, weight_decay=0.01)
it = iter(train_dl)
ppls, step_cnt = [], 0
t0 = time.time()

for block in range(N_BLOCKS):
    for cyc in range(N_PER_BLOCK):
        progress = cyc / max(N_PER_BLOCK - 1, 1)
        sync_strength = 0.05 * 0.5 * (1 + math.cos(math.pi * progress))
        lr = 2e-4 * 0.5 * (1 + math.cos(math.pi * progress))
        sgd.set_lr(lr)

        w_before = _lm.weight.data.cpu().clone()
        try: b = next(it)
        except StopIteration: it = iter(train_dl); b = next(it)
        b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
        step_cnt += 1
        try: als.solve_block(b_dev, block_size=512)
        except Exception as e:
            if "OOM" not in str(e): print(f"  ALS fail: {e}")

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
        total_cyc = block * N_PER_BLOCK + cyc + 1
        ppl_str = f"{ppl:.1f}" if ppl < 1e10 else "inf"
        restart = " [RESTART]" if cyc == 0 and block > 0 else ""
        print(f"  B{block+1}C{cyc+1:<2} (total {total_cyc:2d}): ppl={ppl_str}, sync={sync_strength:.3f}, lr={lr:.6f}{restart}")
        if ppl > 1e10: break

elapsed = time.time() - t0
del m; torch.cuda.empty_cache()

result = {
    "ppls": ppls, "elapsed": elapsed,
    "n_blocks": N_BLOCKS, "n_per_block": N_PER_BLOCK,
    "final_ppl": ppls[-1], "best_ppl": min(ppls),
}
print(f"\nA-CYCLE: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
print(f"  PPL: {' -> '.join(f'{x:.1f}' for x in ppls)}")
print(f"  Best: {min(ppls):.1f}, Final: {ppls[-1]:.1f}")

# Compare
print(f"\n--- COMPARISON ---")
print(f"  A-CYCLE 3x8 restart:  PPL {ppls[0]:.1f} -> {min(ppls):.1f}")
print(f"  A-SYNC 16-cosine:     PPL 59.7 -> 10.5")
print(f"  A-SYNC 32-cosine:     PPL 59.9 -> 13.2 (plateau C20)")

with open("runs/a_cycle_7b.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print("Saved runs/a_cycle_7b.json")
