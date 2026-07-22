"""A-SYNC on Qwen2.5-7B (28L) — the critical depth boundary test.

A-SYNC: ALS computes optimal lm_head delta direction, injects it as
momentum-style gradient bias in SGD. Head and body co-evolve.

If this works on 28L, it's the first Protocol A variant to cross the depth boundary.
"""
import json, math, time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer
from altopt.perturbation import PerturbationScheduler

MODEL = "Qwen/Qwen2.5-7B"
N_CYCLES = 4
device = torch.device("cuda:0")
print(f"Device: {device}")

print("Loading model & tokenizer...")
DTYPE = torch.bfloat16  # 7B in float32 = 28GB → too big
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(device)

# Find lm_head
lm_head = None
lm_head_name = None
for n, mod in model.named_modules():
    if isinstance(mod, nn.Linear) and ("lm_head" in n or "score" in n):
        lm_head = mod; lm_head_name = n; break
print(f"lm_head: {lm_head_name}, shape={list(lm_head.weight.shape)}")

print("Loading data...")
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
eval_dl = DataLoader(eval_ds, batch_size=2, collate_fn=collate)

def evaluate(m):
    m.eval(); total_l, total_n = 0.0, 0
    with torch.no_grad():
        for b in eval_dl:
            b = {k: v.to(device) for k, v in b.items()}
            try:
                out = m(**b)
            except Exception as e:
                return float("inf")
            if out.loss is None or torch.isnan(out.loss) or torch.isinf(out.loss):
                return float("inf")
            total_l += out.loss.item() * b["attention_mask"].sum().item()
            total_n += b["attention_mask"].sum().item()
    m.train()
    avg = total_l / max(total_n, 1)
    return math.exp(avg) if avg < 700 else float("inf")

baseline = evaluate(model)
print(f"Baseline PPL: {baseline:.2f}")

def run_a_sync(sync_strength=0.05, sync_decay=0.8, cycle_sgd_steps=50, name="") -> dict:
    print(f"\n{'='*50}\nA-SYNC {name} (strength={sync_strength}, decay={sync_decay})\n{'='*50}")
    torch.cuda.empty_cache()

    # Use both GPUs to fit 7B: model across GPU 0+1, optimizer on GPU 0
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="auto")

    _lm = None
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and ("lm_head" in n or "score" in n):
            _lm = mod; break

    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    perturb = PerturbationScheduler(m, initial_scale=1e-3)
    sgd = SGDPhaseOptimizer(m, lr=2e-4, momentum=0.0, weight_decay=0.01)
    it = iter(train_dl)

    ppls = []
    step_cnt = 0
    t0 = time.time()
    current_strength = sync_strength

    for cyc in range(N_CYCLES):
        # Snapshot on CPU to save GPU memory
        w_before = _lm.weight.data.cpu().clone()

        b = None
        try: b = next(it)
        except StopIteration: it = iter(train_dl); b = next(it)
        b = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}

        step_cnt += 1
        try:
            als.solve_block(b, block_size=512)  # smaller block = less float32 overhead
        except Exception as e:
            print(f"  ALS failed: {e}")

        # Compute delta, then revert — both on device to avoid CPU round-trip
        als_delta_cpu = (_lm.weight.data.cpu() - w_before).cpu()
        _lm.weight.data.copy_(w_before.to(_lm.weight.data.device))  # Revert

        for j in range(cycle_sgd_steps):
            step_cnt += 1
            try: b2 = next(it)
            except StopIteration: it = iter(train_dl); b2 = next(it)
            sgd.step(b2)

            # Inject sync gradient (CPU delta, slice onto device grad)
            if als_delta_cpu is not None and _lm.weight.grad is not None:
                sync = current_strength * als_delta_cpu.to(
                    device=_lm.weight.grad.device, dtype=_lm.weight.grad.dtype,
                )
                _lm.weight.grad.add_(sync)

        step_cnt += 1
        perturb.apply_noise(scale=1e-3)

        ppl = evaluate(m)
        ppls.append(ppl)
        ppl_str = f"{ppl:.1f}" if ppl < 1e10 else "inf"
        print(f"  Cycle {cyc+1}: step={step_cnt}, ppl={ppl_str}, sync={current_strength:.4f}")

        current_strength *= sync_decay

        if ppl > 1e10:
            print("  DIVERGED — stopping")
            break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}


# Run configs
results = {}

# Config A: moderate sync, fast decay
results["sync_0.05_d0.8"] = run_a_sync(
    sync_strength=0.05, sync_decay=0.8, name="moderate sync, decay"
)

# Config B: weak sync, slow decay
results["sync_0.02_d0.95"] = run_a_sync(
    sync_strength=0.02, sync_decay=0.95, name="weak sync, slow decay"
)

# Config C: very weak sync, no decay
results["sync_0.01_d1.0"] = run_a_sync(
    sync_strength=0.01, sync_decay=1.0, name="minimal sync, constant"
)

print(f"\n{'='*60}")
print(f"Qwen2.5-7B (28L) A-SYNC RESULTS — Baseline PPL={baseline:.2f}")
for label, r in results.items():
    p = r["ppls"]
    pstr = " -> ".join(f"{x:.1f}" if x < 1e10 else "inf" for x in p)
    print(f"  {label:<25} {pstr}")

with open("runs/a_sync_qwen7b.json", "w") as f:
    json.dump({"baseline_ppl": baseline, "results": results}, f, indent=2, default=str)
print("Saved runs/a_sync_qwen7b.json")
