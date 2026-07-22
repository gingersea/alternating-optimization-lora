"""Protocol A-RAPID: Rapid ALS-SGD interleaving to minimize head-body gap.

Hypothesis: Protocol A diverges because 1 ALS step followed by 50 SGD steps
creates a drift of ~50*lr*amplification ≈ 0.44 between head and body
in 28-layer models. A-RAPID interleaves 10×(ALS → 5 SGD) per cycle,
keeping the head-body gap below 5 SGD steps.

Also tests A-SYNC: ALS delta is injected as a momentum-style direction
in SGD, keeping the head update aligned with body optimization direction.
"""
import json, math, time, copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer
from altopt.perturbation import PerturbationScheduler

MODEL = "Qwen/Qwen2.5-0.5B"
N_CYCLES = 4
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

def collate(b):
    r = {k: torch.stack([x[k] for x in b]) for k in b[0]}
    r["labels"] = r["input_ids"].clone()
    return r

train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate)
eval_dl = DataLoader(eval_ds, batch_size=4, collate_fn=collate)

def evaluate(m):
    m.eval(); total_l, total_n = 0.0, 0
    with torch.no_grad():
        for b in eval_dl:
            b = {k: v.to(device) for k, v in b.items()}
            out = m(**b)
            if out.loss is None or torch.isnan(out.loss) or torch.isinf(out.loss):
                return float("inf")
            total_l += out.loss.item() * b["attention_mask"].sum().item()
            total_n += b["attention_mask"].sum().item()
    m.train()
    return math.exp(total_l / max(total_n, 1)) if total_l / max(total_n, 1) < 700 else float("inf")

def get_cycle_batch(train_iter):
    try: return next(train_iter)
    except StopIteration: return None

# ── Baseline Protocol A ─────────────────────────────────────────────

def run_baseline() -> dict:
    print("\n=== Protocol A BASELINE (1×ALS → 50 SGD → perturb) ===")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    perturb = PerturbationScheduler(m, initial_scale=1e-3)
    sgd = SGDPhaseOptimizer(m, lr=1e-4, momentum=0.9, weight_decay=0.01)
    it = iter(train_dl)
    ppls, step_cnt = [], 0

    for cyc in range(N_CYCLES):
        b = get_cycle_batch(it) or (it := iter(train_dl), next(it))[1]
        b = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
        step_cnt += 1
        try: als.solve_block(b, block_size=1024)
        except: pass

        for _ in range(50):
            step_cnt += 1
            b2 = get_cycle_batch(it) or (it := iter(train_dl), next(it))[1]
            sgd.step(b2)

        step_cnt += 1
        perturb.apply_noise(scale=1e-3)
        ppl = evaluate(m); ppls.append(ppl)
        print(f"  Cycle {cyc+1}: {ppl:.1f}" if ppl < 1e10 else f"  Cycle {cyc+1}: DIVERGED")
        if ppl > 1e10: break

    del m; torch.cuda.empty_cache()
    return {"ppls": ppls}

# ── A-RAPID: Rapid ALS-SGD interleaving ────────────────────────────

def run_rapid(sub_cycles=10, sgd_per_als=5) -> dict:
    print(f"\n=== A-RAPID: {sub_cycles}×(ALS → {sgd_per_als} SGD) per cycle ===")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    perturb = PerturbationScheduler(m, initial_scale=1e-3)
    sgd = SGDPhaseOptimizer(m, lr=1e-4, momentum=0.9, weight_decay=0.01)
    it = iter(train_dl)
    ppls, step_cnt = [], 0

    for cyc in range(N_CYCLES):
        for sc in range(sub_cycles):
            b = get_cycle_batch(it) or (it := iter(train_dl), next(it))[1]
            b = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
            step_cnt += 1
            try: als.solve_block(b, block_size=1024)
            except: pass
            for _ in range(sgd_per_als):
                step_cnt += 1
                b2 = get_cycle_batch(it) or (it := iter(train_dl), next(it))[1]
                sgd.step(b2)

        step_cnt += 1
        perturb.apply_noise(scale=1e-3)
        ppl = evaluate(m); ppls.append(ppl)
        print(f"  Cycle {cyc+1}: {ppl:.1f}" if ppl < 1e10 else f"  Cycle {cyc+1}: DIVERGED")
        if ppl > 1e10: break

    del m; torch.cuda.empty_cache()
    return {"ppls": ppls}

# ── A-SYNC: ALS delta injected as SGD momentum direction ────────────

def run_sync(sync_strength=0.1) -> dict:
    """A-SYNC: After ALS, inject the lm_head delta direction into SGD momentum.

    ALS computes W_new. The delta W_new-W_old is a high-quality direction signal
    from exact least squares. Instead of immediately applying it (creating head-body
    mismatch), we blend it into SGD momentum so the head moves gradually toward
    the ALS target WHILE the body also moves.
    """
    print(f"\n=== A-SYNC: ALS delta → momentum injection (strength={sync_strength}) ===")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    perturb = PerturbationScheduler(m, initial_scale=1e-3)
    sgd = SGDPhaseOptimizer(m, lr=1e-4, momentum=0.9, weight_decay=0.01)
    it = iter(train_dl)

    # Find lm_head
    lm_head = None
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and ("lm_head" in n or "score" in n):
            lm_head = mod; break

    ppls, step_cnt = [], 0

    for cyc in range(N_CYCLES):
        # Snapshot lm_head before ALS
        w_before = lm_head.weight.data.clone() if lm_head is not None else None

        b = get_cycle_batch(it) or (it := iter(train_dl), next(it))[1]
        b = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
        step_cnt += 1
        try: als.solve_block(b, block_size=1024)
        except: pass

        # Compute ALS delta direction on lm_head
        als_delta = None
        if w_before is not None:
            als_delta = lm_head.weight.data - w_before

        # Revert lm_head (SGD will move it gradually)
        if als_delta is not None:
            lm_head.weight.data.copy_(w_before)

        for j in range(50):
            step_cnt += 1
            b2 = get_cycle_batch(it) or (it := iter(train_dl), next(it))[1]
            sgd.step(b2)

            # Inject ALS delta into lm_head gradient as momentum boost
            if als_delta is not None and lm_head.weight.grad is not None:
                # Add sync_strength * als_delta direction to current gradient
                sync_grad = sync_strength * als_delta.to(lm_head.weight.grad.dtype)
                lm_head.weight.grad.add_(sync_grad)

        step_cnt += 1
        perturb.apply_noise(scale=1e-3)
        ppl = evaluate(m); ppls.append(ppl)
        print(f"  Cycle {cyc+1}: {ppl:.1f}" if ppl < 10**12 else f"  Cycle {cyc+1}: DIVERGED")
        if ppl > 1e12: break

    del m; torch.cuda.empty_cache()
    return {"ppls": ppls}


# ── Main ────────────────────────────────────────────────────────────

m0 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
baseline_ppl = evaluate(m0)
del m0; torch.cuda.empty_cache()
print(f"Baseline PPL: {baseline_ppl:.2f}")

results = {}
results["baseline"] = run_baseline()
results["rapid"] = run_rapid(sub_cycles=10, sgd_per_als=5)
results["sync"] = run_sync(sync_strength=0.1)

print(f"\n{'='*60}")
print(f"Qwen0.5B (24L) Protocol Variants — Baseline PPL={baseline_ppl:.2f}")
print(f"{'Protocol':<30} {'PPL trajectory':>45}")
for label, r in results.items():
    p = r["ppls"]
    pstr = " -> ".join(f"{x:.1f}" if x < 1e10 else "inf" for x in p)
    print(f"  {label:<28} {pstr:>45}")

with open("runs/protocol_variants_qwen05b.json", "w") as f:
    json.dump({"baseline_ppl": baseline_ppl, "results": results}, f, indent=2, default=str)
print("Saved runs/protocol_variants_qwen05b.json")
