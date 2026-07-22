"""Protocol A-CASCADE: ALS lm_head + SGD body-only (lm_head frozen).

Core hypothesis: Protocol A diverges because ALS changes lm_head while body
stays frozen, creating 8.7x residual amplification in deep models. CASCADE
inverts this: after ALS on lm_head, FREEZE lm_head and run SGD on body only.
SGD adapts body representations to match the ALS-optimized head — the body
wants the head's new mapping, no amplification needed.

Design:
  Cycle = ALS(lm_head) -> freeze lm_head -> SGD(body) x N steps
         -> perturb(whole model) -> unfreeze lm_head -> repeat

Tests: Qwen0.5B (24L). If converges, Qwen2.5-7B (28L).
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

MODEL = "Qwen/Qwen2.5-0.5B"
N_CYCLES = 4
SGD_STEPS = 50
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

def collate(batch):
    r = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    r["labels"] = r["input_ids"].clone()
    return r

train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate)
eval_dl = DataLoader(eval_ds, batch_size=4, collate_fn=collate)

def evaluate(m):
    m.eval()
    total_l, total_n = 0.0, 0
    with torch.no_grad():
        for b in eval_dl:
            b = {k: v.to(device) for k, v in b.items()}
            out = m(**b)
            if out.loss is None or torch.isnan(out.loss) or torch.isinf(out.loss):
                return float("inf")
            mask = b.get("attention_mask", torch.ones_like(b["input_ids"]))
            n = mask.sum().item()
            total_l += out.loss.item() * n
            total_n += n
    m.train()
    avg_loss = total_l / max(total_n, 1)
    return math.exp(avg_loss) if avg_loss < 700 else float("inf")


def find_lm_head(m):
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and ("lm_head" in name or "score" in name):
            return name, mod
    return None, None


def freeze_lm_head(m):
    """Freeze only the lm_head parameters (not embedding if tied)."""
    name, mod = find_lm_head(m)
    if mod is None:
        return
    for p in mod.parameters():
        p.requires_grad = False


def unfreeze_lm_head(m):
    name, mod = find_lm_head(m)
    if mod is None:
        return
    for p in mod.parameters():
        p.requires_grad = True


def run_a_cascade(k: int, name: str) -> dict:
    """Protocol A-CASCADE: ALS on lm_head, then SGD on body only.

    Args:
        k: multi_layer_depth for ALS (1 = lm_head only, standard)
        name: label for logging
    """
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    torch.cuda.empty_cache()

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)

    als = ALSBlockSolver(
        m, reg_lambda=1e-3, step_size=0.01,
        multi_layer_depth=k,
        clip_catastrophic=10.0,
    )
    perturb = PerturbationScheduler(m, initial_scale=1e-3)
    # SGD optimizes ALL parameters — but we freeze lm_head before SGD phase
    sgd = SGDPhaseOptimizer(m, lr=1e-4, momentum=0.9, weight_decay=0.01)
    train_iter = iter(train_dl)

    ppls = []
    step = 0
    t0 = time.time()

    for cycle in range(N_CYCLES):
        try: b = next(train_iter)
        except StopIteration: train_iter = iter(train_dl); b = next(train_iter)
        b_device = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}

        # Phase 1: ALS on lm_head
        step += 1
        try:
            als_loss = als.solve_block(b_device, block_size=1024)
        except Exception as e:
            print(f"  ALS failed step {step}: {e}")
            als_loss = 0.0

        # Phase 2: Freeze lm_head, SGD on body only
        freeze_lm_head(m)

        for j in range(SGD_STEPS):
            step += 1
            try: b2 = next(train_iter)
            except StopIteration: train_iter = iter(train_dl); b2 = next(train_iter)
            sgd_loss = sgd.step(b2)

        # Unfreeze lm_head for next cycle
        unfreeze_lm_head(m)

        # Phase 3: Perturb whole model
        step += 1
        perturb.apply_noise(scale=1e-3)

        ppl = evaluate(m)
        ppls.append(ppl)
        ppl_str = f"{ppl:.2f}" if ppl < 1e10 else "inf"
        print(f"  Cycle {cycle+1}: step={step}, ppl={ppl_str}, als_loss={als_loss:.4f}, sgd_loss={sgd_loss:.4f}")

        if ppl > 1e10:
            print("  DIVERGED — stopping")
            break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}


def run_a_baseline(name: str) -> dict:
    """Standard Protocol A: ALS -> SGD (whole model) -> Perturb."""
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    torch.cuda.empty_cache()

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    als = ALSBlockSolver(m, reg_lambda=1e-3, step_size=0.01, clip_catastrophic=10.0)
    perturb = PerturbationScheduler(m, initial_scale=1e-3)
    sgd = SGDPhaseOptimizer(m, lr=1e-4, momentum=0.9, weight_decay=0.01)
    train_iter = iter(train_dl)

    ppls = []
    step = 0
    t0 = time.time()

    for cycle in range(N_CYCLES):
        try: b = next(train_iter)
        except StopIteration: train_iter = iter(train_dl); b = next(train_iter)
        b_device = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}

        step += 1
        try:
            als_loss = als.solve_block(b_device, block_size=1024)
        except Exception as e:
            print(f"  ALS failed: {e}")
            als_loss = 0.0

        for j in range(SGD_STEPS):
            step += 1
            try: b2 = next(train_iter)
            except StopIteration: train_iter = iter(train_dl); b2 = next(train_iter)
            sgd.step(b2)

        step += 1
        perturb.apply_noise(scale=1e-3)

        ppl = evaluate(m)
        ppls.append(ppl)
        ppl_str = f"{ppl:.2f}" if ppl < 1e10 else "inf"
        print(f"  Cycle {cycle+1}: step={step}, ppl={ppl_str}")

        if ppl > 1e10:
            print("  DIVERGED — stopping")
            break

    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}


# ── Main ────────────────────────────────────────────────────────────

m0 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
baseline = evaluate(m0)
del m0; torch.cuda.empty_cache()
print(f"Baseline PPL: {baseline:.2f}")

results = {}

# Protocol A baseline
results["A-baseline"] = run_a_baseline("PROTOCOL A BASELINE (lm_head + full SGD)")

# Protocol A-CASCADE k=1 (lm_head only)
results["A-CASCADE-k1"] = run_a_cascade(k=1, name="A-CASCADE k=1 (lm_head ALS, body SGD)")

print(f"\n{'='*50}")
print(f"Qwen0.5B (24L) RESULTS: Baseline PPL={baseline:.2f}")
for label, r in results.items():
    p = r["ppls"]
    pstr = " -> ".join(f"{x:.2f}" if x < 1e10 else "inf" for x in p)
    print(f"  {label:20s}: {pstr}  ({r['elapsed']:.0f}s)")

with open("runs/cascade_qwen05b.json", "w") as f:
    json.dump({"baseline_ppl": baseline, "results": results}, f, indent=2, default=str)
print("Saved runs/cascade_qwen05b.json")
