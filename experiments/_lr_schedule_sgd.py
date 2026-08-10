"""Verify the re-interpretation: lr schedule drives the variant ranking.

The no-op diagnosis showed the A-SYNC "CONSTANT vs Cosine" difference must
be a LEARNING-RATE SCHEDULE effect, not an injection-strength effect. This
experiment tests that directly with PURE SGD (no ALS machinery at all):

  1. SGD lr=2e-4 FIXED        (mirrors "CONSTANT")
  2. SGD lr cosine-decay→0    (mirrors "Cosine 32c")
  3. SGD lr exp-decay ×0.8    (mirrors "Vanilla" schedule)

On OPT-125m, 16 cycles × 50 steps. If fixed-lr beats the decays in exactly
the way CONSTANT beat Cosine/Vanilla on 7B, the re-interpretation holds.
"""
import json, math, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

MODEL = "facebook/opt-125m"
N_CYCLES = 16
SGD_STEPS = 50
DTYPE = torch.float32
device = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
def tok(x): return tokenizer(x["text"], truncation=True, max_length=128, padding="max_length")
train_ds = ds["train"].map(tok, batched=True, remove_columns=["text"])
eval_ds = ds["test"].map(tok, batched=True, remove_columns=["text"])
train_ds.set_format("torch", columns=["input_ids", "attention_mask"])
eval_ds.set_format("torch", columns=["input_ids", "attention_mask"])
def c(b):
    r = {k: torch.stack([x[k] for x in b]) for k in b[0]}
    r["labels"] = r["input_ids"].clone(); return r
train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=c)
eval_dl = DataLoader(eval_ds, batch_size=8, collate_fn=c)

def evaluate(m):
    m.eval(); tl, tn = 0.0, 0
    with torch.no_grad():
        for b in eval_dl:
            b = {k: v.to(device) for k, v in b.items()}
            out = m(**b)
            if out.loss is None or math.isnan(out.loss.item()): return float("inf")
            tl += out.loss.item() * b["attention_mask"].sum().item()
            tn += b["attention_mask"].sum().item()
    m.train(); avg = tl/max(tn,1)
    return math.exp(avg) if avg < 700 else float("inf")

def run(label, mode):
    print(f"\n=== {label} (mode={mode}) ===", flush=True)
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(device)
    base_lr = 2e-4
    sgd_opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad],
                              lr=base_lr, momentum=0.0, weight_decay=0.01)
    it = iter(train_dl)
    ppls, step_cnt = [], 0

    for cyc in range(N_CYCLES):
        # Set lr per cycle
        progress = cyc / max(N_CYCLES - 1, 1)
        if mode == "fixed":
            lr = base_lr
        elif mode == "cosine":
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        elif mode == "exp":
            lr = base_lr * (0.8 ** cyc)
        for g in sgd_opt.param_groups:
            g["lr"] = lr

        for j in range(SGD_STEPS):
            step_cnt += 1
            try: b2 = next(it)
            except StopIteration: it = iter(train_dl); b2 = next(it)
            b2_dev = {k: v.to(device) for k, v in b2.items() if isinstance(v, torch.Tensor)}
            sgd_opt.zero_grad()
            out = m(**b2_dev); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            sgd_opt.step()
        ppl = evaluate(m); ppls.append(ppl)
        print(f"  C{cyc+1:2d}: PPL={ppl:.1f}, lr={lr:.2e}", flush=True)

    del m; torch.cuda.empty_cache()
    return ppls

results = {}
for mode, label in [("fixed", "SGD lr FIXED (2e-4)"),
                    ("cosine", "SGD lr cosine-decay"),
                    ("exp", "SGD lr exp-decay x0.8")]:
    results[label] = run(label, mode)

print("\n" + "="*60)
print("LR-SCHEDULE VERIFICATION — pure SGD, OPT-125m")
print("="*60)
for label, ppls in results.items():
    print(f"{label}: final={ppls[-1]:.1f}, best={min(ppls):.1f}")

with open("runs/lr_schedule_sgd_opt125m.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved runs/lr_schedule_sgd_opt125m.json")
