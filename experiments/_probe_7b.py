"""Protocol A-PROBE on Qwen2.5-7B (28L) — the real test.

Core idea: Insert a low-rank probe (3584->64->3584) before lm_head.
ALS solves only the 64-dim probe output projection (64x64 Cholesky, ~trivial).
Body SGD + probe input projection learn to produce representations through
the bottleneck. ALS never touches lm_head — zero residual amplification.

Compares against prior baselines:
  A-SYNC: PPL 59.7 -> 10.5
  Pure SGD: PPL 60.5 -> 22.5 (plateaus)
  A-PROBE: ???
"""
import json, math, time
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

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

def evaluate(m, lm_head, probe):
    m.eval(); tl, tn = 0.0, 0
    orig = lm_head.forward
    def aug(x): return orig(x) + probe(x)
    lm_head.forward = aug
    with torch.no_grad():
        for b in eval_dl:
            b = {k: v.to(device) for k, v in b.items()}
            try: out = m(**b)
            except: lm_head.forward = orig; return float("inf")
            if out.loss is None or torch.isnan(out.loss) or torch.isinf(out.loss):
                lm_head.forward = orig; return float("inf")
            tl += out.loss.item() * b["attention_mask"].sum().item()
            tn += b["attention_mask"].sum().item()
    lm_head.forward = orig
    m.train()
    return math.exp(tl / max(tn, 1)) if tl / max(tn, 1) < 700 else float("inf")


class Probe(nn.Module):
    def __init__(self, d, v, r=64):
        super().__init__()
        self.inp = nn.Linear(d, r, bias=False)
        self.out = nn.Linear(r, v, bias=False)
        nn.init.normal_(self.inp.weight, std=0.02 / (r**0.5))
        nn.init.zeros_(self.out.weight)
    def forward(self, x): return self.out(self.inp(x))


def solve_probe_out(probe_out, Z, labels, alpha=0.05, lam=1e-3):
    w = probe_out.weight.data; v, r = w.shape
    Zf = Z.detach().float(); N = Zf.shape[0]
    labs = labels.reshape(-1)[:N].to(device=Z.device, dtype=torch.long).clamp(0, v-1)
    bs = 4096; nb = (v + bs - 1) // bs
    reg = lam * torch.eye(r, device=Zf.device, dtype=torch.float32)
    for i in range(nb):
        s, e = i*bs, min((i+1)*bs, v)
        msk = (labs >= s) & (labs < e)
        if not msk.any(): continue
        Zm = Zf[msk]; Y = torch.zeros((msk.sum().item(), e-s), device=Zf.device, dtype=torch.float32)
        Y[torch.arange(msk.sum().item(), device=Zf.device), labs[msk]-s] = 1.0
        ZtY = Zm.T @ Y
        try:
            L = torch.linalg.cholesky(Zm.T @ Zm + reg)
            Wn = torch.cholesky_solve(ZtY, L).T
        except RuntimeError:
            Wn = torch.linalg.lstsq(Zm.T @ Zm + reg, ZtY).solution.T
        Wc = w[s:e, :].detach().float()
        w[s:e, :] = ((1-alpha)*Wc + alpha*Wn).to(device=w.device, dtype=w.dtype)


print(f"\nA-PROBE on Qwen2.5-7B (28L), rank=64, {N_CYCLES} cycles")
torch.cuda.empty_cache()

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="auto")
lm_head = m.lm_head
lm_device = lm_head.weight.device
d_model, v_dim = lm_head.in_features, lm_head.out_features
probe = Probe(d_model, v_dim, 64).to(device=lm_device, dtype=DTYPE)

# All body params + probe (probe on lm_head device, body on whatever device_map chose)
all_params = list(m.parameters()) + list(probe.parameters())
opt = torch.optim.SGD(all_params, lr=2e-4, momentum=0.0, weight_decay=0.01, foreach=False)
it = iter(train_dl)
ppls, step_cnt = [], 0
t0 = time.time()

for cyc in range(N_CYCLES):
    # Cosine schedule
    prog = cyc / max(N_CYCLES-1, 1)
    alpha = 0.05 * 0.5 * (1 + math.cos(math.pi * prog))
    lr = 0.0002 * 0.5 * (1 + math.cos(math.pi * prog))
    for pg in opt.param_groups: pg['lr'] = lr

    # Phase 1: ALS on probe_out
    try: b = next(it)
    except StopIteration: it = iter(train_dl); b = next(it)
    b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
    labels = b_dev["labels"]

    hidden_states = []
    def capture(_mod, _inp, out): hidden_states.append(out[0].detach())
    handle = m.model.norm.register_forward_hook(capture)
    try:
        with torch.no_grad():
            _ = m(input_ids=b_dev["input_ids"], attention_mask=b_dev["attention_mask"])
    finally: handle.remove()

    if hidden_states:
        hs = hidden_states[0]; hs = hs.reshape(-1, d_model) if hs.dim()==3 else hs
        with torch.no_grad(): z = probe.inp(hs.to(device=lm_device, dtype=DTYPE))
        step_cnt += 1
        solve_probe_out(probe.out, z, labels.to(lm_device), alpha=alpha, lam=1e-3)

    # Phase 2: SGD on body + probe_in (probe_out frozen via grad zeroing)
    for _ in range(50):
        step_cnt += 1
        try: b2 = next(it)
        except StopIteration: it = iter(train_dl); b2 = next(it)
        b2_dev = {k: v.to(device) for k, v in b2.items() if isinstance(v, torch.Tensor)}
        opt.zero_grad()
        orig_fwd = lm_head.forward
        lm_head.forward = lambda x: orig_fwd(x) + probe(x)
        out = m(**b2_dev)
        loss = out.loss if hasattr(out, "loss") else out[0]
        lm_head.forward = orig_fwd
        loss.backward()
        probe.out.weight.grad = None  # freeze
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()

    ppl = evaluate(m, lm_head, probe); ppls.append(ppl)
    print(f"  C{cyc+1:2d}: ppl={ppl:.1f}" if ppl < 1e10 else f"  C{cyc+1}: DIVERGED")
    if ppl > 1e10: break

elapsed = time.time() - t0
del m, probe; torch.cuda.empty_cache()

result = {"ppls": ppls, "elapsed": elapsed}
pts = " -> ".join(f"{x:.1f}" for x in ppls)
print(f"\nA-PROBE 7B 16-cycle: {elapsed:.0f}s")
print(f"  PPL: {pts}")
print(f"  Final: {ppls[-1]:.1f}" if ppls else "  No results")

# Compare against prior baselines
print(f"\n--- COMPARISON ---")
print(f"  A-PROBE (rank=64): PPL {ppls[0]:.1f} -> {ppls[-1]:.1f}")
print(f"  A-SYNC (cosine):   PPL 59.7 -> 10.5")
print(f"  Pure SGD:          PPL 60.5 -> 22.5 (plateau)")

with open("runs/probe_7b.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print("Saved runs/probe_7b.json")
