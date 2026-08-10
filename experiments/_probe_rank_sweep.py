"""Probe-rank sweep on OPT-125m — does low-rank ALS probe add quality value?

A-PROBE audit confirmed it is a GENUINE ALS mechanism (directly modifies
probe_out.weight; not a no-op like A-SYNC's injection). The open question:
at what rank (if any) does the low-rank ALS probe BEAT pure SGD?

Conditions (all fixed lr=2e-4, 16 cycles x 50 SGD steps, OPT-125m):
  - pure_sgd: no probe (baseline)
  - probe_r64 / probe_r256 / probe_r1024: A-PROBE at varying bottleneck rank

Key A-PROBE mechanics (from _probe_7b.py audit):
  1. ALS solves probe_out.weight in closed form (direct modification — REAL)
  2. lm_head.forward patched: orig_fwd(x) + probe(x)
  3. SGD updates body + probe.inp; probe_out.weight.grad = None (frozen)

If any rank beats pure SGD on OPT-125m → escalate to Qwen7B stability+quality.
If no rank beats pure SGD → ALS probe adds no post-training value; the
"low-rank bottleneck" path is closed too.
"""
import json, math, time
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

MODEL = "facebook/opt-125m"
N_CYCLES = 16
SGD_STEPS = 50
DTYPE = torch.float32
device = torch.device("cuda:0")
LR = 2e-4

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

class Probe(nn.Module):
    def __init__(self, d, v, r):
        super().__init__()
        self.inp = nn.Linear(d, r, bias=False)
        self.out = nn.Linear(r, v, bias=False)
        nn.init.normal_(self.inp.weight, std=0.02/(r**0.5))
        nn.init.zeros_(self.out.weight)
    def forward(self, x): return self.out(self.inp(x))

def solve_probe_out(probe_out, Z, labels, alpha=0.05, lam=1e-3):
    """ALS closed-form solve of probe_out.weight (direct modification)."""
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

def run(label, mode, rank=None):
    print(f"\n=== {label} ===", flush=True)
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(device)
    lm_head = m.lm_head
    d_model, v_dim = lm_head.in_features, lm_head.out_features
    probe = None
    if mode == "probe":
        probe = Probe(d_model, v_dim, rank).to(device)

    all_params = list(m.parameters()) + (list(probe.parameters()) if probe else [])
    opt = torch.optim.SGD(all_params, lr=LR, momentum=0.0, weight_decay=0.01, foreach=False)
    it = iter(train_dl)
    ppls = []

    for cyc in range(N_CYCLES):
        if mode == "probe":
            # Phase 1: ALS on probe_out (direct modification — genuine ALS)
            try: b = next(it)
            except StopIteration: it = iter(train_dl); b = next(it)
            b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
            labels = b_dev["labels"]
            hidden_states = []
            def capture(_mod, _inp, out): hidden_states.append(out[0].detach())
            handle = m.model.decoder.final_layer_norm.register_forward_hook(capture)
            try:
                with torch.no_grad():
                    _ = m(input_ids=b_dev["input_ids"], attention_mask=b_dev["attention_mask"])
            finally: handle.remove()
            if hidden_states:
                hs = hidden_states[0].reshape(-1, d_model)
                with torch.no_grad(): z = probe.inp(hs.to(device))
                solve_probe_out(probe.out, z, labels.to(device), alpha=0.05, lam=1e-3)

        # Phase 2: SGD on body + (probe.inp if probe); probe_out frozen
        for _ in range(SGD_STEPS):
            try: b2 = next(it)
            except StopIteration: it = iter(train_dl); b2 = next(it)
            b2_dev = {k: v.to(device) for k, v in b2.items() if isinstance(v, torch.Tensor)}
            opt.zero_grad()
            if mode == "probe":
                orig_fwd = lm_head.forward
                lm_head.forward = lambda x: orig_fwd(x) + probe(x)
                out = m(**b2_dev)
                loss = out.loss if hasattr(out, "loss") else out[0]
                lm_head.forward = orig_fwd
                loss.backward()
                probe.out.weight.grad = None  # freeze probe_out (ALS owns it)
            else:
                out = m(**b2_dev)
                loss = out.loss if hasattr(out, "loss") else out[0]
                loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()

        ppl = evaluate(m); ppls.append(ppl)
        print(f"  C{cyc+1:2d}: PPL={ppl:.1f}", flush=True)

    del m; torch.cuda.empty_cache()
    return ppls

results = {}
for label, mode, rank in [("pure_sgd", "pure", None),
                          ("probe_r64", "probe", 64),
                          ("probe_r256", "probe", 256),
                          ("probe_r1024", "probe", 1024)]:
    results[label] = run(label, mode, rank)

print("\n" + "="*60)
print("PROBE-RANK SWEEP — OPT-125m (16c, fixed lr=2e-4)")
print("="*60)
for label, ppls in results.items():
    print(f"{label}: final={ppls[-1]:.1f}, best={min(ppls):.1f}")
with open("runs/probe_rank_sweep_opt125m.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved runs/probe_rank_sweep_opt125m.json")
