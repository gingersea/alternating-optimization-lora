"""A-KD: teacher-soft-target ALS — the one ALS objective that is NOT redundant.

Prior findings:
  1. A-SYNC gradient injection = timing no-op (proven).
  2. A-PROBE low-rank ALS on hard labels = no quality gain at ANY rank
     (50.5-50.8 vs pure SGD 50.6 on valid task).
     Reason: closed-form solve of one-hot CE ≈ gradient of CE — same
     direction as SGD, so ALS is redundant with the body's SGD.

The ONE genuinely non-redundant ALS objective is DISTILLATION:
match a teacher's soft logit distribution. A closed-form ALS solve of the
probe output head against teacher logits (soft targets) does something SGD
on the body cannot trivially replicate — it can move the probe head to the
exact best linear readout of the current hidden states in one shot, while
SGD must integrate the same correction across many steps.

Design (OPT-125m, valid task = OPT tokenizer + pad-masked labels):
  - Teacher: pure SGD 16c (the 50.6 model from the sweep baseline)
  - kd_sgd: student probe, SGD on soft-KL loss (teacher logits)
  - kd_als: student probe, closed-form ALS solve on soft targets each cycle
  - eval: standard PPL on wikitext-2 test (both students)

If kd_als ≥ kd_sgd at equal cycles: closed-form soft-target ALS has
incremental value → the first genuinely non-redundant ALS mechanism.
"""
import json, math, time
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

MODEL = "facebook/opt-125m"
N_CYCLES = 12
SGD_STEPS = 50
DTYPE = torch.float32
device = torch.device("cuda:0")
LR = 2e-4
KD_TEMP = 2.0
LAM = 1e-3

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
if tokenizer.pad_token is None:
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
    r["labels"][r["attention_mask"] == 0] = -100
    return r
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

def soft_target_solve(probe_out, Z, soft_labels, alpha=1.0, lam=LAM):
    """Closed-form ALS solve of probe_out.weight against SOFT targets.

    soft_labels: [N, V] float distribution (teacher softmax, temp-scaled).
    Solve ridge-regularized least squares: W = argmin ||ZW^T - Y||^2 + lam||W||^2
    per vocab block. Uses only valid (non-pad) positions.
    """
    w = probe_out.weight.data; v, r = w.shape
    Zf = Z.detach().float()
    N = Zf.shape[0]
    Y = soft_labels.float().to(Zf.device)
    assert Y.shape[0] == N, f"soft labels {Y.shape} vs Z {Zf.shape}"
    bs = 4096; nb = (v + bs - 1) // bs
    reg = lam * torch.eye(r, device=Zf.device, dtype=torch.float32)
    for i in range(nb):
        s, e = i*bs, min((i+1)*bs, v)
        Yb = Y[:, s:e]
        nrm = Yb.norm(dim=1)
        msk = nrm > 1e-8
        if not msk.any(): continue
        Zm = Zf[msk]; Ym = Yb[msk]
        ZtY = Zm.T @ Ym
        try:
            L = torch.linalg.cholesky(Zm.T @ Zm + reg)
            Wn = torch.cholesky_solve(ZtY, L).T
        except RuntimeError:
            Wn = torch.linalg.lstsq(Zm.T @ Zm + reg, ZtY).solution.T
        Wc = w[s:e, :].detach().float()
        w[s:e, :] = ((1-alpha)*Wc + alpha*Wn).to(device=w.device, dtype=w.dtype)

@torch.no_grad()
def teacher_logits(m, b):
    """Teacher logits on a batch (before softmax, temp-scaled)."""
    out = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    return out.logits.float()

def run(label, student_mode, teacher):
    """Train a probe student. student_mode: 'sgd' or 'als'."""
    print(f"\n=== {label} (student_mode={student_mode}) ===", flush=True)
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(device)
    lm_head = m.lm_head
    d_model, v_dim = lm_head.in_features, lm_head.out_features
    probe = Probe(d_model, v_dim, 256).to(device)  # fixed r=256 (mid sweep point)

    all_params = list(m.parameters()) + list(probe.parameters())
    opt = torch.optim.SGD(all_params, lr=LR, momentum=0.0, weight_decay=0.01, foreach=False)
    it = iter(train_dl)
    ppls = []

    for cyc in range(N_CYCLES):
        # Phase 1: KD target update (teacher logits on one batch)
        try: b = next(it)
        except StopIteration: it = iter(train_dl); b = next(it)
        b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            tlogits = teacher_logits(teacher, b_dev)
        # soft labels from teacher (temp-scaled softmax)
        soft = torch.softmax(tlogits / KD_TEMP, dim=-1)
        # mask pad positions (teacher sees same attn mask)
        am = b_dev["attention_mask"]
        soft = soft * am.unsqueeze(-1)

        # capture student hidden states for ALS phase
        hidden_states = []
        def capture(_mod, _inp, out):
            # LayerNorm returns a single tensor [B, L, D]; keep the FULL batch
            hidden_states.append(out.detach())
        handle = m.model.decoder.final_layer_norm.register_forward_hook(capture)
        try:
            with torch.no_grad():
                _ = m(input_ids=b_dev["input_ids"], attention_mask=b_dev["attention_mask"])
        finally: handle.remove()

        if student_mode == "als" and hidden_states:
            hs = hidden_states[0].reshape(-1, d_model)
            with torch.no_grad(): z = probe.inp(hs.to(device))
            soft_flat = soft.reshape(-1, v_dim)
            # Z covers full batch; soft_flat aligns with reshape(-1)
            soft_target_solve(probe.out, z, soft_flat, alpha=1.0)

        # Phase 2: SGD on body + probe.inp (KD loss on teacher logits)
        orig_fwd = lm_head.forward
        def patched_fwd(x):
            return orig_fwd(x) + probe(x)
        for _ in range(SGD_STEPS):
            try: b2 = next(it)
            except StopIteration: it = iter(train_dl); b2 = next(it)
            b2_dev = {k: v.to(device) for k, v in b2.items() if isinstance(v, torch.Tensor)}
            opt.zero_grad()
            lm_head.forward = patched_fwd
            with torch.no_grad():
                tlog2 = teacher_logits(teacher, b2_dev)
            out = m(input_ids=b2_dev["input_ids"], attention_mask=b2_dev["attention_mask"])
            lm_head.forward = orig_fwd
            # KD loss: KL(student || teacher) over valid positions
            logits = out.logits.float()
            log_p = torch.log_softmax(logits / KD_TEMP, dim=-1)
            p_t = torch.softmax(tlog2 / KD_TEMP, dim=-1)
            kl = (p_t * (p_t.log() - log_p)).sum(-1)
            am2 = b2_dev["attention_mask"]
            loss = (kl * am2).sum() / am2.sum().clamp(min=1)
            loss.backward()
            probe.out.weight.grad = None  # freeze probe_out (ALS owns it)
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()

        ppl = evaluate(m); ppls.append(ppl)
        print(f"  C{cyc+1:2d}: PPL={ppl:.1f}", flush=True)

    del m; torch.cuda.empty_cache()
    return ppls

# 1) Train teacher: pure SGD 16c (fixed lr, valid task) — reuse sweep config
print("\n=== Training teacher (pure SGD 16c) ===", flush=True)
teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(device)
topt = torch.optim.SGD([p for p in teacher.parameters() if p.requires_grad],
                       lr=LR, momentum=0.0, weight_decay=0.01)
it = iter(train_dl)
for cyc in range(16):
    for _ in range(SGD_STEPS):
        try: b2 = next(it)
        except StopIteration: it = iter(train_dl); b2 = next(it)
        b2_dev = {k: v.to(device) for k, v in b2.items() if isinstance(v, torch.Tensor)}
        topt.zero_grad()
        out = teacher(**b2_dev); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
        topt.step()
teacher.eval()
print(f"Teacher trained (16c).", flush=True)

# 2) Students
results = {}
results["kd_sgd"] = run("kd_sgd (SGD soft-KL)", "sgd", teacher)
results["kd_als"] = run("kd_als (closed-form soft-target ALS)", "als", teacher)

print("\n" + "="*60)
print("A-KD SWEEP — OPT-125m (12c, teacher=16c pure SGD, r=256)")
print("="*60)
for label, ppls in results.items():
    print(f"{label}: final={ppls[-1]:.1f}, best={min(ppls):.1f}")
with open("runs/kd_als_opt125m.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved runs/kd_als_opt125m.json")
