"""Protocol A-PROBE: Low-rank probe bottleneck eliminates residual amplification.

Core idea: Insert a low-rank probe (3584->64->3584) between the last hidden
layer and lm_head.  ALS solves only the 64-dim probe output projection
(well-conditioned, ~0 OOM risk).  Body SGD + probe input projection learn
to produce representations the probe can use.

Cycle:
  1. ALS on probe output projection (64 x 3584, label-based targets)
  2. Freeze probe output, SGD on body + probe input (50 steps)
  3. Unfreeze, repeat

Key advantage over ALL prior approaches: ALS never touches lm_head.
Zero residual amplification.  The probe acts as a regularization
bottleneck — if body representations work through the probe, they'll
work even better with the full lm_head.

Compared against A-SYNC and pure SGD on Qwen0.5B (24L).
"""
import json, math, time
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.als import ALSBlockSolver
from altopt.sgd import SGDPhaseOptimizer

MODEL = "Qwen/Qwen2.5-0.5B"
N_CYCLES = 12
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
eval_dl = DataLoader(eval_ds, batch_size=4, collate_fn=c)

def evaluate(m):
    m.eval(); tl, tn = 0.0, 0
    with torch.no_grad():
        for b in eval_dl:
            b = {k: v.to(device) for k, v in b.items()}
            out = m(**b)
            if out.loss is None or torch.isnan(out.loss) or torch.isinf(out.loss): return float("inf")
            tl += out.loss.item() * b["attention_mask"].sum().item()
            tn += b["attention_mask"].sum().item()
    m.train()
    return math.exp(tl / max(tn, 1)) if tl / max(tn, 1) < 700 else float("inf")


class ProbeBottleneck(nn.Module):
    def __init__(self, d_model, v_dim, rank=64):
        super().__init__()
        self.probe_in = nn.Linear(d_model, rank, bias=False)
        self.probe_out = nn.Linear(rank, v_dim, bias=False)
        nn.init.normal_(self.probe_in.weight, std=0.02 / (rank**0.5))
        nn.init.zeros_(self.probe_out.weight)

    def forward(self, x):
        return self.probe_out(self.probe_in(x))


class ProbeAugmentedModel(nn.Module):
    def __init__(self, base_model, probe_rank=64):
        super().__init__()
        self.base = base_model
        d_model = base_model.lm_head.in_features
        v_dim = base_model.lm_head.out_features
        self.probe = ProbeBottleneck(d_model, v_dim, rank=probe_rank)
        # Disable base lm_head — use probe->base lm_head path
        # Actually: we ADD probe output to base lm_head output
        self.probe.to(next(base_model.parameters()).device)

    def forward(self, **kwargs):
        # Intercept: use base to get hidden states, then apply probe
        # Strategy: replace lm_head with (lm_head + probe)
        orig_forward = self.base.lm_head.forward

        def probe_forward(x):
            return orig_forward(x) + self.probe(x)

        self.base.lm_head.forward = probe_forward.__get__(self.base.lm_head)
        try:
            return self.base(**kwargs)
        finally:
            self.base.lm_head.forward = orig_forward


# ── A-PROBE ─────────────────────────────────────────────────────────

def run_probe(rank=64):
    print(f"\n{'='*50}\nA-PROBE: rank={rank}, {N_CYCLES} cycles\n{'='*50}")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    d_model = m.lm_head.in_features
    v_dim = m.lm_head.out_features

    probe = ProbeBottleneck(d_model, v_dim, rank=rank).to(device)

    # We need ALS to solve probe_out (rank x v_dim, label-based)
    # Then freeze probe_out, train body + probe_in via SGD
    # This means we wrap ALS differently: solve probe_out only

    it = iter(train_dl)
    sgd_opt = torch.optim.SGD(
        list(m.parameters()) + list(probe.parameters()),
        lr=1e-4, momentum=0.9, weight_decay=0.01, foreach=False,
    )
    ppls, step_cnt = [], 0
    t0 = time.time()

    # Find lm_head for hooking
    lm_head = m.lm_head

    for cyc in range(N_CYCLES):
        try: b = next(it)
        except StopIteration: it = iter(train_dl); b = next(it)
        b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
        labels = b_dev["labels"]

        # Phase 1: ALS on probe_out via label-based target
        # Capture hidden states at the point probe_in outputs
        hidden_states = []

        def capture_hidden(_mod, _inp, out):
            hidden_states.append(out[0].detach())

        handle = m.model.norm.register_forward_hook(capture_hidden)
        try:
            with torch.no_grad():
                _ = m(input_ids=b_dev["input_ids"], attention_mask=b_dev["attention_mask"])
        finally:
            handle.remove()

        if hidden_states:
            hs = hidden_states[0]
            if hs.dim() == 3: hs = hs.reshape(-1, d_model)
            # Apply probe_in to get bottleneck reps
            with torch.no_grad():
                z = probe.probe_in(hs)  # [N, rank]
            # Solve probe_out via ALS on z -> labels
            step_cnt += 1
            _solve_probe_out(probe.probe_out, z, labels, alpha=0.05, lambda_reg=1e-3)

        # Phase 2: Freeze probe_out, SGD on body + probe_in
        for pg in probe.probe_out.parameters():
            for p in pg: pass  # will freeze via requires_grad
        # Actually: just SGD on all with lr for body+probe_in, 0 for probe_out
        for j in range(50):
            step_cnt += 1
            try: b2 = next(it)
            except StopIteration: it = iter(train_dl); b2 = next(it)
            b2_dev = {k: v.to(device) for k, v in b2.items() if isinstance(v, torch.Tensor)}
            sgd_opt.zero_grad()

            # Forward: lm_head + probe
            orig_forward = lm_head.forward
            lm_head.forward = lambda x: orig_forward(x) + probe(x)
            out = m(**b2_dev)
            loss = out.loss if hasattr(out, "loss") else out[0]
            lm_head.forward = orig_forward

            loss.backward()
            # Zero out probe_out gradients (frozen)
            probe.probe_out.weight.grad = None
            torch.nn.utils.clip_grad_norm_(
                [p for p in m.parameters() if p.requires_grad] +
                [probe.probe_in.weight] if probe.probe_in.weight.requires_grad else [],
                1.0,
            )
            sgd_opt.step()

        ppl = evaluate_wrapper(m, probe, lm_head)
        ppls.append(ppl)
        print(f"  C{cyc+1:2d}: ppl={ppl:.1f}" if ppl < 1e10 else f"  C{cyc+1}: DIVERGED")
        if ppl > 1e10: break

    elapsed = time.time() - t0
    del m, probe; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}


def _solve_probe_out(probe_out, Z, labels, alpha=0.05, lambda_reg=1e-3):
    """ALS block solve for probe_out: W_new = argmin ||Z @ W^T - Y_target||^2.
    Z: [N, rank], labels: [N] token ids.
    W: [v_dim, rank], solved per-block with label-based masking.
    """
    weight = probe_out.weight.data  # [v_dim, rank]
    v_dim, rank = weight.shape
    Z_f32 = Z.detach().float()
    N = Z_f32.shape[0]
    labels_flat = labels.reshape(-1)[:N].to(device=Z.device, dtype=torch.long)
    labels_flat = torch.clamp(labels_flat, 0, v_dim - 1)

    block_size = 4096
    n_blocks = (v_dim + block_size - 1) // block_size
    reg = lambda_reg * torch.eye(rank, device=Z_f32.device, dtype=torch.float32)
    ZtZ = Z_f32.T @ Z_f32 + reg

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, v_dim)
        mask = (labels_flat >= start) & (labels_flat < end)
        if not mask.any():
            continue
        Z_masked = Z_f32[mask]
        target_tokens = labels_flat[mask] - start
        Y = torch.zeros((mask.sum().item(), end - start), device=Z_f32.device, dtype=torch.float32)
        Y[torch.arange(mask.sum().item(), device=Z_f32.device), target_tokens] = 1.0
        ZtY = Z_masked.T @ Y  # [rank, block]

        try:
            L = torch.linalg.cholesky(Z_masked.T @ Z_masked + reg)
            W_new = torch.cholesky_solve(ZtY, L).T
        except RuntimeError:
            W_new = torch.linalg.lstsq(Z_masked.T @ Z_masked + reg, ZtY).solution.T

        W_cur = weight[start:end, :].detach().float()
        damped = (1 - alpha) * W_cur + alpha * W_new
        weight[start:end, :] = damped.to(device=weight.device, dtype=weight.dtype)


def evaluate_wrapper(m, probe, lm_head):
    """Evaluate model with probe augmenting lm_head."""
    m.eval(); tl, tn = 0.0, 0
    orig_forward = lm_head.forward
    lm_head.forward = lambda x: orig_forward(x) + probe(x)
    with torch.no_grad():
        for b in eval_dl:
            b = {k: v.to(device) for k, v in b.items()}
            out = m(**b)
            if out.loss is None or torch.isnan(out.loss) or torch.isinf(out.loss):
                lm_head.forward = orig_forward; return float("inf")
            tl += out.loss.item() * b["attention_mask"].sum().item()
            tn += b["attention_mask"].sum().item()
    lm_head.forward = orig_forward
    m.train()
    return math.exp(tl / max(tn, 1)) if tl / max(tn, 1) < 700 else float("inf")


# ── Pure SGD baseline ───────────────────────────────────────────────

def run_pure_sgd():
    print(f"\n{'='*50}\nPure SGD, {N_CYCLES} cycles, 52 steps/cycle\n{'='*50}")
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    opt = torch.optim.SGD(m.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.01, foreach=False)
    it = iter(train_dl)
    ppls, step_cnt = [], 0
    t0 = time.time()
    for cyc in range(N_CYCLES):
        for j in range(52):
            try: b = next(it)
            except StopIteration: it = iter(train_dl); b = next(it)
            b_dev = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
            opt.zero_grad()
            out = m(**b_dev)
            loss = out.loss if hasattr(out, "loss") else out[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            step_cnt += 1
        ppl = evaluate(m); ppls.append(ppl)
        print(f"  C{cyc+1:2d}: ppl={ppl:.1f}" if ppl < 1e10 else f"  C{cyc+1}: DIVERGED")
    elapsed = time.time() - t0
    del m; torch.cuda.empty_cache()
    return {"ppls": ppls, "elapsed": elapsed}


# ── Main ──
results = {}
results["probe"] = run_probe(64)
results["pure_sgd"] = run_pure_sgd()

print(f"\n{'='*60}")
print("A-PROBE vs Pure SGD on Qwen0.5B (24L):")
for label, r in results.items():
    p = r["ppls"]
    pts = " -> ".join(f"{x:.1f}" if x < 1e10 else "inf" for x in p)
    print(f"  {label:12s}: {pts}")
    if p and not math.isinf(p[-1]):
        print(f"  {'':12s}  Final={p[-1]:.1f}")

with open("runs/probe_vs_sgd_05b.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Saved runs/probe_vs_sgd_05b.json")
