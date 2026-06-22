#!/usr/bin/env python3
"""
F2: Full ASP Crossover — OPT-125m with real Cholesky ALS.
Validates whether true ASP (ALS+SGD+Perturbation) catches AdamW at 1200+ steps.

This is the paper's longest-standing open question: does the three-phase
ASP cycle actually surpass AdamW within the stable depth regime (L ≤ 24)?

Implementation: uses altopt.als.ALSBlockSolver for Cholesky-based ALS on
nn.Linear modules (OPT-125m has standard nn.Linear layers throughout).
No simplified noise-injection — this is the REAL ASP.

ASP schedule: ALS(1)→SGD(50)→Perturb(1), repeated for ~23 cycles = 1200 steps.
AdamW baseline at same step budget for comparison.
"""

import json, sys, time, gc, os, logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from altopt.als import ALSBlockSolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("f2")

# Config
MODEL = "facebook/opt-125m"
ML, BS = 512, 4
NTr, NEv, LR, WD = 400, 100, 1e-4, 0.01
MAX_STEPS = 1200
LOG_EVERY = 100
ALS_BLOCK_SIZE = 512
ALS_REG = 1e-4
PERTURB_SIGMA = 1e-3
SGD_PER_CYCLE = 50
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "runs/f2_full_asp"

torch.manual_seed(SEED)


def build_dl(tok, split, n):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    ds = ds.select(range(min(n, len(ds))))
    ds = ds.map(lambda ex: tok(ex["text"], truncation=True, max_length=ML, padding="max_length"),
                 batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(ds, batch_size=BS, shuffle=(split == "train"),
                       collate_fn=lambda b: {
                           "input_ids": torch.stack([x["input_ids"] for x in b]),
                           "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                           "labels": torch.stack([x["input_ids"] for x in b])})


def ppl_eval(model, dl, dev):
    model.eval() if not isinstance(model, dict) else None
    try:
        model.eval()
    except Exception:
        pass
    tl, tt = 0.0, 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            lo = model(**b).loss
            nt = b["attention_mask"].sum().item()
            tl += lo.item() * nt
            tt += nt
    return round(float(torch.exp(torch.tensor(tl / max(tt, 1))).item()), 2)


def run_adamw_baseline(tok, tr_dl, ev_dl, dev):
    """AdamW full-rank baseline at MAX_STEPS."""
    logger.info(">>> ADAMW: %d steps", MAX_STEPS)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=False, local_files_only=True)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    history = []

    model.train()
    gs = 0
    for b in tr_dl:
        if gs >= MAX_STEPS:
            break
        b = {k: v.to(dev) for k, v in b.items()}
        opt.zero_grad()
        loss = model(**b).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        gs += 1
        if gs % LOG_EVERY == 0:
            p = ppl_eval(model, ev_dl, dev)
            history.append({"step": gs, "ppl": p})
            logger.info("  AdamW step=%d ppl=%.2f", gs, p)
            model.train()

    p = ppl_eval(model, ev_dl, dev)
    elapsed = time.time() - t0
    logger.info("  FINAL AdamW: ppl=%.2f (%.0fs)", p, elapsed)
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return {"optimizer": "AdamW", "ppl": p, "time_s": int(elapsed), "history": history}


def run_asp_full(tok, tr_dl, ev_dl, dev):
    """Full ASP: ALS (Cholesky) → SGD → Perturb cycles."""
    n_cycles = MAX_STEPS // (SGD_PER_CYCLE + 2)  # 1 ALS + SGD + 1 Perturb
    logger.info(">>> ASP (REAL ALS): %d cycles, %d total steps", n_cycles, n_cycles * (SGD_PER_CYCLE + 2))
    t0 = time.time()

    # Load model in float32 for ALS Cholesky stability
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=False, local_files_only=True,
        torch_dtype=torch.float32)
    model = model.to(dev)

    # Initialize ALS solver (model-specific, handles depth-boundary)
    als = ALSBlockSolver(model)

    history = []
    gs = 0

    for cycle in range(n_cycles):
        if gs >= MAX_STEPS:
            break

        # Phase I: ALS (real Cholesky on all nn.Linear layers)
        logger.info("  Cycle %d/%d: ALS phase (step %d)", cycle + 1, n_cycles, gs)
        try:
            # Get a batch for ALS
            for b in tr_dl:
                b = {k: v.to(dev) for k, v in b.items()}
                als.solve_block(b, block_size=ALS_BLOCK_SIZE)
                break  # one batch per ALS cycle
        except Exception as e:
            logger.warning("  ALS failed: %s — skipping to SGD", e)
        gs += 1

        # Phase II: SGD
        opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD)
        model.train()
        ss = 0
        for b in tr_dl:
            if ss >= SGD_PER_CYCLE or gs >= MAX_STEPS:
                break
            b = {k: v.to(dev) for k, v in b.items()}
            opt.zero_grad()
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            gs += 1
            ss += 1
            if gs % LOG_EVERY == 0:
                p = ppl_eval(model, ev_dl, dev)
                history.append({"step": gs, "ppl": p})
                logger.info("  ASP step=%d ppl=%.2f", gs, p)
                model.train()

        # Phase III: Perturbation
        if gs < MAX_STEPS:
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(PERTURB_SIGMA * torch.randn_like(p))
            gs += 1
            logger.info("  Perturbation applied (σ=%.1e)", PERTURB_SIGMA)

    p = ppl_eval(model, ev_dl, dev)
    elapsed = time.time() - t0
    logger.info("  FINAL ASP: ppl=%.2f (%.0fs, %.0fmin)", p, elapsed, elapsed / 60)
    del model, opt, als
    gc.collect()
    torch.cuda.empty_cache()
    return {"optimizer": "ASP (real ALS)", "ppl": p, "time_s": int(elapsed), "history": history}


def main():
    logger.info("=" * 70)
    logger.info("F2: FULL ASP CROSSOVER — OPT-125m with real Cholesky ALS")
    logger.info("%d steps, ALS(SGD=%d)+Perturb, N=%d seeds", MAX_STEPS, SGD_PER_CYCLE, 1)
    logger.info("Device: %s", DEVICE)
    logger.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tr_dl = build_dl(tokenizer, "train", NTr)
    ev_dl = build_dl(tokenizer, "test", NEv)
    dev = torch.device(DEVICE)

    # AdamW baseline
    adamw_result = run_adamw_baseline(tokenizer, tr_dl, ev_dl, dev)

    # ASP with real ALS
    asp_result = run_asp_full(tokenizer, tr_dl, ev_dl, dev)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("F2 CROSSOVER SUMMARY")
    logger.info("OPT-125m @ %d steps:", MAX_STEPS)
    logger.info("  AdamW:       PPL=%.2f (%.0fs)", adamw_result["ppl"], adamw_result["time_s"])
    logger.info("  ASP (REAL):  PPL=%.2f (%.0fs)", asp_result["ppl"], asp_result["time_s"])

    if asp_result["ppl"] < adamw_result["ppl"]:
        logger.info("  ★ ASP CROSSES AdamW! gap=%.2f", adamw_result["ppl"] - asp_result["ppl"])
    else:
        logger.info("  ASP behind AdamW. gap=%.2f", asp_result["ppl"] - adamw_result["ppl"])

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/results.json", "w") as f:
        json.dump({"adamw": adamw_result, "asp": asp_result}, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
