#!/usr/bin/env python3
"""
F1: η mechanism attribution — discriminates between intrinsic dimension vs training budget.
Tests: r=4 at N_train = 400, 800, 1600 on Qwen2.5-0.5B (100 steps, AdamW, seed 42).

Hypotheses:
  A: η ∝ 1/N_samples (training budget dominates)
     → r=4 at N=400: UNDER threshold (higher PPL)
     → r=4 at N=800: NEAR threshold (currently tested: 1.62 vs r=8 1.62)
     → r=4 at N=1600: AT plateau (matching r=8; η halved from 400)

  B: η is task-intrinsic (N_samples-independent)
     → r=4 at ALL N: SAME r8/r4 ratio (~1.01)
     → r=4 is always at plateau, regardless of sample count

  C: η ∝ sqrt(N_samples)
     → r=4 at N=400: UNDER threshold (r8/r4 > 1.5)
     → r=4 at N=800: NEAR threshold
     → r=4 at N=1600: NEAR threshold (slower scaling than Hypothesis A)

Also runs r=8 at each N to normalize out the N_samples effect on plateau PPL.
"""

import json, sys, time, gc, os, math
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("f1")

ML, BS, GA, MS = 1024, 1, 4, 100
LR, SD = 1e-4, 42
TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]
N_TRAIN_VALS = [400, 800, 1600]
N_EVAL = 100
OUT = "runs/f1_eta"

torch.manual_seed(SD)


def dl(tok, sp, n):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=sp)
    ds = ds.select(range(min(n, len(ds))))
    ds = ds.map(lambda ex: tok(ex["text"], truncation=True, max_length=ML, padding="max_length"),
                 batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(ds, batch_size=BS, shuffle=(sp == "train"),
                       collate_fn=lambda b: {"input_ids": torch.stack([x["input_ids"] for x in b]),
                                             "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                                             "labels": torch.stack([x["input_ids"] for x in b])})


def ppl_eval(m, dl, dev):
    m.eval(); tl, tt = 0.0, 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            lo = m(**b).loss; nt = b["attention_mask"].sum().item(); tl += lo.item() * nt; tt += nt
    return round(float(torch.exp(torch.tensor(tl / max(tt, 1))).item()), 4)


def train_and_eval(rank, n_train, tok, ev_dl, dev):
    alpha = int(rank * 2)
    logger.info(">>> r=%d N_train=%d", rank, n_train)
    t0 = time.time()

    # Build training loader with specific N
    tr_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").select(range(n_train))
    tr_ds = tr_ds.map(lambda ex: tok(ex["text"], truncation=True, max_length=ML, padding="max_length"),
                       batched=True, remove_columns=["text"])
    tr_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    tr_dl = DataLoader(tr_ds, batch_size=BS, shuffle=True,
                        collate_fn=lambda b: {"input_ids": torch.stack([x["input_ids"] for x in b]),
                                              "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                                              "labels": torch.stack([x["input_ids"] for x in b])})

    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=False, local_files_only=True)
    dev = next(base.parameters()).device
    m = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, target_modules=TARGETS))
    m.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=LR, weight_decay=0.01)
    m.train(); step, acc = 0, 0
    while step < MS:
        for b in tr_dl:
            b = {k: v.to(dev) for k, v in b.items()}; (m(**b).loss / GA).backward(); acc += 1
            if acc >= GA:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
                step += 1; acc = 0
                if step >= MS: break
    pp = ppl_eval(m, ev_dl, dev); el = time.time() - t0
    logger.info("  PPL=%.4f (%dM params, %.0fs)", pp, n_params // 1_000_000, el)
    del m, base, opt; gc.collect(); torch.cuda.empty_cache()
    return {"rank": rank, "n_train": n_train, "ppl": pp, "params_M": round(n_params / 1e6, 1), "time_s": int(el)}


def main():
    logger.info("F1: η mechanism attribution — N_samples scaling")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=False, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ev_dl = dl(tok, "test", N_EVAL)
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=False, local_files_only=True)
    dev = next(base.parameters()).device
    del base; gc.collect(); torch.cuda.empty_cache()

    results = []
    for n in N_TRAIN_VALS:
        for rank in [4, 8]:
            r = train_and_eval(rank, n, tok, ev_dl, dev)
            results.append(r)

    # Hypothesis test
    logger.info("\n=== HYPOTHESIS DISCRIMINATION ===")
    logger.info("%-15s %8s %8s %8s %8s", "Condition", "r4 PPL", "r8 PPL", "r4/r8", "Hypothesis")
    for n in N_TRAIN_VALS:
        r4 = next(r["ppl"] for r in results if r["rank"] == 4 and r["n_train"] == n)
        r8 = next(r["ppl"] for r in results if r["rank"] == 8 and r["n_train"] == n)
        ratio = r4 / r8

        # Hypothesis test
        if ratio < 1.05:
            hyp = "B(CONSTANT): η independent of N_samples"
        elif n == 400 and ratio > 1.5 and N_TRAIN_VALS.index(n) > 0:
            hyp = "A(1/N): η ∝ 1/N_samples — r4 catches up with more data"
        elif ratio > 1.10 and ratio < 1.30:
            hyp = "C(sqrt): η ∝ 1/sqrt(N) — weak dependence"
        else:
            hyp = "A(trend): check ratio progression"

        logger.info("N=%-11d %8.4f %8.4f %8.4f %s", n, r4, r8, ratio, hyp)

    # Trend analysis
    ratios = []
    for n in N_TRAIN_VALS:
        r4 = next(r["ppl"] for r in results if r["rank"] == 4 and r["n_train"] == n)
        r8 = next(r["ppl"] for r in results if r["rank"] == 8 and r["n_train"] == n)
        ratios.append(r4 / r8)

    logger.info("\nRatio trend: %s", " → ".join(f"{x:.4f}" for x in ratios))
    if ratios[-1] < ratios[0] * 0.95:
        logger.info("✅ Hypothesis A: η DECREASES with more samples (ratio declining)")
        logger.info("   → r_min(1600) ≈ r_min(800) × 800/1600 = 3.1")
        logger.info("   → r=4 should be sufficient at N>800")
    elif max(ratios) - min(ratios) < 0.03:
        logger.info("✅ Hypothesis B: η is CONSTANT (all ratios ≈ %.3f)", np.mean(ratios))
        logger.info("   → r_min is architecture-intrinsic, NOT data-dependent")
    else:
        logger.info("⚠  Mixed: check ratio progression pattern")

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
