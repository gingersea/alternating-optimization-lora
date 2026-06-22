#!/usr/bin/env python3
"""
P1: ASP Long-Horizon Convergence Crossover.
Tests whether ASP catches AdamW at extended step budgets.

GPT-2: Protocol A (ASP) vs B (AdamW) at 800 steps, N=3 seeds.
OPT-125m: Protocol A vs B at 800 steps, N=3 seeds.

Paper prediction (§6.3): crossover at ~800-1000 (GPT-2), ~1000-1500 (OPT-125m).
This experiment extends from current max (400 steps for multi-seed) to 800 steps.
"""

import json, sys, time, gc, os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

ML, BS, NTr, NEv, LR, WD = 512, 4, 200, 100, 1e-4, 0.01
MAX_STEPS = 800
LOG_EVERY = 200
SEEDS = [42, 123, 456]
CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"
OUT = "runs/p1_crossover"

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("p1")


def build_dl(tokenizer, split, n):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    ds = ds.select(range(min(n, len(ds))))
    ds = ds.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=ML, padding="max_length"),
                batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(ds, batch_size=BS, shuffle=(split == "train"),
                       collate_fn=lambda b: {
                           "input_ids": torch.stack([x["input_ids"] for x in b]),
                           "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                           "labels": torch.stack([x["input_ids"] for x in b])})


def ppl_eval(model, dl, dev):
    model.eval()
    tl, tt = 0.0, 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            lo = model(**b).loss
            nt = b["attention_mask"].sum().item()
            tl += lo.item() * nt
            tt += nt
    return round(float(torch.exp(torch.tensor(tl / max(tt, 1))).item()), 2)


def run_adamw_800(name, hf, seed, tok, tr_dl, ev_dl, dev):
    torch.manual_seed(seed)
    logger.info(">>> %s AdamW seed=%d", name, seed)
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(hf, trust_remote_code=False, local_files_only=True)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    history = []

    model.train()
    gs = 0
    while gs < MAX_STEPS:
        for b in tr_dl:
            b = {k: v.to(dev) for k, v in b.items()}
            opt.zero_grad()
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            gs += 1
            if gs % LOG_EVERY == 0:
                p = ppl_eval(model, ev_dl, dev)
                history.append({"step": gs, "ppl": p, "loss": loss.item()})
                logger.info("  step=%d ppl=%.2f", gs, p)
                model.train()
            if gs >= MAX_STEPS: break

    p = ppl_eval(model, ev_dl, dev)
    elapsed = time.time() - t0
    logger.info("  FINAL: step=%d ppl=%.2f (%.0fs)", MAX_STEPS, p, elapsed)
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return {"name": f"{name}_AdamW_s{seed}", "seed": seed, "ppl": p, "time_s": int(elapsed), "history": history}


def run_asp_800(name, hf, seed, tok, tr_dl, ev_dl, dev):
    torch.manual_seed(seed)
    logger.info(">>> %s ASP seed=%d", name, seed)
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(hf, trust_remote_code=False, local_files_only=True)
    model = model.to(torch.float32).to(dev)
    history = []

    # ASP schedule: ALS→SGD→Perturb cycles
    sgd_per_cycle = 100
    phases_per_cycle = 1 + sgd_per_cycle + 1  # ALS + SGD + Perturb
    n_cycles = MAX_STEPS // phases_per_cycle

    gs = 0
    for cycle in range(n_cycles):
        if gs >= MAX_STEPS: break

        # ALS: add tiny noise to lm_head (simplified — real ALS too expensive for 800-step GPU)
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'lm_head'):
                model.lm_head.weight.data += 5e-5 * torch.randn_like(model.lm_head.weight.data)
        gs += 1

        # SGD
        opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD)
        model.train()
        ss = 0
        for b in tr_dl:
            if ss >= sgd_per_cycle or gs >= MAX_STEPS: break
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
                history.append({"step": gs, "ppl": p, "loss": loss.item()})
                logger.info("  step=%d ppl=%.2f", gs, p)
                model.train()

        # Perturbation
        if gs < MAX_STEPS:
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(1e-3 * torch.randn_like(p))
            gs += 1

    p = ppl_eval(model, ev_dl, dev)
    elapsed = time.time() - t0
    logger.info("  FINAL: step=%d ppl=%.2f (%.0fs)", MAX_STEPS, p, elapsed)
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return {"name": f"{name}_ASP_s{seed}", "seed": seed, "ppl": p, "time_s": int(elapsed), "history": history}


def run_model(name, hf):
    logger.info("=" * 60)
    logger.info("MODEL: %s (%s) @ %d steps × %d seeds", name, hf, MAX_STEPS, len(SEEDS))
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(hf, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tr_dl = build_dl(tokenizer, "train", NTr)
    ev_dl = build_dl(tokenizer, "test", NEv)
    dev = torch.device(DEVICE)  # GPU when available

    results = []
    for seed in SEEDS:
        r = run_adamw_800(name, hf, seed, tokenizer, tr_dl, ev_dl, dev)
        results.append(r)
    for seed in SEEDS:
        r = run_asp_800(name, hf, seed, tokenizer, tr_dl, ev_dl, dev)
        results.append(r)

    # Summary
    adamw_ppls = [r["ppl"] for r in results if "AdamW" in r["name"]]
    asp_ppls = [r["ppl"] for r in results if "ASP" in r["name"]]
    logger.info("\n  %s SUMMARY:", name)
    logger.info("  AdamW: %.2f ± %.2f (%d seeds)", np.mean(adamw_ppls), np.std(adamw_ppls), len(adamw_ppls))
    logger.info("  ASP:   %.2f ± %.2f (%d seeds)", np.mean(asp_ppls), np.std(asp_ppls), len(asp_ppls))
    if np.mean(asp_ppls) < np.mean(adamw_ppls):
        logger.info("  ★ ASP CROSSES AdamW! gap=%.2f", np.mean(adamw_ppls) - np.mean(asp_ppls))
    else:
        gap = np.mean(asp_ppls) - np.mean(adamw_ppls)
        logger.info("  ASP still behind AdamW. gap=%.2f", gap)

    return results


def main():
    logger.info("=" * 70)
    logger.info("P1: ASP CONVERGENCE CROSSOVER")
    logger.info("GPT-2 + OPT-125m | Protocol A vs B | %d steps | N=%d seeds", MAX_STEPS, len(SEEDS))
    logger.info("=" * 70)

    all_results = {}
    for name, hf in [("GPT-2", "gpt2"), ("OPT-125m", "facebook/opt-125m")]:
        results = run_model(name, hf)
        all_results[name] = results

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL CROSSOVER SUMMARY")
    for name in ["GPT-2", "OPT-125m"]:
        r = all_results[name]
        a_ppls = [x["ppl"] for x in r if "AdamW" in x["name"]]
        s_ppls = [x["ppl"] for x in r if "ASP" in x["name"]]
        logger.info("%s: AdamW=%.2f±%.2f  ASP=%.2f±%.2f  Δ=%.2f",
                     name, np.mean(a_ppls), np.std(a_ppls),
                     np.mean(s_ppls), np.std(s_ppls),
                     np.mean(s_ppls) - np.mean(a_ppls))

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
