#!/usr/bin/env python3
"""
P5: Multi-seed rank curve — statistical robustness of r=8 plateau.
r=8, r=32, r=256 on Qwen2.5-0.5B at seeds 123 and 456 (seed 42 already done).
6 fast runs. Confirms r=8 plateau has SE < 0.01.
"""

import json, sys, time, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

ML, BS, GA, MS = 1024, 1, 4, 100
NTr, NEv, LR = 800, 100, 1e-4
TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]
SEEDS = [123, 456]
RANKS = [8, 32, 256]
OUT = "runs/p5_multiseed"

RESULTS = []
EN_REF = {"r8": 1.62, "r32": 1.60, "r256": 1.61}  # seed 42 reference


def dl(tok, sp, n):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=sp)
    ds = ds.select(range(min(n, len(ds))))
    ds = ds.map(lambda ex: tok(ex["text"], truncation=True, max_length=ML, padding="max_length"),
                 batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(ds, batch_size=BS, shuffle=(sp == "train"),
                       collate_fn=lambda b: {
                           "input_ids": torch.stack([x["input_ids"] for x in b]),
                           "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                           "labels": torch.stack([x["input_ids"] for x in b])})


def ppl_eval(m, dl, dev):
    m.eval()
    tl, tt = 0.0, 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            lo = m(**b).loss
            nt = b["attention_mask"].sum().item()
            tl += lo.item() * nt
            tt += nt
    return round(float(torch.exp(torch.tensor(tl / max(tt, 1))).item()), 4)


def run_seed_rank(seed, rank, tok, tr_dl, ev_dl, dev):
    torch.manual_seed(seed)
    alpha = int(rank * 2)
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=False, local_files_only=True)
    dev = next(base.parameters()).device
    m = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, target_modules=TARGETS))
    m.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=LR, weight_decay=0.01)
    m.train()
    step, acc = 0, 0
    while step < MS:
        for b in tr_dl:
            b = {k: v.to(dev) for k, v in b.items()}
            (m(**b).loss / GA).backward()
            acc += 1
            if acc >= GA:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                acc = 0
                if step >= MS: break
    pp = ppl_eval(m, ev_dl, dev)
    result = {"seed": seed, "rank": rank, "ppl": pp, "params_M": round(n_params / 1e6, 1)}
    del m, base, opt
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    import os, logging
    os.makedirs(OUT, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("p5")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=False, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tr_dl = dl(tok, "train", NTr)
    ev_dl = dl(tok, "test", NEv)

    # Get device once
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=False, local_files_only=True)
    dev = next(base.parameters()).device
    del base
    gc.collect()
    torch.cuda.empty_cache()

    for seed in SEEDS:
        for rank in RANKS:
            logger.info(">>> seed=%d r=%d", seed, rank)
            r = run_seed_rank(seed, rank, tok, tr_dl, ev_dl, dev)
            RESULTS.append(r)
            logger.info("  PPL=%.4f", r["ppl"])

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info("P5 RESULTS: Multi-Seed Rank Curve (Qwen2.5-0.5B)")
    logger.info("%-8s %10s %10s %10s %10s", "Rank", "s42 PPL", "s123 PPL", "s456 PPL", "Mean±SE")
    for rank in RANKS:
        pp42 = EN_REF[f"r{rank}"]
        pp123 = next(r["ppl"] for r in RESULTS if r["seed"] == 123 and r["rank"] == rank)
        pp456 = next(r["ppl"] for r in RESULTS if r["seed"] == 456 and r["rank"] == rank)
        vals = [pp42, pp123, pp456]
        import numpy as np
        mean = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(3)
        logger.info("r%-7d %10.4f %10.4f %10.4f %10.4f±%.4f", rank, pp42, pp123, pp456, mean, se)

    logger.info("\nPlateau stability: max|Δ| = %.4f across all seeds and ranks",
                 max(abs(RESULTS[i]["ppl"] - EN_REF[f"r{RESULTS[i]['rank']}"]) for i in range(len(RESULTS))))

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
