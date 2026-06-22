#!/usr/bin/env python3
"""
P3: M-index cross-scale calibration.
C4 PPL on Qwen2.5-0.5B at r=8, r=32, r=256, r=512 + full-rank.
Uses existing checkpoints from param-matched baseline and xval runs.
Extends M-index from 2 extreme points (3M, 7B) to intermediate scale.
"""

import json, sys, time, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

ML, BS, NEv = 2048, 2, 300
CKPT_DIR = "runs/rank_curve"
OUT = "runs/p3_m_index"


def build_c4_dl(tokenizer, n_samples):
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = [ex["text"] for _, ex in zip(range(n_samples), ds)]
    enc = tokenizer(texts, truncation=True, max_length=ML, padding="max_length", return_tensors="pt")
    from torch.utils.data import TensorDataset
    return DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"]),
                       batch_size=BS, shuffle=False,
                       collate_fn=lambda b: {
                           "input_ids": torch.stack([x[0] for x in b]),
                           "attention_mask": torch.stack([x[1] for x in b]),
                           "labels": torch.stack([x[0] for x in b])})


def ppl(m, dl):
    m.eval()
    tl, tt = 0.0, 0
    dev = next(m.parameters()).device
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            lo = m(**b).loss
            nt = b["attention_mask"].sum().item()
            tl += lo.item() * nt
            tt += nt
    return round(float(torch.exp(torch.tensor(tl / max(tt, 1))).item()), 2)


def eval_ckpt(name, hf, ckpt_path, is_peft, rank, alpha):
    """Evaluate saved checkpoint on C4."""
    base = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=False,
                                                 local_files_only=True)
    sd = torch.load(ckpt_path, map_location="cpu")
    if is_peft:
        base = get_peft_model(base, LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]))
    base.load_state_dict(sd, strict=False)
    return base


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("p3")

    logger.info("P3: M-index cross-scale calibration")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=False, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dl = build_c4_dl(tok, NEv)

    results = []

    # Baseline
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16,
                                              device_map="auto", trust_remote_code=False, local_files_only=True)
    results.append({"model": "baseline", "c4_ppl": ppl(m, dl), "trainable_M": 0})
    del m
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate r=8, r=32, r=256 from rank_curve checkpoints
    # These checkpoints were saved as part of param-matched baseline experiments
    # We need to re-create them since old checkpoints were in a different format

    logger.info("Training fresh C4 evaluations on Qwen2.5-0.5B...")
    for rank in [8, 32, 256, 512]:
        alpha = int(rank * 2)
        logger.info(">>> r=%d (α=%d)", rank, alpha)

        base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16,
                                                     device_map="auto", trust_remote_code=False, local_files_only=True)
        dev = next(base.parameters()).device
        m = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05,
                                              target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]))
        m.gradient_checkpointing_enable()
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

        # Train on wikiText-2 (same config as all other experiments)
        from datasets import load_dataset as ld
        wds = ld("wikitext", "wikitext-2-raw-v1", split="train").select(range(800))
        wt = wds.map(lambda ex: tok(ex["text"], truncation=True, max_length=1024, padding="max_length"),
                       batched=True, remove_columns=["text"])
        wt.set_format(type="torch", columns=["input_ids", "attention_mask"])
        tr_dl = DataLoader(wt, batch_size=1, shuffle=True,
                           collate_fn=lambda b: {
                               "input_ids": torch.stack([x["input_ids"] for x in b]),
                               "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                               "labels": torch.stack([x["input_ids"] for x in b]).clone()})

        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-4, weight_decay=0.01)
        m.train()
        step, acc = 0, 0
        while step < 100:
            for b in tr_dl:
                b = {k: v.to(dev) for k, v in b.items()}
                (m(**b).loss / 4).backward()
                acc += 1
                if acc >= 4:
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step(); opt.zero_grad()
                    step += 1; acc = 0
                    if step >= 100: break

        c4p = ppl(m, dl)
        logger.info("  C4 PPL=%.2f (WT2 PPL reference from xval)", c4p)
        results.append({"model": f"r{rank}", "trainable_M": round(n_params / 1e6, 1), "c4_ppl": c4p})
        del m, base, opt
        gc.collect()
        torch.cuda.empty_cache()

    # Add full-rank (from existing xval checkpoint if available, or train fresh)
    logger.info(">>> Full-rank")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=False, local_files_only=True)
    dev = next(base.parameters()).device
    base.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in base.parameters() if p.requires_grad)

    # Use same training config
    opt = torch.optim.AdamW(base.parameters(), lr=1e-4, weight_decay=0.01)
    base.train()
    step, acc = 0, 0
    while step < 100:
        for b in tr_dl:
            b = {k: v.to(dev) for k, v in b.items()}
            (base(**b).loss / 4).backward()
            acc += 1
            if acc >= 4:
                torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                step += 1; acc = 0
                if step >= 100: break

    c4p = ppl(base, dl)
    results.append({"model": "full-rank", "trainable_M": round(n_params / 1e6, 1), "c4_ppl": c4p})
    logger.info("  C4 PPL=%.2f", c4p)

    # ── M-index computation ──
    import os, math
    os.makedirs(OUT, exist_ok=True)

    # WT2 PPL reference values (from xval, matching config):
    wt2_ref = {"r8": 1.62, "r32": 1.60, "r256": 1.61, "r512": 1.64, "full-rank": 44.4}
    N_data = 800

    logger.info("\n" + "=" * 60)
    logger.info("M-INDEX RESULTS")
    logger.info("%-12s %8s %8s %8s %8s %10s",
                 "Model", "Np(M)", "WT2 PPL", "C4 PPL", "M-index", "log10(Np/Nd)")
    for r in results:
        name = r["model"]
        np_val = r["trainable_M"]
        c4p = r["c4_ppl"]
        wt2 = wt2_ref.get(f"r{name}" if name.startswith("r") else name.replace("-",""), wt2_ref.get(name, 0))
        m_idx = round(wt2 / c4p, 2) if wt2 and c4p else 0
        log_np_nd = round(math.log10(max(np_val * 1e6 / N_data, 1)), 2)
        logger.info("%-12s %8d %8.2f %8.2f %8.2f %10.2f",
                     name, np_val, wt2, c4p, m_idx, log_np_nd)

    # Refine β: log M = log k + β·log(N_d/N_p)
    # Using all points where WT2 and C4 values are valid
    xs, ys = [], []
    for r in results:
        np_val = r["trainable_M"] * 1e6
        if np_val == 0: continue
        wt2 = wt2_ref.get(r["model"], 0)
        c4p = r["c4_ppl"]
        if wt2 == 0: continue
        M = wt2 / c4p
        xs.append(math.log(N_data / np_val))
        ys.append(math.log(M))

    if len(xs) >= 2:
        import numpy as np
        xs_a = np.array(xs)
        ys_a = np.array(ys)
        beta = sum(xs_a * ys_a) / sum(xs_a * xs_a)
        log_k = ys_a.mean() - beta * xs_a.mean()
        k_new = math.exp(log_k)
        logger.info("\nREFINED M-index parameters (from %d points):", len(xs))
        logger.info("  β_new = %.4f (vs β_old = 0.28)", beta)
        logger.info("  k_new = %.1f (vs k_old = 37)", k_new)

    with open(f"{OUT}/m_index_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
