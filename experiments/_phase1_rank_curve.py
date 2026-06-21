#!/usr/bin/env python3
"""
Phase 1: Rank Curve Completion — r=16/32/64/128 on Qwen2.5-0.5B
Fills the critical gap between r=8 (3M) and r=256 (35M).
AdamW, 100 steps, seed 42, WikiText-2 eval, same config as param_matched_baseline.
"""

import json, logging, sys, time, gc
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rank-curve")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
RANKS = [16, 32, 64, 128]
ALPHA_RATIO = 2.0  # α = 2·r
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
MAX_SEQ_LEN, BATCH_SIZE, GRAD_ACCUM = 1024, 1, 4
N_TRAIN, N_EVAL, LR, MAX_STEPS = 800, 100, 1e-4, 100
SEED = 42
OUT_DIR = Path("runs/rank_curve")
OUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)


def build_dataloader(tokenizer, split, n_samples):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    ds = ds.select(range(min(n_samples, len(ds))))

    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def collate(b):
        ids = torch.stack([x["input_ids"] for x in b])
        mask = torch.stack([x["attention_mask"] for x in b])
        return {"input_ids": ids, "attention_mask": mask, "labels": ids.clone()}
    return DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=(split == "train"), collate_fn=collate)


def ppl_eval(model, dl, device):
    model.eval(); tl = 0.0; tt = 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            lo = model(**b).loss; nt = b["attention_mask"].sum().item()
            tl += lo.item() * nt; tt += nt
    return float(torch.exp(torch.tensor(tl / max(tt, 1))).item())


def train_and_eval(label, rank, alpha):
    logger.info(">>> %s (r=%d, α=%d)", label, rank, alpha)
    t0 = time.time()

    # Load base model
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=False, local_files_only=True)
    device = next(base.parameters()).device

    # Apply LoRA
    lora_cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05,
                          target_modules=TARGET_MODULES)
    model = get_peft_model(base, lora_cfg)
    model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Trainable: %.1fM", n_params / 1e6)

    # Train
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=0.01)
    train_dl = build_dataloader(tokenizer, "train", N_TRAIN)
    eval_dl = build_dataloader(tokenizer, "test", N_EVAL)

    step, acc = 0, 0
    model.train()
    while step < MAX_STEPS:
        for b in train_dl:
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss / GRAD_ACCUM
            loss.backward(); acc += 1
            if acc >= GRAD_ACCUM:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                step += 1; acc = 0
                if step >= MAX_STEPS: break

    ppl = ppl_eval(model, eval_dl, device)
    elapsed = time.time() - t0
    logger.info("  DONE: PPL=%.4f (%.0fs)", ppl, elapsed)

    result = {"rank": rank, "alpha": alpha, "trainable_params_M": round(n_params / 1e6, 1),
              "ppl": round(ppl, 4), "wall_time_s": int(elapsed), "max_steps": MAX_STEPS}

    del model, base; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    logger.info("=" * 60)
    logger.info("Phase 1: Rank Curve Completion — r=16/32/64/128 on Qwen2.5-0.5B")
    logger.info("=" * 60)

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    results = []
    for rank in RANKS:
        alpha = int(rank * ALPHA_RATIO)
        r = train_and_eval(f"r{rank}", rank, alpha)
        results.append(r)

    out_file = OUT_DIR / "phase1_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nSaved: %s", out_file)

    # Summary
    logger.info("\n=== Rank Curve (Qwen2.5-0.5B, AdamW, 100 steps) ===")
    logger.info("%-10s %10s %10s", "Config", "Params(M)", "PPL")
    for r in results:
        logger.info("r=%-7d %10.1f %10.4f", r["rank"], r["trainable_params_M"], r["ppl"])

    # Reminder of existing data
    logger.info("\nPreviously measured (same config):")
    logger.info("  r=8:    ~3M, PPL=32.2")
    logger.info("  r=256: 34.6M, PPL=1.61")
    logger.info("  r=512: 69.2M, PPL=1.64")
    logger.info("  full: 494M, PPL=44.4")


if __name__ == "__main__":
    sys.exit(main())
