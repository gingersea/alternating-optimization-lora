#!/usr/bin/env python3
"""
Parameter-matched LoRA baseline on Qwen2.5-0.5B.

Runs AdamW+LoRA (Protocol D) with high-rank settings (r=256, r=512)
to isolate parameter form effects from parameter count effects.

Key comparisons:
  - r=8  (~3M params)  vs r=256 (~36M) vs r=512 (~72M) vs full-rank (~494M)
  - All with AdamW optimizer, identical step budgets
  - Tests whether PPL scales with param count or exhibits a form effect

Based on review feedback: "add a parameter-matched LoRA baseline to isolate
form from count" (Round 6 R3, severity 9/10).
"""

import json
import logging
import sys
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("param-matched")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

DATASET_NAME = "wikitext-2-raw-v1"
MAX_SEQ_LEN = 2048
BATCH_SIZE = 2
GRAD_ACCUM = 8         # effective batch = 16
N_TRAIN = 1600
N_EVAL = 200
LR = 1e-4
WEIGHT_DECAY = 0.01

LORA_RANKS = [256, 512]
LORA_ALPHA_RATIO = 2.0   # alpha = ratio * r
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

STEPS_LIST = [100, 200, 400]
SEED = 42
OUT_DIR = Path("runs/param_matched_baseline")


def count_lora_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_dataloader(tokenizer, split, n_samples):
    ds = load_dataset("wikitext", DATASET_NAME, split=split)
    ds = ds.select(range(min(n_samples, len(ds))))

    def tokenize(ex):
        return tokenizer(
            ex["text"], truncation=True,
            max_length=MAX_SEQ_LEN, padding="max_length"
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def collate(batch):
        ids = torch.stack([x["input_ids"] for x in batch])
        mask = torch.stack([x["attention_mask"] for x in batch])
        return {"input_ids": ids, "attention_mask": mask, "labels": ids.clone()}

    return DataLoader(
        tokenized, batch_size=BATCH_SIZE,
        shuffle=(split == "train"), collate_fn=collate,
    )


def compute_perplexity(model, eval_dl, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in eval_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            n_tokens = batch["attention_mask"].sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return torch.exp(torch.tensor(avg_loss)).item()


def run_experiment(model, train_dl, eval_dl, max_steps, label, device):
    """Train with AdamW and evaluate at checkpoints."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    model.train()
    model.gradient_checkpointing_enable()

    global_step = 0
    accumulation_counter = 0
    results = []

    while global_step < max_steps:
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            accumulation_counter += 1

            if accumulation_counter >= GRAD_ACCUM:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                accumulation_counter = 0

                if global_step >= max_steps:
                    break

    # Final evaluation
    ppl = compute_perplexity(model, eval_dl, device)
    logger.info("  %s: step=%d, ppl=%.2f", label, max_steps, ppl)
    return ppl


def main():
    logger.info("=" * 60)
    logger.info("Parameter-Matched LoRA Baseline: Qwen2.5-0.5B")
    logger.info("Ranks: %s | Steps: %s", LORA_RANKS, STEPS_LIST)
    logger.info("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ── Load tokenizer once ──
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=False, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dl = build_dataloader(tokenizer, "train", N_TRAIN)
    eval_dl = build_dataloader(tokenizer, "test", N_EVAL)

    # ── For each rank and step count ──
    for rank in LORA_RANKS:
        alpha = int(rank * LORA_ALPHA_RATIO)
        label = f"LoRA_r{rank}"

        logger.info("\n--- %s (alpha=%d) ---", label, alpha)

        for max_steps in STEPS_LIST:
            exp_label = f"{label}_s{max_steps}"
            logger.info("Training %s...", exp_label)
            t0 = time.time()

            try:
                # Load fresh base model for each run (PEFT modifies in-place)
                logger.info("  Loading base model...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, torch_dtype=torch.bfloat16,
                    device_map="auto", trust_remote_code=False,
                    local_files_only=True,
                )
                device = next(base_model.parameters()).device

                lora_config = LoraConfig(
                    r=rank, lora_alpha=alpha, lora_dropout=0.05,
                    target_modules=TARGET_MODULES,
                    task_type=TaskType.CAUSAL_LM,
                )
                model = get_peft_model(base_model, lora_config)
                n_trainable = count_lora_params(model)
                logger.info("  Trainable: %.1fM params", n_trainable / 1e6)

                ppl = run_experiment(model, train_dl, eval_dl,
                                     max_steps, exp_label, device)
                elapsed = time.time() - t0
                logger.info("  DONE: ppl=%.2f, time=%.0fs", ppl, elapsed)

                result = {
                    "model": "Qwen2.5-0.5B",
                    "rank": rank, "alpha": alpha,
                    "trainable_params": n_trainable,
                    "max_steps": max_steps,
                    "perplexity": ppl, "wall_time_s": elapsed,
                }
                all_results[exp_label] = result

            except Exception as e:
                logger.error("  FAILED: %s", e, exc_info=True)
                all_results[exp_label] = {"rank": rank, "steps": max_steps,
                                          "error": str(e)}
            finally:
                del base_model
                try:
                    del model
                except Exception:
                    pass
                torch.cuda.empty_cache()

    # ── Save results ──
    out_file = OUT_DIR / "results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nResults saved: %s", out_file)

    # ── Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Parameter-Matched LoRA on Qwen2.5-0.5B")
    logger.info("-" * 60)
    logger.info("%-20s %6s %10s", "Experiment", "Steps", "PPL")
    for key in sorted(all_results):
        r = all_results[key]
        if "error" not in r:
            logger.info("%-20s %6d %10.2f", key, r["max_steps"], r["perplexity"])

    # Reference values from paper
    logger.info("\nReference (from paper Table 1, 100 steps):")
    logger.info("  Protocol B (full-rank, 494M):   44.4")
    logger.info("  Protocol D (LoRA r=8, ~3M):     32.2")
    logger.info("  LoRA r=256 (~36M):               %s",
                 all_results.get("LoRA_r256_s100", {}).get("perplexity", "N/A"))
    logger.info("  LoRA r=512 (~72M):               %s",
                 all_results.get("LoRA_r512_s100", {}).get("perplexity", "N/A"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
