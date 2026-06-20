#!/usr/bin/env python3
"""
Protocol A — AltOpt + Full-Rank via PyTorch FSDP (torchrun × 2).

Usage:
    torchrun --nproc_per_node=2 -m experiments.run_7b_fsdp <seed> [steps]

Hardware: 2× RTX 5090 (32GB each), 251GB CPU RAM.
Memory: ~23GB peak/GPU (FSDP FULL_SHARD + CPU offload).
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.trainer import AltOptTrainer, TrainerConfig
from altopt.framework import Phase, PhaseConfig, PhaseSchedule
from altopt.evaluation import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("FSDP-A")

MODEL_NAME = "Qwen/Qwen2.5-7B"
DATASET_NAME = "wikitext-2-raw-v1"
MAX_SEQ_LEN = 2048
BATCH_SIZE = 1
GRAD_ACCUM = 16
N_TRAIN = 1600
N_EVAL = 200
EVAL_EVERY = 100
OUT_DIR = Path("runs/qwen25_7b_800s")


def build_dataloader(tokenizer, split, max_len, batch_size, n_samples=None):
    ds = load_dataset("wikitext", DATASET_NAME, split=split)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    def tokenize(ex):
        return tokenizer(
            ex["text"], truncation=True,
            max_length=max_len, padding="max_length",
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def collate(batch):
        ids = torch.stack([x["input_ids"] for x in batch])
        mask = torch.stack([x["attention_mask"] for x in batch])
        return {"input_ids": ids, "attention_mask": mask, "labels": ids.clone()}

    return DataLoader(
        tokenized, batch_size=batch_size,
        shuffle=(split == "train"), collate_fn=collate,
    )


def main():
    # ── Distributed init ──
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Parse args
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 800

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label = f"Qwen25-7B_PA_{n_steps}s_s{seed}"

    logger.info("=" * 60)
    logger.info("Protocol A (AltOpt+Full-Rank) via FSDP")
    logger.info(f"Model: {MODEL_NAME}, Seed: {seed}, Steps: {n_steps}")
    logger.info(f"GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
    logger.info("=" * 60)

    # ── Load model to CPU ──
    logger.info("Loading model to CPU: %s", MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=None,  # CPU — FSDP manages placement
        trust_remote_code=False,
        local_files_only=True,
    )
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=False, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dl = build_dataloader(tokenizer, "train", MAX_SEQ_LEN, BATCH_SIZE, N_TRAIN)
    eval_dl = build_dataloader(tokenizer, "test", MAX_SEQ_LEN, BATCH_SIZE, N_EVAL)

    # ── Build schedule ──
    # 2 cycles × (ALS 1 + SGD 350 + Perturb 1) = 704 steps
    schedule = PhaseSchedule(
        phases=[
            PhaseConfig(phase=Phase.ALS, steps=1, block_size=512),
            PhaseConfig(phase=Phase.SGD, steps=350, lr=5e-5),
            PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=5e-4),
        ],
        cycles=2,
    )

    config = TrainerConfig(
        protocol="A",
        optimizer_type="altopt",
        parameter_form="full_rank",
        max_steps=n_steps,
        lr=5e-5,
        run_dir=str(OUT_DIR / f"ckpt_{label}"),
        seed=seed,
        eval_every=EVAL_EVERY,
        save_every=10000,  # no checkpointing during run
        use_fsdp=True,
        phase_schedule=schedule,
    )

    try:
        t0 = time.time()
        trainer = AltOptTrainer(model, config, eval_dataloader=eval_dl, tokenizer=tokenizer)
        state = trainer.train(train_dl)
        elapsed = time.time() - t0

        # Evaluate (rank 0 only)
        if local_rank == 0:
            evaluator = Evaluator(["perplexity", "loss"], eval_dl)
            metrics = evaluator.evaluate(model)

            result = {
                "model": "Qwen2.5-7B",
                "protocol": "A",
                "optimizer": "altopt",
                "parameter_form": "full_rank",
                "seed": seed,
                "n_steps": n_steps,
                "perplexity": metrics.get("perplexity"),
                "loss": metrics.get("loss"),
                "best_perplexity": state.best_perplexity,
                "best_loss": state.best_loss,
                "flops": state.cumulative_flops if hasattr(state, 'cumulative_flops') else None,
                "peak_memory_mb": state.peak_memory_mb if hasattr(state, 'peak_memory_mb') else None,
                "wall_time_s": elapsed,
                "status": "success",
            }
            logger.info("SUCCESS: ppl=%.2f, time=%.0fs",
                        metrics.get('perplexity', float('nan')), elapsed)

            out_file = OUT_DIR / f"{label}.json"
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)
        else:
            result = {"status": "success"}  # placeholder for non-rank-0

    except Exception as e:
        logger.error("FAIL: %s", e, exc_info=True)
        result = {
            "model": "Qwen2.5-7B",
            "protocol": "A",
            "optimizer": "altopt",
            "parameter_form": "full_rank",
            "seed": seed,
            "n_steps": n_steps,
            "status": "failed",
            "error": str(e),
        }
        if local_rank == 0:
            out_file = OUT_DIR / f"{label}.json"
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)

    # Clean up FSDP
    import gc
    if hasattr(model, "_reset_parameters"):
        pass
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.destroy_process_group()

    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
