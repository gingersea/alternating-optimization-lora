#!/usr/bin/env python3
"""
B3: Qwen2.5-7B 2×2 Factorial Experiment (DeepSpeed ZeRO-2, 2×RTX 5090)

Protocols:
  A: AltOpt + Full-Rank  (ASP, our method)
  B: AdamW + Full-Rank   (traditional full fine-tune)
  C: AltOpt + LoRA        (ASP on low-rank parameters)
  D: AdamW + LoRA         (standard LoRA, dominant baseline)

Memory budget: ~43 GB total, ~21.5 GB/GPU with ZeRO-2
Step budget: 800 steps for matrix, >2000 for crossover test
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

from altopt.trainer import AltOptTrainer, TrainerConfig
from altopt.framework import Phase, PhaseConfig, PhaseSchedule
from altopt.evaluation import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("B3-7B")

# ── Configuration ──
MODEL_NAME = "Qwen/Qwen2.5-7B"
DATASET_NAME = "wikitext-2-raw-v1"
MAX_SEQ_LEN = 2048
BATCH_SIZE = 2
GRAD_ACCUM = 8              # effective batch = 16
N_TRAIN = 1600              # training samples
N_EVAL = 200                # eval samples
MAX_STEPS = 800             # target steps
EVAL_EVERY = 100
SAVE_EVERY = 200
SEEDS = [42, 123, 456]      # multi-seed
OUT_DIR = Path("runs/qwen25_7b_800s")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DeepSpeed config
DEEPSPEED_CFG = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": GRAD_ACCUM,
    "gradient_clipping": 1.0,
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"},
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    },
    "wall_clock_breakdown": False,
}


def build_dataloader(tokenizer, split, max_len, batch_size, n_samples=None):
    """Build tokenized dataloader for WikiText-2."""
    ds = load_dataset("wikitext", DATASET_NAME, split=split)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    def tokenize(ex):
        return tokenizer(
            ex["text"], truncation=True,
            max_length=max_len, padding="max_length"
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def collate(batch):
        ids = torch.stack([x["input_ids"] for x in batch])
        mask = torch.stack([x["attention_mask"] for x in batch])
        return {"input_ids": ids, "attention_mask": mask, "labels": ids.clone()}

    return DataLoader(
        tokenized, batch_size=batch_size,
        shuffle=(split == "train"), collate_fn=collate
    )


def build_altopt_schedule(n_steps: int, param_form: str = "full_rank") -> PhaseSchedule:
    """Build ASP phase schedule for n_steps total.

    For full_rank: ALS -> SGD -> Perturb cycles.
    For lora: SGD -> Perturb only (ALS targets full-rank modules,
    incompatible with LoRA parameterization).
    """
    sgd_per_cycle = max(10, n_steps // 4)
    # Account for phases per cycle when computing n_cycles
    if param_form == "lora":
        n_phases_per_cycle = 2  # SGD + Perturb
    else:
        n_phases_per_cycle = 3  # ALS + SGD + Perturb
    n_cycles = max(1, n_steps // (sgd_per_cycle + n_phases_per_cycle))
    phases = []
    if param_form != "lora":
        phases.append(PhaseConfig(phase=Phase.ALS, steps=1, block_size=2048))
    phases.append(PhaseConfig(phase=Phase.SGD, steps=sgd_per_cycle, lr=5e-5))
    phases.append(PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=5e-4))
    return PhaseSchedule(phases=phases, cycles=n_cycles)


def run_protocol(protocol_label, opt_type, param_form, seed, n_steps):
    """Run a single protocol × seed combination."""
    label = f"Qwen25-7B_P{protocol_label}_{n_steps}s_s{seed}"
    logger.info("=" * 60)
    logger.info(f"Running: {label} (optimizer={opt_type}, param_form={param_form})")
    logger.info("=" * 60)

    # Load model (fresh for each run to avoid state leakage)
    logger.info("Loading model: %s", MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
        local_files_only=True,
    )
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False,
                                              local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dl = build_dataloader(tokenizer, "train", MAX_SEQ_LEN, BATCH_SIZE, N_TRAIN)
    eval_dl = build_dataloader(tokenizer, "test", MAX_SEQ_LEN, BATCH_SIZE, N_EVAL)

    use_ds = (param_form == "full_rank")  # DeepSpeed only for full-rank 7B

    config = TrainerConfig(
        protocol=protocol_label,
        optimizer_type=opt_type,
        parameter_form=param_form,
        max_steps=n_steps,
        lr=5e-5 if param_form == "full_rank" else 1e-4,
        run_dir=str(OUT_DIR / f"ckpt_{label}"),
        seed=seed,
        eval_every=EVAL_EVERY,
        save_every=SAVE_EVERY,
        use_deepspeed=use_ds,
        deepspeed_zero_stage=2,
        deepspeed_bf16=True,
        gradient_accumulation_steps=GRAD_ACCUM,
        lora_r=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    if opt_type == "altopt":
        config.phase_schedule = build_altopt_schedule(n_steps, param_form)

    try:
        t0 = time.time()
        trainer = AltOptTrainer(model, config, eval_dataloader=eval_dl, tokenizer=tokenizer)
        state = trainer.train(train_dl)
        elapsed = time.time() - t0

        # Evaluate
        evaluator = Evaluator(["perplexity", "loss"], eval_dl)
        metrics = evaluator.evaluate(model)

        result = {
            "model": "Qwen2.5-7B",
            "protocol": protocol_label,
            "optimizer": opt_type,
            "parameter_form": param_form,
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
        logger.info(f"  SUCCESS: ppl={metrics.get('perplexity', 'N/A')}, "
                     f"time={elapsed:.0f}s, mem={result['peak_memory_mb']}MB")

    except Exception as e:
        logger.error(f"  FAIL: {e}", exc_info=True)
        result = {
            "model": "Qwen2.5-7B",
            "protocol": protocol_label,
            "optimizer": opt_type,
            "parameter_form": param_form,
            "seed": seed,
            "n_steps": n_steps,
            "status": "failed",
            "error": str(e),
        }
    finally:
        # Aggressive cleanup to free GPU memory between runs
        import gc
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Save individual result
    out_file = OUT_DIR / f"{label}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    """Run all 4 protocols × N seeds."""
    logger.info("=" * 70)
    logger.info("B3: Qwen2.5-7B 2×2 Factorial (800 steps, DeepSpeed ZeRO-2)")
    logger.info(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"  GPU {i}: {props.name}, {props.total_memory/1e9:.1f}GB")
    logger.info("=" * 70)

    # Full-rank protocols (A, B) need DeepSpeed ZeRO-2 (2 GPUs).
    # LoRA protocols (C, D) have only ~590K trainable params, single GPU suffices.
    protocols = [
        ("A", "altopt", "full_rank"),
        ("B", "adamw", "full_rank"),
        ("C", "altopt", "lora"),
        ("D", "adamw", "lora"),
    ]

    all_results = []
    for proto_label, opt_type, param_form in protocols:
        for seed in SEEDS:
            result = run_protocol(proto_label, opt_type, param_form, seed, MAX_STEPS)
            all_results.append(result)

    # Summary
    success = [r for r in all_results if r.get("status") == "success"]
    failed = [r for r in all_results if r.get("status") != "success"]
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {len(success)} success, {len(failed)} failed")
    logger.info(f"{'='*60}")

    # Print summary table
    for proto_label in ["A", "B", "C", "D"]:
        proto_results = [r for r in success if r["protocol"] == proto_label]
        if proto_results:
            ppls = [r["perplexity"] for r in proto_results]
            logger.info(f"  Protocol {proto_label}: "
                         f"ppl={np.mean(ppls):.1f}±{np.std(ppls):.1f} "
                         f"(N={len(ppls)})")

    # Save combined
    combined_file = OUT_DIR / "combined_results.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {combined_file}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())