"""
Experiment runner — thin CLI wrapper around AltOptTrainer.

Executes the 2x2 factorial comparison protocol (A/B/C/D) by constructing
a TrainerConfig for each protocol and running them sequentially.

The heavy lifting moved to altopt/trainer.py. This file is now a
configuration + execution entry point.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.trainer import AltOptTrainer, TrainerConfig, TrainerState

logger = logging.getLogger(__name__)

PROTOCOL_LABELS = {
    "A": ("altopt", "full_rank"),
    "B": ("adamw", "full_rank"),
    "C": ("altopt", "lora"),
    "D": ("adamw", "lora"),
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataloader(dataset_name: str, tokenizer, split: str, max_length: int, batch_size: int) -> DataLoader:
    dataset = load_dataset(dataset_name, split=split)
    tokenized = dataset.map(
        lambda examples: tokenizer(
            examples["text"], truncation=True, max_length=max_length, padding="max_length"
        ),
        batched=True,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(tokenized, batch_size=batch_size, shuffle=(split == "train"))


def run_single_protocol(
    label: str,
    model_name: str,
    dataset_name: str,
    max_seq_length: int,
    batch_size: int,
    eval_batch_size: int,
    total_budget_flops: float,
    run_dir: str,
    max_steps: Optional[int],
    seed: int,
) -> TrainerState:
    opt_type, param_form = PROTOCOL_LABELS[label]

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dl = build_dataloader(dataset_name, tokenizer, "train", max_seq_length, batch_size)
    eval_dl = build_dataloader(dataset_name, tokenizer, "test", max_seq_length, eval_batch_size)

    run_path = f"{run_dir}/protocol_{label}"
    os.makedirs(run_path, exist_ok=True)

    config = TrainerConfig(
        protocol=label,
        optimizer_type=opt_type,
        parameter_form=param_form,
        total_budget_flops=total_budget_flops,
        max_steps=max_steps,
        run_dir=run_path,
        seed=seed,
    )

    trainer = AltOptTrainer(model, config, eval_dataloader=eval_dl, tokenizer=tokenizer)
    state = trainer.train(train_dl)

    trainer.export_results()
    return state


def run_all_protocols(config_path: str) -> dict[str, TrainerState]:
    cfg = load_config(config_path)
    base = cfg.get("base", {})
    model_name = base.get("model", "gpt2")
    dataset_name = base.get("dataset", "wikitext-2-raw-v1")
    max_seq_length = base.get("max_seq_length", 512)
    batch_size = base.get("per_device_batch_size", 4)
    eval_batch_size = base.get("eval_batch_size", 8)
    budget_cfg = base.get("total_budget", {})
    total_budget_flops = budget_cfg.get("value", 1e15)
    seed = base.get("seed", 42)
    run_dir = cfg.get("logging", {}).get("local", {}).get("log_dir", "logs/")

    results: dict[str, TrainerState] = {}
    for label in ["A", "B", "C", "D"]:
        logger.info("Running Protocol %s: %s + %s", label, *PROTOCOL_LABELS[label])
        state = run_single_protocol(
            label=label, model_name=model_name, dataset_name=dataset_name,
            max_seq_length=max_seq_length, batch_size=batch_size,
            eval_batch_size=eval_batch_size, total_budget_flops=total_budget_flops,
            run_dir=run_dir, max_steps=100, seed=seed,
        )
        results[label] = state
        logger.info("Protocol %s: loss=%.4f, ppl=%.2f, flops=%.2e, mem=%.0fMB",
                     label, state.best_loss, state.best_perplexity,
                     state.cumulative_flops, state.peak_memory_mb)

    combined = {
        label: {
            "protocol": label,
            "optimizer": PROTOCOL_LABELS[label][0],
            "parameter_form": PROTOCOL_LABELS[label][1],
            "final_loss": state.loss_history[-1] if state.loss_history else None,
            "best_perplexity": state.best_perplexity,
            "total_flops": state.cumulative_flops,
            "peak_memory_mb": state.peak_memory_mb,
        }
        for label, state in results.items()
    }
    with open(Path(run_dir) / "combined_results.json", "w") as f:
        json.dump(combined, f, indent=2)
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config_path = sys.argv[1] if len(sys.argv) > 1 else "experiments/configs/base.yaml"
    results = run_all_protocols(config_path)
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for label, state in results.items():
        print(f"Protocol {label}: loss={state.best_loss:.4f}, ppl={state.best_perplexity:.2f}")
