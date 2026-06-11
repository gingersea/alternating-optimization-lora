"""
Experiment #004: Ablation + Reproducibility

Phase 1 — Run all RQ2-RQ6 with lightweight settings to verify correctness.
Phase 2 — Re-run RQ6 (ALS:SGD ratio) with seed=123 to check reproducibility.
Phase 3 — Analyze results for anomalies.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.trainer import AltOptTrainer, TrainerConfig, TrainerState
from altopt.framework import Phase, PhaseConfig, PhaseSchedule
from altopt.evaluation import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("exp004")


def make_dataloader(dataset_name, tokenizer, split, max_len, batch_size, n_samples=None):
    dataset = load_dataset("wikitext", dataset_name, split=split)
    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=max_len,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attn = torch.stack([item["attention_mask"] for item in batch])
        return {"input_ids": input_ids, "attention_mask": attn, "labels": input_ids.clone()}

    return DataLoader(tokenized, batch_size=batch_size, shuffle=(split == "train"),
                      collate_fn=collate_fn)


def run_protocol(model, tokenizer, train_dl, eval_dl, protocol_label, overrides, n_steps):
    lora_target = overrides.get("lora_target_modules")
    cfg = TrainerConfig(
        protocol=protocol_label,
        optimizer_type=overrides.get("optimizer_type", "altopt"),
        parameter_form=overrides.get("parameter_form", "full_rank"),
        max_steps=n_steps,
        lr=overrides.get("lr", 1e-4),
        lora_r=overrides.get("lora_r", 8),
        lora_alpha=overrides.get("lora_alpha", 16.0),
        lora_target_modules=lora_target,
        run_dir=f"/tmp/exp004_{overrides.get('label', 'tmp')}",
        seed=overrides.get("seed", 42),
        eval_every=10000,
        save_every=10000,
    )
    if overrides.get("phase_schedule"):
        cfg.phase_schedule = overrides["phase_schedule"]

    trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_dl, tokenizer=tokenizer)
    state = trainer.train(train_dl)

    eval_result = Evaluator(["perplexity", "loss"], eval_dl).evaluate(model)

    train_losses = [
        l for i, l in enumerate(state.loss_history)
        if i >= len(getattr(state, 'loss_types', [])) or
        getattr(state, 'loss_types', [])[i] != 'noise_energy'
    ]
    final_train_loss = train_losses[-1] if train_losses else float("inf")

    return {
        "protocol": protocol_label,
        "label": overrides.get("label", "?"),
        "final_train_loss": final_train_loss,
        "final_eval_loss": eval_result.get("loss", float("inf")),
        "final_perplexity": eval_result.get("perplexity", float("inf")),
        "total_flops": state.cumulative_flops,
        "peak_memory_mb": state.peak_memory_mb,
        "elapsed_seconds": state.elapsed_seconds,
        "loss_history": state.loss_history,
        "n_steps": state.step,
    }


def detect_lora_modules(model_name):
    if "gpt2" in model_name.lower():
        return None  # GPT-2 uses Conv1D, not compatible with standard LoRA
    if "opt" in model_name.lower():
        return ["q_proj", "v_proj", "k_proj", "out_proj"]
    if "llama" in model_name.lower() or "mistral" in model_name.lower():
        return ["q_proj", "v_proj", "k_proj", "o_proj"]
    return ["q_proj", "v_proj", "k_proj", "o_proj"]


def supports_lora(model_name):
    return detect_lora_modules(model_name) is not None


def run_phase1(model_name, dataset_name, max_len, batch_size, n_samples, n_steps):
    """
    Phase 1: Run all RQs with lightweight settings.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dl = make_dataloader(dataset_name, tokenizer, "train", max_len, batch_size, n_samples)
    eval_dl = make_dataloader(dataset_name, tokenizer, "test", max_len, batch_size, n_samples=min(n_samples, 20))

    lora_modules = detect_lora_modules(model_name)
    has_lora = lora_modules is not None

    results = {}

    for rq_id, runs in [
        ("RQ2", [
            {"label": "A_altopt_full", "protocol": "A", "optimizer_type": "altopt", "parameter_form": "full_rank"},
            {"label": "B_adamw_full", "protocol": "B", "optimizer_type": "adamw", "parameter_form": "full_rank"},
        ]),
        ("RQ3", [
            {"label": "A_with_perturb", "protocol": "A", "optimizer_type": "altopt", "parameter_form": "full_rank",
             "phase_schedule": PhaseSchedule(phases=[
                 PhaseConfig(phase=Phase.ALS, steps=1, block_size=256),
                 PhaseConfig(phase=Phase.SGD, steps=10, lr=1e-4),
                 PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
             ], cycles=1)},
            {"label": "A_no_perturb", "protocol": "A", "optimizer_type": "altopt", "parameter_form": "full_rank",
             "phase_schedule": PhaseSchedule(phases=[
                 PhaseConfig(phase=Phase.ALS, steps=1, block_size=256),
                 PhaseConfig(phase=Phase.SGD, steps=11, lr=1e-4),
             ], cycles=1)},
        ]),
        ("RQ4", [
            {"label": "A", "protocol": "A", "optimizer_type": "altopt", "parameter_form": "full_rank"},
            {"label": "B", "protocol": "B", "optimizer_type": "adamw", "parameter_form": "full_rank"},
            {"label": "C", "protocol": "C", "optimizer_type": "altopt", "parameter_form": "lora",
             "lora_target_modules": lora_modules, "skip_if_no_lora": True},
            {"label": "D", "protocol": "D", "optimizer_type": "adamw", "parameter_form": "lora",
             "lora_target_modules": lora_modules, "skip_if_no_lora": True},
        ]),
        ("RQ5", [
            {"label": "C_lora_altopt", "protocol": "C", "optimizer_type": "altopt", "parameter_form": "lora",
             "lora_target_modules": lora_modules, "skip_if_no_lora": True,
             "phase_schedule": PhaseSchedule(phases=[
                 PhaseConfig(phase=Phase.SGD, steps=10, lr=1e-4),
                 PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=5e-4),
             ], cycles=1)},
            {"label": "D_lora_adamw", "protocol": "D", "optimizer_type": "adamw", "parameter_form": "lora",
             "lora_target_modules": lora_modules, "skip_if_no_lora": True},
        ]),
    ]:
        rq_results = []
        for run_cfg in runs:
            label = run_cfg["label"]

            if run_cfg.pop("skip_if_no_lora", False) and not has_lora:
                logger.info("=== %s / %s === SKIPPED (model %s does not support LoRA)",
                            rq_id, label, model_name)
                rq_results.append({"label": label, "skipped": True,
                                   "reason": f"Model {model_name} does not support standard LoRA (Conv1D architecture)"})
                continue

            logger.info("=== %s / %s ===", rq_id, label)
            t0 = time.time()

            model = AutoModelForCausalLM.from_pretrained(model_name)
            try:
                r = run_protocol(model, tokenizer, train_dl, eval_dl, run_cfg["protocol"],
                                 run_cfg, n_steps=n_steps)
                r["wall_time"] = time.time() - t0
                rq_results.append(r)
                logger.info("%s/%s: train_loss=%.4f ppl=%.2f flops=%.2e time=%.0fs",
                            rq_id, label, r["final_train_loss"], r["final_perplexity"],
                            r["total_flops"], r["wall_time"])
            except Exception as e:
                logger.error("%s/%s FAILED: %s", rq_id, label, e)
                rq_results.append({"label": label, "error": str(e)})

        results[rq_id] = rq_results

    return results


def check_anomalies(results):
    issues = []

    for rq_id, rq_data in results.items():
        valid = [r for r in rq_data if "error" not in r and not r.get("skipped")]
        losses = [r.get("final_train_loss") or r.get("final_loss", float("inf")) for r in valid]
        ppls = [r.get("final_perplexity") for r in valid if r.get("final_perplexity") != float("inf")]

        if not valid:
            continue

        if losses and any(l == float("inf") or (isinstance(l, float) and l > 1e6) for l in losses):
            issues.append(f"{rq_id}: extreme train loss values (inf or >1e6) — insufficient steps for convergence")
        if ppls and max(ppls) / max(min(ppls), 1) > 5000:
            issues.append(f"{rq_id}: perplexity range too large ({min(ppls):.0f}-{max(ppls):.0f}) — possible divergence or insufficient steps")

    return issues


def run_phase2_reproducibility(model_name, dataset_name, max_len, batch_size, n_samples, n_steps):
    """
    Phase 2: Re-run RQ6 (ALS:SGD ratio) with seed=123 to check reproducibility.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dl = make_dataloader(dataset_name, tokenizer, "train", max_len, batch_size, n_samples)
    eval_dl = make_dataloader(dataset_name, tokenizer, "test", max_len, batch_size, n_samples=min(n_samples, 20))

    ratios = [(1, 10), (1, 20), (1, 50)]
    results_seed42 = {}
    results_seed123 = {}

    for seed in [42, 123]:
        seed_results = {}
        for als_s, sgd_s in ratios:
            per_cycle = als_s + sgd_s
            n_cycles = max(1, n_steps // per_cycle)
            actual_steps = n_cycles * per_cycle
            ratio_key = f"1:{sgd_s}"

            schedule = PhaseSchedule(
                phases=[
                    PhaseConfig(phase=Phase.ALS, steps=als_s, block_size=256),
                    PhaseConfig(phase=Phase.SGD, steps=sgd_s, lr=1e-4),
                ],
                cycles=n_cycles,
            )

            model = AutoModelForCausalLM.from_pretrained(model_name)
            logger.info("RQ6 seed=%d ratio=%s (steps=%d)", seed, ratio_key, actual_steps)
            t0 = time.time()

            try:
                r = run_protocol(model, tokenizer, train_dl, eval_dl, "A",
                                 {"label": f"rq6_s{seed}_{ratio_key}", "phase_schedule": schedule,
                                  "seed": seed},
                                 n_steps=actual_steps)
                r["wall_time"] = time.time() - t0
                r["seed"] = seed
                seed_results[ratio_key] = r
                logger.info("RQ6 seed=%d ratio=%s: ppl=%.2f flops=%.2e",
                            seed, ratio_key, r["final_perplexity"], r["total_flops"])
            except Exception as e:
                logger.error("RQ6 seed=%d ratio=%s FAILED: %s", seed, ratio_key, e)
                seed_results[ratio_key] = {"error": str(e), "seed": seed}

        if seed == 42:
            results_seed42 = seed_results
        else:
            results_seed123 = seed_results

    # Compute cross-seed deltas
    delta = {}
    for ratio_key in results_seed42:
        if ratio_key in results_seed123:
            p1 = results_seed42[ratio_key].get("final_perplexity", float("inf"))
            p2 = results_seed123[ratio_key].get("final_perplexity", float("inf"))
            delta[ratio_key] = {
                "ppl_seed42": p1,
                "ppl_seed123": p2,
                "delta_ppl": p2 - p1,
                "delta_pct": (abs(p2 - p1) / max(abs(p1), 1)) * 100 if p1 != float("inf") else float("inf"),
            }

    return {
        "seed_42": results_seed42,
        "seed_123": results_seed123,
        "cross_seed_delta": delta,
    }


def main():
    model_name = "gpt2"
    dataset_name = "wikitext-2-raw-v1"
    max_len = 64
    batch_size = 1
    n_samples_train = 20
    n_steps_per_protocol = 12
    output_dir = Path("runs/exp_004")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1
    logger.info("=" * 60)
    logger.info("PHASE 1: Run RQ2-RQ5 (lightweight verification)")
    logger.info("=" * 60)
    phase1_results = run_phase1(model_name, dataset_name, max_len, batch_size,
                                 n_samples_train, n_steps_per_protocol)

    with open(output_dir / "phase1_results.json", "w") as f:
        json.dump(phase1_results, f, indent=2, default=str)

    issues = check_anomalies(phase1_results)
    if issues:
        logger.warning("ANOMALIES DETECTED:")
        for issue in issues:
            logger.warning("  - %s", issue)
    else:
        logger.info("No anomalies detected in Phase 1.")

    # Phase 2
    logger.info("=" * 60)
    logger.info("PHASE 2: Reproducibility check (RQ6, seed=42 vs seed=123)")
    logger.info("=" * 60)
    phase2_results = run_phase2_reproducibility(model_name, dataset_name, max_len, batch_size,
                                                  n_samples_train, n_steps_per_protocol)

    with open(output_dir / "phase2_reproducibility.json", "w") as f:
        json.dump(phase2_results, f, indent=2, default=str)

    for ratio_key, d in phase2_results["cross_seed_delta"].items():
        logger.info("RQ6 %s: seed42=%.2f seed123=%.2f delta=%.2f (%.1f%%)",
                    ratio_key, d["ppl_seed42"], d["ppl_seed123"],
                    d["delta_ppl"], d["delta_pct"])

    # Summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT #004 SUMMARY")
    logger.info("=" * 60)

    for rq_id, rq_data in phase1_results.items():
        for r in rq_data:
            if r.get("skipped"):
                logger.info("%s/%s: SKIPPED — %s", rq_id, r.get("label", "?"), r.get("reason", "unknown"))
            elif "error" in r:
                logger.warning("%s/%s: FAILED — %s", rq_id, r.get("label", "?"), r["error"])
            else:
                logger.info("%s/%s: ppl=%.2f, flops=%.2e, time=%.0fs",
                            rq_id, r.get("label", "?"), r["final_perplexity"],
                            r["total_flops"], r.get("wall_time", 0))

    logger.info("Results saved to %s/", output_dir)
    return phase1_results, phase2_results


if __name__ == "__main__":
    main()
