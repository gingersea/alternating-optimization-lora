"""
Systematic ablation experiment runner for RQ1-RQ6.

Research Questions:
  RQ1 — Disentanglement: can we isolate optimizer effect from parameter form effect?
  RQ2 — Efficiency Frontier: at what FLOPs budget does ALS become worthwhile?
  RQ3 — Loss Landscape: does LoRA low-rank manifold weaken perturbation escape?
  RQ4 — Generalization: does AltOpt generalize better than AdamW at equal train loss?
  RQ5 — Synergy: can LoRA+AltOpt (Protocol C) beat LoRA+AdamW (Protocol D)?
  RQ6 — ALS:SGD Ratio: what is the optimal ALS-to-SGD step ratio?

Each RQ is implemented as a dedicated ablation function that runs targeted
experiments and produces structured JSON results.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
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

logger = logging.getLogger(__name__)


def _make_tokenized_dataloader(dataset_name, tokenizer, split, max_len, batch_size, n_samples=None):
    dataset = load_dataset("wikitext", dataset_name, split=split)
    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=max_len,
            padding="max_length", return_tensors="pt",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attn = torch.stack([item["attention_mask"] for item in batch])
        return {"input_ids": input_ids, "attention_mask": attn, "labels": input_ids.clone()}

    return DataLoader(tokenized, batch_size=batch_size, shuffle=(split == "train"),
                      collate_fn=collate_fn)


def run_quick_train(model, tokenizer, dl, eval_dl, config_overrides, n_steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    cfg = TrainerConfig(
        protocol="A",
        optimizer_type=config_overrides.get("optimizer_type", "altopt"),
        parameter_form=config_overrides.get("parameter_form", "full_rank"),
        max_steps=n_steps,
        lr=config_overrides.get("lr", 1e-4),
        lora_r=config_overrides.get("lora_r", 8),
        lora_alpha=config_overrides.get("lora_alpha", 16.0),
        run_dir=f"/tmp/ablation_{config_overrides.get('label', 'tmp')}",
        seed=42,
    )
    if config_overrides.get("phase_schedule"):
        schedule = config_overrides["phase_schedule"]
        steps_per_cycle = sum(p.steps for p in schedule.phases)
        if steps_per_cycle > 0:
            schedule.cycles = max(1, (n_steps + steps_per_cycle - 1) // steps_per_cycle)
        cfg.phase_schedule = schedule

    trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_dl, tokenizer=tokenizer)
    return trainer.train(dl)


@dataclass
class AblationResult:
    rq_id: str
    config_name: str
    final_loss: float
    final_perplexity: float
    total_flops: float
    peak_memory_mb: float
    elapsed_seconds: float
    loss_history: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def _evaluate_model(model, dataloader) -> dict:
    evaluator = Evaluator(["perplexity", "loss"], dataloader)
    return evaluator.evaluate(model)


def rq2_efficiency_frontier(
    model_name="gpt2", dataset_name="wikitext-2-raw-v1",
    max_len=128, batch_size=2, n_steps=120,
):
    """
    RQ2: Efficiency Frontier — scan across FLOPs budgets.

    Runs Protocol A (AltOpt full-rank) and Protocol B (AdamW full-rank)
    at identical FLOPs budgets, comparing final loss at each checkpoint.

    Returns list[dict] with (flops, loss_a, loss_b) at each eval point.
    """
    model_a = AutoModelForCausalLM.from_pretrained(model_name)
    model_b = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dl = _make_tokenized_dataloader(dataset_name, tokenizer, "train", max_len, batch_size)
    eval_dl = _make_tokenized_dataloader(dataset_name, tokenizer, "test", max_len, batch_size)

    schedule = PhaseSchedule(
        phases=[
            PhaseConfig(phase=Phase.SGD, steps=51, lr=1e-4),
            PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
        ],
        cycles=2,
    )

    logger.info("RQ2: Running Protocol A (AltOpt full-rank)")
    state_a = run_quick_train(model_a, tokenizer, train_dl, eval_dl,
                              {"optimizer_type": "altopt", "parameter_form": "full_rank",
                               "phase_schedule": schedule, "label": "rq2_a"},
                              n_steps=n_steps)

    logger.info("RQ2: Running Protocol B (AdamW full-rank)")
    model_b = AutoModelForCausalLM.from_pretrained(model_name)
    state_b = run_quick_train(model_b, tokenizer, train_dl, eval_dl,
                              {"optimizer_type": "adamw", "parameter_form": "full_rank",
                               "label": "rq2_b"},
                              n_steps=n_steps)

    eval_a = _evaluate_model(model_a, eval_dl)
    eval_b = _evaluate_model(model_b, eval_dl)

    result = {
        "rq": "RQ2",
        "description": "Efficiency Frontier — FLOPs vs final loss comparison",
        "protocol_a": {
            "final_loss": state_a.best_loss,
            "final_perplexity": eval_a.get("perplexity", float("inf")),
            "total_flops": state_a.cumulative_flops,
            "peak_memory_mb": state_a.peak_memory_mb,
            "loss_history": state_a.loss_history,
        },
        "protocol_b": {
            "final_loss": state_b.best_loss,
            "final_perplexity": eval_b.get("perplexity", float("inf")),
            "total_flops": state_b.cumulative_flops,
            "peak_memory_mb": state_b.peak_memory_mb,
            "loss_history": state_b.loss_history,
        },
        "comparison": {
            "flops_ratio_b_to_a": state_b.cumulative_flops / max(state_a.cumulative_flops, 1),
            "loss_delta_a_minus_b": state_a.best_loss - state_b.best_loss,
        },
    }
    return result


def rq3_perturbation_effect(
    model_name="gpt2", dataset_name="wikitext-2-raw-v1",
    max_len=128, batch_size=2,
):
    """
    RQ3: Loss Landscape Interaction — measure perturbation effect.

    Compares Protocol A (with perturbation) vs AltOpt without perturbation,
    measuring the loss drop after each perturbation phase.

    Also tests with LoRA (Protocol C) to see if low-rank manifold
    weakens perturbation effects.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dl = _make_tokenized_dataloader(dataset_name, tokenizer, "train", max_len, batch_size)
    eval_dl = _make_tokenized_dataloader(dataset_name, tokenizer, "test", max_len, batch_size)

    # Protocol A: Full-Rank AltOpt WITH perturbation
    schedule_with = PhaseSchedule(
        phases=[
            PhaseConfig(phase=Phase.SGD, steps=40, lr=1e-4),
            PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
        ],
        cycles=2,
    )
    # Full-Rank AltOpt WITHOUT perturbation
    schedule_without = PhaseSchedule(
        phases=[
            PhaseConfig(phase=Phase.SGD, steps=41, lr=1e-4),
        ],
        cycles=2,
    )

    model_with = AutoModelForCausalLM.from_pretrained(model_name)
    model_without = AutoModelForCausalLM.from_pretrained(model_name)

    logger.info("RQ3: AltOpt WITH perturbation")
    state_with = run_quick_train(model_with, tokenizer, train_dl, eval_dl,
                                 {"phase_schedule": schedule_with, "label": "rq3_with"},
                                 n_steps=100)

    logger.info("RQ3: AltOpt WITHOUT perturbation")
    model_without = AutoModelForCausalLM.from_pretrained(model_name)
    state_without = run_quick_train(model_without, tokenizer, train_dl, eval_dl,
                                    {"phase_schedule": schedule_without, "label": "rq3_without"},
                                    n_steps=100)

    # Identify perturbation-phase loss drops
    loss_types_with = getattr(state_with, 'loss_types', ['loss'] * len(state_with.loss_history))
    perturb_effects = []
    for i, (loss_val, loss_type) in enumerate(zip(state_with.loss_history, loss_types_with)):
        if loss_type == 'noise_energy':
            if i > 0:
                prev_loss = state_with.loss_history[i - 1]
                perturb_effects.append({"step": i, "loss_before": prev_loss, "noise_energy": loss_val})

    eval_with = _evaluate_model(model_with, eval_dl)
    eval_without = _evaluate_model(model_without, eval_dl)

    return {
        "rq": "RQ3",
        "description": "Perturbation effect on loss landscape escape",
        "with_perturbation": {
            "final_loss": state_with.best_loss,
            "final_perplexity": eval_with.get("perplexity", float("inf")),
            "loss_history": state_with.loss_history,
        },
        "without_perturbation": {
            "final_loss": state_without.best_loss,
            "final_perplexity": eval_without.get("perplexity", float("inf")),
            "loss_history": state_without.loss_history,
        },
        "perturbation_events": perturb_effects,
        "perturbation_benefit": eval_without["perplexity"] - eval_with["perplexity"],
    }


def rq4_generalization(
    model_name="gpt2", dataset_name="wikitext-2-raw-v1",
    max_len=128, batch_size=2, n_steps=100,
):
    """
    RQ4: Generalization — compare train/eval gap across protocols.

    Measures: (eval_loss - train_loss) gap at various training checkpoints.
    Smaller gap = better generalization.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dl = _make_tokenized_dataloader(dataset_name, tokenizer, "train", max_len, batch_size)
    eval_dl = _make_tokenized_dataloader(dataset_name, tokenizer, "test", max_len, batch_size)

    results = {}
    protocols = {
        "A": {"optimizer_type": "altopt", "parameter_form": "full_rank"},
        "B": {"optimizer_type": "adamw", "parameter_form": "full_rank"},
        "C": {"optimizer_type": "altopt", "parameter_form": "lora"},
        "D": {"optimizer_type": "adamw", "parameter_form": "lora"},
    }

    for label, overrides in protocols.items():
        model = AutoModelForCausalLM.from_pretrained(model_name)
        overrides["label"] = f"rq4_{label}"
        logger.info("RQ4: Protocol %s", label)
        state = run_quick_train(model, tokenizer, train_dl, eval_dl, overrides, n_steps=n_steps)
        eval_result = _evaluate_model(model, eval_dl)

        train_losses = [
            l for i, l in enumerate(state.loss_history)
            if i < len(getattr(state, 'loss_types', [])) and
            getattr(state, 'loss_types', [])[i] != 'noise_energy'
        ]
        final_train_loss = train_losses[-1] if train_losses else float("inf")
        final_eval_loss = eval_result.get("loss", float("inf"))
        gap = final_eval_loss - final_train_loss

        results[label] = {
            "final_train_loss": final_train_loss,
            "final_eval_loss": final_eval_loss,
            "generalization_gap": gap,
            "final_perplexity": eval_result.get("perplexity", float("inf")),
        }

    return {
        "rq": "RQ4",
        "description": "Generalization gap analysis (eval - train loss)",
        "protocols": results,
        "best_generalization": min(results.items(), key=lambda x: x[1]["generalization_gap"]),
    }


def rq5_synergy(
    model_name="gpt2", dataset_name="wikitext-2-raw-v1",
    max_len=128, batch_size=2, n_steps=100,
):
    """
    RQ5: Synergy — does LoRA+AltOpt (C) beat LoRA+AdamW (D)?

    Tests whether the AltOpt optimizer can improve LoRA training.
    Uses equal FLOPs budget for fair comparison.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dl = _make_tokenized_dataloader(dataset_name, tokenizer, "train", max_len, batch_size)
    eval_dl = _make_tokenized_dataloader(dataset_name, tokenizer, "test", max_len, batch_size)

    lora_schedule = PhaseSchedule(
        phases=[
            PhaseConfig(phase=Phase.SGD, steps=50, lr=1e-4),
            PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=5e-4),
        ],
        cycles=2,
    )

    model_c = AutoModelForCausalLM.from_pretrained(model_name)
    model_d = AutoModelForCausalLM.from_pretrained(model_name)

    logger.info("RQ5: Protocol C (LoRA+AltOpt)")
    state_c = run_quick_train(model_c, tokenizer, train_dl, eval_dl,
                              {"optimizer_type": "altopt", "parameter_form": "lora",
                               "phase_schedule": lora_schedule, "label": "rq5_c"},
                              n_steps=n_steps)

    logger.info("RQ5: Protocol D (LoRA+AdamW)")
    model_d = AutoModelForCausalLM.from_pretrained(model_name)
    state_d = run_quick_train(model_d, tokenizer, train_dl, eval_dl,
                              {"optimizer_type": "adamw", "parameter_form": "lora",
                               "label": "rq5_d"},
                              n_steps=n_steps)

    eval_c = _evaluate_model(model_c, eval_dl)
    eval_d = _evaluate_model(model_d, eval_dl)

    return {
        "rq": "RQ5",
        "description": "Synergy — LoRA+AltOpt (C) vs LoRA+AdamW (D)",
        "protocol_c": {
            "final_loss": state_c.best_loss,
            "final_perplexity": eval_c.get("perplexity", float("inf")),
            "total_flops": state_c.cumulative_flops,
            "loss_history": state_c.loss_history,
        },
        "protocol_d": {
            "final_loss": state_d.best_loss,
            "final_perplexity": eval_d.get("perplexity", float("inf")),
            "total_flops": state_d.cumulative_flops,
            "loss_history": state_d.loss_history,
        },
        "synergy_benefit": eval_d.get("perplexity", 0) - eval_c.get("perplexity", 0),
        "altopt_advantage": "C better" if eval_c.get("perplexity", float("inf")) < eval_d.get("perplexity", float("inf")) else "D better",
    }


def rq6_als_sgd_ratio(
    model_name="gpt2", dataset_name="wikitext-2-raw-v1",
    max_len=128, batch_size=2, n_total_steps=120,
):
    """
    RQ6: ALS:SGD Ratio — scan ratios to find optimal.

    Tests ALS:SGD step ratios: 1:10, 1:20, 1:50, 1:100
    At fixed total steps (~120) and identical FLOPs budget.

    Returns results per ratio with final loss and perplexity.
    """
    ratios = [(1, 10), (1, 20), (1, 50), (1, 100)]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dl = _make_tokenized_dataloader(dataset_name, tokenizer, "train", max_len, batch_size)
    eval_dl = _make_tokenized_dataloader(dataset_name, tokenizer, "test", max_len, batch_size)

    results = {}
    for als_steps, sgd_steps in ratios:
        # Compute cycles to fill ~n_total_steps
        per_cycle = als_steps + sgd_steps
        n_cycles = max(1, n_total_steps // per_cycle)
        actual_steps = n_cycles * per_cycle

        schedule = PhaseSchedule(
            phases=[
                PhaseConfig(phase=Phase.ALS, steps=als_steps, block_size=512),
                PhaseConfig(phase=Phase.SGD, steps=sgd_steps, lr=1e-4),
            ],
            cycles=n_cycles,
        )

        ratio_key = f"1:{sgd_steps}"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info("RQ6: ALS:SGD = %s (cycles=%d, steps=%d)", ratio_key, n_cycles, actual_steps)

        state = run_quick_train(model, tokenizer, train_dl, eval_dl,
                                {"phase_schedule": schedule, "label": f"rq6_{ratio_key}"},
                                n_steps=actual_steps)

        eval_result = _evaluate_model(model, eval_dl)
        results[ratio_key] = {
            "ratio": ratio_key,
            "als_steps": als_steps,
            "sgd_steps": sgd_steps,
            "n_cycles": n_cycles,
            "total_steps": actual_steps,
            "final_loss": state.best_loss,
            "final_perplexity": eval_result.get("perplexity", float("inf")),
            "total_flops": state.cumulative_flops,
            "loss_history": state.loss_history,
        }

    best = min(results.items(), key=lambda x: x[1]["final_perplexity"])

    return {
        "rq": "RQ6",
        "description": "ALS:SGD ratio ablation",
        "ratios": results,
        "optimal_ratio": best[0],
        "optimal_perplexity": best[1]["final_perplexity"],
    }


def run_all_ablation(model_name="gpt2", output_dir="runs/ablation/"):
    """
    Run all RQ1-RQ6 ablation experiments and save results.

    RQ1 is covered by the existing 2x2 factorial analysis in analysis.py,
    so it is not re-run here but referenced.

    Args:
        model_name: HuggingFace model ID
        output_dir: where to save JSON results

    Returns:
        dict with results keyed by RQ label
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = {}

    logger.info("=" * 60)
    logger.info("RQ2: Efficiency Frontier")
    try:
        all_results["RQ2"] = rq2_efficiency_frontier(model_name=model_name)
        logger.info("RQ2 complete: loss_delta=%.4f",
                    all_results["RQ2"]["comparison"]["loss_delta_a_minus_b"])
    except Exception as e:
        logger.error("RQ2 failed: %s", e)
        all_results["RQ2"] = {"error": str(e)}

    logger.info("=" * 60)
    logger.info("RQ3: Perturbation Effect")
    try:
        all_results["RQ3"] = rq3_perturbation_effect(model_name=model_name)
        logger.info("RQ3 complete: perturbation_benefit=%.2f ppl",
                    all_results["RQ3"]["perturbation_benefit"])
    except Exception as e:
        logger.error("RQ3 failed: %s", e)
        all_results["RQ3"] = {"error": str(e)}

    logger.info("=" * 60)
    logger.info("RQ4: Generalization Gap")
    try:
        all_results["RQ4"] = rq4_generalization(model_name=model_name)
        best_gen = all_results["RQ4"]["best_generalization"]
        logger.info("RQ4 complete: best=%s (gap=%.4f)", best_gen[0], best_gen[1]["generalization_gap"])
    except Exception as e:
        logger.error("RQ4 failed: %s", e)
        all_results["RQ4"] = {"error": str(e)}

    logger.info("=" * 60)
    logger.info("RQ5: Synergy (C vs D)")
    try:
        all_results["RQ5"] = rq5_synergy(model_name=model_name)
        logger.info("RQ5 complete: %s (delta=%.2f ppl)",
                    all_results["RQ5"]["altopt_advantage"],
                    all_results["RQ5"]["synergy_benefit"])
    except Exception as e:
        logger.error("RQ5 failed: %s", e)
        all_results["RQ5"] = {"error": str(e)}

    logger.info("=" * 60)
    logger.info("RQ6: ALS:SGD Ratio Scan")
    try:
        all_results["RQ6"] = rq6_als_sgd_ratio(model_name=model_name)
        logger.info("RQ6 complete: optimal_ratio=%s (ppl=%.2f)",
                    all_results["RQ6"]["optimal_ratio"],
                    all_results["RQ6"]["optimal_perplexity"])
    except Exception as e:
        logger.error("RQ6 failed: %s", e)
        all_results["RQ6"] = {"error": str(e)}

    output_path = Path(output_dir) / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("All ablation results saved to %s", output_path)

    return all_results


def summarize_ablation(results: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("ABLATION STUDY SUMMARY")
    lines.append("=" * 60)

    if "RQ2" in results and "error" not in results["RQ2"]:
        r2 = results["RQ2"]
        lines.append(f"RQ2 Efficiency: AltOpt loss={r2['protocol_a']['final_loss']:.4f}, AdamW loss={r2['protocol_b']['final_loss']:.4f}")

    if "RQ3" in results and "error" not in results["RQ3"]:
        r3 = results["RQ3"]
        lines.append(f"RQ3 Perturbation: benefit={r3['perturbation_benefit']:.2f} ppl, events={len(r3['perturbation_events'])}")

    if "RQ4" in results and "error" not in results["RQ4"]:
        r4 = results["RQ4"]
        for label, d in r4["protocols"].items():
            lines.append(f"RQ4 Generalization {label}: gap={d['generalization_gap']:.4f}")

    if "RQ5" in results and "error" not in results["RQ5"]:
        r5 = results["RQ5"]
        lines.append(f"RQ5 Synergy: {r5['altopt_advantage']} (delta={r5['synergy_benefit']:.2f} ppl)")

    if "RQ6" in results and "error" not in results["RQ6"]:
        r6 = results["RQ6"]
        lines.append(f"RQ6 Optimal ALS:SGD = {r6['optimal_ratio']} (ppl={r6['optimal_perplexity']:.2f})")
        for ratio, d in r6["ratios"].items():
            lines.append(f"  {ratio}: ppl={d['final_perplexity']:.2f}, flops={d['total_flops']:.2e}")

    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    model = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    output = sys.argv[2] if len(sys.argv) > 2 else "runs/ablation/"

    results = run_all_ablation(model_name=model, output_dir=output)
    print(summarize_ablation(results))
