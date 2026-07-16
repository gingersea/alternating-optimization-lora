"""
P1.1: Component Attribution — Nested Ablation of ASP on OPT-125m.

Decomposes ASP's three components (ALS, SGD, Perturbation) via a 4-condition
nested design, measuring each component's marginal contribution to eval perplexity.

Conditions:
  SGD-only:   No ALS, No Perturb  — pure gradient baseline
  ALS+SGD:    ALS once, then SGD  — ASP without perturbation
  SGD+Perturb: SGD + cyclic noise — current Protocol C (without ALS)
  Full ASP:   ALS+SGD+Perturb     — current Protocol A

Fixed: OPT-125m (12L), WikiText-2, 200 steps, 3 seeds, step_size=0.01, reg_lambda=1e-3.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from altopt.trainer import AltOptTrainer, TrainerConfig
from altopt.framework import Phase, PhaseConfig, PhaseSchedule
from altopt.evaluation import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("p1.1")

MODEL_NAME = "facebook/opt-125m"
DATASET_NAME = "wikitext-2-raw-v1"
MAX_LEN = 128
BATCH_SIZE = 2
N_SAMPLES_TRAIN = 400
N_STEPS = 200
SEEDS = [42, 123, 456]
OUTPUT_DIR = Path("runs/p1.1_ablation")
EVAL_SAMPLES = 100


def make_dataloader(tokenizer, split, max_len, batch_size, n_samples=None):
    dataset = load_dataset("wikitext", DATASET_NAME, split=split)
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


def run_one(label, schedule, seed, tokenizer, train_dl, eval_dl):
    """Run single condition × seed. Returns result dict."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    t0 = time.time()

    cfg = TrainerConfig(
        protocol="A",         # full-rank altopt framework
        optimizer_type="altopt",
        parameter_form="full_rank",
        max_steps=N_STEPS,
        lr=1e-4,
        run_dir=f"/tmp/p11_{label}_s{seed}",
        seed=seed,
        eval_every=50,
        save_every=10000,
    )
    cfg.phase_schedule = schedule

    evaluator = Evaluator(["perplexity", "loss"], eval_dl)

    try:
        trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_dl, tokenizer=tokenizer)
        state = trainer.train(train_dl)
        eval_result = evaluator.evaluate(model)

        train_losses = [
            l for i, l in enumerate(state.loss_history)
            if i >= len(getattr(state, 'loss_types', [])) or
            getattr(state, 'loss_types', [])[i] != 'noise_energy'
        ]

        return {
            "condition": label,
            "seed": seed,
            "final_train_loss": train_losses[-1] if train_losses else float("inf"),
            "final_eval_loss": eval_result.get("loss", float("inf")),
            "final_perplexity": eval_result.get("perplexity", float("inf")),
            "total_flops": state.cumulative_flops,
            "peak_memory_mb": state.peak_memory_mb,
            "wall_time": time.time() - t0,
            "loss_history": state.loss_history,
            "eval_history": state.eval_history,
            "n_steps": state.step,
            "status": "success",
        }
    except Exception as e:
        logger.error("FAILED %s s%d: %s", label, seed, e)
        return {
            "condition": label,
            "seed": seed,
            "error": str(e),
            "status": "failed",
        }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    train_dl = make_dataloader(tokenizer, "train", MAX_LEN, BATCH_SIZE, N_SAMPLES_TRAIN)
    eval_dl = make_dataloader(tokenizer, "test", MAX_LEN, BATCH_SIZE, n_samples=EVAL_SAMPLES)

    # ── Four conditions ──────────────────────────────────────────

    # C1: SGD-only — pure gradient, no ALS, no perturbation
    sgd_only = PhaseSchedule(
        phases=[PhaseConfig(phase=Phase.SGD, steps=N_STEPS, lr=1e-4)],
        cycles=1,
    )

    # C2: ALS+SGD — ALS once at start, then pure SGD
    als_sgd = PhaseSchedule(
        phases=[
            PhaseConfig(phase=Phase.ALS, steps=1, block_size=1024),
            PhaseConfig(phase=Phase.SGD, steps=N_STEPS - 1, lr=1e-4),
        ],
        cycles=1,
    )

    # C3: SGD+Perturb — cyclic SGD + perturbation (no ALS)
    sgd_perturb = PhaseSchedule(
        phases=[
            PhaseConfig(phase=Phase.SGD, steps=50, lr=1e-4),
            PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
        ],
        cycles=4,
    )

    # C4: Full ASP — ALS + SGD + Perturb (current Protocol A)
    full_asp = PhaseSchedule(
        phases=[
            PhaseConfig(phase=Phase.ALS, steps=1, block_size=1024),
            PhaseConfig(phase=Phase.SGD, steps=50, lr=1e-4),
            PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
        ],
        cycles=4,
    )

    conditions = [
        ("sgd_only", sgd_only),
        ("als_sgd", als_sgd),
        ("sgd_perturb", sgd_perturb),
        ("full_asp", full_asp),
    ]

    # ── Run ───────────────────────────────────────────────────────

    all_results = []
    total_start = time.time()

    for label, schedule in conditions:
        for seed in SEEDS:
            logger.info("=" * 60)
            logger.info("Running %s seed=%d", label, seed)
            logger.info("=" * 60)

            r = run_one(label, schedule, seed, tokenizer, train_dl, eval_dl)
            all_results.append(r)

            if r["status"] == "success":
                logger.info("DONE %s s%d: ppl=%.2f loss=%.4f time=%.0fs",
                            label, seed, r["final_perplexity"],
                            r["final_eval_loss"], r["wall_time"])
            else:
                logger.error("FAILED %s s%d: %s", label, seed, r.get("error"))

    total_elapsed = time.time() - total_start

    # ── Summary ───────────────────────────────────────────────────

    summary = {}
    for label, _ in conditions:
        seeds_data = [r for r in all_results if r["condition"] == label and r["status"] == "success"]
        if seeds_data:
            ppls = [r["final_perplexity"] for r in seeds_data]
            losses = [r["final_eval_loss"] for r in seeds_data]
            summary[label] = {
                "n_seeds": len(seeds_data),
                "ppl_mean": float(np.mean(ppls)),
                "ppl_std": float(np.std(ppls)),
                "loss_mean": float(np.mean(losses)),
                "loss_std": float(np.std(losses)),
                "wall_time_mean": float(np.mean([r["wall_time"] for r in seeds_data])),
            }

    # ── Marginal contributions (factorial decomposition) ──────────

    def get_mean(label):
        return summary[label]["ppl_mean"] if label in summary else float("nan")

    sgd_ppl = get_mean("sgd_only")
    als_sgd_ppl = get_mean("als_sgd")
    sgd_perturb_ppl = get_mean("sgd_perturb")
    full_asp_ppl = get_mean("full_asp")

    # Main effects
    als_effect = sgd_ppl - als_sgd_ppl      # + = ALS helps
    perturb_effect = sgd_ppl - sgd_perturb_ppl  # + = perturbation helps

    # Interaction: does ALS × Perturb produce more than sum of parts?
    # Full ASP effect = Full_ASP - SGD_only
    # Expected additive = ALSeffect + PerturbEffect
    full_effect = sgd_ppl - full_asp_ppl
    expected_additive = als_effect + perturb_effect
    interaction = full_effect - expected_additive  # + = synergy, - = antagonism

    decomposition = {
        "sgd_only_ppl": float(sgd_ppl),
        "als_sgd_ppl": float(als_sgd_ppl),
        "sgd_perturb_ppl": float(sgd_perturb_ppl),
        "full_asp_ppl": float(full_asp_ppl),
        "als_main_effect": float(als_effect),
        "perturb_main_effect": float(perturb_effect),
        "full_asp_effect": float(full_effect),
        "expected_additive": float(expected_additive),
        "interaction": float(interaction),
        "interpretation": (
            "synergy (ALS×Perturb > additive)" if interaction > 0.5
            else "antagonism (ALS×Perturb < additive)" if interaction < -0.5
            else "additive (no interaction)"
        ),
    }

    output = {
        "experiment": "p1.1_component_attribution",
        "model": MODEL_NAME,
        "n_steps": N_STEPS,
        "n_seeds_per": len(SEEDS),
        "conditions": summary,
        "decomposition": decomposition,
        "individual_results": [{
            k: v for k, v in r.items() if k not in ("loss_history", "eval_history")
        } for r in all_results],
        "total_wall_time_s": total_elapsed,
        "git_commit": "HEAD",
    }

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("P1.1 COMPONENT ATTRIBUTION — RESULTS")
    print("=" * 60)
    for label, s in summary.items():
        print(f"  {label:16s}  PPL={s['ppl_mean']:.1f} ± {s['ppl_std']:.1f}  (N={s['n_seeds']})")
    print()
    print("DECOMPOSITION:")
    print(f"  ALS main effect:       {als_effect:+.1f} PPL ({'helps' if als_effect > 0 else 'hurts'})")
    print(f"  Perturb main effect:   {perturb_effect:+.1f} PPL ({'helps' if perturb_effect > 0 else 'hurts'})")
    print(f"  Full ASP total effect: {full_effect:+.1f} PPL")
    print(f"  Expected additive:     {expected_additive:+.1f} PPL")
    print(f"  Interaction:           {interaction:+.1f} PPL → {decomposition['interpretation']}")
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
