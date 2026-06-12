"""
Matrix experiment: 2 models × 4 step counts × 4 protocols.

Tests the crossover hypothesis across temporal regimes:
- 50 steps: pre-digestion (ALS loss dominates)
- 100 steps: early digestion
- 200 steps: mid digestion (Round 5 data)
- 400 steps: late digestion (Round 6 data)

Outputs a 2×4×4 tensor: (model, steps, protocol) → PPL.
"""

import json, logging, sys, time
from pathlib import Path
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from altopt.trainer import AltOptTrainer, TrainerConfig
from altopt.framework import Phase, PhaseConfig, PhaseSchedule
from altopt.evaluation import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("matrix")

STEPS_LIST = [50, 100, 200, 400, 800]
SEED = 42
OUT = Path("runs/matrix_experiment")


def make_dl(tokenizer, split, max_len, batch_size, ds_name, n_samples):
    dataset = load_dataset("wikitext", ds_name, split=split)
    if n_samples: dataset = dataset.select(range(min(n_samples, len(dataset))))
    def tok(ex): return tokenizer(ex["text"], truncation=True, max_length=max_len, padding="max_length")
    tokenized = dataset.map(tok, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    def collate(b):
        ids = torch.stack([x["input_ids"] for x in b])
        return {"input_ids": ids, "attention_mask": torch.stack([x["attention_mask"] for x in b]), "labels": ids.clone()}
    return DataLoader(tokenized, batch_size=batch_size, shuffle=(split=="train"), collate_fn=collate)


def build_schedules(n_steps):
    """Build phase schedules that adapt to total step count."""
    sgd_per_cycle = max(10, n_steps // 4)
    n_cycles = n_steps // (sgd_per_cycle + 2)
    if n_cycles < 1: n_cycles = 1

    altopt_full = PhaseSchedule(phases=[
        PhaseConfig(phase=Phase.ALS, steps=1, block_size=1024),
        PhaseConfig(phase=Phase.SGD, steps=sgd_per_cycle, lr=1e-4),
        PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
    ], cycles=n_cycles)

    altopt_lora = PhaseSchedule(phases=[
        PhaseConfig(phase=Phase.SGD, steps=sgd_per_cycle, lr=1e-4),
        PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=5e-4),
    ], cycles=n_cycles)

    return altopt_full, altopt_lora


def run_model_matrix(model_name, model_type, lora_targets, max_len, batch_size, ds_name, n_train, n_eval):
    """Run all protocols at all step counts for one model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=(model_type=="qwen"))
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    train_dl = make_dl(tokenizer, "train", max_len, batch_size, ds_name, n_train)
    eval_dl = make_dl(tokenizer, "test", max_len, batch_size, ds_name, n_eval)

    results = {}
    total = len(STEPS_LIST) * 4
    count = 0

    for n_steps in STEPS_LIST:
        altopt_s, lora_s = build_schedules(n_steps)
        protocols = [
            ("A", "altopt", "full_rank", altopt_s, None, n_steps),
            ("B", "adamw", "full_rank", None, None, n_steps),
            ("C", "altopt", "lora", lora_s, lora_targets, n_steps),
            ("D", "adamw", "lora", None, lora_targets, n_steps),
        ]

        for proto, opt, form, sched, lt, steps in protocols:
            count += 1
            label = f"{model_type}_{proto}_{steps}s"
            logger.info(f"[{count}/{total}] {label}")

            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=(model_type=="qwen"))
            t0 = time.time()

            cfg = TrainerConfig(
                protocol=proto, optimizer_type=opt, parameter_form=form,
                max_steps=steps, lr=5e-5 if model_type=="qwen" else 1e-4,
                lora_r=8, lora_alpha=16.0, lora_target_modules=lt,
                run_dir=f"/tmp/matrix_{label}", seed=SEED,
                eval_every=max(10, steps//4), save_every=10000,
            )
            if sched: cfg.phase_schedule = sched

            try:
                trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_dl, tokenizer=tokenizer)
                state = trainer.train(train_dl)
                eval_r = Evaluator(["perplexity", "loss"], eval_dl).evaluate(model)

                train_losses = [l for i, l in enumerate(state.loss_history)
                                if i >= len(getattr(state, 'loss_types', [])) or
                                getattr(state, 'loss_types', [])[i] != 'noise_energy']

                key = f"{proto}_{steps}s"
                results[key] = {
                    "protocol": proto, "steps": steps,
                    "final_ppl": eval_r["perplexity"],
                    "final_train_loss": train_losses[-1] if train_losses else float("inf"),
                    "final_eval_loss": eval_r.get("loss", float("inf")),
                    "total_flops": state.cumulative_flops,
                    "wall_time": time.time() - t0,
                    "eval_curve": [(e["step"], e.get("perplexity", float("inf"))) for e in state.eval_history],
                }
                logger.info(f"  ppl={eval_r['perplexity']:.1f} flops={state.cumulative_flops:.2e} time={time.time()-t0:.0f}s")
            except Exception as e:
                logger.error(f"  FAIL: {e}")
                results[f"{proto}_{steps}s"] = {"error": str(e)}

    return results


def compute_matrix(results_opt, results_qwen):
    """Build the numerical matrix for analysis."""
    matrix = {}
    for model_name, res in [("OPT-125m", results_opt), ("Qwen2.5-0.5B", results_qwen)]:
        matrix[model_name] = {}
        for steps in STEPS_LIST:
            row = {}
            for proto in ["A", "B", "C", "D"]:
                key = f"{proto}_{steps}s"
                if key in res and "error" not in res[key]:
                    row[proto] = res[key]["final_ppl"]
                    row[f"{proto}_flops"] = res[key]["total_flops"]
                else:
                    row[proto] = None
            row["A_minus_B"] = (row["A"] - row["B"]) if row["A"] and row["B"] else None
            row["C_minus_D"] = (row["C"] - row["D"]) if row["C"] and row["D"] else None
            row["interaction"] = (row["A_minus_B"] - row["C_minus_D"]) if row["A_minus_B"] and row["C_minus_D"] else None
            matrix[model_name][steps] = row
    return matrix


def print_matrix(matrix):
    print("\n" + "=" * 70)
    print("MATRIX EXPERIMENT RESULTS: PPL vs Steps × Protocol")
    print("=" * 70)

    for model_name in ["OPT-125m", "Qwen2.5-0.5B"]:
        print(f"\n--- {model_name} ---")
        print(f"{'Steps':<8} {'A (Alt/F)':<12} {'B (Adam/F)':<12} {'C (Alt/L)':<12} {'D (Adam/L)':<12} {'A-B':<10} {'C-D':<10}")
        print("-" * 76)
        for steps in STEPS_LIST:
            row = matrix[model_name][steps]
            vals = [
                row.get("A"), row.get("B"), row.get("C"), row.get("D"),
                row.get("A_minus_B"), row.get("C_minus_D"),
            ]
            line = f"{steps:<8}"
            for v in vals:
                if v is None: line += f"{'ERR':<12}"
                else: line += f"{v:<12.1f}"
            print(line)

    print("\n--- A-B gap convergence (crossover test) ---")
    for model_name in ["OPT-125m", "Qwen2.5-0.5B"]:
        gaps = [str(matrix[model_name][s].get("A_minus_B", "N/A")) for s in STEPS_LIST]
        print(f"{model_name}: {' → '.join(gaps)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MODEL 1: OPT-125m")
    logger.info("=" * 60)
    opt_results = run_model_matrix(
        "facebook/opt-125m", "opt",
        lora_targets=["q_proj", "v_proj", "k_proj", "out_proj"],
        max_len=128, batch_size=2, ds_name="wikitext-2-raw-v1",
        n_train=400, n_eval=100,
    )

    logger.info("=" * 60)
    logger.info("MODEL 2: Qwen2.5-0.5B")
    logger.info("=" * 60)
    qwen_results = run_model_matrix(
        "Qwen/Qwen2.5-0.5B", "qwen",
        lora_targets=["q_proj", "v_proj", "k_proj", "o_proj"],
        max_len=128, batch_size=1, ds_name="wikitext-2-raw-v1",
        n_train=200, n_eval=50,
    )

    matrix = compute_matrix(opt_results, qwen_results)
    print_matrix(matrix)

    full_results = {"opt": opt_results, "qwen": qwen_results, "matrix": matrix}
    with open(OUT / "results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
