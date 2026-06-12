"""
Round 6: Long SGD cycles + built-in LoRA + crossover test.

Key changes from Round 5:
1. Built-in LoRALayer (NOT PEFT) for Protocol C/D — fixes ~30x PEFT discrepancy
2. ALS(1)+SGD(200) × 2 cycles — tests H2 (crossover at SGD > 150/cycle)
3. 400 total steps per protocol
4. Eval at 100, 200, 300, 400 to track convergence trajectory

Hypothesis: With SGD=200 per ALS cycle (4x longer than Round 5),
AltOpt should significantly narrow or close the gap with AdamW.
"""

from __future__ import annotations
import json, logging, sys, time
from pathlib import Path
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from altopt.trainer import AltOptTrainer, TrainerConfig
from altopt.framework import Phase, PhaseConfig, PhaseSchedule
from altopt.evaluation import Evaluator
from altopt.lora import LoRAConfig, LoRABaseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("round6")

MODEL, DS, ML, BS, NS = "facebook/opt-125m", "wikitext-2-raw-v1", 128, 2, 400
SEEDS = [42, 123]
OUT = Path("runs/round6_long_cycles")


def make_dl(tokenizer, split, max_len, batch_size, n_samples=None):
    dataset = load_dataset("wikitext", DS, split=split)
    if n_samples: dataset = dataset.select(range(min(n_samples, len(dataset))))
    def tok(ex): return tokenizer(ex["text"], truncation=True, max_length=max_len, padding="max_length")
    tokenized = dataset.map(tok, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    def collate(batch):
        ids = torch.stack([b["input_ids"] for b in batch])
        return {"input_ids": ids, "attention_mask": torch.stack([b["attention_mask"] for b in batch]), "labels": ids.clone()}
    return DataLoader(tokenized, batch_size=batch_size, shuffle=(split=="train"), collate_fn=collate)


def run_protocol(protocol, opt_type, param_form, seed, tokenizer, train_dl, eval_dl,
                 phase_schedule=None, use_builtin_lora=False, lora_r=8, lora_alpha=16.0):
    """Run a single protocol with optional built-in LoRA."""
    from altopt.trainer import AltOptTrainer, TrainerConfig  # noqa: E402

    model = AutoModelForCausalLM.from_pretrained(MODEL)
    t0 = time.time()

    if use_builtin_lora:
        lora_cfg = LoRAConfig(r=lora_r, alpha=lora_alpha, target_modules=["q_proj", "v_proj", "k_proj", "out_proj"])
        lora_baseline = LoRABaseline(model, lora_cfg, lr=1e-4)
        model = lora_baseline.model

        if opt_type == "altopt" and phase_schedule:
            cfg = TrainerConfig(
                protocol=protocol, optimizer_type="altopt", parameter_form="full_rank",
                max_steps=NS, lr=1e-4, run_dir=f"/tmp/r6_{protocol}_s{seed}", seed=seed,
                eval_every=100, save_every=10000, phase_schedule=phase_schedule,
            )
            trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_dl, tokenizer=tokenizer)
            state = trainer.train(train_dl)
        else:
            cfg = TrainerConfig(
                protocol=protocol, optimizer_type="adamw", parameter_form="full_rank",
                max_steps=NS, lr=1e-4, run_dir=f"/tmp/r6_{protocol}_s{seed}", seed=seed,
                eval_every=100, save_every=10000,
            )
            trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_dl, tokenizer=tokenizer)
            state = trainer.train(train_dl)
    else:
        if phase_schedule:
            cfg = TrainerConfig(protocol=protocol, optimizer_type=opt_type, parameter_form=param_form,
                max_steps=NS, lr=1e-4, run_dir=f"/tmp/r6_{protocol}_s{seed}", seed=seed,
                eval_every=100, save_every=10000, phase_schedule=phase_schedule)
        else:
            cfg = TrainerConfig(protocol=protocol, optimizer_type=opt_type, parameter_form=param_form,
                max_steps=NS, lr=1e-4, run_dir=f"/tmp/r6_{protocol}_s{seed}", seed=seed,
                eval_every=100, save_every=10000)
        trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_dl, tokenizer=tokenizer)
        state = trainer.train(train_dl)

    eval_result = Evaluator(["perplexity", "loss"], eval_dl).evaluate(model)
    eval_curve = [(e["step"], e.get("perplexity", float("inf"))) for e in state.eval_history]

    train_losses = [l for i, l in enumerate(state.loss_history)
                    if i >= len(getattr(state, 'loss_types', [])) or
                    getattr(state, 'loss_types', [])[i] != 'noise_energy']

    return {
        "protocol": protocol, "seed": seed,
        "final_train_loss": train_losses[-1] if train_losses else float("inf"),
        "final_eval_loss": eval_result.get("loss", float("inf")),
        "final_perplexity": eval_result.get("perplexity", float("inf")),
        "eval_curve": eval_curve,  # [(step, ppl), ...]
        "total_flops": state.cumulative_flops, "wall_time": time.time() - t0,
        "loss_history": state.loss_history, "n_steps": state.step,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    train_dl = make_dl(tokenizer, "train", ML, BS, NS)
    eval_dl = make_dl(tokenizer, "test", ML, BS, 100)

    # Schedules
    altopt_long = PhaseSchedule(phases=[
        PhaseConfig(phase=Phase.ALS, steps=1, block_size=1024),
        PhaseConfig(phase=Phase.SGD, steps=200, lr=1e-4),  # KEY: 200 SGD/cycle
    ], cycles=2)

    lora_altopt_long = PhaseSchedule(phases=[
        PhaseConfig(phase=Phase.SGD, steps=200, lr=1e-4),
        PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=5e-4),
    ], cycles=2)

    protocols = [
        ("A", "altopt", "full_rank", altopt_long, False),
        ("B", "adamw", "full_rank", None, False),
        ("C", "altopt", "lora", lora_altopt_long, True),
        ("D", "adamw", "lora", None, True),
    ]

    results = {}
    total_start = time.time()

    for protocol, opt_type, param_form, schedule, builtin_lora in protocols:
        seed_results = {}
        for seed in SEEDS:
            label = f"{protocol}_s{seed}"
            logger.info(f"{'='*60}\n{label}: {opt_type}/{param_form} (builtin_lora={builtin_lora})\n{'='*60}")
            try:
                r = run_protocol(protocol, opt_type, param_form, seed, tokenizer,
                                train_dl, eval_dl, schedule, builtin_lora)
                seed_results[str(seed)] = r
                logger.info(f"DONE {label}: ppl={r['final_perplexity']:.2f} flops={r['total_flops']:.2e} time={r['wall_time']:.0f}s")
            except Exception as e:
                logger.error(f"FAIL {label}: {e}")
                seed_results[str(seed)] = {"error": str(e)}

        ppls = [r["final_perplexity"] for r in seed_results.values()
                if "error" not in r and r.get("final_perplexity", float("inf")) != float("inf")]
        flops_vals = [r["total_flops"] for r in seed_results.values() if "error" not in r]

        results[protocol] = {
            "protocol": protocol, "optimizer": opt_type, "parameter_form": param_form,
            "builtin_lora": builtin_lora, "seeds": seed_results,
            "mean_ppl": float(np.mean(ppls)) if ppls else float("inf"),
            "std_ppl": float(np.std(ppls)) if len(ppls) > 1 else 0.0,
            "mean_flops": float(np.mean(flops_vals)) if flops_vals else 0,
            "convergence": {
                f"ppl@{s}": {str(seed): next((p for step, p in r.get("eval_curve", []) if step >= s), None)
                             for seed, r in seed_results.items() if "error" not in r}
                for s in [100, 200, 300, 400]
            },
        }

        logger.info(f"Protocol {protocol}: ppl={results[protocol]['mean_ppl']:.2f}±{results[protocol]['std_ppl']:.2f}")

    # Comparison matrix
    a = results["A"]["mean_ppl"]; b = results["B"]["mean_ppl"]
    c = results["C"]["mean_ppl"]; d = results["D"]["mean_ppl"]

    summary = {
        "experiment": "round6", "model": MODEL, "n_steps": NS, "n_seeds": len(SEEDS),
        "total_wall_time_s": time.time() - total_start,
        "protocols": results,
        "comparison": {
            "A_vs_B_opt_effect_full": a - b,
            "C_vs_D_opt_effect_lora": c - d,
            "A_vs_C_param_effect_altopt": a - c,
            "B_vs_D_param_effect_adamw": b - d,
            "interaction": (a-b) - (c-d),
            "crossover_delta": {
                "A_minus_B_at_100": _delta_at(results, "A", "B", 100),
                "A_minus_B_at_200": _delta_at(results, "A", "B", 200),
                "A_minus_B_at_300": _delta_at(results, "A", "B", 300),
                "A_minus_B_at_400": _delta_at(results, "A", "B", 400),
            },
        },
    }

    output = OUT / "results.json"
    with open(output, "w") as f: json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n{'='*60}\nROUND 6 COMPLETE")
    logger.info(f"  A (AltOpt full, SGD=200/cyc): ppl={a:.1f}±{results['A']['std_ppl']:.1f}")
    logger.info(f"  B (AdamW full):              ppl={b:.1f}±{results['B']['std_ppl']:.1f}")
    logger.info(f"  C (AltOpt builtin-LoRA):     ppl={c:.1f}±{results['C']['std_ppl']:.1f}")
    logger.info(f"  D (AdamW builtin-LoRA):      ppl={d:.1f}±{results['D']['std_ppl']:.1f}")
    logger.info(f"  A-B (opt effect full):       {a-b:.1f}")
    logger.info(f"  C-D (opt effect LoRA):       {c-d:.1f}")
    logger.info(f"  Crossover trend: 100→200→300→400: "
                f"{summary['comparison']['crossover_delta']['A_minus_B_at_100']:.0f} → "
                f"{summary['comparison']['crossover_delta']['A_minus_B_at_200']:.0f} → "
                f"{summary['comparison']['crossover_delta']['A_minus_B_at_300']:.0f} → "
                f"{summary['comparison']['crossover_delta']['A_minus_B_at_400']:.0f}")
    logger.info(f"Results: {output}")

    # Verify built-in LoRA is working: C should have much lower FLOPs than A
    if results["C"]["mean_flops"] < results["A"]["mean_flops"] * 0.5:
        logger.info("✓ Built-in LoRA active: Protocol C FLOPs << Protocol A FLOPs")
    else:
        logger.warning("✗ Protocol C FLOPs ~= Protocol A FLOPs — built-in LoRA may not be active!")

    return summary


def _delta_at(results, proto_a, proto_b, step):
    """Compute mean perplexity difference at a specific step."""
    a_vals = []
    b_vals = []
    for seed, r in results[proto_a].get("seeds", {}).items():
        if "error" in r: continue
        curve = r.get("eval_curve", [])
        ppls_at_step = [p for s, p in curve if s >= step]
        if ppls_at_step: a_vals.append(ppls_at_step[0])
    for seed, r in results[proto_b].get("seeds", {}).items():
        if "error" in r: continue
        curve = r.get("eval_curve", [])
        ppls_at_step = [p for s, p in curve if s >= step]
        if ppls_at_step: b_vals.append(ppls_at_step[0])
    if a_vals and b_vals:
        return float(np.mean(a_vals) - np.mean(b_vals))
    return float("nan")


if __name__ == "__main__":
    main()
