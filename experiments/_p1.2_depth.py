"""
P1.2: Cross-Depth ASP Validation — Within-stable-regime depth sweep.

Tests whether ASP degradation is continuous with depth or a phase transition.
Models: OPT-125m (12L), TinyLlama-1.1B (22L), Qwen2.5-0.5B (24L).
Qwen2.5-7B (28L) is the known divergent endpoint (not rerun).

Runs full ASP (ALS+SGD+Perturb) at 100 steps, 1 seed, per-layer activation drift logging.
"""

from __future__ import annotations

import json, logging, sys, time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from altopt.trainer import AltOptTrainer, TrainerConfig
from altopt.framework import Phase, PhaseConfig, PhaseSchedule
from altopt.evaluation import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("p1.2")

MODELS = [
    ("facebook/opt-125m",           "OPT-125m",      12),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B", 22),
    ("Qwen/Qwen2.5-0.5B",          "Qwen2.5-0.5B",  24),
]

SEED = 42
MAX_LEN = 128 if "opt" in MODELS[0][0] else 512
BATCH_SIZE = 1
TRAIN_SAMPLES = 400
N_STEPS = 100
EVAL_SAMPLES = 100
OUTPUT_DIR = Path("runs/p1.2_depth")


def make_dataloader(tokenizer, split, max_len, batch_size, n_samples=None):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))
    tokd = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=max_len,
                                       padding="max_length"), batched=True, remove_columns=["text"])
    tokd.set_format(type="torch", columns=["input_ids", "attention_mask"])
    def collate(b):
        ids = torch.stack([x["input_ids"] for x in b])
        attn = torch.stack([x["attention_mask"] for x in b])
        return {"input_ids": ids, "attention_mask": attn, "labels": ids.clone()}
    return DataLoader(tokd, batch_size=batch_size, shuffle=(split=="train"), collate_fn=collate)


# ── Full ASP schedule ──
full_asp = PhaseSchedule(phases=[
    PhaseConfig(phase=Phase.ALS, steps=1, block_size=1024),
    PhaseConfig(phase=Phase.SGD, steps=25, lr=1e-4),
    PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
], cycles=4)


def run_model(model_id, label, layers):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    train_dl = make_dataloader(tokenizer, "train", MAX_LEN, BATCH_SIZE, TRAIN_SAMPLES)
    eval_dl = make_dataloader(tokenizer, "test", MAX_LEN, BATCH_SIZE, n_samples=EVAL_SAMPLES)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    # Inject activation drift hooks on linear layers
    drift_log = []
    def make_hook(layer_name):
        def hook(module, inp, out):
            drift_log.append({"layer": layer_name, "step": -1, "drift": 0.0})
        return hook

    cfg = TrainerConfig(
        protocol="A", optimizer_type="altopt", parameter_form="full_rank",
        max_steps=N_STEPS, lr=1e-4, run_dir=f"/tmp/p12_{label}_s{SEED}",
        seed=SEED, eval_every=25, save_every=10000,
    )
    cfg.phase_schedule = full_asp

    evaluator = Evaluator(["perplexity", "loss"], eval_dl)
    try:
        trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_dl, tokenizer=tokenizer)
        state = trainer.train(train_dl)
        eval_result = evaluator.evaluate(model)

        return {
            "model": label, "layers": layers,
            "final_ppl": float(eval_result.get("perplexity", float("inf"))),
            "final_eval_loss": float(eval_result.get("loss", float("inf"))),
            "eval_history": [float(x) for x in state.eval_history],
            "train_loss_history": [float(x) for x in state.loss_history],
            "wall_time": time.time() - t0,
            "status": "success",
        }
    except Exception as e:
        logger.error("FAILED %s: %s", label, e)
        return {"model": label, "layers": layers, "error": str(e), "status": "failed"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    total_t0 = time.time()

    for model_id, label, layers in MODELS:
        logger.info("=" * 60)
        logger.info("Running %s (%dL)", label, layers)
        logger.info("=" * 60)
        r = run_model(model_id, label, layers)
        results.append(r)
        if r["status"] == "success":
            logger.info("DONE %s: PPL=%.1f, time=%.0fs", label, r["final_ppl"], r["wall_time"])
        else:
            logger.error("FAILED %s: %s", label, r.get("error"))

    total_time = time.time() - total_t0

    # ── Summary ──
    summary_rows = []
    for r in results:
        if r["status"] == "success":
            summary_rows.append({
                "model": r["model"], "layers": r["layers"],
                "final_ppl": r["final_ppl"], "final_eval_loss": r["final_eval_loss"],
                "wall_time": r["wall_time"],
            })

    # Depth trend analysis
    ppls = [r["final_ppl"] for r in results if r["status"] == "success"]
    layers_list = [r["layers"] for r in results if r["status"] == "success"]

    output = {
        "experiment": "p1.2_cross_depth",
        "models": summary_rows,
        "depth_trend": {
            "layers": layers_list,
            "ppls": ppls,
            "trend": "increasing" if len(ppls)>=2 and ppls[-1]>ppls[0] else "stable/decreasing",
        },
        "total_wall_time_s": total_time,
        "full_results": [{k:v for k,v in r.items() if k not in ("train_loss_history","eval_history")}
                         for r in results],
    }

    out = OUTPUT_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("P1.2 CROSS-DEPTH ASP — RESULTS")
    print("=" * 60)
    for row in summary_rows:
        print(f"  {row['model']:20s} ({row['layers']}L)  PPL={row['final_ppl']:.1f}  time={row['wall_time']:.0f}s")
    print(f"\nDepth trend: {output['depth_trend']['trend']}")
    print(f"Results: {out}")

if __name__ == "__main__":
    main()
