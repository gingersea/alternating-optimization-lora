"""
P1.3: Implicit Regularization Replication — Train/Eval Gap on WikiText-2 + C4.

Tests whether ASP's train≈eval property is genuine or just slower convergence.
Compares ASP vs AdamW vs early-stopped AdamW on OPT-125m at multiple step budgets.

Three conditions:
  1. ASP Full (ALS+SGD+Perturb): 1200 steps, record train/eval every 50 steps
  2. AdamW (constant LR): 1200 steps, record train/eval every 50 steps
  3. Early-stopped AdamW: stop at min eval loss checkpoint

Primary metric: C4 cross-domain PPL (not WikiText-2 in-distribution).
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
logger = logging.getLogger("p1.3")

MODEL_NAME = "facebook/opt-125m"
MAX_LEN = 128
BATCH_SIZE = 2
TRAIN_SAMPLES = 400
EVAL_SAMPLES = 100
C4_SAMPLES = 500
N_STEPS = 1200
SEEDS = [42, 123]
OUTPUT_DIR = Path("runs/p1.3_regularization")


def make_dataloader(tokenizer, split, max_len, batch_size, n_samples, dataset_name="wikitext-2-raw-v1"):
    ds = load_dataset("wikitext", dataset_name, split=split)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    tokd = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=max_len,
                                       padding="max_length"), batched=True, remove_columns=["text"])
    tokd.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def collate(b):
        ids = torch.stack([x["input_ids"] for x in b])
        attn = torch.stack([x["attention_mask"] for x in b])
        return {"input_ids": ids, "attention_mask": attn, "labels": ids.clone()}

    return DataLoader(tokd, batch_size=batch_size, shuffle=(split == "train"), collate_fn=collate)


def make_c4_dataloader(tokenizer, split, max_len, batch_size, n_samples):
    """C4 dataset loader — different from WikiText-2 for cross-domain eval."""
    ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
    samples = []
    for i, example in enumerate(ds):
        if i >= n_samples:
            break
        text = example["text"][:max_len * 10]
        samples.append(text)
    tokd = tokenizer(samples, truncation=True, max_length=max_len, padding="max_length",
                     return_tensors="pt")
    class C4Dataset(torch.utils.data.Dataset):
        def __init__(self, inp, mask):
            self.inp = inp
            self.mask = mask
        def __len__(self): return len(self.inp)
        def __getitem__(self, i):
            return {"input_ids": self.inp[i], "attention_mask": self.mask[i],
                    "labels": self.inp[i].clone()}
    ds = C4Dataset(tokd["input_ids"], tokd["attention_mask"])
    return DataLoader(ds, batch_size=batch_size)


def run_asp(tokenizer, train_dl, eval_dl, c4_dl, seed):
    """Full ASP (ALS+SGD+Perturb) for N_STEPS."""
    schedule = PhaseSchedule(phases=[
        PhaseConfig(phase=Phase.ALS, steps=1, block_size=1024),
        PhaseConfig(phase=Phase.SGD, steps=50, lr=1e-4),
        PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3),
    ], cycles=N_STEPS // 52 + 1)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    cfg = TrainerConfig(
        protocol="A", optimizer_type="altopt", parameter_form="full_rank",
        max_steps=N_STEPS, lr=1e-4, run_dir=f"/tmp/p13_asp_s{seed}",
        seed=seed, eval_every=50, save_every=10000,
    )
    cfg.phase_schedule = schedule

    trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_dl, tokenizer=tokenizer)
    state = trainer.train(train_dl)

    wt_evaluator = Evaluator(["perplexity", "loss"], eval_dl)
    wt_result = wt_evaluator.evaluate(model)

    c4_evaluator = Evaluator(["perplexity", "loss"], c4_dl)
    c4_result = c4_evaluator.evaluate(model)

    return {
        "condition": "ASP",
        "seed": seed,
        "wt2_ppl": float(wt_result.get("perplexity", float("inf"))),
        "wt2_loss": float(wt_result.get("loss", float("inf"))),
        "c4_ppl": float(c4_result.get("perplexity", float("inf"))),
        "c4_loss": float(c4_result.get("loss", float("inf"))),
        "train_loss_final": float(state.loss_history[-1]) if state.loss_history else float("inf"),
        "eval_history": [float(x) if not isinstance(x, dict) else float('nan')
                         for x in state.eval_history],
        "loss_history": [float(x) if not isinstance(x, dict) else float('nan')
                         for x in state.loss_history],
        "wall_time": state.elapsed_seconds,
    }


def run_adamw(tokenizer, train_dl, eval_dl, c4_dl, seed):
    """AdamW for N_STEPS, recording best eval checkpoint."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda()
    cfg = TrainerConfig(
        protocol="B", optimizer_type="adamw", parameter_form="full_rank",
        max_steps=N_STEPS, lr=1e-4, run_dir=f"/tmp/p13_adamw_s{seed}",
        seed=seed, eval_every=50, save_every=10000,
    )

    trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_dl, tokenizer=tokenizer)
    state = trainer.train(train_dl)

    # Find best eval step
    best_eval = float("inf")
    best_step = 0
    for i, (e, s) in enumerate(zip(state.eval_history, range(0, N_STEPS, 50))):
        v = float(e) if not isinstance(e, dict) else float('inf')
        if v < best_eval:
            best_eval = v
            best_step = s

    wt_evaluator = Evaluator(["perplexity", "loss"], eval_dl)
    wt_final = wt_evaluator.evaluate(model)

    c4_evaluator = Evaluator(["perplexity", "loss"], c4_dl)
    c4_final = c4_evaluator.evaluate(model)

    return {
        "condition": "AdamW",
        "seed": seed,
        "best_eval_loss": float(best_eval),
        "best_eval_step": best_step,
        "wt2_ppl_final": float(wt_final.get("perplexity", float("inf"))),
        "wt2_loss_final": float(wt_final.get("loss", float("inf"))),
        "c4_ppl_final": float(c4_final.get("perplexity", float("inf"))),
        "c4_loss_final": float(c4_final.get("loss", float("inf"))),
        "train_loss_final": float(state.loss_history[-1]) if state.loss_history else float("inf"),
        "eval_history": [float(x) if not isinstance(x, dict) else float('nan')
                         for x in state.eval_history],
        "loss_history": [float(x) if not isinstance(x, dict) else float('nan')
                         for x in state.loss_history],
        "wall_time": state.elapsed_seconds,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    train_dl = make_dataloader(tokenizer, "train", MAX_LEN, BATCH_SIZE, TRAIN_SAMPLES)
    eval_dl = make_dataloader(tokenizer, "test", MAX_LEN, BATCH_SIZE, n_samples=EVAL_SAMPLES)

    logger.info("Loading C4 dataset for cross-domain evaluation...")
    try:
        c4_dl = make_c4_dataloader(tokenizer, "validation", MAX_LEN, BATCH_SIZE, C4_SAMPLES)
    except Exception as e:
        logger.warning("C4 not available: %s. Falling back to WikiText-2 only.", e)
        c4_dl = eval_dl  # fallback

    total_start = time.time()
    all_results = []

    for seed in SEEDS:
        for cond_fn, cond_name in [(run_asp, "ASP"), (run_adamw, "AdamW")]:
            logger.info("=" * 60)
            logger.info("Running %s seed=%d", cond_name, seed)
            logger.info("=" * 60)
            r = cond_fn(tokenizer, train_dl, eval_dl, c4_dl, seed)
            all_results.append(r)

    total_time = time.time() - total_start

    # ── Summary ──
    summary = {}
    for cond in ["ASP", "AdamW"]:
        seeds_data = [r for r in all_results if r["condition"] == cond]
        if seeds_data:
            if cond == "ASP":
                wt2_ppls = [r["wt2_ppl"] for r in seeds_data]
                c4_ppls = [r["c4_ppl"] for r in seeds_data]
                train_losses = [r["train_loss_final"] for r in seeds_data]
                summary[cond] = {
                    "n_seeds": len(seeds_data),
                    "wt2_ppl_mean": float(np.mean(wt2_ppls)),
                    "c4_ppl_mean": float(np.mean(c4_ppls)),
                    "train_eval_gap": float(np.mean([abs(t - e[-1] if e else float('inf'))
                                                     for t, e in zip(train_losses,
                                                                      [r.get("eval_history", []) for r in seeds_data])])),
                }
            else:
                summary[cond] = {
                    "n_seeds": len(seeds_data),
                    "best_eval_step": int(np.mean([r["best_eval_step"] for r in seeds_data])),
                    "wt2_ppl_final": float(np.mean([r["wt2_ppl_final"] for r in seeds_data])),
                    "c4_ppl_final": float(np.mean([r["c4_ppl_final"] for r in seeds_data])),
                    "best_eval_loss": float(np.mean([r["best_eval_loss"] for r in seeds_data])),
                }

    output = {
        "experiment": "p1.3_implicit_regularization",
        "n_steps": N_STEPS,
        "summary": summary,
        "full_results": all_results,
        "total_wall_time": total_time,
    }

    out = OUTPUT_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("P1.3 IMPLICIT REGULARIZATION — RESULTS")
    print("=" * 60)
    for cond, s in summary.items():
        print(f"  {cond}: {json.dumps(s, indent=2)}")
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
