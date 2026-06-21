#!/usr/bin/env python3
"""
Evaluate saved 7B checkpoints on downstream tasks using lm-evaluation-harness.

Usage:
  python experiments/_eval_downstream.py --baseline --tasks hellaswag --limit 10
  python experiments/_eval_downstream.py --protocols B --seeds 42 --tasks hellaswag
  python experiments/_eval_downstream.py --protocols B,D --tasks hellaswag,mmlu
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval-downstream")

MODEL_NAME = "Qwen/Qwen2.5-7B"
MODEL_LOCAL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/"
    "snapshots/d149729398750b98c0af14eb82c78cfe92750796"
)
RUNS_DIR = Path("runs/qwen25_7b_800s")
OUTPUT_FILE = "runs/qwen25_7b_800s/downstream_eval.json"


def load_base_model():
    """Load base Qwen2.5-7B from local cache."""
    logger.info("Loading base model from local cache...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_LOCAL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_LOCAL_PATH,
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_checkpoint_into_model(model, ckpt_dir: Path, is_peft: bool):
    """
    Load checkpoint state_dict into model (in-place).
    Returns meta dict from checkpoint metadata.
    """
    weights_path = ckpt_dir / "model_weights.pt"
    logger.info("Loading state_dict from %s (%.1f GB)...",
                 weights_path, weights_path.stat().st_size / 1e9)

    state_dict = torch.load(str(weights_path), map_location="cpu")

    if is_peft:
        # Infer LoRA config from state_dict
        lora_targets = set()
        lora_rank = None
        for k in state_dict:
            if "lora_A" in k:
                parts = k.split(".")
                lora_idx = parts.index("lora_A")
                if lora_idx >= 2:
                    lora_targets.add(parts[lora_idx - 1])
                if lora_rank is None:
                    lora_rank = state_dict[k].shape[0]

        logger.info("PEFT: r=%s, targets=%s", lora_rank, sorted(lora_targets))
        lora_config = LoraConfig(
            r=lora_rank or 8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=sorted(lora_targets) if lora_targets
            else ["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)

    # Load state_dict — use strict=False since we already have base weights
    logger.info("Loading %d keys into model...", len(state_dict))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.debug("Missing keys: %d", len(missing))
    if unexpected:
        logger.debug("Unexpected keys: %d", len(unexpected))
    logger.info("State dict loaded successfully.")

    # Read metadata
    meta = {}
    for fname in ["metadata.json", "altopt_state.json"]:
        p = ckpt_dir / fname
        if p.exists():
            with open(p) as f:
                meta.update(json.load(f))
    return model, meta


def save_model_hf(model, tokenizer, output_dir: Path):
    """Save model in HF format. Uses safetensors for speed."""
    logger.info("Saving model to %s (this may take a minute)...", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))


def run_lm_eval(model_path: str, tasks: list[str], num_fewshot: int = 0,
                limit: int = None, batch_size: str = "auto"):
    """Evaluate using lm-eval-harness."""
    import lm_eval

    logger.info("Running lm-eval %s: tasks=%s, fewshot=%d, limit=%s",
                lm_eval.__version__, tasks, num_fewshot, limit)

    # lm-eval uses a YAML config approach — use simple_evaluate
    from lm_eval.api.registry import get_model
    from lm_eval.evaluator import simple_evaluate

    t0 = time.time()
    results = simple_evaluate(
        model="hf",
        model_args={
            "pretrained": model_path,
            "trust_remote_code": False,
            "local_files_only": True,
        },
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        bootstrap_iters=1000,
    )
    elapsed = time.time() - t0
    logger.info("Evaluation done in %.0fs (%.1f min)", elapsed, elapsed / 60)
    return results


def main():
    parser = argparse.ArgumentParser(description="Downstream eval for 7B checkpoints")
    parser.add_argument("--tasks", default="hellaswag",
                        help="Comma-separated task names")
    parser.add_argument("--protocols", default="B,D",
                        help="Protocols to eval: A,B,C,D")
    parser.add_argument("--seeds", default="42,123,456",
                        help="Seeds to eval")
    parser.add_argument("--step", type=int, default=800)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit examples per task (for testing)")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--batch_size", default="auto")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    protocols = [p.strip() for p in args.protocols.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    logger.info("=" * 60)
    logger.info("Downstream Eval: Qwen2.5-7B")
    logger.info("Tasks: %s | Protocols: %s | Seeds: %s | Fewshot: %d",
                 tasks, protocols, seeds, args.num_fewshot)
    logger.info("=" * 60)

    all_results = {}
    temp_dirs = []

    # ── Baseline ──
    if args.baseline:
        logger.info("\n>>> BASELINE: Untrained Qwen2.5-7B")
        # Use the local cache directly — no need to reload
        try:
            r = run_lm_eval(MODEL_LOCAL_PATH, tasks, args.num_fewshot,
                            args.limit, args.batch_size)
            all_results["baseline"] = r
        except Exception as e:
            logger.error("Baseline failed: %s", e, exc_info=True)

    # ── Checkpoints ──
    for proto in protocols:
        for seed in seeds:
            label = f"P{proto}_s{seed}"
            ckpt_name = f"ckpt_Qwen25-7B_P{proto}_{args.step}s_s{seed}"
            ckpt_dir = RUNS_DIR / ckpt_name / "checkpoints" / f"step_{args.step:05d}"

            if not ckpt_dir.exists():
                logger.warning("Not found: %s — skipping", ckpt_dir)
                continue

            logger.info("\n>>> PROTOCOL %s SEED %d", proto, seed)

            try:
                # Load base model
                model, tokenizer = load_base_model()

                # Load checkpoint
                is_peft = proto in ("C", "D")  # LoRA protocols
                model, meta = load_checkpoint_into_model(model, ckpt_dir, is_peft)

                # Save as HF model for lm-eval
                tmpdir = Path(tempfile.mkdtemp(prefix=f"qwen_p{proto}_s{seed}_"))
                temp_dirs.append(tmpdir)
                save_model_hf(model, tokenizer, tmpdir)

                # Free memory
                del model, tokenizer
                torch.cuda.empty_cache()

                # Evaluate
                r = run_lm_eval(str(tmpdir), tasks, args.num_fewshot,
                                args.limit, args.batch_size)
                all_results[label] = {
                    "protocol": f"P{proto}", "seed": seed, "step": args.step,
                    "best_perplexity": meta.get("best_perplexity"),
                    "best_loss": meta.get("best_loss"),
                    **r,
                }

                # Print key metrics
                for t, m in r.get("results", {}).items():
                    logger.info("  %s: %s", t, json.dumps(m, indent=None, default=str))

            except Exception as e:
                logger.error("FAILED %s: %s", label, e, exc_info=True)
                all_results[label] = {"error": str(e)}

    # ── Save results ──
    out = Path(OUTPUT_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nSaved: %s", out)

    # ── Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY TABLE")
    logger.info("-" * 60)
    for label in ["baseline"] + [f"P{p}_s{s}" for p in protocols for s in seeds]:
        data = all_results.get(label)
        if data is None:
            continue
        if "error" in data:
            logger.info("%-15s ERROR", label)
        else:
            for t, m in data.get("results", {}).items():
                acc = m.get("acc,none") or m.get("acc")
                acc_norm = m.get("acc_norm,none") or m.get("acc_norm")
                ppl = data.get("best_perplexity", "N/A")
                logger.info("%-15s %s: acc=%s  acc_norm=%s  ppl=%s",
                             label, t, acc, acc_norm, ppl)

    # ── Cleanup temp dirs ──
    logger.info("\nCleaning up temp dirs...")
    for d in temp_dirs:
        try:
            shutil.rmtree(d)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
