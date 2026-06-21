#!/usr/bin/env python3
"""
Evaluate 7B checkpoints on C4 perplexity (second dataset).

Addresses reviewer concern about single-dataset evaluation (R5 H5, R6 R1).
C4 (Colossal Clean Crawled Corpus) is web text, testing cross-domain generalization.
"""

import json
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval-c4")

MODEL_NAME = "Qwen/Qwen2.5-7B"
RUNS_DIR = Path("runs/qwen25_7b_800s")
OUTPUT_FILE = RUNS_DIR / "c4_eval.json"

MAX_SEQ_LEN = 2048
BATCH_SIZE = 2
N_EVAL = 500  # C4 validation samples
SEED = 42     # Use seed 42 checkpoints for efficiency
EVAL_STEP = 800
PROTOCOLS = ["B", "D"]


def build_dataloader(tokenizer, n_samples):
    """Build C4 validation dataloader."""
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    # Collect n_samples
    texts = []
    for i, example in enumerate(ds):
        if i >= n_samples:
            break
        texts.append(example["text"])

    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_tensors="pt",
    )

    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
    )

    def collate(batch):
        ids = torch.stack([x[0] for x in batch])
        mask = torch.stack([x[1] for x in batch])
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": ids.clone(),
        }

    # Shuffle the indices for better coverage
    return DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate,
    )


def compute_ppl(model, eval_dl, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in eval_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            n_tokens = batch["attention_mask"].sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    return float(torch.exp(torch.tensor(avg_loss)).item())


def main():
    logger.info("=" * 60)
    logger.info("C4 Perplexity Evaluation: Qwen2.5-7B Checkpoints")
    logger.info("Protocols: %s | Seed: %d | Samples: %d",
                PROTOCOLS, SEED, N_EVAL)
    logger.info("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=False, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build C4 dataloader
    logger.info("Loading C4 validation set...")
    t0 = time.time()
    eval_dl = build_dataloader(tokenizer, N_EVAL)
    logger.info("Dataloader ready in %.0fs", time.time() - t0)

    # Baseline (untrained model)
    logger.info("\n>>> BASELINE: Untrained Qwen2.5-7B on C4")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=False, local_files_only=True,
    )
    device = next(base_model.parameters()).device
    baseline_ppl = compute_ppl(base_model, eval_dl, device)
    logger.info("Baseline C4 PPL: %.2f", baseline_ppl)
    del base_model
    torch.cuda.empty_cache()

    results = {"baseline": {"dataset": "C4", "ppl": baseline_ppl, "n_samples": N_EVAL}}

    # Evaluate checkpoints
    for proto in PROTOCOLS:
        ckpt_name = f"ckpt_Qwen25-7B_P{proto}_{EVAL_STEP}s_s{SEED}"
        ckpt_dir = RUNS_DIR / ckpt_name / "checkpoints" / f"step_{EVAL_STEP:05d}"

        if not ckpt_dir.exists():
            logger.warning("Not found: %s — skipping", ckpt_dir)
            continue

        logger.info("\n>>> PROTOCOL %s SEED %d", proto, SEED)

        try:
            # Load model + checkpoint
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=False, local_files_only=True,
            )

            # Load checkpoint
            logger.info("Loading state_dict...")
            sd = torch.load(str(ckpt_dir / "model_weights.pt"), map_location="cpu")

            if proto == "D":
                # PEFT format — need LoRA wrapper
                from peft import LoraConfig, get_peft_model
                # Infer config from state_dict
                lora_targets = set()
                lora_r = None
                for k in sd:
                    if "lora_A" in k:
                        parts = k.split(".")
                        lora_targets.add(parts[parts.index("lora_A") - 1])
                        if lora_r is None:
                            lora_r = sd[k].shape[0]

                lora_config = LoraConfig(
                    r=lora_r or 8, lora_alpha=16, lora_dropout=0.05,
                    target_modules=sorted(lora_targets)
                    if lora_targets else ["q_proj", "v_proj", "k_proj", "o_proj"],
                )
                model = get_peft_model(model, lora_config)

            model.load_state_dict(sd, strict=False)

            ppl = compute_ppl(model, eval_dl, device)
            logger.info("C4 PPL: %.2f", ppl)
            results[f"P{proto}_s{SEED}"] = {
                "protocol": f"P{proto}", "seed": SEED,
                "dataset": "C4", "ppl": ppl, "n_samples": N_EVAL,
            }

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error("FAILED: %s", e, exc_info=True)
            results[f"P{proto}_s{SEED}"] = {"error": str(e)}

    # Save and summarize
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nSaved: %s", OUTPUT_FILE)

    # Quick comparison
    logger.info("\n=== C4 vs WikiText-2 PPL ===")
    logger.info("%-15s %10s %10s", "Model", "C4 PPL", "WikiText-2 PPL")
    logger.info("%-15s %10.2f %10s", "Baseline", baseline_ppl, "133.16")
    for label, data in results.items():
        if label == "baseline":
            continue
        if "error" not in data:
            # WikiText-2 ref values from paper
            wt2_ref = {"PB_s42": "1.26", "PD_s42": "10.41"}.get(
                label, "N/A"
            )
            logger.info("%-15s %10.2f %10s", label, data["ppl"], wt2_ref)

    return 0


if __name__ == "__main__":
    sys.exit(main())
