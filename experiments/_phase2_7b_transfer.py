#!/usr/bin/env python3
"""
Phase 2: Cross-Architecture Transfer — r=64 on Qwen2.5-7B
Validates Phase 1 rank saturation law transferability.
1 run, AdamW, 100 steps, seed 42.
"""

import json, logging, sys, time, gc
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase2")

MODEL_NAME = "Qwen/Qwen2.5-7B"
RANK, ALPHA = 64, 128
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
MAX_SEQ_LEN, BATCH_SIZE, GRAD_ACCUM = 1024, 1, 4
N_TRAIN, N_EVAL, LR, MAX_STEPS = 800, 100, 1e-4, 100
SEED = 42
OUT_DIR = Path("runs/rank_curve")
OUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)


def build_dataloader(tokenizer, split, n_samples):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    ds = ds.select(range(min(n_samples, len(ds))))

    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def collate(b):
        ids = torch.stack([x["input_ids"] for x in b])
        mask = torch.stack([x["attention_mask"] for x in b])
        return {"input_ids": ids, "attention_mask": mask, "labels": ids.clone()}
    return DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=(split == "train"), collate_fn=collate)


def ppl_eval(model, dl, device):
    model.eval(); tl = 0.0; tt = 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            lo = model(**b).loss; nt = b["attention_mask"].sum().item()
            tl += lo.item() * nt; tt += nt
    return float(torch.exp(torch.tensor(tl / max(tt, 1))).item())


def main():
    logger.info("=" * 60)
    logger.info("Phase 2: r=64 on Qwen2.5-7B — cross-architecture transfer")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading Qwen2.5-7B base model...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=False, local_files_only=True)
    device = next(base.parameters()).device

    logger.info("Applying LoRA r=%d α=%d...", RANK, ALPHA)
    model = get_peft_model(base, LoraConfig(
        r=RANK, lora_alpha=ALPHA, lora_dropout=0.05, target_modules=TARGET_MODULES))
    model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable: %.1fM params", n_params / 1e6)

    train_dl = build_dataloader(tokenizer, "train", N_TRAIN)
    eval_dl = build_dataloader(tokenizer, "test", N_EVAL)

    # Phase 1 prediction: r=64 PPL ≈ 1.7-2.0 on 0.5B
    # Transfer prediction: r(7B) ≈ r(0.5B) * sqrt(7000/494)^{0.5}
    # PPL(7B, r=64) should be in [1.5, 2.5] if formula transfers
    logger.info("Training (AdamW, lr=1e-4, 100 steps)...")
    t0 = time.time()
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=0.01)
    step, acc = 0, 0
    model.train()
    while step < MAX_STEPS:
        for b in train_dl:
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss / GRAD_ACCUM
            loss.backward(); acc += 1
            if acc >= GRAD_ACCUM:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                step += 1; acc = 0
                if step >= MAX_STEPS: break

    ppl = ppl_eval(model, eval_dl, device)
    elapsed = time.time() - t0
    logger.info("DONE: PPL=%.4f (%.0fs, %.1fmin)", ppl, elapsed, elapsed / 60)

    result = {
        "model": "Qwen2.5-7B", "rank": RANK, "alpha": ALPHA,
        "trainable_params_M": round(n_params / 1e6, 1),
        "ppl": round(ppl, 4), "wall_time_s": int(elapsed),
        "max_steps": MAX_STEPS, "seed": SEED,
        "prediction_from_0.5B": f"r=64: PPL ≈ 1.7-2.0",
    }

    out_file = OUT_DIR / "phase2_7b_r64.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved: %s", out_file)

    # Validation
    logger.info("\n=== Transfer Validation ===")
    logger.info("0.5B r=64 prediction:  PPL ≈ 1.7-2.0")
    logger.info("7B r=64 measured:      PPL = %.4f", ppl)
    if 1.5 <= ppl <= 2.5:
        logger.info("✅ IN RANGE — rank saturation law transfers!")
    else:
        logger.info("⚠️  OUT OF RANGE — transfer formula needs recalibration")

    del model, base; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
