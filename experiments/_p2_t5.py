#!/usr/bin/env python3
"""
P2: Encoder-Decoder Architecture Validation.
T5-3B encoder AND decoder stacks. LoRA on attention modules.
Tests whether r_min prediction (§6.9.3): r_min ≈ 5.4 per stack → r=8 sufficient.
Also tests: encoder may need LESS rank (frozen input embeddings → less correction).
"""
import json, sys, time, gc, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("p2")

ML, BS, GA, MS = 1024, 1, 4, 100
NTr, NEv, LR, SD = 800, 100, 1e-4, 42
RANKS = [4, 8, 32]
T5_TARGETS = ["q", "k", "v", "o"]  # T5 uses different naming
OUT = "runs/p2_t5"
torch.manual_seed(SD)


def dl(tok, sp, n):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=sp)
    ds = ds.select(range(min(n, len(ds))))
    ds = ds.map(lambda ex: tok(ex["text"], truncation=True, max_length=ML, padding="max_length"),
                 batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(ds, batch_size=BS, shuffle=(sp == "train"),
                       collate_fn=lambda b: {"input_ids": torch.stack([x["input_ids"] for x in b]),
                                             "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                                             "labels": torch.stack([x["input_ids"] for x in b])})


def ppl(m, dl, dev):
    m.eval(); tl, tt = 0.0, 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            lo = m(**b).loss; nt = b["attention_mask"].sum().item(); tl += lo.item() * nt; tt += nt
    return round(float(torch.exp(torch.tensor(tl / max(tt, 1))).item()), 4)


def run_rank(rank, tok, tr_dl, ev_dl, model_path):
    alpha = int(rank * 2)
    logger.info(">>> r=%d", rank)
    t0 = time.time()

    # T5 needs special handling — it's encoder-decoder
    base = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                        device_map="auto", trust_remote_code=False,
                                                        local_files_only=True)
    dev = next(base.parameters()).device
    m = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05,
                                         target_modules=T5_TARGETS))
    m.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=LR, weight_decay=0.01)
    m.train(); step, acc = 0, 0
    try:
        while step < MS:
            for b in tr_dl:
                b = {k: v.to(dev) for k, v in b.items()}; (m(**b).loss / GA).backward(); acc += 1
                if acc >= GA:
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()
                    step += 1; acc = 0
                    if step >= MS: break
    except Exception as e:
        logger.error("FAIL at step %d: %s", step, e)
    pp = ppl(m, ev_dl, dev); el = time.time() - t0
    logger.info("  PPL=%.4f (%dM params, %.0fs)", pp, n_params // 1_000_000, el)
    del m, base, opt; gc.collect(); torch.cuda.empty_cache()
    return {"rank": rank, "ppl": pp, "params_M": round(n_params / 1e6, 1), "time_s": int(el)}


def main():
    model_path = "google-t5/t5-3b"
    logger.info("P2: T5-3B encoder-decoder rank sufficiency test")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tr_dl = dl(tok, "train", NTr); ev_dl = dl(tok, "test", NEv)

    # Baseline
    logger.info(">>> Baseline")
    base = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                        device_map="auto", trust_remote_code=False,
                                                        local_files_only=True)
    dev = next(base.parameters()).device
    bl = ppl(base, ev_dl, dev)
    logger.info("  Baseline: %.2f", bl)
    del base; gc.collect(); torch.cuda.empty_cache()

    results = []
    for rank in RANKS:
        r = run_rank(rank, tok, tr_dl, ev_dl, model_path)
        results.append(r)

    # Verify predictions
    logger.info("\n=== T5-3B RESULTS ===")
    logger.info("%-8s %10s %10s", "Rank", "PPL", "vs r=8")
    r8 = next(r["ppl"] for r in results if r["rank"] == 8)
    for r in sorted(results, key=lambda x: x["rank"]):
        vs_r8 = r["ppl"] / r8 if r["ppl"] else 999
        logger.info("r%-7d %10.4f %10.3f", r["rank"], r["ppl"], vs_r8)

    # Test §6.9.3 prediction: r=8 should be at plateau
    r4 = next((r["ppl"] for r in results if r["rank"] == 4), None)
    r32 = next((r["ppl"] for r in results if r["rank"] == 32), None)

    logger.info("\n§6.9.3 PREDICTION: r_min(T5)=5.4 → r=8 at plateau")
    if r4 and r8 and r8 / r4 < 1.10:
        logger.info("  r=4 already at plateau → r_min ≤ 4 ✓")
    elif r8 and r32 and r32 / r8 < 1.10:
        logger.info("  r=8 at plateau (r=32 matches) → r_min ≤ 8 ✓")
    else:
        logger.info("  r=8 does NOT plateau → r_min > 8 ⚠")

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
