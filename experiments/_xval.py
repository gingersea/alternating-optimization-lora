#!/usr/bin/env python3
"""Cross-arch rank curve: 4 cached models × 5 runs each."""

import json, logging, sys, time, gc
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("xval")

ML, BS, GA, MS = 1024, 1, 4, 100
NT, NE, LR, SD = 800, 100, 1e-4, 42
torch.manual_seed(SD)

MODEL_CFG = [
    # Reference model (matching config to confirm phase transition)
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B",
     ["q_proj", "v_proj", "k_proj", "o_proj"]),
    # 7B-scale LoRA-only (B_full OOM on single GPU)
    ("Mistral-7B-lora", "mistralai/Mistral-7B-v0.3",
     ["q_proj", "v_proj", "k_proj", "o_proj"], True),  # skip B_full
]
RANKS = [8, 32, 256]
OUT_DIR = Path("runs/cross_arch")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_dl(tokenizer, split, n_samples):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    ds = ds.select(range(min(n_samples, len(ds))))
    tokenized = ds.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=ML, padding="max_length"),
        batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(tokenized, batch_size=BS,
                      shuffle=(split == "train"),
                      collate_fn=lambda b: {
                          "input_ids": torch.stack([x["input_ids"] for x in b]),
                          "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                          "labels": torch.stack([x["input_ids"] for x in b]).clone()})


def calc_ppl(model, dl, device):
    model.eval()
    tl, tt = 0.0, 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            lo = model(**b).loss
            nt = b["attention_mask"].sum().item()
            tl += lo.item() * nt
            tt += nt
    return round(float(torch.exp(torch.tensor(tl / max(tt, 1))).item()), 4)


def train_loop(model, optimizer, train_dl, device):
    model.train()
    step, acc = 0, 0
    while step < MS:
        for b in train_dl:
            b = {k: v.to(device) for k, v in b.items()}
            (model(**b).loss / GA).backward()
            acc += 1
            if acc >= GA:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                acc = 0
                if step >= MS:
                    break


def run_one(name, hf, targets):
    logger.info("\n" + "=" * 60)
    logger.info("MODEL: %s (%s)", name, hf)
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(hf, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dl = build_dl(tokenizer, "train", NT)
    eval_dl = build_dl(tokenizer, "test", NE)

    results = []

    # ── Baseline ──
    logger.info(">>> Baseline")
    m = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16,
                                              device_map="auto", trust_remote_code=False,
                                              local_files_only=True)
    dev = next(m.parameters()).device
    bl = calc_ppl(m, eval_dl, dev)
    total_params = sum(p.numel() for p in m.parameters())
    results.append({"run": f"{name}_baseline", "ppl": bl})
    logger.info("  Baseline PPL: %.4f (%dM total params)", bl, total_params // 1_000_000)
    del m
    gc.collect()
    torch.cuda.empty_cache()

    # ── Protocol B: full-rank ──
    logger.info(">>> B_full")
    m = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16,
                                              device_map="auto", trust_remote_code=False,
                                              local_files_only=True)
    dev = next(m.parameters()).device
    m.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    t0 = time.time()
    train_loop(m, opt, train_dl, dev)
    pp = calc_ppl(m, eval_dl, dev)
    elapsed = time.time() - t0
    results.append({"run": f"{name}_B_full", "ppl": pp, "time_s": int(elapsed)})
    logger.info("  PPL=%.4f (%.0fs)", pp, elapsed)
    del m, opt
    gc.collect()
    torch.cuda.empty_cache()

    # ── Protocol D: LoRA ranks ──
    for rank in RANKS:
        alpha = int(rank * 2)
        logger.info(">>> D_r%d (α=%d)", rank, alpha)
        base = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16,
                                                     device_map="auto", trust_remote_code=False,
                                                     local_files_only=True)
        dev = next(base.parameters()).device
        m = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05,
                                             target_modules=targets))
        m.gradient_checkpointing_enable()
        np_sum = sum(p.numel() for p in m.parameters() if p.requires_grad)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()),
                                lr=LR, weight_decay=0.01)
        t0 = time.time()
        train_loop(m, opt, train_dl, dev)
        pp = calc_ppl(m, eval_dl, dev)
        elapsed = time.time() - t0
        results.append({"run": f"{name}_D_r{rank}", "ppl": pp,
                        "params_M": round(np_sum / 1e6, 1), "time_s": int(elapsed)})
        logger.info("  PPL=%.4f (%dM trainable, %.0fs)", pp, np_sum // 1_000_000, elapsed)
        del m, base, opt
        gc.collect()
        torch.cuda.empty_cache()

    # Save
    out_file = OUT_DIR / f"{name.replace('/', '_')}_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", out_file)
    return results


def main():
    all_results = []
    for name, hf, targets in MODEL_CFG:
        try:
            r = run_one(name, hf, targets)
            all_results.extend(r)
        except Exception as e:
            logger.error("FAIL %s: %s", name, e, exc_info=True)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-ARCH RANK CURVE SUMMARY")
    logger.info("%-25s %8s %8s %8s %8s %8s",
                 "Model", "Baseline", "B_full", "D_r8", "D_r32", "D_r256")
    for name, _, _ in MODEL_CFG:
        rs = [r for r in all_results if r["run"].startswith(name)]
        vals = {}
        for r in rs:
            run_name = r["run"].split("_", 1)[1]
            vals[run_name] = r.get("ppl", r.get("error", "ERR"))
        logger.info("%-25s %8s %8s %8s %8s %8s",
                     name,
                     str(vals.get("baseline", "?")),
                     str(vals.get("B_full", "?")),
                     str(vals.get("D_r8", "?")),
                     str(vals.get("D_r32", "?")),
                     str(vals.get("D_r256", "?")))

    # Phase transition check
    logger.info("\nPhase Transition Verification:")
    for name, _, _ in MODEL_CFG:
        rs = {r["run"]: r for r in all_results if r["run"].startswith(name)}
        d8 = rs.get(f"{name}_D_r8", {}).get("ppl")
        d32 = rs.get(f"{name}_D_r32", {}).get("ppl")
        bf = rs.get(f"{name}_B_full", {}).get("ppl")
        if d8 and d32:
            ratio = d8 / d32 if d32 else float("inf")
            status = "✅" if ratio > 3 else ("⚠️" if ratio > 1.5 else "❌")
            logger.info("  %s: r8/r32=%.1fx %s r32 vs B: %s %s",
                         name, ratio, status,
                         f"r32={d32} vs B={bf}" if bf else "N/A",
                         "(r32 << B? OK)" if d32 and bf and d32 < bf else "")

    with open(OUT_DIR / "summary_cross_arch.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
