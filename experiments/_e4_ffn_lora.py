#!/usr/bin/env python3
"""E4: FFN LoRA — test break condition #3: does adapting FFN layers lower r_min?"""
import json, sys, time, gc, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader

torch.manual_seed(42)
ML, BS, GA, MS = 1024, 1, 4, 100
NTr, NEv, LR = 800, 100, 1e-4
MODEL = "Qwen/Qwen2.5-0.5B"
ATTN = ["q_proj", "v_proj", "k_proj", "o_proj"]
FFN = ["gate_proj", "up_proj", "down_proj"]
OUT = "runs/e4_ffn_lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False, local_files_only=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

def dl_fn(split, n):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    ds = ds.select(range(min(n, len(ds))))
    ds = ds.map(lambda ex: tokenizer(ex["text"], truncation=True, max_length=ML, padding="max_length"),
                 batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    def collate(b):
        return {"input_ids": torch.stack([x["input_ids"] for x in b]),
                "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                "labels": torch.stack([x["input_ids"] for x in b])}
    return DataLoader(ds, batch_size=BS, shuffle=(split == "train"), collate_fn=collate)

tr_dl = dl_fn("train", NTr)
ev_dl = dl_fn("test", NEv)

def ppl_eval(m, dl, dev):
    m.eval(); tl, tt = 0.0, 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            lo = m(**b).loss; nt = b["attention_mask"].sum().item(); tl += lo.item() * nt; tt += nt
    return round(float(torch.exp(torch.tensor(tl / max(tt, 1))).item()), 4)

def run(label, rank, targets):
    alpha = int(rank * 2)
    print(f">>> {label} (r={rank}, targets={targets})", flush=True)
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto",
                                                  trust_remote_code=False, local_files_only=True)
    dev = next(base.parameters()).device
    m = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, target_modules=targets))
    m.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=LR, weight_decay=0.01)
    m.train(); step, acc = 0, 0
    while step < MS:
        for b in tr_dl:
            b = {k: v.to(dev) for k, v in b.items()}; (m(**b).loss / GA).backward(); acc += 1
            if acc >= GA: torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad(); step += 1; acc = 0
            if step >= MS: break
    pp = ppl_eval(m, ev_dl, dev); elapsed = time.time() - t0
    print(f"  PPL={pp} ({n_params/1e6:.1f}M params, {elapsed:.0f}s)", flush=True)
    del m, base, opt; gc.collect(); torch.cuda.empty_cache()
    return {"run": label, "rank": rank, "ppl": pp, "params_M": round(n_params / 1e6, 1), "time_s": int(elapsed)}

def main():
    results = []
    results.append(run("attn_r4", 4, ATTN))
    results.append(run("attn_r8", 8, ATTN))
    results.append(run("attn+ffn_r4", 4, ATTN + FFN))
    results.append(run("ffn_r4", 4, FFN))

    print()
    print("=== E4 RESULTS: FFN LoRA ===")
    print(f"{'Config':<20s} {'Rank':>5s} {'Params(M)':>10s} {'PPL':>8s} {'vs attn_r8':>10s}")
    attn_r8_ppl = results[1]["ppl"]
    for r in results:
        vs = r["ppl"] / attn_r8_ppl
        print(f'{r["run"]:<20s} {r["rank"]:5d} {r["params_M"]:10.1f} {r["ppl"]:8.4f} {vs:10.3f}')

    print()
    attn_r4 = results[0]["ppl"]
    print(f"attn_r4 / attn_r8 = {attn_r4/attn_r8_ppl:.4f} (ref: should be ~1.01)")
    ffn_r4 = results[2]["ppl"]
    print(f"attn+ffn_r4 / attn_r8 = {ffn_r4/attn_r8_ppl:.4f}")
    if ffn_r4 < attn_r4 * 0.98:
        print("✅ FFN+attention IMPROVES over attention-only — r_min lowered")
    elif ffn_r4 <= attn_r4 * 1.02:
        print("= FFN+attention matches attention-only — r_min unchanged")
    else:
        print("⚠  FFN+attention is worse — FFN LoRA degrades performance")

    ff_only = results[3]["ppl"]
    print(f"ffn_only_r4 / attn_r8 = {ff_only/attn_r8_ppl:.4f}")
    if ff_only < attn_r8_ppl * 1.05:
        print("✅ FFN-only LoRA at r=4 reaches plateau — r_min even lower with FFN targets")
    elif ff_only < attn_r8_ppl * 1.20:
        print("= FFN-only LoRA close to plateau — r_min possibly lower")
    else:
        print("⚠  FFN-only LoRA significantly worse — attention modules essential")

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    sys.exit(main())
