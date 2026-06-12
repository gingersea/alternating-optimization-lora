"""R1: Multi-seed matrix experiment (3 seeds, 2 models, 5 step counts, Protocol A+B)."""
import json, logging, sys, time; from pathlib import Path
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset; from torch.utils.data import DataLoader
from altopt.trainer import AltOptTrainer, TrainerConfig
from altopt.framework import Phase, PhaseConfig, PhaseSchedule
from altopt.evaluation import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("R1")

MODELS = [("facebook/opt-125m", "opt", 2, 400, 100, False),
          ("Qwen/Qwen2.5-0.5B", "qwen", 1, 200, 50, True)]
STEPS = [50, 100, 200, 400, 800]; SEEDS = [42, 123, 456]
OUT = Path("runs/multi_seed_matrix")

def make_dl(tok, split, ml, bs, ds, n):
    ds_data = load_dataset("wikitext", ds, split=split)
    if n: ds_data = ds_data.select(range(min(n, len(ds_data))))
    def t(ex): return tok(ex["text"], truncation=True, max_length=ml, padding="max_length")
    td = ds_data.map(t, batched=True, remove_columns=["text"])
    td.set_format(type="torch", columns=["input_ids", "attention_mask"])
    def c(b):
        ids = torch.stack([x["input_ids"] for x in b])
        return {"input_ids":ids,"attention_mask":torch.stack([x["attention_mask"] for x in b]),"labels":ids.clone()}
    return DataLoader(td, batch_size=bs, shuffle=(split=="train"), collate_fn=c)

def build_sched(n_steps):
    sgd_pc = max(10, n_steps // 4); nc = max(1, n_steps // (sgd_pc + 2))
    return PhaseSchedule(phases=[
        PhaseConfig(phase=Phase.ALS, steps=1, block_size=1024),
        PhaseConfig(phase=Phase.SGD, steps=sgd_pc, lr=1e-4),
        PhaseConfig(phase=Phase.PERTURB, steps=1, noise_scale=1e-3)], cycles=nc)

for m_name, m_type, bs, n_train, n_eval, trust_rc in MODELS:
    tok = AutoTokenizer.from_pretrained(m_name, trust_remote_code=trust_rc)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    train_dl = make_dl(tok, "train", 128, bs, "wikitext-2-raw-v1", n_train)
    eval_dl = make_dl(tok, "test", 128, bs, "wikitext-2-raw-v1", n_eval)

    for n_steps in STEPS:
        sched = build_sched(n_steps)
        for proto, opt_type in [("A", "altopt"), ("B", "adamw")]:
            for seed in SEEDS:
                label = f"{m_type}_{proto}_{n_steps}s_s{seed}"
                logger.info(f"{label}")
                model = AutoModelForCausalLM.from_pretrained(m_name, trust_remote_code=trust_rc)
                cfg = TrainerConfig(protocol=proto, optimizer_type=opt_type, parameter_form="full_rank",
                    max_steps=n_steps, lr=5e-5 if m_type=="qwen" else 1e-4,
                    run_dir=f"/tmp/r1_{label}", seed=seed, eval_every=max(10,n_steps//4), save_every=10000)
                if opt_type == "altopt": cfg.phase_schedule = sched
                try:
                    trainer = AltOptTrainer(model, cfg, eval_dataloader=eval_dl, tokenizer=tok)
                    state = trainer.train(train_dl)
                    er = Evaluator(["perplexity","loss"], eval_dl).evaluate(model)
                    r = {"protocol":proto,"seed":seed,"steps":n_steps,"ppl":er["perplexity"],
                         "flops":state.cumulative_flops,"model":m_type}
                    logger.info(f"  ppl={er['perplexity']:.1f}")
                    # Write incrementally
                    out_f = OUT / f"{label}.json"
                    out_f.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_f, "w") as f: json.dump(r, f)
                except Exception as e:
                    logger.error(f"  FAIL: {e}")

logger.info("R1 complete")
