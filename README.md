# Alternating Optimization for LLM Post-Training

**From Divergence to Verified Solutions** — a research project investigating why ALS-based post-training diverges on deep LLMs, and what provably works instead.

**Author**: jianghuanyun · **Supervisor**: Prof. Hoi To Wai · CUHK CSE

---

## Overview

Alternating Least Squares (ALS) offers a closed-form alternative to gradient-based fine-tuning — one matrix solve replaces hundreds of gradient steps. Applied to LLM post-training, however, ALS **works on shallow models (≤24 layers) but catastrophically diverges on deep models (≥28 layers)** — 11/11 observed attempts on Qwen2.5-7B failed.

This project establishes, through controlled experiments and a 2×2 factorial framework (optimizer × parameter form):

1. **The cause** — ALS weight modification (not noise, optimizer, or hook overhead) is the necessary-and-sufficient cause of divergence, via a five-condition ablation.
2. **The theory** — residual connections amplify ALS perturbations by ρ≈1.08 per layer, predicting a 25–28-layer depth boundary consistent with 8 architectures.
3. **The failed repairs** — gradient injection, low-rank probing, and soft-target distillation all provably fail matched-budget controls.
4. **The verified solutions** — AdamW full-rank (1.25 PPL, 58× below baseline), AdamW+LoRA (10.41 PPL, 7×), plain SGD with fixed lr (6.83 PPL, 10.7×) on Qwen2.5-7B.

---

## Key Deliverables

| Deliverable | File | Description |
|---|---|---|
| **Poster** | [`jianghuanyun_Poster.pdf`](jianghuanyun_Poster.pdf) | 36×48in portrait, judged-poster presentation |
| **Final Report** | [`Final_Report.pdf`](Final_Report.pdf) | Full report with references |
| **Final Report (no refs)** | [`Final_Report_NoRefs.pdf`](Final_Report_NoRefs.pdf) | Report without reference section |
| **Final Report + VeriGuide** | [`Final_Report_with_VeriGuide.pdf`](Final_Report_with_VeriGuide.pdf) | Report merged with AI-detection report |
| **Presentation speech** | [`docs/presentation_speech.md`](docs/presentation_speech.md) | 10–15 min English talk, timed, with Q&A |
| **Problem-solution report** | [`docs/problem-solution-report.md`](docs/problem-solution-report.md) | Concise problem → solution summary |
| **Final report (md)** | [`docs/final-report.md`](docs/final-report.md) | Detailed findings, methods, evidence |

## Documentation

| Doc | Description |
|------|-------------|
| [`docs/diag-injection-report.md`](docs/diag-injection-report.md) | Timing-no-op diagnosis of gradient injection |
| [`docs/fair_comparison_methodology.md`](docs/fair_comparison_methodology.md) | FLOPs-normalized 2×2 factorial design |
| [`docs/math-analysis.md`](docs/math-analysis.md) | ALS reconstruction loss, convergence theory |
| [`docs/causal_depth_boundary.md`](docs/causal_depth_boundary.md) | Residual-amplification causal theory |
| [`docs/experiment-registry.md`](docs/experiment-registry.md) | Full experiment matrix with evidence tags |
| [`docs/claims-audit.md`](docs/claims-audit.md) | Claim → artifact traceability |
| [`docs/all-findings.md`](docs/all-findings.md) | All findings with evidence strength |
| [`docs/p2-synthesis.md`](docs/p2-synthesis.md) | P2 comprehensive assessment |

## Key Experiments (reproducible)

| Script | Experiment | Result file |
|--------|-----------|-------------|
| [`experiments/_diverge_cause_7b.py`](experiments/_diverge_cause_7b.py) | 5-condition controlled ablation | [`runs/diverge_cause_7b.json`](runs/diverge_cause_7b.json) |
| [`experiments/_pure_sgd_96c_7b.py`](experiments/_pure_sgd_96c_7b.py) | Decisive pure-SGD control (96c) | [`runs/pure_sgd_96c_7b.json`](runs/pure_sgd_96c_7b.json) |
| [`experiments/_probe_rank_sweep.py`](experiments/_probe_rank_sweep.py) | Low-rank probe sweep (r=64/256/1024) | [`runs/probe_rank_sweep_opt125m.json`](runs/probe_rank_sweep_opt125m.json) |
| [`experiments/_lr_schedule_sgd.py`](experiments/_lr_schedule_sgd.py) | lr-schedule verification | [`runs/lr_schedule_sgd_opt125m.json`](runs/lr_schedule_sgd_opt125m.json) |
| [`experiments/_kd_als.py`](experiments/_kd_als.py) | Soft-target (distillation) ALS | [`runs/kd_als_opt125m.json`](runs/kd_als_opt125m.json) |
| [`experiments/_flops_sweep.py`](experiments/_flops_sweep.py) | FLOPs-normalized comparison | [`runs/flops_sweep_opt125m.json`](runs/flops_sweep_opt125m.json) |

## Quick Start

```bash
pip install -e .
pytest tests/  # framework tests
python experiments/_diverge_cause_7b.py   # ablation (requires GPU + Qwen2.5-7B)
```

## Repository Structure

```
├── altopt/                    # Core framework (ALS, SGD, LoRA, trainer, evaluator)
├── experiments/               # Experiment scripts (_*.py are key verified runs)
├── docs/                      # Reports, methodology, theory, speech
│   ├── archive/               # Historical superseded reports
│   ├── figures/               # Generated figures
│   └── reference/             # Educational deep-dives
├── paper/                     # Paper draft, LaTeX, reviews
├── runs/                      # Machine-readable experiment results
├── tests/                     # Framework unit tests
├── tutorials/                 # Step-by-step framework walkthrough
├── figures/                   # Paper figures
├── jianghuanyun_Poster.pdf    # Poster deliverable
├── Final_Report*.pdf          # Report deliverables
└── README.md                  # This file
```

## License

MIT
