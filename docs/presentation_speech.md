# Presentation Speech — 10-15 minutes
# Alternating Optimization for LLM Post-Training: From Divergence to Verified Solutions
# jianghuanyun — Supervisor: Prof. Hoi To Wai
# CUHK Department of Computer Science and Engineering
#
# Narrative (per author's request): Background -> Research Question -> Logic of the
# 2x2 framework -> Focus on ONE problem (ASP column fails) -> Experiments -> Conclusions.
# Speaker notes: target ~140 words/min. Bold = emphasis. Each section header notes its 海报板块 (poster v9 block).
# Total ≈ 1750 words ≈ 12.5 min at 140 wpm. Cut suggestions marked (CUT) for 10-min.
#
# JUDGING-CRITERIA MAP (Faculty panel):
#   C1 Problem solving/contribution  -> "Contributions" poster block + Sec 2-5
#   C2 Results and significance      -> quantified headline (58x/10.7x/7x) + Finding 1-3
#   C3 Presentation/delivery         -> verdict strip + story arc + one-sentence summary
#   C4 Q&A knowledge                 -> 12 prepared answers below, organized by criterion

---

**[0:00-1:30] 1. RESEARCH BACKGROUND — why this problem matters (海报板块: 顶部 Header + 判决带 "ALS & its 3 repairs → DIVERGE | SGD / AdamW / LoRA → CONVERGE")**

Good morning. Thank you for coming to my poster. Let me start with the context that motivates this project.

Fine-tuning large language models is the backbone of modern AI deployment — from instruction tuning to domain adaptation. But it is extremely expensive: fine-tuning a 7-billion-parameter model can take hundreds of GPU-hours, because the standard approach — AdamW on every parameter — needs thousands of gradient steps.

This cost motivates a fundamentally different idea: **Alternating Least Squares, or ALS.** ALS comes from the collaborative-filtering era — it made the Netflix Prize solvable. Instead of descending a loss by gradients, ALS solves a *closed-form* least-squares problem on a layer's weights:

W* = (XᵀX + λI)⁻¹XᵀY

One matrix solve replaces hundreds of gradient steps. So the idea of this project was to **apply ALS to the output layer of a large language model during post-training** — hoping to get the same closed-form acceleration for LLM customization.

The protocol we built — which we call **ASP** — alternates three phases per cycle: an ALS solve on the language-model head, an SGD phase on the whole model, and a small perturbation phase.

**[1:30-3:00] 2. THE RESEARCH QUESTION (海报板块: 左上 "Research Question" — Q1/Q2)**

But when we ran it, we hit a sharp puzzle. **On shallow models, ALS works.** Twelve-layer OPT and twenty-four-layer Qwen-0.5B both converge. **On a deep model — Qwen2.5-7B, twenty-eight layers — it diverges catastrophically**, to NaN, within one to three cycles.

We tried every reactive patch in the literature: damping, layer-skipping, norm-clipping. **Eleven independent configurations. All eleven diverged.**

So the research question is twofold, and it defines everything that follows:

1. **Why does ALS-based post-training diverge on deep models — and can it be fixed?**
2. **What actually works for post-training, compared fairly, on the same deep model?**

**[3:00-5:30] 3. THE 2×2 FRAMEWORK — its logic (海报板块: 左栏 "The 2×2 Framework — Four Post-Training Algorithms")**

To answer both questions rigorously, we built a systematic comparison framework — and I want to explain it slowly, because it shapes every experiment that follows.

The framework is a **2×2 factorial design**. Two design choices define how any model is fine-tuned, and each choice has two options.

**Choice 1 — which optimizer drives the update?** The optimizer is the "steering algorithm" that decides how to adjust weights based on the loss. We compare two: **AdamW**, the standard gradient-based optimizer used across the industry, versus **ASP**, our alternating-optimization method — which alternates an ALS closed-form solve, SGD, and a perturbation phase.

**Choice 2 — how many parameters do we train?** Either **full-rank** — every weight in the model is updated — or **LoRA** — we freeze the original model and train only a tiny low-rank adapter ΔW = BA with rank r far smaller than the dimension d. LoRA has orders of magnitude fewer trainable parameters.

Crossing these two choices gives four post-training algorithms — let me describe each one in plain terms:

|  | Full-rank (train ALL params) | LoRA (train a small adapter) |
|---|---|---|
| **AdamW** (standard gradient optimizer) | **B** — gradient descent updates every weight of the whole model. This is the industry standard for fine-tuning. | **D** — same optimizer, but only the tiny adapter is trained; the original model is frozen. Far fewer parameters to update. |
| **ASP** (alternating: ALS + SGD + perturbation) | **A** — each cycle, ALS *replaces* the output-layer weights with a closed-form least-squares solution, then SGD and noise follow. This is the cell that **diverges on deep models.** | **C** — the same alternating schedule, but applied to the low-rank adapter. It converges, but quality is an order of magnitude worse than AdamW. |

**Why is each cell a different algorithm?** Because the optimizer and the parameter form interact. Training B (full AdamW) and D (LoRA AdamW) share the same steering algorithm but update different sets of weights. Training A and C share the same ALS machinery but apply it to different targets. So the grid answers three questions at once: Does the optimizer matter? Does the parameter form matter? And critically — **do they interact?**

One design principle makes the comparison fair: **FLOPs normalization.** One ALS solve costs orders of magnitude more compute than one AdamW step — comparing by step count would be meaningless. So every protocol runs until it consumes the *same total floating-point operations*, with shared data, shared seeds, and identical evaluation. Everything else is held fixed.

Now — here is the key thing about the 2×2 grid. Look at the ASP column. **Both ASP cells fail, in two different ways.** Under full-rank, protocol A diverges. Under LoRA, protocol C converges but is an order of magnitude worse than AdamW on the same parameter form. The AdamW column is stable at every depth.

So the 2×2 framework localizes the problem precisely: **the failure lives in the ASP optimizer, not in the model and not in the parameter form.** That is the one problem I want to focus on for the rest of the talk.

**[6:00-9:00] 4. FOCUS: WHY DOES ASP FAIL? — controlled ablation (海报板块: 右上 "Why ALS Diverges — The Theory" + "Finding 1 — ALS weight modification is the sole cause")**

ASP has three components: ALS, SGD, and perturbation. Which one causes the divergence? Prior work blamed vague things — "numerical instability," "optimizer mismatch," "the noise." I wanted evidence, so I ran a **five-condition controlled ablation** on Qwen2.5-7B, holding data, seeds, and budget fixed, and turning components on and off one at a time.

(Point at the ablation chart.)

- **SGD alone: converges**, 53.6 perplexity.
- **Perturbation noise alone: converges.**
- **ALS machinery with its weight update suppressed — a no-op ALS: converges.**
- **But ALS with a real weight modification: diverges in a single solve.** Perplexity jumps from 73 to 2 *billion* in one step.
- And ALS-plus-SGD, the full protocol: also diverges.

The verdict is unambiguous: **ALS weight modification — not the noise, not the optimizer, not hook overhead — is the necessary-and-sufficient cause of divergence within the tested space.** That was the first time this failure was established by controlled evidence rather than assumed.

And there is a causal explanation, not just a correlation. The key is the **residual connection** — the architectural feature that makes deep transformers trainable at all. Each layer computes h_{l+1} = h_l + f_l(h_l). When ALS modifies the output layer, the perturbation propagates backward through every residual connection, and each layer multiplies it by roughly 1.08 — the identity path preserves it, and the nonlinearity adds about eight percent. Over 27 layers that is an **eightfold amplification**, while SGD's per-cycle recovery is only about 0.005 — a sixteen-hundred-to-one asymmetry. The model cannot heal faster than the perturbation grows. The theory predicts the boundary **between 25 and 28 layers**, consistent with the observed 24-converges / 28-diverges split. (Honest caveat: 25–27 layers were not directly tested — this is a calibrated prediction.)

**[9:00-11:45] 5. FOCUS: CAN IT BE FIXED? — three repairs, all fail (海报板块: "Finding 2 — Gradient injection" + 左栏 "Repair analysis")**

Now the natural response to any diagnosis is to fix it. I designed three repairs — and rigorously establishing that they fail is one of the most valuable results of this project. Let me be honest about each.

**(1) Gradient injection — A-SYNC.** If the hard weight jump is the problem, why not inject the ALS solution as a *gradient bias* instead? Compute the ALS delta, revert the weights, and add it to the gradient. We called this A-SYNC. And here is the first lesson — a *methodological* one: **when we audited the implementation, we found the injection was a timing no-op.** It was added to the gradient buffer *after* the optimizer step and wiped by the next zero-grad. It never touched a single parameter — the convergence we had attributed to A-SYNC was just plain SGD.

We fixed the timing and ran the decisive control: **matched-budget pure SGD versus A-SYNC on Qwen2.5-7B — identical steps, identical learning rate, identical evaluation points.** The trajectories are indistinguishable: final perplexity 6.82 versus 6.83, trajectory correlation 0.99981. The injection contributes nothing.

**(2) The re-interpretation.** With the injection proven vacuous, our earlier twelve-variant "algorithm ranking" had to be explained by something else — and the suspect was the learning-rate schedule, which differed across variants. We tested it with pure SGD, no ALS machinery: fixed learning rate 50.7 PPL, cosine decay 54.7, exponential decay 57.1 — the same ranking as the 7B variants. **The perceived algorithm ranking was a learning-rate-schedule effect, not an injection-strength effect.**

**(3) Low-rank probe and soft-target ALS.** Two more repairs, both negative. Confining ALS to a low-rank probe head eliminates divergence — confirming the amplification mechanism — but caps quality: 22.8 PPL versus 6.83 for plain SGD on the 7B model. Widening the bottleneck, r = 64 to 1024, changes nothing; all within noise of pure SGD. Why? Because a closed-form solve of the cross-entropy objective recovers *the same descent direction* as the gradient itself — the ALS solve is structurally redundant with SGD. And the one objective where a closed-form solve might not be redundant — soft-target distillation, matching a teacher's logit distribution — also buys nothing: 52.6 versus 52.4, within noise.

So the design space of "salvaging ALS" is closed. In every case, a matched SGD control matches or beats the ALS mechanism.

**[11:45-13:30] 6. RESOLVING THE RESEARCH QUESTION — what works (海报板块: "Finding 3 — The 2×2 matrix" + 右下 "Verified Solutions")**

Let me now close the loop and answer the two research questions I posed at the start — because that is what "solving" means for this project: not salvaging a broken method, but answering the questions definitively.

**Question 1 — why does ALS-based post-training diverge on deep models, and can it be fixed?** The answer is: ALS weight modification interacting with residual amplification is the cause, established by controlled evidence; and the three natural repairs cannot fix it, each proven redundant against a matched SGD baseline. The honest conclusion is a *negative resolution*: this design space is closed, and future work should not re-enter it.

**Question 2 — what actually works, compared fairly, on the same deep model?** The 2×2 framework gives the clean, quantified answer, on Qwen2.5-7B at 800 steps, three seeds:

- **AdamW, full-rank: 1.25 perplexity.** Best absolute quality.
- **AdamW + LoRA: 10.41**, at roughly 0.1% of trainable parameters — the lowest trainable-parameter cost. (A memory advantage; FLOPs through frozen weights are similar.)
- And the surprising one: **plain SGD with a sustained learning rate — 6.83 perplexity in 4800 steps**, with a power-law tail toward about 3.5. No ALS, no perturbation, no injection — just gradient descent, with a learning rate that *doesn't decay to zero*.

For context, the pretrained Qwen2.5-7B baseline is **73.1 PPL** — so every verified solution is a large, real improvement over doing nothing: **AdamW 58×, plain SGD 10.7×, LoRA 7×.**

So the research arc is complete: **a research question (why does ALS diverge), observed through the 2×2 experiments, several solutions attempted (three repairs, all rigorously falsified), and a definitive conclusion** — on deep models, remove the ALS machinery and let gradient descent do its job with a sustained learning rate; best quality still requires AdamW.

**[13:30-14:15] 7. CLOSING — value statement (海报板块: 底部 "Takeaway" band + "Contributions")**

Let me close on what this project's lasting value is, because it is easy to misread a negative result as a failure.

**First, a rigorous, controlled answer to a long-standing failure mode** — the ALS-divergence story had been explained away with hand-waving; we established the specific mechanism by controlled evidence and made it falsifiable.

**Second, a causal theory with predictive power** — the residual-amplification framework predicts the depth boundary, not just explains it in hindsight.

**Third, and most important: a negative result that saves other researchers from a plausible but ineffective design.** We showed — with matched-budget controls, at two scales — that gradient repair of ALS doesn't work, and *why*: the closed-form solve is redundant with the gradient. That is a saved research direction, not a dead end. And along the way we built a reproducibility discipline — including the evaluation-harness audit that corrected our own earlier numbers — that transfers beyond this project.

Thank you. My poster walks through each of these results with the experiment charts. I'm happy to take questions — especially challenges; the negative results are the part I'd defend hardest. Thank you.

---

## TIMING REFERENCE
- 0:00  Research background (why post-training is expensive; ALS promise)
- 1:30  Research question (shallow converges / deep diverges; 11/11 failures)
- 3:00  The 2×2 framework — its logic + the four algorithms explained
- 6:00  Focus: why ASP fails — controlled ablation + causal theory
- 9:00  Focus: can it be fixed — three repairs fail
- 11:45 Resolving the research question — what works, quantified
- 13:30 Closing / value statement
- 14:15 → Q&A

## ONE-SENTENCE SUMMARY (if asked)
"On deep LLMs, ALS weight modification causes divergence via residual amplification; three natural repairs are provably ineffective against matched SGD baselines; and the verified alternatives — AdamW, LoRA, plain SGD — deliver up to 58× improvement over the pretrained baseline, all established within a fair 2×2 factorial framework."

## ANTICIPATED Q&A — 预备问答（总体 + 各算法细节）

> Organized in two parts: **A. 总体问题** (framing, methodology, claims) and **B. 各算法细节** (per-algorithm deep dives with exact numbers). Answers are grounded in `docs/final-report.md`, `docs/experiment-registry.md`, and the corrected-evaluation results. Each answer ends with the honest caveat if one exists.

### A. 总体问题 — Overall & framing

**A1. "Summarize the project in one sentence."**
See ONE-SENTENCE SUMMARY above.

**A2. "So your own method doesn't work?"**
The ALS family doesn't work *on deep models* (≥28 layers), and that's the finding, not a failure. The contribution is fourfold: (1) a **diagnosis** — first controlled evidence that ALS weight modification (not noise/optimizer/hook overhead) causes the divergence; (2) a **causal theory** with a falsifiable depth boundary; (3) a **saved design space** — three plausible repairs proven ineffective against matched-budget controls; (4) **verified alternatives** with a corrected evaluation harness.

**A3. "Why the 2×2 framework — couldn't you just compare optimizers directly?"**
Because optimizer and parameter form **interact**. Protocol A (ASP full-rank) diverges, but Protocol C (ASP+LoRA) only underperforms (135.4 PPL vs 10.41) — a one-dimensional comparison would misattribute the failure. The factorial design separates the optimizer effect, the parameter-form effect, and their interaction, with FLOPs normalization making steps comparable. The key output: the failure lives in the *optimizer*, not the parameter form.

**A4. "What's the single headline number?"**
Relative to the 73.1 PPL pretrained Qwen2.5-7B baseline: AdamW full-rank **1.25** (58×), plain SGD fixed-lr **6.83** (10.7×), AdamW+LoRA **10.41** (7×). Decisive control: A-SYNC 6.82 vs pure SGD 6.83, trajectory correlation 0.99981 — injection contributes nothing.

**A5. "Why is AdamW 1.25 but plain SGD 6.83 — which is better?"**
Different budgets, both valid operating points. AdamW at 800 steps is heavily trained on a small in-distribution set (overfitting risk — its downstream HellaSwag drops to 56.74% vs baseline 59.91%). Plain SGD at 4800 steps is the *fair matched control* to A-SYNC, still on a power-law tail toward ~3.5. LoRA (10.41) splits the difference on parameter cost and generalizes best downstream. "Better" depends on the axis: absolute quality → AdamW; parameter cost → LoRA; stability/dependency-free → SGD.

**A6. "What's the significance of the negative results?"**
They close three plausible research directions with matched-budget evidence and identify *why* each fails — the closed-form solve is redundant with the CE gradient direction. A saved direction (future researchers won't re-enter it), not a dead end. Plus a methodology lesson: a hybrid component (A-SYNC's injection) can be a no-op and masquerade as a contribution until rigorously controlled.

**A7. "How was the FLOPs budget matched?"**
All protocols run until they consume equal total FLOPs (forward+backward+optimizer+ALS amortized), because one ALS solve ≫ one AdamW step — comparing by step count would be meaningless. Data, seeds, and evaluation are shared. Note the asymmetry: LoRA's advantage is trainable-parameter/memory cost (~0.1% of params), **not** FLOPs (backprop through frozen weights dominates). On OPT-125m: LoRA 0.013 TFLOPs vs AdamW 0.911 TFLOPs — ~70× less compute, PPL-per-TFLOP 2812 vs 25.5.

**A8. "How do the ablation and the 2×2 framework relate?"**
Complementary questions. The ablation isolates *which component* of ASP causes failure (mechanism *inside* the method); the 2×2 isolates *where* the failure lives relative to baselines (the method's *standing* in the landscape). One gives the causal unit, the other gives the comparison frame.

**A9. "Is trajectory correlation 0.99981 meaningful?"**
It's on the full 96-cycle trajectory, with mean per-cycle difference 0.116 PPL. Honest caveat: Pearson correlation on trending (non-stationary) series is inflated. That's why we also report the per-cycle mean difference and matched final PPL (6.82 vs 6.83) — the curves are indistinguishable under those metrics too, so the conclusion doesn't rest on correlation alone.

**A10. "A 25–28 layer boundary from only a few model depths?"**
It's a two-point boundary (≤24 converge / ≥28 diverge), consistent across **8 architectures**: GPT-2 (12L), OPT-125m (12L), TinyLlama-1.1B (22L), Qwen2.5-0.5B (24L) converge; Qwen2.5-7B (28L), DeepSeek-1.5B (28L), SmolLM2-135M (30L), Mistral-7B (32L) diverge. The *trend* is predicted by theory (ρ≈1.08 per layer), the constant (L_max≈26) is calibrated, and 25–27 layers were not directly tested. The poster says "consistent with observation" — deliberately not "proven."

**A11. "Why WikiText-2 only?"**
Acknowledged limitation. The ALS-divergence diagnosis and theory are validated on WikiText-2; cross-domain (C4, HellaSwag, MMLU) validation is future work. PPL is a proxy for in-distribution fit, not generalization — which is exactly why we *did* run downstream evals for the verified solutions (see B6.2), where LoRA beats full-rank.

**A12. "What was the evaluation-harness audit, and why does it matter?"**
We found **two bugs** that had corrupted all OPT-125m absolute PPL numbers (but not the 7B results, which use the model's own tokenizer): (1) a cross-vocabulary tokenizer mismatch (GPT-2 ids fed to OPT — only 2 of 50,257 tokens map identically), and (2) unmasked padding labels inflating/deflating PPL. After correction, the OPT pretrained baseline is 73.7 (was an absurd "2246"). It matters because it shows the negative results are **not** artifacts of a broken task — training on the corrected task is a net improvement (73.7→50.7), so the nulls (probe, A-KD) are real.

**A13. "What's the one thing I should remember?"**
On deep LLMs, remove the ALS machinery and let gradient descent do its job with a *sustained* learning rate — the failure lives in the ALS solve, not in gradient training. Best absolute quality still needs AdamW; best parameter efficiency is LoRA.

**A14. "What was the actual story arc?"**
Research background (why post-training is expensive) → research question (shallow converges / deep diverges, 11/11) → the 2×2 framework's logic → focus on the one failing cell (ASP): five-condition ablation → residual-amplification theory → three failed repairs → resolve the question with verified solutions and quantified gains.

---

### B. 各算法细节 — Per-algorithm deep dive

#### B1. ASP / ALS full-rank (Protocol A) — the diverging cell

**B1.1 "What exactly does ASP do each cycle?"**
Three phases per cycle: (1) an **ALS solve** on the language-model head — compute W\* = (XᵀX + λI)⁻¹XᵀY and *replace* the output-layer weights; (2) an **SGD phase** on the whole model; (3) a small **perturbation phase**. The closed-form solve is the distinctive part: one matrix solve replaces hundreds of gradient steps.

**B1.2 "Why is one solve worth hundreds of gradient steps?"**
ALS solves the least-squares objective argmin ‖XWᵀ − Y‖² in closed form, the same trick that made the Netflix Prize solvable in collaborative filtering. The gradient of that objective is Xᵀ(XW−Y), so gradient descent must iterate down this slope step-by-step, while ALS jumps directly to the argmin. On shallow models this converges (OPT-125m, TinyLlama-1.1B, Qwen-0.5B all train fine); the problem is exclusively on deep models.

**B1.3 "Which component causes divergence, and what's the evidence?"**
The five-condition ablation on Qwen2.5-7B, holding data/seeds/budget fixed: SGD-only 53.6 ✓, perturb-only 94.4 ✓, ALS(no-op)+SGD 54.7 ✓, **ALS-only 2×10⁹ ✗** (73→2 billion in one step), ALS+SGD 3×10⁸ ✗. Every condition with a *real* weight solve diverges; every ALS-free condition converges. Verdict: ALS weight modification — not noise, not SGD, not hook overhead — is the necessary-and-sufficient cause within the tested space.

**B1.4 "Why do residual connections amplify the perturbation?"**
Each layer computes h_{l+1} = h_l + f_l(h_l). Linearizing the response to a perturbation: δ_{l+1} = (I + J_l)δ_l, with ‖I + J_l‖ ≈ 1.08 — the identity path preserves the perturbation exactly, and the nonlinearity adds ~8%. Over 27 layers that's ≈8× amplification (1.08²⁷), while SGD's per-cycle recovery is only ≈0.005 — a **1600:1 asymmetry**. The model cannot heal faster than the perturbation grows.

**B1.5 "Why does the perturbation grow across cycles?"**
The measured ALS delta δ grows 0.085 → 0.196 (×2.3) across cycles, because SGD's recovery shifts the activation distribution, so the next ALS solve finds a *larger* discrepancy. That's a positive feedback loop — the divergence is self-reinforcing, which is precisely why reactive patches (damping/clipping) only delay it, never prevent it.

**B1.6 "What patches did you try, and why did they fail?"**
Depth-aware damping, layer-skipping, and norm-clipping — 11 independent configurations, all diverged on 7B. They reduce symptoms (the δ magnitude per step) but never address the cause (the amplification that grows δ across cycles). The feedback loop (B1.5) explains why any fixed-threshold patch is eventually overwhelmed.

**B1.7 "Is 8 architectures enough to claim a universal boundary?"**
It's strong but not universal proof. 8 architectures across 4 model families (GPT-2, OPT, Llama, Qwen, plus DeepSeek/SmolLM/Mistral at the deep end) is unusually broad for a single-project negative result. Honest caveat: it's a *necessary-condition* pattern (all deep models diverge), not a proof that no deep model could ever be stabilized by a different ALS variant.

#### B2. A-SYNC / gradient injection (repair 1)

**B2.1 "What was A-SYNC supposed to do?"**
Keep ALS's direction-finding while avoiding the fatal hard weight jump: compute the ALS delta δ = W_als − W_before, **revert** the weights, then add δ to the *gradient* as a bias (strength `sync`), so SGD steps in a direction informed by the ALS solution rather than having weights replaced outright.

**B2.2 "What exactly was the timing no-op?"**
When we audited the implementation, the injection was added to the gradient buffer **after** `optimizer.step()` and wiped by the next `zero_grad()`. It never touched a single parameter. The "convergence" we had attributed to A-SYNC was literally just plain SGD running — a perfect demonstration of why hybrid components must be verified to actually modify what they claim, when they claim to.

**B2.3 "What did the corrected-timing experiment show?"**
The decisive matched-budget control on Qwen2.5-7B (identical 4800 steps, lr=2e-4, eval points): A-SYNC 6.82 vs pure SGD 6.83, correlation 0.99981, mean per-cycle diff 0.116 PPL. The injection contributes **nothing**. At OPT-125m, a corrected-timing sync sweep (strength 0.05/0.5/2.0) never beat pure SGD and was mildly harmful. Both scales agree.

**B2.4 "Why is the injection redundant with the gradient?"**
Because the ALS closed-form solve of the cross-entropy objective recovers **the same descent direction** as the gradient itself — the gradient of the least-squares objective at the current weights *is* (XᵀX + λI)W − XᵀY, and its zero is exactly W_als. So injecting W_als − W as a gradient bias points in a direction SGD already follows. There is no "new information" in the injection to exploit.

#### B3. Low-rank probe (repair 2)

**B3.1 "What is the probe, and why does it eliminate divergence?"**
Instead of ALS-replacing the full lm_head, confine the closed-form solve to a tiny low-rank probe head (rank r). It eliminates divergence — which *confirms* the amplification mechanism: a low-rank head's weight jump produces a much smaller effective perturbation δ, below the amplification runaway threshold.

**B3.2 "Why does it cap quality?"**
On 7B, probe r=64 reaches only 22.8 PPL vs 6.83 for plain SGD. The probe *removes* divergence but also removes the capacity the solve would need to help — you're deliberately shrinking the ALS's reach, so the best it can do is "not break anything," never "improve anything."

**B3.3 "What did the rank sweep show?"**
r = 64 / 256 / 1024 → 50.7 / 50.6 / 50.7 PPL, all within seed noise of pure SGD's 50.7 (OPT-125m, corrected harness). Widening the bottleneck from 64 to 1024 (16×) changes nothing. Conclusion: it's not a capacity ceiling — the closed-form solve is simply redundant with the CE gradient at *every* rank.

#### B4. A-KD / soft-target ALS (repair 3)

**B4.1 "What is A-KD?"**
The one objective where a closed-form solve *might* not be redundant with the gradient: soft-target distillation — matching a teacher's logit distribution (KL loss) rather than the one-hot CE target. We tested closed-form ALS against SGD on this same objective.

**B4.2 "Why is 52.6 vs 52.4 'within noise' rather than a real gain?"**
Because the closed-form solver's 52.6 is *indistinguishable* from SGD's 52.4 on the same objective (OPT-125m, corrected harness). If the solve had a real edge on this non-redundant objective, we'd expect a separation — instead it's within noise. We report it honestly, which is what distinguishes this from a cherry-picked claim.

**B4.3 "So is every ALS salvage path now closed?"**
The three natural repairs — gradient injection, low-rank confinement, soft-target distillation — are all provably redundant or harmful against matched SGD baselines. The one genuinely *untested* direction is applying ALS to objectives where gradient descent is structurally weak (e.g., attention-logit alignment), rather than objectives whose gradient ALS recovers. That's future work, not something we've falsified.

#### B5. AdamW full-rank (Protocol B) — best absolute quality

**B5.1 "Why is it 1.25 PPL?"**
800 steps, N=3 seeds (1.25 ± 0.01) on Qwen2.5-7B — 58× below the 73.1 baseline. AdamW's per-weight adaptive step size (moment estimates) plus decoupled weight decay descends the loss surface quickly and stably at any depth. It's the industry standard for a reason.

**B5.2 "Doesn't it overfit?"**
Yes, on small data. Round 9 showed overfitting at 400–1600 samples, and its downstream HellaSwag drops to 56.74% vs the 59.91% baseline — in-distribution PPL 1.25 does **not** imply better generalization. We state this explicitly: the 58× headline is an in-distribution number.

**B5.3 "What's the compute cost?"**
The highest of the verified solutions: full forward+backward on all parameters, thousands of steps, hundreds of GPU-hours for 7B. On OPT-125m it's 0.911 TFLOPs vs LoRA's 0.013. That cost is exactly the problem ALS was meant to solve — and the project's answer is that the *cheap* closed-form shortcut doesn't survive on deep models.

#### B6. AdamW + LoRA (Protocol D) — best parameter efficiency

**B6.1 "Why 10.41, and what's the parameter count?"**
AdamW applied to a low-rank adapter ΔW = BA (r ≪ d, ~0.1% of trainable parameters), base model frozen. 10.41 ± 0.01 PPL at 800 steps, 7× below baseline. The adapter's low rank forces a compact, generalizable correction.

**B6.2 "Why does LoRA generalize better on downstream tasks?"**
Despite worse in-distribution PPL (10.41 vs 1.25), LoRA wins downstream: HellaSwag 59.74% (baseline 59.91%, vs full-rank's 56.74%), MMLU 76.34% (+4.18pp over full-rank), ARC-Challenge +3.25pp. The low-rank constraint acts as implicit regularization — full-rank AdamW memorizes the small training set, LoRA can't. This is the "parameter form matters" evidence the 2×2 was built to surface.

**B6.3 "FLOPs vs memory — what's the real efficiency claim?"**
Nearly the same FLOPs as full-rank (backprop through frozen weights dominates), but optimizer state and adapter memory are orders of magnitude smaller — and serving many task-adapters from one frozen base is far cheaper. The honest phrasing (also on the poster) is "lowest trainable-parameter cost," **not** "compute efficiency."

#### B7. Plain SGD, fixed lr — the verified surprise

**B7.1 "Why is plain SGD a 'verified solution' and not just a baseline?"**
Because it *is* the finding: a 28-layer model that diverged 11/11 under ALS converges to 6.83 PPL under nothing but plain SGD with a fixed learning rate (4800 steps, lr=2e-4, momentum=0, wd=0.01). It's the decisive control that proves the failure lives in the ALS solve, not in gradient training — and it's a clean, reproducible number the field can build on.

**B7.2 "Why does fixed lr beat cosine/exponential decay?"**
A decaying schedule (lr→0) starves the tail of SGD progress and plateaus early (Cosine plateaus at 13.2 on 7B). A *sustained* learning rate keeps making progress — the CONSTANT schedule reaches 6.82. This is the re-interpretation of our earlier "variant ranking" (see B8).

**B7.3 "What's the power-law tail and the ~3.5 asymptote?"**
The 96-cycle curve fits PPL(c) = 3.48 + 61.3·c^−0.70, still improving at −0.013 PPL/cycle by cycle 88, asymptote ≈3.5. It means 6.83 is a *budget-limited* snapshot, not a convergence floor — more cycles would keep improving toward ~3.5.

**B7.4 "Is SGD a contribution or an admission that ALS failed?"**
Both, honestly. As a *result*, it's the admission: ALS's closed-form advantage evaporates against a matched SGD control. As a *deliverable*, it's a contribution: a minimal, dependency-free, reproducible baseline (6.83 PPL on 28L) that future ALS-variant claims must now be measured against.

#### B8. The lr-schedule re-interpretation

**B8.1 "How did the 12-variant ranking turn out to be a schedule effect?"**
We originally ranked A-SYNC variants (Vanilla 25.8 < No-Perturb 16.6 < Cosine 13.2 < CONSTANT 6.82) as if injection strength mattered. Once the injection was shown to be a no-op (B2.2), the *only* variable left differing across variants was the learning-rate schedule. Tested directly with pure SGD (no ALS machinery): fixed lr 50.7 < cosine 54.7 < exp×0.8 57.1 — the **identical ranking**. The perceived algorithm ranking was a schedule effect, full stop.

**B8.2 "What's the evidence it's the schedule, not the injection?"**
Three independent lines: (1) the no-op audit (injection never reached any parameter); (2) the corrected-timing sync sweep (injection strength 0.05/0.5/2.0 all ≈ pure SGD, mildly harmful); (3) the pure-SGD lr-schedule verification reproducing the variant ranking without any ALS machinery. Any one line alone is suggestive; together they're conclusive.

#### B9. ASP + LoRA (Protocol C) — the other failing cell

**B9.1 "Why does Protocol C converge but stay bad?"**
Same alternating ALS+SGD+noise schedule, but applied to the low-rank adapter instead of full weights — 135.36 ± 9.05 PPL on 7B (800 steps), an order of magnitude worse than AdamW+LoRA's 10.41. The low-rank adapter's small perturbation avoids the amplification runaway (so no NaN), but the closed-form solve still provides no advantage over gradient descent — so you pay the ALS complexity and get nothing back.

**B9.2 "What does C's failure tell us about optimizer vs parameter form?"**
It's the interaction term of the 2×2. Full-rank ALS *diverges*; LoRA ALS *underperforms*. So the failure mode changes with parameter form — but the conclusion is invariant: **in both ASP cells, ALS loses to its AdamW counterpart**. The optimizer is the culprit in every row; parameter form only changes *how* ALS fails.

#### B10. Reproducibility & hardware (if pressed)

**B10.1 "Can I reproduce this?"**
Yes — everything is in the repo: core scripts `experiments/_diverge_cause_7b.py` (ablation), `_pure_sgd_96c_7b.py` (decisive control), `_probe_rank_sweep.py`, `_kd_als.py`, `_lr_schedule_sgd.py`, `_flops_sweep.py`; machine-readable results in `runs/`. The historical `_a_sync_*.py` scripts are retained *with* the no-op diagnosis documented.

**B10.2 "What hardware did this run on?"**
2× RTX 5090 (32GB each), 251GB RAM, PyTorch 2.12 + DeepSpeed ZeRO-2. Protocol B used DeepSpeedCPUAdam with CPU optimizer offload (24GB/GPU); Protocols C/D used device_map="auto" + 8-bit AdamW on a single GPU.

## CUT LIST (for a strict 10-minute version)
- The "timing no-op lesson" sentence in Repair 1.
- The "positive feedback loop" detail in the theory section (keep one sentence).
- One of the three repair details in the A-KD paragraph.
- The FLOPs-normalization design-principle paragraph can compress to one sentence.
