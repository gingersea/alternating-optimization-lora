# Presentation Speech — 10-15 minutes
# Alternating Optimization for LLM Post-Training: From Divergence to Verified Solutions
# jianghuanyun — Supervisor: Prof. Hoi To Wai
# CUHK Department of Computer Science and Engineering
#
# Narrative (per author's request): Background -> Research Question -> Logic of the
# 2x2 framework -> Focus on ONE problem (ASP column fails) -> Experiments -> Conclusions.
# Speaker notes: target ~140 words/min. Bold = emphasis. [SLIDE] maps to poster sections.
# Total ≈ 1750 words ≈ 12.5 min at 140 wpm. Cut suggestions marked (CUT) for 10-min.
#
# JUDGING-CRITERIA MAP (Faculty panel):
#   C1 Problem solving/contribution  -> "Contributions" poster block + Sec 2-5
#   C2 Results and significance      -> quantified headline (58x/10.7x/7x) + Finding 1-5
#   C3 Presentation/delivery         -> verdict strip + story arc + one-sentence summary
#   C4 Q&A knowledge                 -> 12 prepared answers below, organized by criterion

---

**[0:00-1:30] 1. RESEARCH BACKGROUND — why this problem matters (SLIDE: header)**

Good morning. Thank you for coming to my poster. Let me start with the context that motivates this project.

Fine-tuning large language models is the backbone of modern AI deployment — from instruction tuning to domain adaptation. But it is extremely expensive: fine-tuning a 7-billion-parameter model can take hundreds of GPU-hours, because the standard approach — AdamW on every parameter — needs thousands of gradient steps.

This cost motivates a fundamentally different idea: **Alternating Least Squares, or ALS.** ALS comes from the collaborative-filtering era — it made the Netflix Prize solvable. Instead of descending a loss by gradients, ALS solves a *closed-form* least-squares problem on a layer's weights:

W* = (XᵀX + λI)⁻¹XᵀY

One matrix solve replaces hundreds of gradient steps. So the idea of this project was to **apply ALS to the output layer of a large language model during post-training** — hoping to get the same closed-form acceleration for LLM customization.

The protocol we built — which we call **ASP** — alternates three phases per cycle: an ALS solve on the language-model head, an SGD phase on the whole model, and a small perturbation phase.

**[1:30-3:00] 2. THE RESEARCH QUESTION (SLIDE: Background & Problem)**

But when we ran it, we hit a sharp puzzle. **On shallow models, ALS works.** Twelve-layer OPT and twenty-four-layer Qwen-0.5B both converge. **On a deep model — Qwen2.5-7B, twenty-eight layers — it diverges catastrophically**, to NaN, within one to three cycles.

We tried every reactive patch in the literature: damping, layer-skipping, norm-clipping. **Eleven independent configurations. All eleven diverged.**

So the research question is twofold, and it defines everything that follows:

1. **Why does ALS-based post-training diverge on deep models — and can it be fixed?**
2. **What actually works for post-training, compared fairly, on the same deep model?**

To answer both questions rigorously, we built a systematic comparison framework — and I want to explain its logic, because it shapes every experiment that follows.

**[3:00-5:00] 3. THE 2×2 FRAMEWORK — its logic (SLIDE: Methods + Finding 4)**

The framework is a **2×2 factorial design**. Two factors, each with two levels.

The first factor is the **optimizer**: our ASP method versus AdamW, the gradient-based standard. The second factor is the **parameter form**: full-rank fine-tuning versus low-rank LoRA adapters. Crossing them gives four protocols:

|  | Full-rank | LoRA |
|---|---|---|
| **AdamW** | B | D |
| **ASP** | A | C |

Four cells, each answering a different question: Does the optimizer matter? Does the parameter form matter? And critically — **do they interact?**

The framework has one design principle that I want to emphasize: **FLOPs normalization.** One ALS solve costs orders of magnitude more than one AdamW step. Comparing by step count would be meaningless — so every protocol runs until it consumes the *same total floating-point operations*, with shared data, shared seeds, and identical evaluation. Everything else is held fixed. This is what makes the comparison fair rather than anecdotal.

Now — here is the key thing about the 2×2 grid. Look at the ASP column. **Both ASP cells fail, in two different ways.** Under full-rank, protocol A diverges. Under LoRA, protocol C converges but is an order of magnitude worse than AdamW on the same parameter form. The AdamW column is stable at every depth.

So the 2×2 framework localizes the problem precisely: **the failure lives in the ASP optimizer, not in the model and not in the parameter form.** That is the one problem I want to focus on for the rest of the talk.

**[5:00-8:30] 4. FOCUS: WHY DOES ASP FAIL? — controlled ablation (SLIDE: Finding 5 + ablation chart)**

ASP has three components: ALS, SGD, and perturbation. Which one causes the divergence? Prior work blamed vague things — "numerical instability," "optimizer mismatch," "the noise." I wanted evidence, so I ran a **five-condition controlled ablation** on Qwen2.5-7B, holding data, seeds, and budget fixed, and turning components on and off one at a time.

(Point at the ablation chart.)

- **SGD alone: converges**, 53.6 perplexity.
- **Perturbation noise alone: converges.**
- **ALS machinery with its weight update suppressed — a no-op ALS: converges.**
- **But ALS with a real weight modification: diverges in a single solve.** Perplexity jumps from 73 to 2 *billion* in one step.
- And ALS-plus-SGD, the full protocol: also diverges.

The verdict is unambiguous: **ALS weight modification — not the noise, not the optimizer, not hook overhead — is the necessary-and-sufficient cause of divergence within the tested space.** That was the first time this failure was established by controlled evidence rather than assumed.

And there is a causal explanation, not just a correlation. The key is the **residual connection** — the architectural feature that makes deep transformers trainable at all. Each layer computes h_{l+1} = h_l + f_l(h_l). When ALS modifies the output layer, the perturbation propagates backward through every residual connection, and each layer multiplies it by roughly 1.08 — the identity path preserves it, and the nonlinearity adds about eight percent. Over 27 layers that is an **eightfold amplification**, while SGD's per-cycle recovery is only about 0.005 — a sixteen-hundred-to-one asymmetry. The model cannot heal faster than the perturbation grows. The theory predicts the boundary **between 25 and 28 layers**, consistent with the observed 24-converges / 28-diverges split. (Honest caveat: 25–27 layers were not directly tested — this is a calibrated prediction.)

**[8:30-11:30] 5. FOCUS: CAN IT BE FIXED? — three repairs, all fail (SLIDE: Findings 1-3)**

Now the natural response to any diagnosis is to fix it. I designed three repairs — and rigorously establishing that they fail is one of the most valuable results of this project. Let me be honest about each.

**(1) Gradient injection — A-SYNC.** If the hard weight jump is the problem, why not inject the ALS solution as a *gradient bias* instead? Compute the ALS delta, revert the weights, and add it to the gradient. We called this A-SYNC. And here is the first lesson — a *methodological* one: **when we audited the implementation, we found the injection was a timing no-op.** It was added to the gradient buffer *after* the optimizer step and wiped by the next zero-grad. It never touched a single parameter — the convergence we had attributed to A-SYNC was just plain SGD.

We fixed the timing and ran the decisive control: **matched-budget pure SGD versus A-SYNC on Qwen2.5-7B — identical steps, identical learning rate, identical evaluation points.** The trajectories are indistinguishable: final perplexity 6.82 versus 6.83, trajectory correlation 0.99981. The injection contributes nothing.

**(2) The re-interpretation.** With the injection proven vacuous, our earlier twelve-variant "algorithm ranking" had to be explained by something else — and the suspect was the learning-rate schedule, which differed across variants. We tested it with pure SGD, no ALS machinery: fixed learning rate 50.7 PPL, cosine decay 54.7, exponential decay 57.1 — the same ranking as the 7B variants. **The perceived algorithm ranking was a learning-rate-schedule effect, not an injection-strength effect.**

**(3) Low-rank probe and soft-target ALS.** Two more repairs, both negative. Confining ALS to a low-rank probe head eliminates divergence — confirming the amplification mechanism — but caps quality: 22.8 PPL versus 6.83 for plain SGD on the 7B model. Widening the bottleneck, r = 64 to 1024, changes nothing; all within noise of pure SGD. Why? Because a closed-form solve of the cross-entropy objective recovers *the same descent direction* as the gradient itself — the ALS solve is structurally redundant with SGD. And the one objective where a closed-form solve might not be redundant — soft-target distillation, matching a teacher's logit distribution — also buys nothing: 52.6 versus 52.4, within noise.

So the design space of "salvaging ALS" is closed. In every case, a matched SGD control matches or beats the ALS mechanism.

**[11:30-13:15] 6. RESOLVING THE RESEARCH QUESTION — what works (SLIDE: Finding 4 + Verified Solutions)**

Let me now close the loop and answer the two research questions I posed at the start — because that is what "solving" means for this project: not salvaging a broken method, but answering the questions definitively.

**Question 1 — why does ALS-based post-training diverge on deep models, and can it be fixed?** The answer is: ALS weight modification interacting with residual amplification is the cause, established by controlled evidence; and the three natural repairs cannot fix it, each proven redundant against a matched SGD baseline. The honest conclusion is a *negative resolution*: this design space is closed, and future work should not re-enter it.

**Question 2 — what actually works, compared fairly, on the same deep model?** The 2×2 framework gives the clean, quantified answer, on Qwen2.5-7B at 800 steps, three seeds:

- **AdamW, full-rank: 1.25 perplexity.** Best absolute quality.
- **AdamW + LoRA: 10.41**, at roughly 0.1% of trainable parameters — the lowest trainable-parameter cost. (A memory advantage; FLOPs through frozen weights are similar.)
- And the surprising one: **plain SGD with a sustained learning rate — 6.83 perplexity in 4800 steps**, with a power-law tail toward about 3.5. No ALS, no perturbation, no injection — just gradient descent, with a learning rate that *doesn't decay to zero*.

For context, the pretrained Qwen2.5-7B baseline is **73.1 PPL** — so every verified solution is a large, real improvement over doing nothing: **AdamW 58×, plain SGD 10.7×, LoRA 7×.**

So the research arc is complete: **a research question (why does ALS diverge), observed through the 2×2 experiments, several solutions attempted (three repairs, all rigorously falsified), and a definitive conclusion** — on deep models, remove the ALS machinery and let gradient descent do its job with a sustained learning rate; best quality still requires AdamW.

**[13:15-14:00] 7. CLOSING — value statement (SLIDE: footer)**

Let me close on what this project's lasting value is, because it is easy to misread a negative result as a failure.

**First, a rigorous, controlled answer to a long-standing failure mode** — the ALS-divergence story had been explained away with hand-waving; we established the specific mechanism by controlled evidence and made it falsifiable.

**Second, a causal theory with predictive power** — the residual-amplification framework predicts the depth boundary, not just explains it in hindsight.

**Third, and most important: a negative result that saves other researchers from a plausible but ineffective design.** We showed — with matched-budget controls, at two scales — that gradient repair of ALS doesn't work, and *why*: the closed-form solve is redundant with the gradient. That is a saved research direction, not a dead end. And along the way we built a reproducibility discipline — including the evaluation-harness audit that corrected our own earlier numbers — that transfers beyond this project.

Thank you. My poster walks through each of these results with the experiment charts. I'm happy to take questions — especially challenges; the negative results are the part I'd defend hardest. Thank you.

---

## TIMING REFERENCE
- 0:00  Research background (why post-training is expensive; ALS promise)
- 1:30  Research question (shallow converges / deep diverges; 11/11 failures)
- 3:00  The 2×2 framework — its logic (two factors, FLOPs normalization)
- 5:00  Focus: why ASP fails — controlled ablation + causal theory
- 8:30  Focus: can it be fixed — three repairs fail
- 11:30 Resolving the research question — what works, quantified
- 13:15 Closing / value statement
- 14:00 → Q&A

## ONE-SENTENCE SUMMARY (if asked)
"On deep LLMs, ALS weight modification causes divergence via residual amplification; three natural repairs are provably ineffective against matched SGD baselines; and the verified alternatives — AdamW, LoRA, plain SGD — deliver up to 58× improvement over the pretrained baseline, all established within a fair 2×2 factorial framework."

## ANTICIPATED Q&A — organized by judging criteria

### Criterion 1: Problem solving (contribution)
1. "So you're saying your own method doesn't work?" — The ALS family doesn't work *on deep models*, and we established the specific mechanism by controlled evidence, plus why the repairs can't fix it. The contribution is fourfold: a diagnosis (first controlled evidence isolating ALS weight modification), a causal theory with a falsifiable boundary, a saved design space (three plausible repairs proven ineffective), and verified alternatives with a corrected evaluation harness.
2. "Why the 2×2 framework — couldn't you just compare optimizers directly?" — Because optimizer and parameter form interact. Protocol A diverges but protocol C only underperforms — a one-dimensional comparison would misattribute the failure. The factorial design separates the optimizer effect, the parameter-form effect, and their interaction, with FLOPs normalization making steps comparable.

### Criterion 2: Results and significance (achievement)
3. "What's the headline number?" — Relative to the 73.1 PPL pretrained baseline: AdamW 1.25 (58×), plain SGD 6.83 (10.7×), LoRA 10.41 (7×). Decisive control: A-SYNC 6.82 vs pure SGD 6.83, correlation 0.99981.
4. "Why is AdamW 1.25 but plain SGD 6.83?" — Different budgets: AdamW at 800 steps is heavily trained on a small set (overfitting risk — in-distribution val only); SGD at 4800 steps is the fair matched control to A-SYNC. Both are valid operating points; LoRA splits the difference on parameter cost.
5. "What's the significance of the negative results?" — They close three plausible research directions with matched-budget evidence and identify *why* each fails (the closed-form solve is redundant with the CE gradient direction). A saved direction, not a dead end.

### Criterion 3: Presentation and delivery (clarity and quality)
6. "Summarize the project in one sentence." — See ONE-SENTENCE SUMMARY above.
7. "How is the story structured?" — Research background → research question → the 2×2 framework's logic → focus on the one problem (ASP column): ablation → causal theory → three failed repairs → resolve the research question with verified solutions, baselines, and quantified gains.

### Criterion 4: Q&A (knowledge of subjects)
8. "Is correlation 0.99981 meaningful?" — It's on the full 96-cycle trajectory, mean per-cycle difference 0.116 PPL. Honest caveat: Pearson correlation on trending series is inflated — we also report the per-cycle mean difference and matched final PPL; the curves are indistinguishable under those metrics too.
9. "Boundary 25–28 layers from only a few model depths?" — It's a two-point boundary (≤24 / ≥28), consistent across 8 architectures; the *trend* is predicted, the constant calibrated, and 25–27 layers were not directly tested. The poster says "consistent with observation."
10. "Why WikiText-2 only?" — Acknowledged limitation; cross-domain (C4, HellaSwag) is future work. PPL is a proxy for in-distribution fit, not generalization.
11. "A-KD 52.6 vs 52.4 — isn't that just noise?" — Exactly the point: the closed-form solver's advantage is *within noise* — nonexistent at this scale. We report it honestly with the noise caveat, which is what distinguishes it from a cherry-picked claim.
12. "How was the FLOPs budget matched across protocols?" — All protocols consume equal total FLOPs (forward+backward+optimizer+ALS amortized); LoRA's advantage is trainable-parameter/memory cost, not FLOPs — the poster says "lowest trainable-parameter cost," not "compute efficiency."
13. "What's the connection between your ablation and the 2×2 framework?" — The ablation isolates *which component* of the ASP optimizer causes failure; the 2×2 framework isolates *where* the failure lives relative to baselines. They answer complementary questions — mechanism inside the method, and method's standing in the landscape.

## CUT LIST (for a strict 10-minute version)
- The "timing no-op lesson" sentence in Repair 1.
- The "positive feedback loop" detail in the theory section (keep one sentence).
- One of the three repair details in the A-KD paragraph.
- The FLOPs-normalization design-principle paragraph can compress to one sentence.
