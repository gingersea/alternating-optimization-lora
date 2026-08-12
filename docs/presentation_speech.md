# Presentation Speech — 10-15 minutes
# Alternating Optimization for LLM Post-Training: From Divergence to Verified Solutions
# jianghuanyun — Supervisor: Prof. Hoi To Wai
# CUHK Department of Computer Science and Engineering
#
# Speaker notes: target ~140 words/min. Bold = emphasis. [SLIDE] markers map to poster sections.
# Total ≈ 1850 words ≈ 13.5 minutes at 140 wpm. Cut suggestions marked (CUT) for the 10-min version.

---

**[0:00-0:30] OPENING — the puzzle (SLIDE: header + Background)**

Good morning. Thank you for coming to my poster. My project is about *alternating optimization* for large-language-model post-training — and the puzzle that drives the whole project is this: **an optimization method that works beautifully on shallow models catastrophically diverges the moment you apply it to a deep model.** Over the course of this project I not only explained *why* it diverges — I also discovered that the most natural fixes for it don't work, and I verified what *does* work instead. Let me show you.

**[0:30-2:30] BACKGROUND — what ALS promises (SLIDE: Background & Problem)**

First, the method. **Alternating Least Squares — ALS — is the classic workhorse of collaborative filtering.** Instead of taking thousands of gradient steps, ALS solves a *closed-form* least-squares problem on a layer's weights:

W* = (XᵀX + λI)⁻¹XᵀY

One matrix solve replaces hundreds of gradient steps. For the Netflix Prize era, that was a revolution. The idea for this project: **apply that same closed-form solve to the output layer of a large language model during post-training.** The method — which we call the ASP protocol — alternates an ALS solve on the language-model head, a phase of SGD on the whole model, and a small perturbation phase.

Here is the puzzle. On shallow models — 12-layer OPT, 24-layer Qwen-0.5B — **ALS converges.** But on a 28-layer model, Qwen2.5-7B, it diverges. Not slowly — **catastrophically, to NaN, within one to three cycles. Every single attempt.** We tried eleven independent configurations — damping, layer-skipping, norm-clipping — all the reactive patches in the literature. **All eleven diverged.**

So the question became: *why?* And that's where the real work of this project began.

**[2:30-5:30] DIAGNOSIS — controlled ablation (SLIDE: Finding 5 + ablation chart)**

The first contribution is a **controlled ablation** that isolates the cause. A lot of prior work blamed vague things: "numerical instability," "optimizer mismatch," "the perturbation noise." I wanted to test each candidate *independently*. So I ran a five-condition experiment on Qwen2.5-7B, holding data, seeds, and budget fixed, and turning components on and off.

The result is unambiguous. (Point at chart.)

- **SGD alone: converges**, 53.6 perplexity.
- **Perturbation noise alone: converges.**
- **ALS machinery with its weight update suppressed — a no-op ALS: converges.**
- **But ALS with a real weight modification: diverges in a single solve.** Perplexity jumps from 73 to 2 *billion* in one step.
- And ALS-plus-SGD, the full protocol: also diverges.

So the verdict: **ALS weight modification — not noise, not the optimizer, not hook overhead — is the necessary-and-sufficient cause of divergence within the tested space.** That was the first time this failure was established by controlled evidence rather than assumed.

**[5:30-7:30] WHY — the causal theory (SLIDE: Methods)**

But a proof of *what* isn't an explanation of *why*. The second contribution is a causal theory. The key is the **residual connection** — the architectural feature that makes deep transformers trainable at all.

Each layer computes h_{l+1} = h_l + f_l(h_l). When ALS modifies the output layer, that perturbation δ propagates backward through every residual connection. Linearizing each layer, the perturbation is multiplied by roughly (I + J_l), where J_l is the layer Jacobian. And here's the structural fact: **for trained transformers, the per-layer amplification factor ‖I + J_l‖ is about 1.08.** The identity path preserves the perturbation exactly, and each layer's nonlinearity adds about eight percent more.

After L layers, that's 1.08^(L-1). At 28 layers — 1.08²⁷ — that's an **eightfold amplification**. Meanwhile, SGD's recovery capacity per cycle is tiny — on the order of 0.005. That's a sixteen-hundred-to-one asymmetry: **the model cannot heal faster than the perturbation grows.**

This theory predicts a critical depth. Solving for where amplification overwhelms recovery places the boundary **between 25 and 28 layers — consistent with the observed 24-converges / 28-diverges split.** (To be honest: 25–27 layers were not directly tested; this is a calibrated prediction.) And it explains a second observation: the perturbation magnitude *grows* across cycles, roughly doubling — because SGD's partial recovery shifts the hidden-state distribution, so the next ALS solve finds a larger discrepancy. It's a positive feedback loop. That's why damping patches only delay the failure — they never stop it.

**[7:30-10:30] THE ATTEMPTED REPAIRS — all three fail (SLIDE: Findings 1-3)**

Now, the natural response to any diagnosis is to fix it. I designed three repairs. **All three failed — and that failure, rigorously established, is one of the most valuable results of this project.** Let me be honest about each.

**(1) Gradient injection — A-SYNC.** If the problem is the hard weight jump, why not inject the ALS solution as a *gradient bias* instead? Compute the ALS delta, revert the weights, and add the delta to the gradient. This is what we called A-SYNC. And here's the first lesson — a *methodological* one: **when we audited the implementation, we found the injection was a timing no-op.** It was added to the gradient buffer *after* the optimizer step, and wiped by the next zero-grad. It never touched a single parameter. The convergence we'd been attributing to A-SYNC was just… plain SGD.

We fixed the timing and ran the decisive control: **matched-budget pure SGD vs. A-SYNC on Qwen2.5-7B, identical steps, identical learning rate, identical evaluation points.** The trajectories are indistinguishable — final perplexity 6.82 versus 6.83, with a trajectory correlation of 0.99981. The injection contributes nothing. (CUT for 10-min: "The lesson is worth stating plainly: before you attribute an effect to a hybrid component, verify that the component actually modifies what it claims, at the time it claims to.")

**(2) The re-interpretation.** With the injection proven vacuous, our earlier twelve-variant "algorithm ranking" had to be explained by something else — and the suspect was the learning rate schedule, which differed across variants. We tested this with pure SGD, no ALS machinery at all: fixed learning rate 50.7 PPL, cosine decay 54.7, exponential decay 57.1 — the same ranking as the seven-billion-parameter variants. **The perceived algorithm ranking was a learning-rate-schedule effect, not an injection-strength effect.**

**(3) The low-rank probe and soft-target ALS.** Two more repairs, both negative. First, confine ALS to a low-rank probe head parallel to the language-model head. This does eliminate divergence — confirming the amplification mechanism — but it caps quality: 22.8 PPL on the 7B model versus 6.83 for plain SGD. And widening the bottleneck, r = 64 to 1024, changes nothing — all within noise of pure SGD. Why? Because a closed-form solve of the cross-entropy objective recovers *the same descent direction* as the gradient itself — the ALS solve is structurally redundant with SGD.

Second, the one objective where a closed-form solve might *not* be redundant: soft-target distillation, matching a teacher's logit distribution. We built it — A-KD — and compared the closed-form solver against SGD on the same objective. **52.6 versus 52.4. Indistinguishable.** The exact solve buys nothing.

So the design space of "salvaging ALS" is closed: gradient injection, no-op; low-rank probing, no quality gain; soft-target distillation, no incremental value. In every case, a matched SGD control matches or beats the ALS mechanism.

**[10:30-12:30] THE VERIFIED SOLUTIONS (SLIDE: Finding 4 + Verified Solutions)**

But a negative result is only valuable if you point to what *does* work. Within our 2×2 factorial framework — optimizer crossed with parameter form — the answers are clean, on Qwen2.5-7B at 800 steps, three seeds:

- **AdamW, full-rank: 1.25 perplexity.** Best absolute quality.
- **AdamW + LoRA: 10.41**, at roughly 0.1% of trainable parameters. Lowest trainable-parameter cost (memory advantage; FLOPs through frozen weights are similar).
- And the surprising one: **plain SGD with a sustained learning rate — 6.83 perplexity in 4800 steps**, with a power-law tail converging toward about 3.5. No ALS, no perturbation, no injection. Just gradient descent, with a learning rate that *doesn't decay to zero*.

For context, the pretrained Qwen2.5-7B baseline on this benchmark is 73.1 PPL — so every verified solution is a large, real improvement over doing nothing.

The practical takeaway is deliberately blunt: **on deep models, remove the ALS machinery and let gradient descent do its job with a sustained learning rate.**

**[12:30-13:30] CONCLUSION + value statement (SLIDE: footer)**

Let me close on what this project's lasting value is, because I think it's easy to misread a negative result as a failure. Three things.

**First, a rigorous, controlled answer to a long-standing failure mode.** The ALS-divergence story had been explained away with hand-waving; we established the specific mechanism by controlled evidence and made it falsifiable.

**Second, a causal theory with predictive power** — the residual-amplification framework predicts the depth boundary across architectures, not just in hindsight.

**Third, and most important for the community: a negative result that saves other researchers from a plausible but ineffective design.** We showed — with matched-budget controls, at two scales — that gradient repair of ALS doesn't work, and we identified *why*: the closed-form solve is redundant with the gradient. That's not a null result in the pejorative sense; it's a *saved* research direction. And along the way we built a reproducibility discipline — including the evaluation-harness audit that corrected our own earlier numbers — that's worth more than the individual findings.

**[13:30-14:00] CLOSING**

Thank you. My poster walks through each of these results with the experiment charts. I'm happy to take questions — especially challenges; the negative results are the part I'd defend hardest. Thank you.

---

## TIMING REFERENCE
- 0:00 Opening
- 0:30 Background
- 2:30 Diagnosis / ablation
- 5:30 Causal theory
- 7:30 Repairs (A-SYNC / lr / probe / A-KD)
- 10:30 Verified solutions
- 12:30 Conclusion
- 13:30 Close → Q&A

## ANTICIPATED Q&A (with crisp answers)
1. "So you're saying your own method doesn't work?" — The ALS family doesn't work *on deep models*, and we established the specific mechanism by controlled evidence, plus why the repairs can't fix it. The project's contribution is the diagnosis and the saved design space, plus the verified alternatives.
2. "Is correlation 0.99981 meaningful?" — It's on the full 96-cycle trajectory, mean per-cycle difference 0.116 PPL. The two curves are visually and statistically indistinguishable.
3. "Why is AdamW 1.25 but plain SGD 6.83?" — Different budgets: AdamW at 800 steps is heavily trained on a small set (overfitting risk noted); SGD at 4800 steps is the fair matched control to A-SYNC. Both are valid operating points; LoRA splits the difference on efficiency.
4. "Boundary 25–28 layers from only a few model depths?" — Honest answer: it's a two-point boundary (≤24 / ≥28), consistent across 8 architectures; the *trend* is predicted, the exact constant is calibrated and 25–27 layers were not directly tested. The poster says "consistent with observation."
5. "Why WikiText-2 only?" — Acknowledged limitation; cross-domain (C4, HellaSwag) is listed as future work in the report.
6. "A-KD 52.6 vs 52.4 — isn't that just noise?" — Exactly the point: the closed-form solver's advantage is *within noise*, i.e., nonexistent at this scale. That's the negative result.

## CUT LIST (for a strict 10-minute version)
- The "timing no-op lesson" sentence in Repair 1.
- The "positive feedback loop" detail in the theory section (keep one sentence).
- One of the three repair details in the A-KD paragraph.
