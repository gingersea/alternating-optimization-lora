# Review Round 5 — Disentangling Optimizer and Parameter Form

**Date**: 2026-06-21
**Reviewers**: 5 parallel agents (Significance, Methodology, Completeness, Clarity, Reproducibility)
**Decision**: **Minor Revision** (Accept after targeted fixes)

---

## Overall Assessment

This paper makes a genuine contribution to the post-training methodology literature through rigorous empirical work across 8 architectures, honest limitation disclosure, and well-substantiated negative results. The depth boundary discovery (ALS-based optimization diverges at ≥28 layers) is the strongest contribution, validated through exhaustive GPU-scale testing (11 Protocol A failure attempts, 2 distributed backends). The 7B-scale full-rank result (PPL 1.25, 3 seeds, full test set validated) is cleanly executed, though its interpretation requires nuanced framing to distinguish in-distribution fitting from generalization.

The factorial design, while careful experimental practice, is better framed as "rigorous methodology" than "novel contribution." The Protocol C asymmetry (no ALS in LoRA space) makes the interaction term uninterpretable — a structural limitation acknowledged but understated. The single-dataset evaluation (WikiText-2 only) and absence of downstream tasks are the most significant empirical gaps, though the paper honestly discloses both.

**The paper should be accepted after targeted revisions.** The empirical execution is unusually thorough for the field (multi-seed, bootstrap stats, full test set validation), the negative results are documented with exemplary transparency, and the depth boundary is a genuine discovery. The textual issues (architecture count mismatch, appendix ordering, phantom Appendix D) are easily fixed.

## Scores

| Lens | Score (1-5) | Key concern |
|------|-------------|-------------|
| Significance & Novelty | 3 | Factorial design overclaimed; memorize vs generalize confound |
| Technical Correctness | 3 | Protocol C breaks factorial symmetry; ANOVA implemented as 1-way |
| Empirical Completeness | 2 | Single dataset; no downstream tasks; perturbation result is 1 datapoint |
| Clarity & Presentation | 3 | Table ordering; architecture count mismatch; phantom appendix |
| Reproducibility & Impact | 3 | Missing LR scheduler; config-paper inconsistencies |
| **Average** | **2.8/5** | |

## Strengths

1. **Exemplary negative-result documentation**: 11 failed Protocol A attempts across DeepSpeed and FSDP backends — saves the community months of unproductive work.
2. **Multi-seed rigor unusual for the field**: N=3-5 seeds, parametric bootstrap, explicit CV reporting, full test set validation (298,938 tokens).
3. **Honest limitations section** (Section 7.3, 9 items) that preempts reviewer concerns rather than burying them.
4. **Deep boundary discovery**: Empirically validated on 8 architectures, with a mechanistic model (residual amplification ρ̄^L) that goes beyond pure phenomenology.
5. **FLOPs-based resource normalization** across protocols — addresses a real confound most optimizer-comparison papers ignore.
6. **Practical Takeaway table** (Section 8) makes conclusions immediately actionable.

## Required Changes

### Critical (Must Fix for Acceptance)

| ID | Issue | Reviewer | Confidence |
|----|-------|----------|-----------|
| C1 | **Architecture count: "9 architectures" claimed in 11 locations; Table 5 lists only 8 rows** | R3, R4 | 100% |
| C2 | **Phantom "Appendix D" cited twice (lines 143, 374) but does not exist** | R4 | 100% |
| C3 | **Appendices appear in reverse order: B, C, A. Appendix A referenced throughout but arrives last** | R4 | 100% |
| C4 | **Tables 4 and 5 out of sequence (5 before 4)** | R4 | 100% |

### High Severity (Should Fix)

| ID | Issue | Reviewer | Confidence |
|----|-------|----------|-----------|
| H1 | **7B PPL=1.25 likely reflects memorization, not generalization** — 1600 training samples + full-rank 7B finetuning on WikiText-2. The paper frames as "parameter form dominates" but doesn't discuss the memorization interpretation. | R1 | 85% |
| H2 | **Factorial design overclaimed as novel** — 2×2 factorial is standard experimental design (Fisher, 1920s). Reframe as "rigorous experimental practice" applied to ML, not methodological innovation. | R1 | 90% |
| H3 | **Protocol C breaks factorial symmetry** — interaction term (A-B)-(C-D) captures (parameter form + ALS presence) jointly, not parameter form alone. The "disentanglement" claim is overstated. | R1, R2 | 95% |
| H4 | **"Two-way ANOVA" is actually one-way between-group comparison** — no multi-seed C/D data exist; code only computes A vs B. | R2 | 85% |
| H5 | **No downstream task evaluation at all** — all claims rest on perplexity only. Practice recommendations in Section 8 are purely perplexity-based. | R1, R3 | 95% |
| H6 | **Parameter count confound**: full-rank (7B params) vs LoRA r=8 (~3M params) — 2300× difference. Finding that more parameters work better is not a "parameter form" effect. | R1 | 85% |
| H7 | **Maximum step count (800) is below paper's own extrapolated crossover threshold** (800–5000). The crossover prediction is not verified. | R3 | 95% |
| H8 | **Perturbation finding rests on a single 12-step experiment** — promoted to Contribution #6 but labeled "preliminary" in text. | R3 | 95% |

### Medium Severity (Nice to Fix)

| ID | Issue | Reviewer | Confidence |
|----|-------|----------|-----------|
| M1 | Depth boundary derivation fits 3+ parameters from only 2 data points — L_max ≈ 26 is illustrative, not predictive. | R1, R2 | 80-85% |
| M2 | No multiple comparison correction across 5 time points — 800-step p=0.039 fails Bonferroni. | R2 | 85% |
| M3 | Cohen's d should be Hedges' g (unbiased) at N=5; no CI reported for effect size. | R2 | 85% |
| M4 | No LR scheduler/warmup specification — critical for 7B training reproducibility. | R5 | 85% |
| M5 | Table 1 mixes 100-step and 800-step results in same table — horizontal comparisons misleading. | R4 | 80% |
| M6 | Config-paper inconsistency: qwen25_7b.yaml says `offload_optimizer: false`, paper says CPU offload was used. | R5 | 80% |
| M7 | LoRA dropout differs between configs (0.0 vs 0.05) but not documented in paper. | R5 | 75% |
| M8 | ASP bundles 3 components into 1 factor — internal component confound prevents attribution of failure causes. | R1 | 90% |
| M9 | Table 1 vs Table 4 Protocol C 18× PPL discrepancy at 100 steps — implementation sensitivity concern. | R2 | 75% |

## Detailed Reviewer Reports

### R1: Significance & Novelty — Score: 3/5

The paper's strongest contribution is the depth boundary discovery, well-substantiated through exhaustive failure testing. The factorial design is careful experimental practice, not novel methodology — 2×2 designs date to Fisher. The 7B full-rank result (PPL=1.25) raises memorization concerns: 1600 training samples + 7B parameters → near-perfect perplexity on in-distribution test data = likely pattern memorization, not generalization. Without downstream evaluation, this finding cannot sustain the "parameter form dominates at scale" framing.

8 findings (2 HIGH, 2 MODERATE, 4 LOW severity).

### R2: Technical Correctness & Methodology — Score: 3/5

The factorial design is structurally compromised: Protocol C lacks ALS, making the optimizer factor asymmetric across rows. The "two-way ANOVA" is implemented as one-way between-group comparison (A vs B only). The depth boundary derivation fits multiple parameters from 2 data points, making the L_max ≈ 26 value illustrative rather than predictive. Statistical methods are directionally correct but multiple comparison correction is missing and effect sizes are biased.

9 findings (2 MAJOR, 3 MODERATE, 4 MINOR severity).

### R3: Empirical Completeness — Score: 2/5

Severe gaps: (1) single dataset (WikiText-2 only), (2) zero downstream task evaluation, (3) maximum step count below extrapolated crossover, (4) perturbation finding rests on a single 12-step experiment, (5) architecture count miscount (8 not 9). The honest limitations section partially mitigates these but does not fix them. The Protocol B full test set validation (298,938 tokens) is a strength.

8 findings (3 HIGH, 4 MEDIUM, 1 LOW severity).

### R4: Clarity & Presentation — Score: 3/5

Multiple structural errors: Table numbering out of sequence (5 before 4), 9-arch count systematically wrong (only 8 listed), Appendix D referenced but nonexistent, appendices in reverse order (B, C, A). Writing is otherwise clear and well-organized, with strong summary tables in Section 8.

8 findings (3 HIGH, 3 MEDIUM, 2 LOW severity).

### R5: Reproducibility & Impact — Score: 3/5

Core hyperparameters and GPU config well-documented. Key gaps: no LR scheduler/warmup specification, config-paper inconsistency on offload_optimizer, LoRA dropout not documented. Sufficient for assessing the paper's claims; insufficient for bitwise reproduction without code.

9 findings (3 MAJOR, 4 MEDIUM, 2 LOW severity).

## Meta-Review

**Highest agreement**: All 5 reviewers flagged the architecture count discrepancy (8 vs claimed 9). 4/5 flagged the single-dataset limitation. 3/5 flagged the interaction term interpretability issue. R1+R3 strongly flagged the memorization vs generalization concern at 7B.

**Divergence**: R3 scored completeness 2/5 (lowest), while R5 scored reproducibility 3/5. The gap reflects the genuine tension between what the paper achieves (honest, well-documented empirical work) and what would be ideal (multiple datasets, downstream tasks, verified crossover).

**Overall**: The paper is significantly stronger than most post-training comparison papers due to multi-seed rigor, open negative-result reporting, and GPU-scale validation. The fixes needed are predominantly textual/clarification (C1-C4, H2-H4) rather than experimental (exception: H7-H8 would benefit from additional experiments but are not blocking).

## Decision

**ACCEPT after Minor Revision**

**Rationale**: The core empirical contributions — depth boundary discovery, 7B 3/4-cell factorial validation, ASP implicit regularization characterization — are solid and well-substantiated. The 4 critical issues are textual/structural. The high-severity issues mostly require reframing and qualification, not new experiments. The paper's exemplary transparency about its limitations makes a revised version suitable for TMLR or a strong workshop.

---
*Review generated by paper-review 5-agent parallel workflow*
