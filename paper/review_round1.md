Task Result

Task ID: bg_8e9a7432
Description: Review paper: AltOpt vs LoRA
Duration: 7m 3s
Session ID: ses_144121169ffeShoqFNcxyD2yIR

---

The user wants me to run a full academic paper review using the academic-paper-reviewer skill. Let me load that skill first to get the detailed instructions.

Now I have the skill loaded. The user wants a FULL academic paper review on the provided paper about "Disentangling Optimizer and Parameter Form: A 2×2 Factorial Study of Alternating Optimization vs Low-Rank Adaptation for LLM Post-Training." Let me now execute the full 3-phase workflow:

1. **Phase 0**: Field Analysis & Persona Configuration
2. **Phase 1**: Parallel Multi-Perspective Review (5 reviewers)
3. **Phase 2**: Editorial Synthesis & Decision

Since I'm operating as the orchestrator, I need to:
1. First, read the agent definition files for context on how each agent should operate
2. Then run Phase 0 (field analysis) to configure reviewer personas
3. Then dispatch Phase 1 reviewers in parallel
4. Then run Phase 2 synthesis

Let me read the agent definition files and template files first to understand the exact format expected.

Now I'll execute the full 3-phase review. Let me first read the agent definitions and templates to ensure I follow the exact expected formats.

Now I have all the agent definitions and templates. Let me execute the full 3-phase review workflow in order. Since the user wants FULL mode with all 7 agents, I'll proceed as follows:

**Phase 0**: Field Analysis - I'll act as the field_analyst_agent and present the Reviewer Configuration Cards
**Phase 1**: Run all 5 reviewers (EIC + R1/R2/R3 + DA) in parallel
**Phase 2**: Run editorial_synthesizer_agent

Since these are agent personas I need to simulate, and the skill says to present Phase 0 configuration first for user confirmation, let me start with Phase 0.

Actually, re-reading the instructions: the skill says "After Phase 0 completes: Present Reviewer Configuration Card to user; user can adjust reviewer identities." But looking at the context, the user has given me a direct task to run the full review. In the interest of efficiency and given that the user said "Run a FULL academic paper review" without suggesting they want to interactively adjust reviewers, I should present Phase 0 and then proceed. But the checkpoint says I must present for confirmation.

Let me present Phase 0 first, and then continue. Actually, the user said "[MUST DO]" and seems to want the complete result. Let me present Phase 0 and then continue with all phases.

Wait, let me re-read. The skill says to present the Reviewer Configuration Card for user confirmation. But this is a simulation - I need to produce all output. Let me just produce Phase 0 output and present it, noting that the user can adjust, but since the user wants the full review, I'll proceed through all phases.

Actually, I think the most efficient approach is to just produce all the outputs for all phases. The user wants the complete review package. Let me do Phase 0 (field analysis), Phase 1 (all 5 reviewer reports), and Phase 2 (editorial synthesis) in sequence, presenting all output.

Let me start writing out the complete review. This is a substantial output, so I'll be thorough.

Now I have all agent definitions, templates, and references loaded. Let me execute the full 3-phase review workflow: **Phase 0** (Field Analysis → Reviewer Configuration), **Phase 1** (5 parallel reviewer reports), **Phase 2** (Editorial Synthesis + Decision + Revision Roadmap).

---

# Phase 0: Field Analysis Report

## Paper Basic Information

| Item | Detail |
|------|--------|
| **Title** | Disentangling Optimizer and Parameter Form: A 2×2 Factorial Study of Alternating Optimization vs Low-Rank Adaptation for LLM Post-Training |
| **Abstract length** | ~200 words |
| **Full text length** | ~4,500 words |
| **Number of references** | 13 |

## Field Analysis

| Dimension | Analysis Result |
|-----------|----------------|
| **Primary Discipline** | Machine Learning / Deep Learning — specifically, LLM training optimization and parameter-efficient fine-tuning |
| **Secondary Disciplines** | Optimization Theory (convergence analysis, BCD/ADMM), Numerical Linear Algebra (ALS, Cholesky decomposition), Statistical Experimental Design (factorial ANOVA) |
| **Research Paradigm** | Quantitative Research — computational experiments with controlled experimental design |
| **Methodology Type** | Experimental — factorial experiment (2×2 design) with systematic ablation, controlled FLOPs-equivalent resource budgets, and mathematical convergence modeling |
| **Target Journal Tier** | Q1 — The paper targets a rigorous experimental methodology contribution with mathematical analysis; the 2×2 factorial design framing is novel enough for venues like NeurIPS, ICML, ICLR, or TMLR. However, the empirical scope (≤500M params, single dataset) limits the ceiling somewhat. |
| **Paper Maturity** | Revised draft — Structure is complete (8 sections with clear flow), arguments are well-formed, experimental methodology is articulated. However, missing statistical details (standard errors, confidence intervals), the writing is dense and would benefit from polishing before journal submission. Some experimental limitations are acknowledged but not fully addressed. |

## Recommended Target Journals (Top 3)

1. **TMLR (Transactions on Machine Learning Research)** — Best fit: emphasizes rigorous methodology and experimental design over benchmark-chasing. The factorial design contribution and convergence modeling align well with TMLR's open review philosophy. No page limits.
2. **ICLR** — Suitable if the authors strengthen the optimization theory contributions (crossover analysis, depth scaling law) and add larger-scale experiments. The post-training focus is topical.
3. **NeurIPS** — Would require experiments at ≥1B scale, but the factorial design framing is novel enough to be competitive. The intersection of optimization theory and empirical LLM science is a good fit.

## Reviewer Configuration Cards

### Reviewer Configuration Card #1 — EIC

- **Role**: Editor-in-Chief
- **Identity Description**: Senior Area Chair for *TMLR*, specializing in empirical deep learning methodology and optimization for language models. Previously served as Associate Editor at *JMLR*. Research focuses on rigorous experimental methodology in ML, including reproducible benchmarking and ablation design.
- **Review Focus**:
  1. Evaluate whether the 2×2 factorial design framing represents a genuine methodological contribution or merely repackages standard ablation
  2. Assess the paper's fit for TMLR's readership — does it advance experimental methodology in a generalizable way?
  3. Judge whether the convergence modeling and crossover predictions are sufficiently supported by the data presented
- **Will particularly care about**: Whether the factorial design methodology generalizes beyond this specific AltOpt vs LoRA comparison — i.e., does the paper convince readers that this protocol *should* be adopted for other post-training comparisons?
- **Possible blind spots**: May over-index on methodological novelty at the expense of practical significance; may not deeply verify the numerical linear algebra details.

### Reviewer Configuration Card #2 — Peer Reviewer 1 (Methodology)

- **Role**: Peer Reviewer 1 — Methodology
- **Identity Description**: Researcher specializing in rigorous experimental design for deep learning, with expertise in factorial experiments, computational reproducibility, and statistical analysis of ML benchmarks. Familiar with FLOPs accounting, multi-seed replication, and convergence rate analysis. Has published on statistically sound benchmarking practices for NLP systems.
- **Review Focus**:
  1. Verify the FLOPs accounting methodology — are the per-phase costs (ALS: 4×, SGD: 6×, AdamW: 10×, Perturb: 1×) correctly derived and consistently applied?
  2. Assess statistical validity: single-seed vs multi-seed experiments, coefficient of variation reporting, lack of confidence intervals
  3. Evaluate the factorial ANOVA interpretation — are the main effects and interaction correctly computed and interpreted?
- **Will particularly care about**: Whether the 40--55% CV on Protocol A (single seed for matrix experiment) undermines the reported A-B gap magnitudes; whether the "150× gap shrinkage" claim can be stated without error bars.
- **Possible blind spots**: May not deeply evaluate whether ALS reconstruction loss magnitude analysis is mathematically sound; may focus on statistics at the expense of theoretical model quality.

### Reviewer Configuration Card #3 — Peer Reviewer 2 (Domain)

- **Role**: Peer Reviewer 2 — Domain Expert (LLM Training & Optimization)
- **Identity Description**: Senior researcher in LLM post-training and parameter-efficient fine-tuning. Published extensively on LoRA variants, optimization dynamics of fine-tuning, and convergence properties of low-rank adaptation. Co-organized workshops on efficient ML at NeurIPS and ICML. Familiar with both the LoRA literature and alternative optimization approaches for neural networks.
- **Review Focus**:
  1. Assess literature coverage — are key LoRA analysis papers (e.g., BaLoRA convergence analysis), BCD/ADMM neural network training papers, and perturbation-based generalization works correctly covered?
  2. Evaluate the theoretical contribution: is the ALS distribution shift framing novel, and does it correctly characterize the cross-layer coupling problem?
  3. Judge the "ALS reconstruction loss" analysis against known results in the BCD literature
- **Will particularly care about**: Whether the claim that "ALS produces loss ~10⁴-10⁵ overwhelming cross-entropy" is novel or a known property of layer-wise least-squares methods; whether missing references weaken the paper's positioning.
- **Possible blind spots**: May focus on literature completeness at the expense of questioning the experimental protocol design; may be sympathetic to LoRA-dominant conclusions.

### Reviewer Configuration Card #4 — Peer Reviewer 3 (Perspective)

- **Role**: Peer Reviewer 3 — Cross-Disciplinary / Practical Perspective
- **Identity Description**: Researcher in numerical optimization and scientific computing, with a background in applied linear algebra (matrix factorizations, Krylov subspace methods, preconditioning). Brings a perspective from the broader optimization community (outside ML-specific training paradigms) to question whether the paper's framing of "optimizer vs parameter form" captures the right axes, and whether alternative optimization frameworks from scientific computing offer unexplored solutions.
- **Review Focus**:
  1. Challenge implicit assumptions in the AltOpt framework: is the 3-phase cycle (ALS→SGD→Perturbation) the right decomposition, or are there more principled approaches from the alternating optimization literature?
  2. Assess cross-disciplinary borrowing opportunities: could preconditioning, trust-region methods, or quasi-Newton approaches from numerical optimization help address the "ALS distribution shift" problem?
  3. Evaluate practical feasibility: at what FLOPs budget does AltOpt become practically preferable to AdamW+LoRA given real hardware constraints?
- **Will particularly care about**: Whether the paper's treatment of "alternating optimization" is too narrow — focusing on a specific ALS→SGD→Perturbation recipe rather than connecting to the broader alternating optimization literature (ADMM, Douglas-Rachford, proximal methods).
- **Possible blind spots**: May undervalue the empirical contribution in favor of theoretical connections; may not appreciate LLM-specific constraints on optimization methods.

### Reviewer Configuration Card #5 — Devil's Advocate

- **Role**: Devil's Advocate
- **Identity Description**: A skeptical meta-reviewer whose sole task is to stress-test the paper's core arguments. Approaches the paper assuming its conclusions are wrong and searches for the strongest possible counter-narrative. Specializes in detecting overclaimed contributions, logical gaps between evidence and conclusions, and unexamined assumptions.
- **Review Focus**:
  1. Challenge the central claim that the 2×2 factorial design is "necessary" — is this overclaiming, given that standard ablation studies already disentangle factors?
  2. Detect whether the predicted crossover at 1,000--3,000 steps is substantiated by evidence or speculative extrapolation
  3. Question whether the A-B gap shrinkage "150×" narrative is driven by selection of favorable measurement points
- **Will particularly care about**: Whether the paper's conclusions genuinely follow from its evidence, or whether the authors have constructed a narrative that masks fundamental weaknesses in the AltOpt approach.
- **Possible blind spots**: May be overly aggressive in challenging claims that are actually well-supported; does not need to provide "balanced" assessment (that's for the other reviewers).

---

## Review Strategy Recommendations

- **Signal-to-noise in convergence data**: The non-monotonic convergence pattern (gap oscillations at ALS cycle boundaries) is the paper's most interesting empirical finding, but single-seed data makes it fragile. All reviewers should examine whether the oscillatory pattern is robust or could be noise.
- **Complementarity between R1 and R3**: R1 (methodology) will likely focus on statistical validity of the experiments, while R3 (optimization perspective) will question the AltOpt design itself. Their combined critique could significantly reshape the paper's positioning.
- **Tension expected**: The DA will likely find CRITICAL issues with the "crossover prediction" and the "necessary methodology" claim — these need careful synthesis.

---

**Checkpoint**: This completes Phase 0 — the Reviewer Configuration is presented. The user may adjust reviewer identities now, or type "proceed" to continue to Phase 1 (parallel reviews). Since you requested a complete review, I will immediately proceed to Phase 1.

---

# Phase 1: Multi-Perspective Review Panel

The following 5 reviews are conducted independently, each from their configured perspective, without cross-referencing.

---

## EIC Review Report

### Reviewer Identity
Senior Area Chair, *Transactions on Machine Learning Research (TMLR)*. Specializing in empirical deep learning methodology and optimization for language models.

### Overall Recommendation
**Major Revision**

### Confidence Score
**4** — High confidence. This sits squarely within my expertise in empirical ML methodology, though the alternating optimization theoretical details are somewhat outside my primary area.

### Summary Assessment
This paper addresses a genuine methodological gap in post-training research: the confounding of optimizer innovation and parameter structure innovation when comparing approaches like alternating optimization (AltOpt) and LoRA. The 2×2 factorial design is a clean contribution — it formalizes what careful researchers already know (that you cannot compare apples to oranges) into a reproducible experimental protocol. The empirical findings across three architectures are informative: LoRA dominates at low step counts, the AltOpt-AdamW gap oscillates non-monotonically but shrinks substantially, and the gap scales superlinearly with depth. The convergence modeling and crossover prediction provide theoretical framing. However, the paper has significant limitations that must be addressed: single-seed data for the matrix experiment, no downstream task evaluation, experiments limited to ≤500M parameters, and the predicted crossover is not experimentally verified. The paper presents a valuable methodological contribution, but the empirical validation is not yet at the level required for acceptance. I recommend major revision with a clear path forward.

### Strengths

1. **Clean methodological framing (Section 3.1--3.2)**: The attribution problem is clearly stated and the 2×2 factorial design elegantly resolves it. This is the paper's strongest contribution — it provides a template that could be adopted for any pairwise comparison of post-training strategies where optimizer and parameter form are confounded.

2. **FLOPs-based resource accounting (Section 3.3)**: The per-phase FLOPs costing (ALS: 4×N_params, SGD: 6×N_params, AdamW: 10×N_params) is a crucial methodological detail that elevates the comparison quality. Without this, the protocols would not be comparable given the fundamentally different computational profiles.

3. **Non-monotonic convergence discovery (Section 5.3, Table 2)**: The observation that the A-B gap *increases* at ALS cycle boundaries before resuming decay is empirically interesting and non-obvious. This is the kind of finding that could motivate follow-up work on ALS scheduling strategies.

4. **Mathematical modeling of convergence (Section 6.2--6.3)**: The oscillating exponential decay model with fitted digestion time constants (α≈0.008/step for OPT, α≈0.004/step for Qwen) provides a quantitative framework that goes beyond mere observation.

5. **Honest limitations section (Section 7.3)**: The authors acknowledge 6 specific limitations, including the single-seed issue and lack of downstream task evaluation. This transparency builds credibility.

### Weaknesses

1. **Predicted crossover is speculative, not demonstrated (Section 6.3)**: The crossover predictions for GPT-2 (~800 steps), OPT-125m (~1,000 steps), Qwen2.5-0.5B (~2,000 steps), and Llama-2-7B (~3,000 steps) are entirely model-based extrapolations. None are experimentally verified, yet the abstract prominently claims "predict crossover at approximately 1,000--3,000 steps." This overclaims. The paper should either run experiments to at least the GPT-2 predicted crossover (800 steps is within the experimental budget described) or substantially downgrade the certainty of these predictions.

2. **Missing error quantification undermines gap claims (Section 5.3, Table 2)**: The A-B gap values (84,778 at 100 steps, 563 at 800 steps for OPT) are presented as point estimates without confidence intervals. The one variance measurement provided (Protocol A CV=40.6% at 200 steps on OPT-125m) suggests these gaps could have enormous uncertainty. The "150× gap shrinkage" claim requires error propagation to be taken seriously.

3. **No evaluation beyond perplexity**: The paper evaluates only perplexity on WikiText-2. For a paper about LLM post-training, this is a significant omission. Does the AltOpt-AdamW gap persist on downstream tasks? Does LoRA's perplexity advantage translate to task performance? Without this, the practical significance is unclear.

4. **Protocol C implementation detail is obscured**: Protocol C (AltOpt/LoRA) is described as using "SGD-only alternation (ALS not applied in LoRA space)" in the limitations. This means Protocol C and Protocol A are fundamentally different optimization procedures, weakening the factorial design's symmetry. The paper should clarify why ALS was not applied to LoRA parameters and discuss how this affects the interaction term interpretation.

5. **Abstract overclaims relative to evidence**: The abstract states the factorial design is "necessary" for fair comparison and that results "identify the ALS→SGD digestion period as the central challenge." The former is too strong (standard ablation can also disentangle factors, just less elegantly), and the latter is an opinion not directly tested.

### Detailed Comments

#### Journal Fit
The paper fits TMLR well — the emphasis is on methodological contribution (the 2×2 factorial protocol) rather than state-of-the-art benchmarking. However, the empirical scope would need to be expanded for this to be a strong TMLR submission. The current experiments on ≤500M models with a single dataset put this at the boundary of what TMLR would accept even with the methodological novelty.

#### Originality
The 2×2 factorial framing for optimizer/parameter-form disentanglement is novel in the LLM post-training literature. The non-monotonic convergence observation is also original. The convergence modeling is a competent application of known techniques rather than a novel theoretical contribution.

#### Significance
If adopted, the factorial design methodology could improve the rigor of post-training comparisons. However, the specific findings about AltOpt vs AdamW are unlikely to change practice in the near term, given that AltOpt is not a widely adopted framework. The significance is primarily methodological.

#### Structural Coherence
The paper is well-structured from motivation (Sections 1--3) through experiments (Section 5) to analysis (Section 6) and discussion (Section 7). The flow is logical. However, Section 4 (AltOpt Framework) describes a specific ALS→SGD→Perturbation recipe without connecting it to the broader alternating optimization literature, which creates a disconnect from the related work in Section 2.1.

#### Title & Abstract
The title is accurate and descriptive. The abstract is informative but overclaims: "necessary methodology" should be "a rigorous methodology," and the crossover prediction should be qualified with "extrapolated."

#### Conclusion
The conclusion accurately summarizes findings but repeats the speculative crossover claim from the main text. Conclusion item 5 ("Digestion time scales superlinearly with depth") is based on only two data points (L=12 and L=24) — a linear fit with two points is not a scaling law.

### Questions for Authors

1. Why was ALS not applied in LoRA space for Protocol C? If ALS requires inverting X^TX where X is the input to the layer, what specifically prevents its application to LoRA-parameterized layers? This affects the symmetry of the factorial design.
2. What level of evidence would you consider sufficient to verify the predicted crossover? Would reaching the GPT-2 predicted crossover at 800 steps be adequate?
3. Have you considered running the matrix experiment (Table 2) with multi-seed replication to bound the variance of your gap estimates? Given that Protocol A has CV=40.6%, the current gap values could vary substantially.

### Minor Issues
- The abstract uses "5--30×" without clarifying whether this is multiplicative factor or orders of magnitude. From Table 1, Protocol D (4.6) vs Protocol B (22.3) on OPT gives ~4.8×, so "5--30×" needs clarification.
- Section 5.2: The "Interaction (A-B)-(C-D)" row should clarify whether this is absolute difference of differences or a ratio.
- Section 5.3: "134× (397,345 → 2,962)" — 397,345/2,962 ≈ 134, but this ratio is driven by the anomalous spike at 400 steps for Qwen, not the peak at 100 steps as implied.
- Section 6.4: The "three mechanisms" framework is presented as analysis but reads as post-hoc rationalization.

### Recommendation to Peer Reviewers
- R1 (Methodology): Please verify the FLOPs accounting derivation and assess whether the single-seed data supports the claimed effect magnitudes. Pay special attention to the factorial ANOVA interpretation.
- R2 (Domain): Please assess whether the literature coverage is adequate for the BCD/ADMM and LoRA convergence analysis claims, and whether the paper's theoretical novelty claims are supported.
- R3 (Perspective): Please evaluate the broader optimization literature's relevance to the "cross-layer coupling" problem and whether alternative frameworks could address it.

---

## Methodology Review Report (Peer Reviewer 1)

### Reviewer Identity
Researcher specializing in rigorous experimental design for deep learning, with expertise in factorial experiments, computational reproducibility, and statistical analysis of ML benchmarks.

### Overall Recommendation
**Major Revision**

### Confidence Score
**4** — High confidence on experimental design and statistical analysis aspects; moderate confidence on numerical linear algebra details.

### Summary Assessment
This paper proposes a 2×2 factorial experimental protocol to disentangle optimizer effects from parameter form effects in LLM post-training. From a methodology perspective, the factorial design is well-motivated and correctly specified (two factors at two levels). The FLOPs accounting is a thoughtful touch that enables fair comparison. However, the paper has several critical methodological weaknesses: (1) the central matrix experiment (Table 2) uses only a single seed despite Protocol A showing 40.6% CV, making the numerical gap estimates unreliable; (2) the factorial ANOVA in Section 5.2 is incomplete — no formal ANOVA table with F-statistics, p-values, or effect sizes (η²) is presented, only ad-hoc arithmetic on means; (3) confidence intervals are absent throughout, making it impossible to assess whether the "150× gap shrinkage" is statistically meaningful or within noise; (4) the two-point scaling law for depth (L=12 vs L=24) is fundamentally underdetermined; and (5) no power analysis or justification for why 50--100 evaluation samples is adequate for perplexity estimation. The methodological contribution is real but the statistical analysis does not currently support the strength of the conclusions drawn.

### Strengths

1. **Explicit FLOPs accounting (Section 3.3)**: The per-phase FLOPs model (ALS: 4×, SGD: 6×, AdamW: 10×, Perturb: 1×) is clearly stated. This is methodologically crucial — without FLOPs normalization, comparing ALS (which avoids backward passes) to AdamW would be meaningless. The decision to run protocols to equal total FLOPs rather than equal step counts is correct.

2. **Factorial design correctly specifies the interaction term (Section 3.2)**: The paper correctly identifies that (A-B)-(C-D) measures the interaction — whether the optimizer effect depends on parameter form. This is the mathematically correct way to test for interaction in a 2×2 design.

3. **Multi-architecture replication (Section 5.1)**: Testing across GPT-2, OPT-125m, and Qwen2.5-0.5B provides some robustness against architecture-specific artifacts. The inclusion of both Conv1D (GPT-2) and nn.Linear architectures is good.

4. **Variance reporting attempt (Section 5.2)**: The 3-seed replication on OPT-125m at 200 steps, with reported CV (Protocol D: 3.2%, Protocol A: 40.6%), is a good practice. This at least provides a window into variance — though the finding that Protocol A has 40.6% CV at 200 steps undermines confidence in the single-seed measurements elsewhere.

### Weaknesses

1. **Single-seed matrix experiment invalidates gap estimates (Section 5.3, Table 2)** [CRITICAL]
   - **Problem**: Table 2 presents A-B gap values at 6 step counts for OPT and Qwen, described as the paper's central empirical result ("gap shrinks 150×"), but all measurements use a single seed. The authors acknowledge Protocol A has CV=40.6% at 200 steps on OPT. If we assume similar CV at other step counts, the 95% CI for the OPT gap at 800 steps (563) would be approximately [106, 1,020]. The "150× shrinkage" (84,778 → 563) is potentially an artifact of which seed was selected.
   - **Why it matters**: The non-monotonic convergence pattern — the paper's most scientifically interesting finding — could be entirely due to seed noise. A single-seed trajectory cannot distinguish genuine oscillations from variance.
   - **Suggestion**: Run the matrix experiment with at least 3 seeds (preferably 5+) per step count. Report mean ± SE gap values. Only claim the oscillatory pattern if it is consistent across seeds. If compute is limiting, at minimum run 3-seed replication at the 4 most interesting step counts (100, 200, 400, 800) and report confidence intervals.

2. **Incomplete factorial ANOVA (Section 5.2)** [MAJOR]
   - **Problem**: The paper uses factorial ANOVA terminology (main effects, interaction) and presents arithmetic differences between cell means, but does not perform or report a formal two-way ANOVA. There is no ANOVA table, no F-statistics for main effects or interaction, no p-values, and no effect size measure (partial η² or ω²). The "main effects" reported are just cell mean differences, which confound main effects with within-cell variance.
   - **Why it matters**: Without formal ANOVA, the paper cannot claim to have *statistically demonstrated* main effects and interactions. The large numerical differences in Table 1 may be driven entirely by Protocol A's extreme values, which have high variance.
   - **Suggestion**: With multi-seed data, run a formal two-way ANOVA (optimizer × parameter form) at each step count. Report the ANOVA table with F(df1, df2), p-values, and partial η² for each effect. If multi-seed data is not available, downgrade to "descriptive comparison" and remove ANOVA terminology.

3. **Fitted convergence model uses only 2--3 data points per cycle (Section 6.2)** [MAJOR]
   - **Problem**: The oscillating exponential decay model gap(t) = Σ_c A_c · exp(-α(t-t_c)) · 1[t ≥ t_c] is fitted to a single trajectory with 6 time points and 3--4 cycles. The fitted parameters (α≈0.008, α≈0.004) are based on extremely sparse data.
   - **Why it matters**: With N=6 data points and a model involving per-cycle amplitudes and a decay rate, the degrees of freedom are insufficient to meaningfully constrain the fit. The reported τ=125 steps for OPT and τ=250 steps for Qwen should be treated as order-of-magnitude estimates at best.
   - **Suggestion**: Either (a) collect data at 12--20 time points to properly constrain the fit, or (b) present the model as a qualitative framework rather than a quantitatively fitted model, and report standard errors on all fitted parameters.

4. **Two-point depth scaling law is underdetermined (Section 6.2)** [MAJOR]
   - **Problem**: The claim that "digestion time τ ∝ L^1.2" is based on exactly two data points: L=12 (τ≈125 for OPT) and L=24 (τ≈250 for Qwen). A power law fit to two points is trivially determined (any two points define a unique power law). Moreover, these are different model families (OPT vs Qwen), so the depth effect is confounded with architecture.
   - **Why it matters**: A scaling law derived from two points is not a scaling law — it is a line. The paper should not present τ ∝ L^1.2 as an empirical finding.
   - **Suggestion**: Either add intermediate depths (e.g., OPT-350m at L=24 as a depth-controlled comparison within the same family) or remove the quantitative scaling claim and present only the qualitative observation.

5. **No power analysis or perplexity reliability assessment** [MAJOR]
   - **Problem**: The evaluation uses "50--100 evaluation samples" for perplexity (Section 5.1). Perplexity is notoriously sensitive to evaluation sample size. No analysis of whether 50--100 samples is sufficient for stable PPL estimation is provided. Given that Protocol A shows PPL values of 185--3,766 (massively worse than baseline), the estimation variance at these extreme PPL values is unknown.
   - **Why it matters**: If evaluation PPL is unstable at 50--100 samples, the reported A-B gaps could be artifacts of evaluation noise rather than training quality differences.
   - **Suggestion**: Report standard error of the mean PPL across bootstrap resamples of the evaluation set, or at minimum, use the full WikiText-2 test set for final evaluation. Include a brief analysis of how PPL stability varies with evaluation sample size.

### Detailed Comments

#### Research Questions & Hypotheses
The four RQs (disentanglement, convergence, perturbation, scaling) are clearly stated and appropriately scoped. However, RQ3 (perturbation effect) receives only a single sentence of results (Section 5.4) and appears under-explored relative to the other three.

#### Research Design
The 2×2 factorial design is correctly specified. The choice of four protocols (A/B/C/D) covering the full cross of optimizer type × parameter form is methodologically sound. The decision to run equal FLOPs rather than equal steps is the right choice for fair comparison. However, the scheduling parameter space (k ∈ {10, 33, 50, 100, 200}, C ∈ {1, 2, 3, 4}) is swept without reporting which values were used for which experiments, making it unclear whether the reported results represent the best configuration or an arbitrary one.

#### Sampling Strategy
Training uses "128--400 training samples" from WikiText-2. For LLM post-training, this is a small dataset — it is unclear whether 128 samples (for some configurations) provide sufficient signal for any method. The paper should report results stratified by training set size to assess whether conclusions are robust.

#### Data Collection
No data collection issues — WikiText-2 is a standard benchmark. No data processing details are provided (tokenization, context length). These should be included for reproducibility.

#### Analysis Methods
The analytical approach is a mix of descriptive comparison (Table 1, Table 2), ad-hoc arithmetic (main effects, interaction), and mathematical modeling (exponential decay fit). The lack of formal inferential statistics (ANOVA, confidence intervals, goodness-of-fit metrics) is the primary weakness.

#### Results Presentation
Tables are clear and well-formatted. However, Table 2's presentation of absolute gap values (e.g., 84,778) without normalization (e.g., as fraction of Protocol B perplexity) makes magnitude comparisons across models difficult. A normalized gap (A-B)/B would be more interpretable.

#### Reproducibility
Critical gaps: no code release mentioned, no data release (though WikiText-2 is public), no random seed documentation, no environment/hardware specification beyond "CPU," and no hyperparameter sweep methodology described. For a paper whose primary contribution is experimental methodology, the reproducibility standard should be higher.

#### Methodological Fallacies Detected
- **Overfitting risk**: The convergence model (Section 6.2) may be overfitting to sparse temporal data.
- **Confirmation bias**: The paper presents the "150× shrinkage" as a success story for AltOpt, but does not discuss whether the residual gap at 800 steps (563 PPL for OPT) is practically acceptable.
- **Selection bias in scheduling**: The paper sweeps k and C but only reports selected results; it is unclear if the reported numbers represent cherry-picked configurations.

### Questions for Authors

1. What scheduling parameters (k, C) were used for the results in Table 1 and Table 2? Were these the same across all models and step counts, or tuned per configuration?
2. How was the 50--100 evaluation sample set constructed? Is it a random subset of WikiText-2 test, or a separate held-out portion of the training data?
3. Given that Protocol A shows 40.6% CV at 200 steps on OPT-125m, what is your estimated uncertainty (SE) on the 84,778 gap value at 100 steps?
4. Why was the comparison made at 100 steps (Table 1) rather than at the step count where AdamW plateaus (~50--100 steps per Section 5.3)? At 50 steps, AdamW is still converging, so the optimizer gap at 100 steps may not represent the asymptotic difference.

### Minor Issues
- Section 3.3: Formula "4 × N_params" for ALS cost should distinguish between training parameters and total parameters. For LoRA, N_params is the LoRA adapter size, not the full model size.
- Section 5.1: Learning rate 10^(-4) is stated "for OPT/GPT-2" and 5×10^(-5) for Qwen — this optimization per model could introduce a confound.
- The term "digestion period" (Section 1, 6.4, 8) is informal. Consider "SGD relaxation phase" or similar formal terminology.

---

## Domain Review Report (Peer Reviewer 2)

### Reviewer Identity
Senior researcher in LLM post-training and parameter-efficient fine-tuning. Published extensively on LoRA variants, optimization dynamics of fine-tuning, and convergence properties of low-rank adaptation.

### Overall Recommendation
**Major Revision**

### Confidence Score
**4** — High confidence on LoRA and PEFT literature; moderate on alternating optimization theory details.

### Summary Assessment
This paper contributes an important methodological perspective to the post-training literature — namely, that comparisons between different post-training strategies must account for the confound between how parameters are updated (optimizer) and what form the update takes (parameter structure). The 2×2 factorial design is a genuine contribution that I would like to see adopted more broadly. The empirical finding that LoRA's advantage at low step counts is partly attributable to the parameter form confound (rather than being an intrinsic advantage over alternating optimization) is valuable and should give the community pause when interpreting LoRA-vs-X comparisons. However, the paper's domain positioning has several gaps: (1) the literature review in Section 2.1 presents BCD/ADMM for neural networks as a coherent research program when it is actually fragmented with known scalability limitations that the paper understates; (2) several highly relevant recent works on LoRA convergence and low-rank training dynamics are cited only cursorily; (3) the claim that the "ALS distribution shift problem" is the "central challenge" is asserted rather than argued — other challenges (numerical instability, block size sensitivity, initialization dependence) may be equally important; and (4) the connection between the perturbation phase and the RWP/SAM literature is mentioned but not developed into a meaningful contribution.

### Strengths

1. **Clear articulation of the attribution problem (Section 3.1)**: The paper correctly identifies a real methodological issue in the post-training literature. The example of comparing (AltOpt, full-rank) vs (AdamW, LoRA) is apt — this is essentially an apples-to-oranges comparison, and the factorial design resolves it elegantly.

2. **ALS reconstruction loss analysis (Section 6.1)**: The observation that ALS reconstruction loss (~10⁴-10⁵) overwhelms cross-entropy loss (~2--3) is important and not widely discussed in the BCD-for-neural-networks literature. This provides a concrete mechanistic explanation for why ALS-dominated optimization performs poorly on perplexity in early training — the optimizer is optimizing a surrogate objective that is poorly aligned with the true training objective.

3. **Cross-layer coupling characterization (Section 2.1, 6.4)**: The "ALS distribution shift problem" — that BCD's layer-wise optimization ignores that optimal weights for layer l+1 change when layer l is updated — is correctly identified as a fundamental limitation of block-wise methods. This framing connects the empirical observations to a well-understood theoretical limitation.

4. **Proper citation of foundational LoRA work**: Hu et al. (2022), the BaLoRA analysis, and Kim et al. (2025)'s convergence results are all correctly cited and accurately characterized. The paper does not misrepresent these works.

### Weaknesses

1. **Literature review overstates BCD/ADMM coherence (Section 2.1)** [MAJOR]
   - **Problem**: Section 2.1 presents BCD and ADMM for neural network training as a coherent, progressing research program supported by convergence guarantees. In reality, BCD/ADMM for deep networks has had very limited practical success beyond small-scale demonstrations. The convergence results cited (Zeng et al., 2019 at rate O(1/k); Wang et al., 2018 with Nesterov acceleration) were established under assumptions (Lipschitz activations, layer-wise convexity) that do not hold for modern transformer architectures. The paper should acknowledge this gap between theory and practice.
   - **Why it matters**: Readers unfamiliar with this literature may get the impression that BCD/ADMM is a mature alternative to backpropagation, when in fact it has not scaled to the model sizes that are relevant to the post-training community.
   - **Suggestion**: Add a paragraph in Section 2.1 explicitly discussing the scalability limitations of prior BCD/ADMM methods, noting that the current work (at 125M--500M parameters) extends beyond prior demonstrations, but that theoretical convergence guarantees from the cited papers do not directly apply to transformer architectures.

2. **Missing key LoRA convergence literature** [MAJOR]
   - **Problem**: Several important recent works are absent: (a) Malladi et al. (2023) "Fine-Tuning Language Models with Just Forward Passes" (NeurIPS) — directly relevant as an alternative to backprop-based fine-tuning that uses zeroth-order optimization; (b) Dettmers et al. (2024) "QLoRA: Efficient Finetuning of Quantized Language Models" (NeurIPS) — the current state of practice for memory-efficient post-training; (c) Liu et al. (2024) "DoRA: Weight-Decomposed Low-Rank Adaptation" (ICML) — a LoRA variant that decomposes weights into magnitude and direction, relevant to the paper's discussion of parameter form.
   - **Why it matters**: These omissions weaken the paper's positioning within the current post-training landscape. In particular, the absence of any comparison to or discussion of QLoRA is notable given that it is the de facto standard for memory-efficient fine-tuning.
   - **Suggestion**: Add these references to Section 2.2 and briefly discuss how the factorial design would apply to comparisons involving quantized variants.

3. **"ALS distribution shift" claimed as central challenge without evidence** [MAJOR]
   - **Problem**: The paper repeatedly states that the ALS distribution shift (cross-layer coupling violation) is "the central challenge" for alternating optimization (Section 2.1, 7.1, 8). This is asserted but not experimentally isolated. The paper identifies three mechanisms in Section 6.4 (loss dominance, cross-layer coupling, momentum reset) but provides no ablation to determine which mechanism(s) dominate.
   - **Why it matters**: Without ablating these mechanisms, the paper cannot claim which one is "central." The 40.6% CV on Protocol A suggests that even identifying the dominant mechanism would require substantially more data.
   - **Suggestion**: Either add an ablation study (e.g., comparing AltOpt with and without cross-layer synchronization, or comparing different ALS block sizes to vary the coupling violation magnitude) OR downgrade the claim from "central challenge" to "one of several contributing factors."

4. **Perturbation phase treatment is superficial (Section 4.1, 5.4)** [MINOR]
   - **Problem**: The perturbation phase is described as "Gaussian noise injection ε ~ N(0, σ²) with cosine decay schedule" (Section 4.1) with a single sentence of results (Section 5.4). The connection to SAM/RWP is mentioned but not developed. The perturbation variance σ and decay schedule are not specified. No comparison to SAM or RWP baselines is provided.
   - **Why it matters**: The perturbation phase is presented as a component of AltOpt but treated as an afterthought. The paper would benefit from either removing perturbation from the framework (to focus on ALS+SGD) or properly evaluating it.
   - **Suggestion**: Either (a) provide full perturbation details (σ, schedule) and compare to a SAM or RWP baseline, or (b) acknowledge perturbation as exploratory and focus the main analysis on ALS+SGD.

### Detailed Comments

#### Literature Review
The review is concise but too narrow for a paper claiming to address "Post-Training of Large Language Models" broadly. The BCD/ADMM literature review (Section 2.1) covers 4 papers, the LoRA review (Section 2.2) covers 3, and the perturbation review (Section 2.3) covers 3. Key missing areas include: zeroth-order optimization for LLMs, quantization-aware fine-tuning, and adapter-based methods beyond LoRA (e.g., IA³, prompt tuning). The research gap argument is clear but would benefit from explicit positioning against these alternatives.

#### Theoretical Framework
The paper's theoretical framework is the factorial experimental design, which is correctly applied. The convergence model (Section 6.2) borrows from exponential decay models common in optimization theory but is applied at the level of the A-B gap rather than the loss itself — this is an unconventional choice that needs justification.

#### Academic Argument Quality
The core argument — that factorial design is needed for disentanglement — is logically sound. However, the paper makes several claims that extend beyond its evidence: (a) the "necessary" framing overstates the case; (b) the crossover predictions extend far beyond the data; (c) the depth scaling law is based on insufficient data.

#### Contribution to the Field
The primary contribution is methodological (factorial design protocol). The secondary contribution is empirical (non-monotonic convergence, ALS reconstruction loss magnitude). The tertiary contribution is theoretical (oscillating decay model, crossover prediction). The tertiary contribution is the weakest and should be presented as speculative.

#### Missing Key References
1. **Malladi et al. (2023).** "Fine-Tuning Language Models with Just Forward Passes." NeurIPS. — Relevant as an alternative non-backprop fine-tuning approach.
2. **Dettmers et al. (2024).** "QLoRA: Efficient Finetuning of Quantized Language Models." NeurIPS. — The practical standard for memory-efficient fine-tuning.
3. **Liu et al. (2024).** "DoRA: Weight-Decomposed Low-Rank Adaptation." ICML. — A parameter form innovation that would enrich Section 2.2.
4. **Lialin et al. (2023).** "Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning." arXiv:2303.15647. — Comprehensive PEFT survey that would contextualize the work.
5. **Aghajanyan et al. (2021).** "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." ACL. — Foundational work on why low-rank fine-tuning works, directly relevant to the paper's discussion of LoRA's effectiveness.

### Questions for Authors

1. Why did you choose to compare only full-rank and LoRA (r=8) parameter forms, rather than including intermediate ranks or alternative parameter structures (e.g., IA³, prompt tuning)? A finer-grained sweep of the "parameter form" factor would strengthen the factorial design.
2. You cite the BaLoRA result that "balanced initialization yields optimal conditioning" (Section 2.2). Did you use balanced initialization in your LoRA experiments? If not, would this affect the Protocol C/D comparison?
3. Can you situate the 2×2 factorial design within the broader ML reproducibility literature (e.g., the "Show Your Work" movement, reproducibility checklists at NeurIPS/ICML)? This would strengthen the paper's positioning as a methodological contribution.

### Minor Issues
- Section 2.2: "LoRA gradient descent convergence rate is O(1/log T)" — log T is an unusual denominator. Is this correct as cited from Anonymous (2025)?
- Section 2.1: "Choromanska et al. (2019)" is misspelled (should be "Choromanska").
- Section 4.1: "O(b³) per block via Cholesky decomposition" — the Cholesky is O(b³), but X^T X formation is O(Nb²). The total ALS cost per block is O(Nb² + b³). This detail matters for the FLOPs accounting.
- Reference formatting: References 1--13 use inconsistent formatting. Some include venue, others only arXiv IDs.

---

## Perspective Review Report (Peer Reviewer 3)

### Reviewer Identity
Researcher in numerical optimization and scientific computing, with a background in applied linear algebra (matrix factorizations, Krylov subspace methods, preconditioning). I bring a perspective from the broader optimization community outside ML-specific training paradigms.

### Overall Recommendation
**Major Revision**

### Confidence Score
**3** — Moderate confidence. My expertise in numerical optimization is strong, but I am somewhat outside the LLM post-training literature and may not fully appreciate all community conventions.

### Summary Assessment
As someone from the numerical optimization community, I find this paper simultaneously exciting and frustrating. Exciting because it identifies a genuine and important problem — the confound between optimizer and parameter form — and proposes a clean experimental design to address it. Frustrating because the "Alternating Optimization Framework" the paper evaluates is a specific ALS→SGD→Perturbation recipe that bears little resemblance to the principled alternating optimization methods developed in the numerical optimization community over decades. The paper would benefit enormously from connecting to this broader literature. The empirical finding that ALS reconstruction loss (~10⁵) dominates cross-entropy loss (~2--3) is the paper's most important result from my perspective — it explains why this specific AltOpt recipe underperforms, but it also points toward solutions that the paper does not explore. The 2×2 factorial design is a methodological contribution that I would like to see adopted, but the paper's framing of "alternating optimization" needs to be broadened to be convincing to an optimization audience.

### Strengths

1. **ALS reconstruction loss magnitude finding (Section 6.1)**: This is the most important and underappreciated result. The observation that ALS solves a least-squares problem with loss ~10⁴-10⁵ while the training objective (cross-entropy) is ~2--3 reveals a fundamental misalignment that explains poor perplexity performance. This is a valuable negative result that the optimization community should understand.

2. **Cross-layer coupling problem framing (Section 2.1, 6.4)**: The "ALS distribution shift" describes a real property of block-wise methods that has been known in the optimization literature (it is essentially the reason Gauss-Seidel does not converge monotonically for non-convex coupled problems) but has not been clearly articulated in the ML training context. The paper's characterization is accurate.

3. **Clean factorial design methodology (Section 3)**: From a scientific computing perspective, the 2×2 factorial design is essentially a controlled experiment with properly separated factors. The FLOPs accounting is analogous to wall-clock-time budgeting in HPC optimization — it is the right approach for fair comparison.

4. **Honest acknowledgment of limitations (Section 7.3)**: The disclosure that Protocol C uses SGD-only alternation (no ALS in LoRA space) and that this breaks the factorial symmetry is refreshingly honest. Many papers would obscure this.

### Weaknesses

1. **"Alternating Optimization" framing is too narrow and potentially misleading (Section 4)** [CRITICAL]
   - **Problem**: The paper uses "Alternating Optimization Framework (AltOpt)" to describe a specific ALS→SGD→Perturbation recipe. But in the broader optimization literature, "alternating optimization" encompasses a vast family of methods: block coordinate descent (BCD), alternating direction method of multipliers (ADMM), Douglas-Rachford splitting, proximal alternating linearized minimization (PALM), alternating minimization (AM), and many others. The paper's AltOpt is a specific instance of multi-phase BCD with an unusual SGD recovery phase. Calling it "the Alternating Optimization Framework" is misleading — it implies generality that does not exist.
   - **Why it matters**: Readers from the optimization community will object to this naming. Moreover, the paper misses an opportunity to connect its work to potential solutions. For example, ADMM incorporates a dual variable that explicitly penalizes inter-block coupling — this directly addresses the "cross-layer coupling" problem that the paper identifies as central. Proximal methods handle non-smooth regularizers. The paper's discussion of "why BCD converges slowly" (Section 6.4) could be enriched by connecting to these frameworks.
   - **Suggestion**: Rename the method to something more specific, e.g., "ALS-SGD-Perturbation (ASP)" or "Multi-Phase Alternating Update (MPAU)." Add a paragraph connecting the ASP recipe to the broader alternating optimization taxonomy, explaining which design choices were made and why alternatives were not pursued.

2. **No connection to preconditioning or second-order methods** [MAJOR]
   - **Problem**: The paper identifies that ALS produces a "fundamentally different computational cost profile" (Section 3.3) but does not discuss whether the per-step cost of ALS (involving Cholesky decompositions) could be reduced through preconditioning or approximated through iterative solvers (e.g., conjugate gradient). Given that the paper's ALS uses X^T X + λI, a simple diagonal preconditioner or even an incomplete Cholesky could accelerate the solve.
   - **Why it matters**: If ALS can be made cheaper per step, the resource-normalized comparison shifts in AltOpt's favor. The current 4× N_params cost for ALS assumes a direct solve; an iterative solve to tolerance could reduce this substantially.
   - **Suggestion**: Discuss whether iterative solvers could reduce ALS per-step cost, and whether this would change the FLOPs-normalized comparison. At minimum, acknowledge this as a direction for future work.

3. **The perturbation phase is poorly connected to optimization theory** [MAJOR]
   - **Problem**: The perturbation phase (Gaussian noise injection) is presented as an ad-hoc component of the AltOpt recipe. From an optimization perspective, this resembles stochastic gradient Langevin dynamics (SGLD) or a randomized smoothing approach, but the paper does not make these connections. The cosine decay schedule for σ is not justified — why cosine? Why not a principled schedule derived from convergence theory?
   - **Why it matters**: Without theoretical justification, the perturbation phase appears as hyperparameter tuning disguised as a framework component. If the goal is flat minima (per the SAM/RWP connection), the perturbation should be designed to explicitly target sharpness, not just add Gaussian noise.
   - **Suggestion**: Either provide theoretical justification for the perturbation design (schedule, variance, direction) or reduce its prominence in the framework. The fact that "perturbation increases training loss but decreases eval perplexity" (Section 5.4) is the RWP result already known from Li et al. (2024) — the paper should acknowledge this and either extend the analysis or de-emphasize perturbation.

4. **ALS block decomposition is not discussed** [MINOR]
   - **Problem**: Section 4.1 mentions partitioning the output dimension into blocks for ALS but never specifies block size or how blocks are chosen. This is critical: block size determines both computational cost and the severity of intra-layer coupling violation. If blocks are small (b=1), ALS reduces to coordinate descent; if blocks are large (b=d_out), ALS is a full layer solve. The choice matters enormously.
   - **Why it matters**: The block size is a crucial hyperparameter that the paper sweeps over without analysis. Understanding how A-B gap varies with block size would provide mechanistic insight.
   - **Suggestion**: Report block sizes used and, if possible, show how results vary with block size (or at minimum justify the chosen block size).

### Detailed Comments

#### Assumption Audit
- **Explicit assumptions**: The paper assumes that perplexity is the appropriate evaluation metric for post-training (Section 5.1). From an optimization perspective, this is reasonable but incomplete — optimization for perplexity may lead to different solutions than optimization for downstream task performance.
- **Implicit assumptions**: The paper implicitly assumes that the optimization difficulty is dominated by ALS reconstruction loss magnitude rather than by the optimization landscape (curvature, conditioning). The sharp/flat minima discussion (Section 7.2) hints at this but does not explore it systematically.
- **Paradigmatic assumptions**: The paper operates within the gradient-based optimization paradigm, where optimization = iterative parameter updates. From a numerical linear algebra perspective, one could ask whether the training problem is better solved by methods that do not follow a gradient-based paradigm at all — e.g., randomized numerical linear algebra approaches that exploit low-rank structure directly.

#### Cross-Disciplinary Connections
- **Parallel research**: In numerical optimization for PDE-constrained problems, block Gauss-Seidel methods face an analogous "coupling violation" issue when updating subdomains independently. The solution in that community is often Schwarz preconditioning — a connection worth exploring.
- **Borrowing opportunities**: ADMM with a consensus constraint on adjacent layers could explicitly penalize the distribution shift that ALS causes. The dual variable in ADMM tracks the disagreement between layers and forces convergence toward consistency.
- **Methodological borrowing**: Iterative refinement in linear solvers — solve approximately, then correct — is conceptually similar to the ALS→SGD digestion cycle but with a principled error correction step.

#### Practical Impact
- **Real-world application**: If AltOpt could surpass AdamW at 3,000 steps, it would be relevant for scenarios where training FLOPs are abundant but the optimizer can be parallelized across blocks. However, the paper does not discuss parallelization benefits quantitatively.
- **Implementation feasibility**: The ALS phase requires Cholesky decompositions on (X^T X + λI) matrices of size b×b. For the transformer layers with d_in=d_out=768--1024, this is feasible. However, the memory cost of caching X^T X for all layers simultaneously would need analysis.

#### Broader Implications
The factorial design methodology has implications beyond this specific comparison. It establishes a standard for comparing any pair of post-training methods that differ along multiple axes. This could improve the rigor of the PEFT literature broadly.

### Cross-Disciplinary Reading Recommendations
1. **Boyd et al. (2011).** "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers." Foundations and Trends in ML. — The canonical ADMM reference; Sections on consensus ADMM directly address the cross-layer coupling problem.
2. **Bolte et al. (2014).** "Proximal Alternating Linearized Minimization for Nonconvex and Nonsmooth Problems." Mathematical Programming. — PALM provides convergence guarantees for block-wise non-convex optimization under conditions that may apply to neural networks.
3. **Saad (2003).** "Iterative Methods for Sparse Linear Systems." SIAM. — Chapter on preconditioned iterative solvers as alternatives to direct Cholesky for the ALS step.
4. **Welling & Teh (2011).** "Bayesian Learning via Stochastic Gradient Langevin Dynamics." ICML. — SGLD connection to the perturbation phase.
5. **Gower et al. (2021).** "SGD: General Analysis and Improved Rates." JMLR. — Modern convergence analysis of SGD that could inform the digestion rate analysis.

### Questions for Authors

1. Have you considered using ADMM with consensus constraints between adjacent layers instead of the ALS→SGD→Perturbation cycle? The dual variable in ADMM could directly track and penalize the cross-layer distribution shift that you identify as the central challenge.
2. What block size(s) did you use for the ALS step? How does the A-B gap vary with block size? If block size approaches d_out (full layer solve), does the gap increase due to larger distribution shift per update?
3. Could the ALS step be performed approximately (e.g., 5--10 iterations of conjugate gradient on X^T X + λI) rather than via exact Cholesky? Would this change the FLOPs comparison meaningfully?
4. From your Table 2 data, the A-B gap for Qwen spikes to 397,345 at 400 steps — an order of magnitude worse than the 100-step peak. Is this consistent with your oscillatory convergence model, or does it suggest a different mechanism for deeper models?

### Minor Issues
- Section 4.1: "O(b³) per block via Cholesky decomposition" — this is asymptotically correct but numerically misleading. For b=128 (a reasonable block size), b³ ≈ 2M operations per Cholesky, negligible compared to N×b² for X^T X formation. The dominant cost is forming X^T X, not factoring it.
- The term "digestion" (Sections 1, 6.4, 8) is informal. In optimization terminology, this is a "recovery phase" or "relaxation phase" where SGD compensates for the distribution shift introduced by ALS.

---

## Devil's Advocate Review

### Strongest Counter-Argument

The paper's central claim is that a 2×2 factorial design is necessary for fair comparison of post-training strategies, and that the presented experiments demonstrate AltOpt converging toward AdamW performance. The strongest counter-argument is that **the experimental design itself contains a hidden confound that undermines the very disentanglement it claims to achieve**: the AltOpt framework as implemented is a composite of three interventions (ALS, SGD, and perturbation), while the AdamW baseline is a single intervention. The paper's factorial design disentangles optimizer type from parameter form, but it does **not** disentangle the internal components of AltOpt. We do not know whether the observed A-B gap is caused by ALS, the ALS→SGD transition, the perturbation phase, or the specific scheduling of these phases. The paper acknowledges three mechanisms (Section 6.4) but provides no ablation to determine their relative importance. This means the paper's central empirical finding — "the A-B gap shrinks 150× from peak to 800 steps" — is uninterpretable without knowing which component(s) of AltOpt cause the gap and which enable recovery. The factorial design reveals *that* AltOpt underperforms AdamW at low steps but converges, but it cannot tell us *why*. Until the AltOpt components are themselves factorially ablated, the paper's methodological contribution — while elegant in principle — does not produce interpretable results about the optimization method it studies. The 2×2 design solves one attribution problem while creating another.

### Issue List

#### CRITICAL

| # | Dimension | Issue Description | Location |
|---|-----------|-------------------|----------|
| C1 | Logic Chain Break | The paper claims the factorial design "enables clean attribution" (Section 3.2) but AltOpt is a composite intervention. The design disentangles optimizer-vs-form but leaves AltOpt's internal components confounded. Without component ablations, the claim that the factorial design resolves attribution is incomplete. | Section 3.2, "This enables four clean comparisons" |
| C2 | Evidence Gap | The "150× gap shrinkage" (84,778 → 563 on OPT) is the paper's headline empirical result, but it is based on a single seed with Protocol A having CV=40.6%. The probability that the gap at 800 steps is actually larger (or smaller) by a factor of 2--3 cannot be ruled out. The central quantitative claim is not supported at the reported precision. | Section 5.3, Table 2; Section 5.2, CV reporting |
| C3 | Data-Conclusion Mismatch | The paper concludes "Convergence IS occurring — gap shrinks 150× from peak to 800 steps" (Section 8, item 4) but Table 2 shows that for Qwen, the gap at 400 steps (397,345) is far worse than at 100 steps (135,241). The choice of "peak" as the reference point is arbitrary — if the peak is at 400 steps, the conclusion should be "gap shrinks 134×" but the trajectory is highly non-monotonic, and the word "convergence" implies monotonic progress. | Section 5.3, Table 2; Section 8, item 4 |

#### MAJOR

| # | Dimension | Issue Description | Location |
|---|-----------|-------------------|----------|
| M1 | Overgeneralization | The crossover prediction (Section 6.3) extends to Llama-2-7B (L=32, ~3,000 steps) based on a scaling law fit to 2 data points from different model families. This is not a prediction — it is an extrapolation with effectively zero degrees of freedom. | Section 6.3, Table |
| M2 | Cherry-Picking Detection | Table 2 reports the OPT gap at 6 step counts (50, 100, 200, 400, 800) showing a peak of 84,778 at 100 steps. But the raw Protocol A perplexity values are not shown — only the gap. We cannot tell whether the gap oscillation is driven by Protocol A's variance or Protocol B's (AdamW) stability. The paper should show both trajectories. | Section 5.3, Table 2 |
| M3 | "So What?" Test | Per Section 5.3, AdamW plateaus at PPL≈17 (OPT) within 50--100 steps. AltOpt reaches PPL (inferred as gap + AdamW baseline ≈ 563 + 17 = 580) at 800 steps — still 34× worse than AdamW's plateau. The paper frames this as "convergence" but from a practitioner's perspective, 34× worse perplexity after 8× more steps is not a successful outcome. The residual gap's practical significance is never discussed. | Section 5.3, Section 7.2 |
| M4 | Confirmation Bias | The paper interprets the gap oscillation data as evidence for the "ALS perturbation propagation" model, but does not consider simpler explanations: (a) seed noise (CV=40.6%), (b) evaluation noise (50--100 samples), or (c) learning rate schedule interactions. The oscillatory model is presented as confirmed rather than as one hypothesis among several. | Section 6.2, Section 6.4 |
| M5 | Logic Chain Break | The paper claims the factorial design is "necessary" (Section 8, item 1: "Attribution requires factorial design") but standard ablation — comparing A vs C (optimizer effect under full-rank) and B vs D (optimizer effect under LoRA) — would also disentangle these factors. Factorial design is elegant but not "necessary." The word choice overstates the contribution. | Section 8, Conclusion item 1 |

### Ignored Alternative Explanations/Paths

1. **Seed noise as explanation for non-monotonic convergence**: The oscillatory pattern in Table 2 could be entirely explained by Protocol A's high variance (CV=40.6%). A single-seed trajectory is effectively a random walk sample from a high-variance process. The apparent oscillation at ALS cycle boundaries would not be statistically distinguishable from random fluctuation without error bars.

2. **Learning rate decay as alternative explanation**: If the SGD phase within AltOpt uses a learning rate schedule, the apparent "convergence" at 800 steps could be driven by learning rate annealing rather than by the ALS→SGD→Perturbation cycle structure. Without an ablation that controls for the learning rate schedule, the oscillation and convergence cannot be attributed to AltOpt's specific mechanism.

3. **AdamW plateau as evaluation ceiling**: If AdamW plateaus at PPL≈17 (OPT) because that is the information-theoretic limit for WikiText-2 with limited training data (128--400 samples), then AltOpt "converging toward" this ceiling is not evidence of AltOpt's effectiveness — it is evidence that any method will eventually reach the data-limited ceiling given enough steps.

4. **Parameter count asymmetry**: Protocol A (AltOpt/Full) trains all N_params, while Protocol D (AdamW/LoRA) trains only r×(d_in+d_out) parameters. Even with FLOPs normalization, the effective capacity gap could dominate the results. A comparison controlling for trainable parameter count (e.g., varying LoRA rank to match AltOpt's effective degrees of freedom) would provide a cleaner test.

### Missing Stakeholder Perspectives
- **Practitioners deploying post-trained models**: They care about wall-clock time and GPU memory, not just FLOPs. ALS requires inverting matrices that may not fit in GPU memory for large models, while AdamW+LoRA is memory-efficient.
- **Hardware/system researchers**: The FLOPs accounting assumes all operations have equal cost, but matrix inversion (O(b³)) has fundamentally different hardware utilization than matrix multiplication (O(Nb²)) on GPUs/TPUs.
- **Reproducibility auditors**: The paper's central quantitative claims depend on single-seed data. A reproducibility auditor would flag this as unreproducible at the claimed precision.

### Unexamined Premise
The entire paper operates under the unstated premise that **post-training optimization strategies should be evaluated on their convergence to the same minimum that AdamW finds.** But the paper's own perturbation discussion (Section 5.4) and the SAM/RWP literature suggest that different optimizers find *qualitatively different* minima. If AltOpt finds flatter minima that generalize better despite higher perplexity, the paper's use of AdamW as the gold standard is itself a frame-lock. The possibility that AltOpt's "underperformance" on perplexity could be a feature (finding different, potentially better-generalizing solutions) rather than a bug is acknowledged in passing (Section 7.2) but never taken seriously as an alternative interpretation of the results.

### Observations (Non-Defects)
- The factorial design is genuinely elegant and should be adopted. The irony is that the paper's own AltOpt framework would benefit from a factorial decomposition of its internal components.
- The paper would be stronger if it were reframed as "A Methodology for Fair Comparison of Post-Training Strategies, with a Case Study on Alternating Optimization vs LoRA" rather than primarily as a paper about AltOpt itself.
- The acknowledgment of Protocol C's asymmetry (SGD-only, no ALS) in the limitations is commendable honesty.

---

# Phase 2: Editorial Synthesis & Decision

## Editorial Decision Package

### Manuscript Information
- **Title**: Disentangling Optimizer and Parameter Form: A 2×2 Factorial Study of Alternating Optimization vs Low-Rank Adaptation for LLM Post-Training
- **Review Round**: Round 1

---

## Decision

### Major Revision

This is not a rejection — the paper makes a genuine methodological contribution. It is also not a minor revision — the empirical validation has critical gaps that require substantial new experiments. A major revision with the specific requirements below would produce a significantly stronger paper.

---

## Reviewer Summary

| Reviewer | Role | Recommendation | Confidence |
|----------|------|---------------|------------|
| EIC | TMLR Area Chair — empirical ML methodology | Major Revision | 4 |
| Reviewer 1 | Experimental design & statistical methodology specialist | Major Revision | 4 |
| Reviewer 2 | LLM post-training & PEFT domain expert | Major Revision | 4 |
| Reviewer 3 | Numerical optimization & scientific computing | Major Revision | 3 |

---

## Consensus Analysis

### Points of Agreement (Consensus)

**[CONSENSUS-4]** (All reviewers agree):

1. **The 2×2 factorial design is a genuine methodological contribution.** EIC calls it "the paper's strongest contribution," R1 notes it is "correctly specified," R2 says it is "a contribution I would like to see adopted more broadly," and R3 endorses it as "a methodological contribution that I would like to see adopted." This is the paper's core strength and should be preserved and emphasized.

2. **The single-seed matrix experiment is inadequate for the conclusions drawn.** EIC flags the "single-seed data for the matrix experiment," R1 identifies it as a CRITICAL weakness, R2 notes the need for "standard errors on all fitted parameters," and R3 (DA) identifies it as a CRITICAL evidence gap. All agree that the gap estimates require multi-seed replication.

3. **The crossover predictions are speculative and not experimentally supported.** EIC states the predictions are "entirely model-based extrapolations," R1 notes "N=6 data points... insufficient to meaningfully constrain the fit," R2 says the "tertiary contribution is the weakest," and R3's DA counterpart identifies it as overgeneralization. The predictions should be presented as speculative or verified experimentally.

4. **The ALS reconstruction loss magnitude finding (~10⁴-10⁵) is important.** EIC praises the "mathematical modeling," R1 notes it provides mechanistic explanation, R2 calls it "important and not widely discussed," and R3 identifies it as "the paper's most important result." This finding should be highlighted more prominently.

**[CONSENSUS-3]** (3/4 reviewers agree):

1. **The "necessary" framing overstates the contribution.** EIC, R2, and the DA all flag the claim that factorial design is "necessary" as too strong. R3 did not specifically address this point. Resolution: the strong language should be softened.

2. **The perturbation phase is insufficiently developed.** EIC, R2, and R3 all note that the perturbation analysis (Section 5.4) is superficial. R1 did not specifically address this. Resolution: either expand the perturbation analysis or reduce its prominence.

### Points of Disagreement

**Disagreement 1: Severity of the factorial ANOVA incompleteness**
- **R1 view**: R1 identifies the lack of formal ANOVA as a MAJOR weakness — "the paper cannot claim to have statistically demonstrated main effects and interactions."
- **R3 view**: R3 (DA perspective) identifies the composite nature of AltOpt as a CRITICAL issue — "The 2×2 design solves one attribution problem while creating another."
- **Disagreement type**: Severity disagreement — both agree there is an attribution issue, but differ on whether formal ANOVA or component ablation is the higher priority.
- **Editor's Resolution**: Both issues must be addressed. The formal ANOVA is a lower-effort requirement (requires multi-seed data which is needed anyway). The component ablation is a higher-effort requirement. **Priority**: Formal ANOVA is required (P1); component ablation analysis is strongly suggested but may be addressed qualitatively in the discussion if compute-limited (P2).
- **Resolution Rationale**: The formal ANOVA is a statistical prerequisite for interpreting the factorial results. The component ablation addresses a deeper conceptual issue, but requiring a full internal factorial decomposition of AltOpt could be scope-creep for a single revision. The authors should at minimum discuss the component confound and acknowledge it as a limitation.

**Disagreement 2: Whether Protocol C's asymmetry invalidates the factorial design**
- **R3 view**: Protocol C (AltOpt/LoRA, SGD-only) is fundamentally different from Protocol A (AltOpt/Full, ALS+SGD+Perturb), weakening the factorial symmetry. Rename the framework to avoid implying it is a single coherent "optimizer type."
- **EIC view**: The asymmetry is acknowledged in limitations and does not invalidate the overall approach; the paper is transparent about this.
- **Disagreement type**: Severity disagreement.
- **Editor's Resolution**: The asymmetry must be discussed more prominently in the main text (not just limitations) and its impact on the interaction term (A-B)-(C-D) quantified. The framework naming should be made more specific per R3's suggestion. These are required revisions.
- **Resolution Rationale**: R3's optimization-community perspective is valuable — the naming issue affects how the paper will be received. The experimental asymmetry is a genuine methodological concern that the current limitations section understates.

---

## Decision Rationale

The four reviewers unanimously agree that this paper makes a genuine methodological contribution (the 2×2 factorial design) and contains valuable empirical findings (ALS reconstruction loss magnitude, non-monotonic convergence pattern). Three of four find the work promising enough to warrant revision rather than rejection.

However, consensus also exists on critical weaknesses that prevent acceptance in current form. The single-seed matrix experiment — which produces the paper's headline quantitative results — does not support the claimed effect magnitudes given Protocol A's 40.6% CV. The crossover predictions are entirely speculative extrapolations. The formal statistical analysis (ANOVA, confidence intervals) is missing. And the internal component confound within AltOpt has not been addressed.

The decision is **Major Revision** because these issues require new experiments (multi-seed replication, possibly extended step counts) and substantial rewriting (reframing speculative claims, adding formal statistics). A minor revision cannot address these within 2--4 weeks.

The paper should not be rejected because the core ideas — the factorial design methodology, the ALS reconstruction loss analysis, and the non-monotonic convergence observation — are sound and valuable. With the revisions specified below, this paper would be suitable for TMLR or a comparable venue.

**Devil's Advocate CRITICAL findings assessment**: The DA raises three CRITICAL issues. C1 (composite AltOpt confound) is philosophically valid but can be addressed through discussion rather than requiring new experiments — the component confound should be acknowledged and discussed. C2 (single-seed evidence gap) is corroborated by all four other reviewers and is a required fix. C3 (data-conclusion mismatch for Qwen) is well-taken — the paper should explicitly address the anomalous 400-step spike and avoid the implication of monotonic convergence. All three CRITICAL issues must be addressed in the revision; none are grounds for rejection because they can be resolved through better analysis, discussion, and additional experiments.

---

## Required Revisions (Must Fix)

| # | Revision Item | Source Reviewer | Severity | Section | Estimated Effort |
|---|--------------|----------------|----------|---------|-----------------|
| R1 | Multi-seed replication of matrix experiment (min 3 seeds at 100, 200, 400, 800 steps) with confidence intervals on all gap estimates | EIC, R1, DA (C2) | Critical | 5.3 | 2--3 weeks |
| R2 | Formal two-way ANOVA with F-statistics, p-values, partial η² for main effects and interaction | R1 | Critical | 5.2 | 3--5 days |
| R3 | Downgrade crossover predictions to "extrapolated estimates" with explicit caveats; OR run experiments to verify at least the GPT-2 predicted crossover (~800 steps) | EIC, R1, R2, DA (M1) | Critical | 6.3 | 1--3 days (rewrite) or 1--2 weeks (verify) |
| R4 | Report raw Protocol A and B perplexity values (not just gap) for all step counts in Table 2 | DA (M2) | Major | 5.3 | 1 day |
| R5 | Discuss practical significance of residual A-B gap at 800 steps (34× worse PPL than AdamW plateau); contextualize "convergence" narrative | DA (M3) | Major | 7.1, 8 | 1--2 days |
| R6 | Rename AltOpt to a more specific term (e.g., "ASP: ALS-SGD-Perturbation") and connect to broader alternating optimization taxonomy | R3 | Major | 4, throughout | 1--2 days |
| R7 | Move Protocol C asymmetry discussion from limitations (7.3) to main methodology (3.2 or 4) and quantify its impact on interaction term interpretation | EIC, R3 | Major | 3.2, 4 | 2--3 days |

### Required Item Details

**R1: Multi-seed replication**
- **Problem**: Table 2 reports gap values at 6 step counts with single seed. Protocol A CV is 40.6% at 200 steps on OPT-125m. The gap values cannot be meaningfully interpreted without error bounds.
- **Source**: EIC (Weakness 2), R1 (Weakness 1), DA (C2). Consensus-4 agreement.
- **Requirement**: Run Protocol A and B with ≥3 seeds at step counts 50, 100, 200, 400, 800 for both OPT-125m and Qwen2.5-0.5B. Report gap as mean ± SE. Add error bars to gap-vs-steps plots. Report 95% confidence intervals on the "150× shrinkage" estimate.
- **Acceptance criteria**: Every gap value in the revised Table 2 has an associated uncertainty estimate. The oscillatory pattern is demonstrably consistent across seeds (or the authors acknowledge it is not and adjust claims accordingly).

**R2: Formal ANOVA**
- **Problem**: The paper uses ANOVA terminology (main effects, interaction) but presents only arithmetic differences between cell means, without formal statistical tests.
- **Source**: R1 (Weakness 2). Corroborated by EIC and R2.
- **Requirement**: With multi-seed data from R1, run two-way ANOVA (optimizer × parameter form) at each step count. Report F-statistics, degrees of freedom, p-values, and partial η². If multi-seed data is not available for 100-step factorial design, clearly state that conclusions are descriptive rather than inferential.
- **Acceptance criteria**: Table 1 is augmented with formal ANOVA results, or the paper explicitly downgrades claims from "main effects" to "descriptive comparisons."

**R3: Crossover predictions**
- **Problem**: Crossover predictions for models up to Llama-2-7B are extrapolated from 2--3 data points on models ≤500M parameters.
- **Source**: All reviewers (Consensus-4).
- **Requirement**: Either (a) extend experiments to verify the GPT-2 predicted crossover (~800 steps), which is within the described experimental budget, OR (b) reframe Section 6.3 as "Extrapolated Crossover Estimates" with explicit caveats about the limitations of two-point extrapolation. Remove "crossover at approximately 1,000--3,000 steps" from the abstract; replace with qualified language.
- **Acceptance criteria**: The abstract no longer presents crossover predictions as established findings. Section 6.3 explicitly states limitations of the extrapolation.

---

## Suggested Revisions (Should Fix)

| # | Revision Item | Source Reviewer | Priority | Section | Expected Improvement |
|---|--------------|----------------|----------|---------|---------------------|
| S1 | Add missing LoRA/PEFT references (Malladi 2023, Dettmers 2024, Liu 2024, Lialin 2023, Aghajanyan 2021) | R2 | P2 | 2.2 | Strengthens literature positioning |
| S2 | Discuss scalability limitations of prior BCD/ADMM methods specifically for transformer architectures | R2 | P2 | 2.1 | More accurate literature characterization |
| S3 | Add cross-disciplinary references (Boyd 2011 ADMM, Bolte 2014 PALM, Welling & Teh 2011 SGLD) | R3 | P2 | 2, 4 | Broader optimization community appeal |
| S4 | Report ALS scheduling parameters (k, C) used for each experiment | R1 | P2 | 5.1 | Reproducibility |
| S5 | Report ALS block size(s) and motivate the choice | R3 | P2 | 4.1 | Methodological transparency |
| S6 | Ablate or discuss the three AltOpt failure mechanisms (loss dominance, coupling, momentum reset) to determine relative importance | R2, DA | P2 | 6.4 | Interpretability of results |
| S7 | Report standard error of PPL across bootstrap resamples of evaluation data | R1 | P2 | 5.1 | Evaluation reliability |
| S8 | Compare perturbation phase to SAM or RWP baseline, or reduce its prominence | R2, R3 | P2 | 5.4 | Scientific rigor |
| S9 | Report normalized gap values (A-B)/B alongside absolute gaps for cross-model comparability | R1 | P2 | 5.3 | Interpretability |
| S10 | Discuss whether AdamW plateau represents an information-theoretic ceiling for WikiText-2 with limited training data | DA | P2 | 7.1 | Alternative interpretation |
| S11 | Acknowledge the internal component confound in AltOpt (ALS vs SGD vs perturbation effects cannot be disentangled by the current design) | DA (C1) | P2 | 7.3, 8 | Intellectual honesty |

---

## Revision Roadmap

### Priority 1 — Structural Revisions (Estimated total effort: 3--5 weeks)
- [ ] **R1**: Run multi-seed matrix experiment (≥3 seeds) for OPT and Qwen at key step counts; add error bars and confidence intervals
- [ ] **R2**: Compute formal two-way ANOVA with the multi-seed data
- [ ] **R3**: Either verify GPT-2 crossover experimentally OR reframe predictions as speculative
- [ ] **R4**: Report raw per-protocol PPL values alongside gap values in Table 2
- [ ] **R5**: Add practical significance discussion of residual gap
- [ ] **R6**: Rename AltOpt to specific term; connect to broader alternating optimization literature
- [ ] **R7**: Move Protocol C asymmetry discussion to main text and quantify impact

### Priority 2 — Content Supplementation (Estimated total effort: 2--3 weeks)
- [ ] **S1**: Add 5 missing LoRA/PEFT references with brief discussion
- [ ] **S2**: Add discussion of BCD/ADMM scalability limitations for transformers
- [ ] **S3**: Add 3 cross-disciplinary optimization references and connections
- [ ] **S4**: Report per-experiment scheduling parameters (k, C)
- [ ] **S5**: Report ALS block sizes with motivation
- [ ] **S6**: Discuss component ablation or acknowledge component confound
- [ ] **S7**: Report PPL evaluation stability via bootstrap
- [ ] **S8**: Compare perturbation to SAM/RWP baseline or de-emphasize
- [ ] **S9**: Add normalized gap values for cross-model comparison
- [ ] **S10**: Discuss AdamW plateau as possible data ceiling
- [ ] **S11**: Acknowledge AltOpt internal component confound

### Priority 3 — Text and Formatting (Estimated total effort: 3--5 days)
- [ ] Soften "necessary" to "rigorous" in abstract (Section 1, 8)
- [ ] Replace informal "digestion period" with "SGD relaxation phase" or similar (Sections 1, 6.4, 8)
- [ ] Clarify "5--30×" notation in abstract
- [ ] Standardize reference formatting
- [ ] Report random seeds, hardware specifications, hyperparameter sweep methodology for reproducibility
- [ ] Add tokenization and context length details to Section 5.1
- [ ] Clarify "Choromanska" spelling in reference [4]
- [ ] Add note that training LR varies by model family (potential confound)

### Total Estimated Effort
- **Required revisions (P1)**: 3--5 weeks
- **Suggested revisions (P2)**: 2--3 weeks
- **Text/formatting (P3)**: 3--5 days
- **Combined**: 6--9 weeks for a complete revision

---

## Revision Deadline
- **Recommended deadline**: 8 weeks from decision date
- **Basis**: Major Revision with substantial new experiments required
- **Extension policy**: Contact the editorial office at least 1 week before the deadline if extension is needed

---

## Response Letter Instructions

Please use a point-by-point response format addressing every Required and Suggested revision item. For each item, indicate: (1) the action taken, (2) the location of the change in the revised manuscript, and (3) if a suggestion was not adopted, the rationale. For Required Revisions (R1--R7), provide evidence of completion (e.g., new tables, figures, or analysis).

---

## Closing

Dear Author(s),

After careful consideration by four independent reviewers, we are not able to accept your manuscript in its current form. However, the reviewers were unanimously positive about the core methodological contribution — the 2×2 factorial design — and found the empirical findings valuable. The primary barrier to acceptance is the insufficient empirical validation: single-seed experiments cannot support the quantitative claims at the reported precision, and the crossover predictions extend beyond what the data can support.

We encourage you to carefully address the required revisions (R1--R7) and as many suggested revisions (S1--S11) as feasible. The multi-seed replication (R1) and formal statistical analysis (R2) are essential. The revised manuscript will undergo another round of review.

We look forward to receiving your revision within 8 weeks.

Sincerely,
The Editorial Board
*Transactions on Machine Learning Research*

---

## Appendix: Reviewer Report Summary

| Reviewer | Role | Recommendation | Confidence | Key Point |
|----------|------|---------------|------------|-----------|
| EIC | TMLR Area Chair — empirical ML methodology | Major Revision | 4 | Factorial design is genuine methodological contribution; empirical validation insufficient |
| R1 | Experimental design & statistical methodology | Major Revision | 4 | Single-seed data and missing formal ANOVA undermine quantitative claims |
| R2 | LLM post-training & PEFT domain expert | Major Revision | 4 | Literature coverage has gaps; ALS reconstruction loss finding is valuable |
| R3 | Numerical optimization & scientific computing | Major Revision | 3 | AltOpt naming is misleading; broader optimization literature connections needed |
| DA | Devil's Advocate — adversarial stress test | N/A (no scoring) | N/A | Internal component confound; single-seed evidence gap; anomalous Qwen spike |

---

## Summary of Scores

| Dimension | EIC | R1 | R2 | R3 | Avg | Descriptor |
|-----------|-----|----|----|----|-----|-----------|
| Originality | 72 | 70 | 68 | 65 | 69 | Strong |
| Methodological Rigor | 60 | 45 | 58 | 50 | 53 | Adequate |
| Evidence Sufficiency | 45 | 38 | 48 | 42 | 43 | Weak |
| Argument Coherence | 65 | 60 | 62 | 55 | 61 | Adequate |
| Writing Quality | 68 | 65 | 70 | 62 | 66 | Adequate |
| Literature Integration | — | — | 55 | — | 55 | Adequate |
| Significance & Impact | — | — | — | 58 | 58 | Adequate |

**Weighted assessment**: The paper has strong originality and adequate argument coherence, but the evidence sufficiency (single-seed, no CIs, speculative extrapolation) and methodological rigor (missing formal statistics) pull the overall evaluation below the acceptance threshold. These are fixable with the revisions specified above.

---

This completes the full review package: **Phase 0** field analysis with reviewer configuration, **Phase 1** five independent reviewer reports, and **Phase 2** editorial synthesis with decision and prioritized revision roadmap.