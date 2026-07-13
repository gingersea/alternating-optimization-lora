# Project Assessment: Alternating Optimization for LoRA Post-Training

**Date**: 2026-07-13
**Status**: Comprehensive assessment, gap analysis, and forward plan.

---

## Part I: Executive Summary

### What this project is

A 2×2 factorial study crossing optimizer type (ASP vs AdamW) with parameter form (full-rank vs LoRA) for LLM post-training. The project discovered a Rank Sufficiency Law ($r_{\min} = \eta \cdot L/d_h$, $\eta \approx 230$) that quantitatively predicts the minimum LoRA rank needed for any Transformer architecture, and demonstrated that full-rank fine-tuning on small data catastrophically overfits.

### Current output

- **Paper v3.3**: 505-line LaTeX manuscript, 5 figures, 28 references
- **Experiments**: 17 completed across 8 architectures (GPT-2 through Qwen2.5-7B)
- **Code**: Full framework in `altopt/`, 53 experiment scripts in `experiments/`
- **Documentation**: 8 docs in `docs/`, 6 rounds of review responses
- **Review status**: 6 formal rounds + 1 Grok review; last verdict "Minor Revision → Accept"

### High-level verdict

**The science is solid. The paper is nearly there. Three things separate v3.3 from submission-ready:**

1. A version synchronization problem (README says v3.4, paper file is v3.3, change log is v0.3)
2. The Grok "Minor Revision" items need explicit tracking and resolution
3. The narrative arc needs tightening — the central contribution is the Rank Sufficiency Law, but the paper oscillates between multiple framings

---

## Part II: Current State — What We Have

### 2.1 Scientific Contributions (all peer-reviewed and experimentally validated)

| # | Contribution | Status | Evidence |
|---|-------------|--------|----------|
| 1 | Rank Sufficiency Law $r_{\min} = \eta \cdot L/d_h$ | ✅ Complete | 5-model cross-arch, 3/3 falsification passed |
| 2 | Full-rank overfits on small data; LoRA preserves accuracy | ✅ Complete | HellaSwag/MMLU/ARC × N=3 seeds |
| 3 | ASP depth boundary at ~26 layers | ✅ Complete | 8 architectures, 11 failed 7B attempts |
| 4 | M-index memorization diagnostic | ✅ Complete | C4 cross-domain, scale-dependent $\beta$ |
| 5 | ASP-LoRA negative synergy | ✅ Complete | 7/7 independent comparisons |
| 6 | 2×2 factorial methodology | ✅ Complete | Reusable template |
| 7 | $\eta$ mechanism: pretraining quality modulation | ✅ Complete | 3 hypotheses tested, 2 falsified |
| 8 | X1: Low-rank ALS solver | ✅ Complete | `torch.linalg.solve` implementation |
| 9 | X2: Causal depth boundary theory | ✅ Complete | SCM framework, 5 predictions |
| 10 | X3: Universal $\eta$ nomogram | ✅ Complete | $R^2 = 0.88$, 7 architectures |

### 2.2 Experiment Inventory (17 total)

| ID | Experiment | Model(s) | Key Result |
|----|-----------|----------|------------|
| P0 | Chinese WikiText | Qwen2.5-0.5B | r=8 plateau language-independent; $\eta \propto H$ falsified |
| P1 | ASP crossover | GPT-2, Qwen-0.5B | AdamW gap shrinks 7.8× (OPT) to 15.8× (Qwen) |
| P2 | T5 encoder-decoder | T5-small | Boundary condition confirmed; PPL undefined on enc-dec |
| P3 | M-index cross-scale | OPT-125m, Qwen-7B | $\beta$ scale-dependent phase transition |
| P4 | SmolLM2 fine-grained | SmolLM2-135M | $r_{\min} \approx 12$ confirmed (±1 rank) |
| P5 | Multi-seed rank curve | Qwen2.5-0.5B | SE < 0.002; plateau statistically robust |
| F1 | $\eta$ mechanism resolution | Cross-arch | H and N_samples eliminated; pretraining quality confirmed |
| F2 | Full ASP Cholesky | OPT-125m (12L) | Non-monotonic convergence confirmed; best PPL=1.87 |
| A | SST-2 classification | Qwen2.5-0.5B | r=4/8/32 all 84.7% — plateau extends to classification |
| E2 | Long-horizon rank stability | Qwen2.5-0.5B | r=8 superior at long horizon; r=256 overfits |
| E4 | FFN LoRA | Qwen2.5-0.5B | attn+FFN r=4 beats attn-only r=8 |
| X1 | Low-rank ALS solver | 0.5B, 7B | Production-ready; `linalg.solve` + lstsq fallback |
| X2 | Causal depth boundary | Theoretical | SCM derivation; 5 falsifiable predictions |
| X3 | $\eta$ nomogram | 7 architectures | $R^2=0.88$, SmolLM2 uniquely at r=12 |
| — | Core 2×2 matrix | GPT-2, OPT, Qwen 0.5B/7B | Full factorial at 0.5B; 3/4 cells at 7B |
| — | Downstream eval | Qwen2.5-7B | HellaSwag, MMLU, ARC — all 3 tasks |
| — | Cross-domain eval | Qwen2.5-7B | C4 PPL; M-index validation |

### 2.3 Codebase

| Component | Files | Quality |
|-----------|-------|---------|
| Core framework (`altopt/`) | 14 Python files | Solid: typed, tested, documented |
| Experiment scripts (`experiments/`) | 53 files | Functional but some duplication |
| Tests (`tests/`) | Present | Coverage unclear |
| Profiling (`altopt/profiling/`) | Present | Used for FLOPs accounting |

### 2.4 Documentation

| File | Content | Status |
|------|---------|--------|
| `README.md` | Project overview, findings, repo map | ✅ Current (v3.4) |
| `todo.md` | Experiment tracker | ✅ Complete |
| `docs/asp_deep_dive.md` | ASP full-rank vs LoRA code walkthrough | ✅ |
| `docs/asp_mathematical_formulation.md` | Rank sufficiency law math | ✅ |
| `docs/als_block_solving_explained.md` | ALS block-wise solving explained | ✅ |
| `docs/fair_comparison_methodology.md` | FLOPs normalization, 2×2 design | ✅ |
| `docs/algorithm_math.md` | 4 protocols math (Chinese) | ⚠️ Outdated vs new formulation |
| `docs/causal_depth_boundary.md` | X2 causal theory | ✅ |
| `paper/paper_v3.3.tex` | LaTeX manuscript | ⚠️ Version sync needed |
| `paper/review_round{1-6}.md` | Review responses | ✅ |
| `paper/revision_plan.md` | R1 revision execution plan | ⚠️ Pre-R6, now outdated |

---

## Part III: Gap Analysis — What Separates Us from Submission

### 3.1 Critical Gaps (Blocking)

#### GAP-1: Version Synchronization

| Artifact | Claims | Reality |
|----------|--------|---------|
| README.md | "Paper v3.4" | Paper file is `paper_v3.3.tex` |
| todo.md | "v3.4 FINAL — Complete" | Paper says "v3.3" in filename |
| `paper_v0.3_updates.md` | Change log from v0.2 to v0.3 | Paper is already at v3.3 — change log versioning is decoupled from paper versioning |

**Action**: Rename paper to v3.4, update version references across all files. Create a single source of truth for version number.

#### GAP-2: Grok Review "Minor Revision" Items — Untracked

The README states "Grok review passed (Minor Revision → Accept)" but the specific minor revision items are not documented in any review round file. We have `review_round1.md` through `review_round6.md` but no `review_grok.md` or equivalent.

**Action**: Document the Grok review comments explicitly and confirm each is resolved in the current manuscript.

#### GAP-3: Paper Claims vs. README Claims — Inconsistency

| Claim | README | Paper v3.3 |
|-------|--------|------------|
| Experiments count | "17 experiments" | "14 hypothesis-driven experiments" (line 389) but "17 hypothesis-driven experiments" (line 409) — contains a contradiction |
| Architecture count | "5 model families" (cross-arch) | 8 architectures tabulated |
| ASP-LoRA ALS | X1 claims "enables Protocol C to include ALS at 7B" | Protocol C description says "without ALS" |

**Action**: Resolve experiment count (14 or 17?), make architecture count consistent, clarify Protocol C status.

#### GAP-4: No Author List

Paper line 23: `\author{[Authors to be determined]}`.

**Action**: Decide authorship and fill in.

#### GAP-5: Target Venue Undecided

No explicit target venue stated. Review style (6 rounds, TMLR-style) suggests TMLR or similar open-review venue. README/revision_plan reference "TMLR" once.

**Action**: Confirm target venue and format accordingly (page limits, style files, reference format).

### 3.2 Significant Gaps (Should Fix)

#### GAP-6: Reference Quality

28 references, but several are arXiv preprints (not peer-reviewed). For a TMLR submission, this is acceptable but weaker than published references. Key missing references:
- No citation for the specific AdamW implementation used
- No citation for the DeepSpeed backend
- The `anonymous2025convergence` reference should be updated if now published

**Action**: Audit references, add missing ones, update preprint references if published.

#### GAP-7: Figure Quality

5 figures referenced in the paper: `fig1_factorial.pdf`, `fig2_convergence.pdf`, `fig3_depth.pdf`, `fig4_overfitting.pdf`, `fig5_als_synergy.pdf`. The figures exist but:
- Generated from Python matplotlib — may need professional styling for publication
- No figure sources (the generating scripts) are referenced

**Action**: Review figure quality against venue standards. Ensure figure generation is reproducible.

#### GAP-8: Reproducibility Package

The paper promises "Code, configurations, and the η nomogram will be released upon publication" but there is no organized release package:
- No `environment.yml` or `requirements.txt` with pinned versions
- No Docker image
- No explicit artifact checklist
- Experiment configs are scattered across scripts

**Action**: Create a reproducibility package: pinned dependencies, experiment configs, pretrained model versions, random seeds.

#### GAP-9: Internal Paper Contradictions

Line 389: "14 hypothesis-driven experiments"
Line 409: "17 hypothesis-driven experiments across 8 architectures"

Line 3 of abstract: "8 architectures" but the paper v0.3 update log says "9 architectures"

**Action**: Fix all internal contradictions through a single careful pass.

### 3.3 Enhancement Gaps (Nice to Have)

#### GAP-10: Future Experiments (P1-P5 in todo.md)

| ID | Experiment | GPU | Value |
|----|-----------|-----|-------|
| E1 | Training budget equation $r_{\min}(N_{\text{samples}})$ | — | 🟡 |
| F3 | Multi-task $\eta$ (GLUE) | 2h | 🟡 |
| F4 | MoE validation (Mixtral) | 45min | 🟡 |
| E3 | LLaMA-3.2 validation | 1h | 🟢 |

These are explicitly marked "Post-Submission" in todo.md and are **not blocking**.

#### GAP-11: $\eta$ Nomogram Full Integration

The $\eta$ nomogram ($R^2=0.88$) is mentioned in the conclusion but only as an extension, not integrated into the main experimental sections. This is the project's strongest practical contribution — a lookup table for practitioners — but it's buried.

**Action**: Consider elevating the nomogram to a main-result figure.

#### GAP-12: PAC-Bayes Analysis Depth

The PAC-Bayes bound in Appendix A.5 is a one-paragraph sketch. For a venue that values theory (TMLR, ICML, NeurIPS), this could be expanded into a substantive result.

**Action**: Expand or remove — a sketch in the appendix is worse than omitting it.

---

## Part IV: Research — What We Need to Learn

### 4.1 Venue Research

We need to answer: where should this paper go?

| Venue | Pros | Cons |
|-------|------|------|
| **TMLR** | Open review, no deadline, accepts negative results, methodology papers | Lower prestige than top-3 |
| **ACL Rolling Review (ARR)** | Broad NLP audience, rolling submission | Reviewer pool highly variable |
| **NeurIPS** | High prestige | December deadline, competitive, may not value negative results |
| **ICLR** | High prestige, values theory | Next deadline ~Oct 2025 |
| **JMLR** | Premium journal, values methodology | Very slow review |

**Recommendation**: TMLR or ARR. The paper has already undergone 6 rounds of review in a TMLR-like format. ARR would reach the NLP community directly (LoRA, PEFT audience).

### 4.2 Competitive Landscape

Recent papers to monitor (post-June 2026):
- Any new LoRA theory papers claiming rank sufficiency results
- New PEFT methods that might supersede standard LoRA
- Follow-ups to Lee et al. (2026) on "Vanilla LoRA May Suffice"

**Action**: Run a literature sweep before submission to ensure claims of novelty remain valid.

### 4.3 Reviewer Expectation Calibration

Based on 6 rounds of review, the pattern is clear:

| Reviewer Concern | Frequency | Our Response |
|-----------------|-----------|-------------|
| Parameter-count confound | R6 (strongest) | ✅ Parameter-matched baseline, full-rank overfitting reframe |
| No downstream evaluation | R6 | ✅ HellaSwag/MMLU/ARC added |
| ASP loses everywhere | R1-R4, R6 | ✅ Reframed as negative result + implicit regularization |
| Presentation errors | R5 | ⚠️ Still need final sweep |
| Protocol C asymmetry | R2 | ✅ X1 closes the gap, quasi-factorial reframe |

### 4.4 The Parameter-Count Confound — Lingering Risk

This was the strongest rejection reason in R6. Our response:
1. Parameter-matched LoRA baseline (r=8 through r=512) — done
2. Downstream task evaluation — done
3. Reframe from "full-rank > LoRA" to "full-rank overfits" — done

**Residual risk**: A reviewer might still argue that comparing 494M vs 3M parameters is unfair regardless of FLOPs normalization. The counter-argument is that FLOPs normalization is the correct fairness metric, and the parameter-count difference IS the independent variable being studied.

---

## Part V: Forward Plan

### Phase 1: Paper Finalization (Priority: CRITICAL, ~2 days)

| Task | Effort | Depends on |
|------|--------|------------|
| 1.1 Version sync: rename paper to v3.4, update README/todo/docs | 30 min | — |
| 1.2 Fix internal contradictions (experiment count, architecture count) | 1h | — |
| 1.3 Document Grok review items and confirm resolution | 1h | Grok review notes |
| 1.4 Final text sweep: typos, stale references, cross-refs | 2h | 1.1-1.3 |
| 1.5 Fill in author list | 30 min | Decision needed |
| 1.6 Figure quality check and regeneration if needed | 2h | 1.4 |
| 1.7 Reference audit: add missing, update preprints | 1h | — |
| 1.8 Abstract polish — align with conclusion | 1h | 1.4 |
| 1.9 PAC-Bayes appendix: expand or remove | 1h | — |

### Phase 2: Reproducibility Package (Priority: HIGH, ~1 day)

| Task | Effort | Depends on |
|------|--------|------------|
| 2.1 Create `requirements.txt` with pinned versions | 30 min | — |
| 2.2 Document experiment configs in a single `configs/README.md` | 2h | — |
| 2.3 Create `reproduce.sh` for key experiments | 2h | 2.1-2.2 |
| 2.4 Archive model versions and random seeds | 30 min | — |
| 2.5 Write `ARTIFACTS.md` checklist | 1h | 2.1-2.4 |

### Phase 3: Venue Selection + Formatting (Priority: HIGH, ~1 day)

| Task | Effort | Depends on |
|------|--------|------------|
| 3.1 Make venue decision (TMLR vs ARR) | — | Discussion |
| 3.2 Reformat paper to venue style | 2h | 3.1 |
| 3.3 Adjust page length, figure placement | 2h | 3.2 |
| 3.4 Write cover letter / submission statement | 1h | 3.1 |

### Phase 4: Optional Enhancements (Priority: MEDIUM, post-submission)

| Task | Effort | Notes |
|------|--------|-------|
| 4.1 E1: Training budget equation | 1 day | Could strengthen $\eta$ story |
| 4.2 F3: Multi-task $\eta$ (GLUE) | 2h GPU | Extends rank sufficiency to classification |
| 4.3 F4: MoE validation | 45min GPU | Boundary condition test |
| 4.4 Elevate $\eta$ nomogram to main figure | 2h | Strongest practical contribution |
| 4.5 Clean up experiment scripts (deduplicate) | 4h | Code quality |

---

## Part VI: Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Another reviewer raises parameter-count confound | Medium | High | Already addressed; strengthen in §3.2 |
| Version/claim inconsistencies surface in review | High | Medium | Phase 1 sweep |
| Competitive work appears before submission | Low | High | Literature sweep in Phase 1 |
| Venue rejects negative results | Medium | High | Choose venue carefully; emphasize positive findings |
| Reproducibility issues on reviewer's hardware | Low | Medium | Phase 2 reproducibility package |
| Author disagreement on narrative/citations | Medium | Low | Author meeting before Phase 1 |

---

## Part VII: Decision Points

The following require explicit decisions before proceeding:

1. **Authorship**: Who are the authors? In what order?
2. **Target venue**: TMLR or ARR? (Recommend: ARR for NLP audience)
3. **Narrative weight**: Should the primary framing be (a) Rank Sufficiency Law, (b) 2×2 factorial methodology, or (c) ASP limitations as cautionary tale? (Recommend: (a) with (b) as methodological contribution)
4. **Future experiments**: Run E1/F3/F4 before submission or after? (Recommend: after — not blocking)
5. **PAC-Bayes appendix**: Expand into full result or remove? (Recommend: remove, it's underdeveloped)
6. **Code release**: GitHub repo public now or at submission? (Recommend: public at submission)
