# RE-REVIEW: Round 2 — Verification Review

**Decision**: MINOR REVISION (upgraded from Round 1's Major Revision)
**Date**: 2026-06-12
**Reviewer**: EIC (TMLR re-review protocol)

## Revision Response Summary

| Category | Total | FULLY ADDRESSED | PARTIALLY | NOT |
|----------|-------|-----------------|-----------|-----|
| Required (R1-R7) | 7 | 6 | 1 (R5) | 0 |
| DA Critical (C1-C3) | 3 | 3 | 0 | 0 |
| Suggested (S1-S11) | 11 | 9 | 1 (S11) | 1 (S5) |
| **Address Rate** | **21** | **18 (86%)** | **2** | **1** |

## New Issues (5, all MINOR)

| ID | Issue | Location |
|----|-------|----------|
| NEW-1 | CI [-1k,14k] crosses zero vs PB ANOVA p=0.039 — tension needs explanation | Section 5.3 |
| NEW-2 | "12 seeds for 80% power" — parameters not specified | Section 5.3 |
| NEW-3 | Cohen's d=1.17 step-count ambiguity | Section 5.3 |
| NEW-4 | PB ANOVA missing test statistic (F-like or bootstrap moments) | Section 5.2 |
| NEW-5 | Instability-as-finding claim needs hedging | Section 7.4 |

## Required Revisions (2, ~2-3h)

RR1: Explain CI-vs-ANOVA tension at 800 steps
RR2: Clarify Cohen's d step count

## Suggested Revisions (5, ~1-2h)

SR1: Power analysis parameters
SR2: Hedge instability claim
SR3: Fix abstract p-value
SR4: Normalized gap values
SR5: PB ANOVA test statistic

## Strengths Post-Revision
1. Multi-seed replication transforms credibility
2. Honest limitation framing
3. Reference coverage: 13→24
4. Clear convergence trajectory with uncertainty
5. Consistent ASP renaming

## Decision Rationale
The revision is substantial and effective. Multi-seed replication (N=3-5) transforms the paper from suggestive single-run observations to a dataset supporting statistical inference. PB ANOVA with partial η², Cohen's d, and bootstrap CIs provide appropriate quantification. The paper is clearly closer to acceptance. Five minor issues remain — none blocking. Estimated acceptance-ready after one more light revision pass.
