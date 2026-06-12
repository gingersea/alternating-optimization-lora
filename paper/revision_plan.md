# Revision Response Plan: Round 1 Review Solutions

**Date**: 2026-06-12  
**Review Decision**: Major Revision  
**Target Venue**: TMLR  

---

## Research Findings: Statistical Methods

### 1. ANOVA Under Heteroscedasticity

**Problem**: Protocol A has 40.6% CV, Protocol B has 2.3% CV — classical F-test is biased.

**Solution**: **Parametric Bootstrap (PB) two-way ANOVA** (Xu et al. 2013, 2015; arXiv:2602.23815).

Why PB over alternatives:
- Classical F-test: biased under heteroscedasticity (overestimates Type-I error)
- Non-parametric bootstrap (NANOVA): requires larger sample sizes for accurate null distribution
- **PB test**: constructs null distribution via parametric resampling, shown to have superior size control and power for heteroscedastic two-way designs with small-to-moderate replications (3-5 per cell)
- Implementation: use `scipy.stats` + manual parametric bootstrap (no external dependency needed)

Alternative if PB is computationally infeasible: **Welch-type heteroscedastic ANOVA** (modified Bartlett test), which is closed-form and requires no resampling.

### 2. Confidence Intervals for Gap Ratios

**Problem**: "150× gap shrinkage" claim needs CI. Delta method has poor coverage for ratios.

**Solution**: **Fieller's method** for ratio CIs.

Why Fieller over alternatives:
- Delta method: assumes symmetry, poor coverage when denominator variance is high
- Fieller's method: produces asymmetric CIs, handles Cauchy-distributed ratios correctly, exact for normal data
- Bootstrap percentile: better for heavy-tailed distributions but computationally intensive
- **Recommendation**: Report both Fieller CI and bootstrap percentile CI for the "gap shrinkage" ratio

Implementation: Fieller CI = solve quadratic \(a\theta^2 + 2b\theta + c = 0\) where a,b,c derived from the joint distribution of numerator and denominator estimates.

### 3. Anomaly Detection for Qwen 400-step Spike

**Problem**: Qwen A-B gap spikes at 400 steps (397k vs typical 3-10k).

**Analysis**: Could be:
1. Seed artifact (single seed, 40.6% CV)
2. ALS cycle boundary effect (400 = 2 × 200 cycle boundary)
3. Genuine instability in 24-layer model

**Solution**: Multi-seed replication will disambiguate. Additional analysis:
- Check if the spike correlates with a specific ALS cycle boundary
- Compare loss trajectories across seeds at 400 steps
- If persistent across seeds: this is a genuine finding about deep model instability

---

## Solution Matrix: All Required Revisions

### R1: Multi-Seed Matrix Experiment [CRITICAL]

| Aspect | Solution |
|--------|----------|
| **Design** | 3 seeds × 2 models × 5 step counts × Protocol A+B = 30 runs |
| **Metrics** | Mean ± SE gap; 95% Fieller CI on ratio |
| **Validation** | Gap shrinkage consistent across all 3 seeds? Oscillatory pattern reproducible? |
| **Effort** | ~6-8h CPU |
| **Verdict** | Execute via modified `matrix_runner.py --seeds 42,123,456` |

### R2: Formal ANOVA [CRITICAL]

| Aspect | Solution |
|--------|----------|
| **Method** | Parametric bootstrap two-way ANOVA (optimizer × param_form) |
| **Output** | F-stat, p-value (PB), partial η², 95% CI for each effect |
| **At which steps** | 100, 200, 400, 800 (where we have multi-seed data) |
| **Software** | Custom implementation using scipy + numpy bootstrap |
| **Effort** | ~2h coding + analysis |
| **Verdict** | Write `experiments/anova_analysis.py` |

### R3: Crossover Prediction Downgrade [CRITICAL]

| Aspect | Solution |
|--------|----------|
| **Option A** | Run GPT-2 at 800 steps to verify crossover prediction (~3h CPU) |
| **Option B** | Reframe as "Extrapolated Crossover Estimates" with explicit caveats |
| **Decision** | Do both: run GPT-2 800-step + add caveats in text |
| **Effort** | ~3h CPU + 1h writing |
| **Verdict** | Execute via `experiments/verify_crossover.py` |

### R4-R7: Text/Analysis Changes [MAJOR]

| # | Solution | Effort |
|---|----------|--------|
| R4 | Add raw PPL values to Table 2 (already have data) | 30min |
| R5 | Compute residual gap ratio (A/B PPL at 800s), discuss AdamW plateau | 1h |
| R6 | Rename AltOpt → ASP throughout paper + codebase | 2h |
| R7 | Move Protocol C asymmetry to Section 3.2, quantify interaction term impact | 1h |

### DA Critical Findings

| # | Solution | Effort |
|---|----------|--------|
| C1 | Add "Component Confound" subsection in Discussion | 1h |
| C2 | Same as R1 (multi-seed) | — |
| C3 | Multi-seed analysis of Qwen 400-step; add explicit discussion of anomalous spike | 1h |

### Suggested Revisions (S1-S11)

| # | Solution | Effort |
|---|----------|--------|
| S1-S3 | Add 8 missing references to Sections 2.1-2.3 | 2h |
| S4-S5 | Report k,C,block_size per experiment in Section 5.1 | 30min |
| S8 | Add perturbation baseline comparison OR de-emphasize | 1h |
| S9 | Add normalized gap (A-B)/B column to Table 2 | 30min |
| S10 | Discuss AdamW plateau as possible data ceiling | 30min |
| S6,S7,S11 | Discussion-level additions | 2h |

---

## Execution Plan

### Phase 1 — Text/Code Changes (now, ~6h)
```
R4: Raw PPL in Table 2          ✅ data exists
R6: Rename AltOpt → ASP          ✅ search-replace
R7: Move Protocol C asymmetry    ✅ text edit
C1: Component confound section   ✅ new subsection
C3: Qwen spike analysis          ✅ new discussion paragraph
S1-S3: Add missing references    ✅ collect + cite
S4-S5: Report hyperparams        ✅ from experiment logs
S8-S11: Discussion additions      ✅ text edits
```

### Phase 2 — Experiments (launch, ~12h CPU)
```
R1: Multi-seed matrix (30 runs)        ~8h
R3: GPT-2 800-step crossover verify    ~3h
```

### Phase 3 — Statistical Analysis (after Phase 2, ~3h)
```
R2: PB two-way ANOVA                   ~2h
R5: Residual gap significance          ~1h
```

---

## Immediate Next Actions

1. **Execute Phase 1 changes** (text/code — no experiments needed)
2. **Launch Phase 2 experiments** in background
3. **Write Phase 3 analysis scripts** while experiments run
