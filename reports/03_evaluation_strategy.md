# Deliverable 3 — Evaluation strategy

The evaluation target is fixed by the assignment: given a person's wave 1-3 persona,
predict their wave-4 answers, score against the real wave-4 answers, and interpret the
result relative to the human test-retest reference line. This document is organised around
that: comparators first, metrics second, protocol third, acceptance criteria last.

---

## 1. The comparator ladder

A number without a comparator is not a result. Every reported score sits on this ladder.

| # | Comparator | Value (measured) | What it rules out |
|---|---|---|---|
| B0 | Uniform random over valid options | — | nothing; sanity check only |
| B1 | **Population mode / mean** per item | **0.471** | "the model just learned the item's marginal" |
| B2 | Demographic-cell mode (age × sex × education) | **0.467** | "the model just learned demographics" — close to B1, so demographics alone do little |
| B3 | kNN on the person's response vector | **0.486** | "the model just learned similarity between respondents" — essentially tied with the best shipped LLM |
| B4 | **Copy-forward** — this person's earlier answer | **0.612** | "the model learned anything beyond repetition" |
| C  | **Human test-retest** | **0.612** | — identical to B4 by construction |
| — | Best shipped LLM (JSON Persona, GPT-4.1) | 0.488 | external reference point |

Two facts about this ladder drive everything downstream.

**B4 and C are the same number.** Test-retest is `P(wave-4 answer == the same person's
earlier answer)`; copy-forward is `predict wave-4 with that earlier answer`. One
computation. So "reached the human test-retest line" and "matched the strongest simple
repeated-item baseline" are the same empirical claim.

**C is not an upper bound.** A human re-test draw contains occasion noise; a model
predicting the conditional mode does not. A genuinely good model *should* exceed C on point
metrics. Treating C as a cap is a modelling error, and reporting "we approached the human
ceiling" as a success is a communication error.

**The band B1→B4 is 0.140 wide, 95% CI [0.137, 0.144].** Every claimed improvement is
quoted as a fraction of this band, never as raw accuracy. The best shipped system captures
about 12% of it.

---

## 2. Metrics by question type

Qualtrics `QuestionType` does not determine the metric — the same `Matrix` type covers
2-point and 100-point items. Classify by observed support (`reports/exploration/01_structure_and_retest.md` §2b):

| Observed scale | Cols | Primary metric | Secondary | Why not accuracy alone |
|---|---:|---|---|---|
| binary | 65 | **Cohen's κ** | accuracy | items run to 80/20; a constant predictor scores 0.80 |
| ordinal_short (≤11 pts) | 43 | **MAE** | within-1 acc, κ | off-by-one ≠ off-by-four |
| ordinal_long | 2 | MAE | Spearman | exact match too strict |
| continuous | 14 | **normalised MAE** | Pearson | human retest exact agreement is 0.136 — exact match is noise |
| numeric text-entry | 2 | normalised MAE | — | numeric fields, not prose |

Human reliability per family (the reference line for each):

| scale | retest exact | retest κ |
|---|---:|---:|
| binary | 0.809 | 0.534 |
| ordinal_short | 0.501 | 0.326 |
| ordinal_long | 0.328 | 0.211 |
| continuous | 0.136 | — |

κ is defined on 110 of 126 columns; it does not exist for continuous support. Report the
denominator every time.

**Aggregation.** Compute per column on valid pairs, then average over columns with equal
weight. Pooling raw pairs lets the 40-column pricing block dominate the 1-column
experiments and silently reweights the benchmark by participant count.

**Normalised score.** For cross-family comparison:

$$\text{NS} = \frac{\text{model} - \text{B1}}{\text{B4} - \text{B1}}$$

NS=0 means no personalisation, NS=1 means copy-forward-equivalent, NS>1 means denoising.
*Caveat, stated because it matters:* on items where B4 ≈ B1 the denominator collapses and
NS explodes. Report NS only where B4−B1 > 0.05, and report the excluded columns.

---

## 3. Distribution-level metrics (non-optional)

Point accuracy alone is insufficient, and this is measurable, not theoretical. Across the
13 shipped systems the median variance ratio is **0.48** (range 0.24–0.87); only one
exceeds 0.85.

The decisive number is a null result: the correlation between accuracy and variance ratio
is **r = +0.06 (n = 13)** — effectively zero. It is not that the two trade off in some
predictable way that a leaderboard could account for; it is that **accuracy carries no
information at all about whether the simulated population is correctly dispersed.**
Gemini-Flash-2.5 preserves variance best (0.87) while scoring near the bottom on accuracy;
the second-most-accurate system is the second-most flattened (0.245). Knowing one number
tells you nothing about the other.

Variance therefore has to be measured and gated in its own right, which is why it appears
below as a ship blocker rather than as a secondary metric.

| Metric | Definition | Target |
|---|---|---|
| **Variance ratio** | var(predicted) / var(human), median over columns | 0.9 – 1.1 |
| **Marginal TV distance** | per item, between predicted and human answer distributions | < 0.10 |
| **Correlation-structure error** | Frobenius norm of (predicted − human) item-item correlation matrix, normalised | report; no threshold yet |
| **Treatment-effect recovery** | for each between-subject experiment, does the simulated effect size CI overlap the human one? | ≥ 80% of experiments overlap |

The last row is the commercial claim — "simulate the experiment instead of running it" — and
it is the only metric that tests it directly. The dataset's 11 between-subject and 5
within-subject experiments make this computable.

---

## 4. Protocol

### 4.1 Splits

| Split | Held out | Reports on | Primary? |
|---|---|---|---|
| **S1 person** | 20% of pids | C1 denoising | secondary |
| **S2 block** | 20% of pids, *and* whole question blocks removed from their persona and from training targets | C2 generalisation to unseen item families | **primary for the full build** |
| **S3 cell** | stratified by demographic cell | fairness | blocking |

S2 is primary for the full build because S1 cannot distinguish a behaviour model from a
lookup table — 0 of 126 wave-4 columns are novel, so on S1 copying is always available. The
dataset ships no such split, and this submission does **not** implement it end to end.

For the full build, splits should be seeded, serialised to `data/derived/splits.json`, and
committed. Model selection should use S2; S1 should be reported but never optimised
against. The bonus POC uses a simpler participant split plus target-item removal from each
prompt, not S2.

### 4.2 Uncertainty

Percentile bootstrap, 1,000 resamples, **resampling participants, not (person, question)
pairs** — answers within a person are correlated and pair-level resampling understates every
interval. Every delta between systems carries a CI. Differences below ~0.01 accuracy are
inside the noise of a 2,058-person sample and are not claims.

### 4.3 Decoding

Sample $n$=20 completions per item at T≈1.0. Report mode for point metrics and the
empirical distribution for §3. Greedy decoding is not an acceptable configuration for any
distributional claim, and is the mechanical cause of the 0.48 variance ratio.

---

## 5. Leakage controls

Five vectors, four verified in `reports/exploration/02_leakage_audit.md`:

1. **`full_persona` contains wave-4 ground truth.** Verified: 100.000% of wave-4 questions
   carry the exact wave-4 answer across 18,900 audited pairs. Banned at the loader.
2. **The dataset README's own usage snippet** passes `wave4_Q_wave4_A` as both prompt input
   and ground truth. Stripping happens in the loader, never in caller code.
3. **`wave4_Q_wave1_3_A` is a baseline, not an input.** Feeding it silently reconstructs
   copy-forward and reports it as a model.
4. **Split granularity.** Split on pid, never on (person, question). Add the S2 block split
   or the model is scored on items it has memorised the marginal of.
5. **Pretraining contamination.** Public on HF since 2025, ships GPT-4.1 and Gemini outputs.
   Cannot be ruled out from inside the benchmark. Mitigation: report open-weight models with
   known cutoffs alongside, and mark frontier-model prompting numbers as
   upper-bounds-of-unknown-validity rather than clean measurements.

**Enforcement is a test, not a convention, but the current test covers only the POC.**
`tests/test_no_leakage.py` checks that `full_persona` is not imported outside the audit,
that the held-out POC prompt does not include the target item, that the copy-forward field
stores the wave1-3 answer, that the supervised label is the wave-4 answer, that participant
splits do not overlap, and that population mode is fit on train participants. It does not
yet string-match every possible wave-4 answer across every future prompt template, and it
does not cover an unimplemented S2 pipeline. A production build should add those checks.

**Tripwire:** any result above ~0.75 accuracy on the repeated-item set should be treated as
a suspected leak until proven otherwise. Human beings only agree with themselves 0.612 of
the time.

---

## 6. Full-build acceptance criteria

Explicit, pre-registered criteria for the full LBM build, with the comparator attached.
The bonus POC is not expected to satisfy these thresholds.

### Must pass (ship blockers)

| # | Criterion | Threshold |
|---|---|---|
| A1 | Beats B1 population mode on S2 | NS > 0.25, CI excludes 0 |
| A2 | Variance ratio | 0.85 – 1.15 on S2 |
| A3 | No leakage | `tests/test_no_leakage.py` green; no accuracy > 0.75 on repeated items |
| A4 | Fairness | no demographic cell with n ≥ 100 scores more than 0.05 below the pooled mean |
| A5 | Calibration | predicted-vs-observed frequency within ±0.05 in every decile |

### Target (the actual goal)

| # | Criterion | Threshold |
|---|---|---|
| T1 | **Beats copy-forward on S1** | NS > 1.0, CI excludes 1.0 — i.e. demonstrates denoising |
| T2 | Beats the best shipped LLM (0.488) on the same 126 columns | CI excludes 0 |
| T3 | Treatment-effect recovery | ≥ 80% of between-subject experiments' CIs overlap |
| T4 | Marginal TV distance | median < 0.10 |

### Falsification

The plan in `reports/02_modeling_plan.md` §7 predicts that classical factorisation/IRT beats
the LLM on C1. **That prediction is falsified if** a Tier-1 or Tier-2 LLM exceeds the Tier-0
model's NS by more than 0.10 with a CI excluding 0 on S2. If that happens, the hybrid
recommendation is wrong and the build should consolidate on the LM path. Stating this in
advance is what makes the recommendation a hypothesis rather than a preference.

---

## 7. What I did not verify

Honesty items, since the assignment asks for them:

- The treatment-effect recovery metric is specified but not implemented; I have confirmed
  the experiments exist in the catalog but have not computed a single effect size.
- The S2 block-held-out split is specified as the right primary evaluation for a full LBM,
  but this repository does not include `data/derived/splits.json` or an S2 dataset builder.
  The POC `held_out` arm is weaker: it removes the target item from a person's prompt, but
  the same target item can appear in training examples for other people.
- The US population benchmarks in `reports/exploration/05_representativeness.md` are approximate ACS/CPS figures entered by
  hand, adequate for showing direction and rough magnitude of skew, not for reweighting.
- The POC (deliverable 6) tests the loop, not the plan. It is far too small to say anything
  about whether Tier 2 would work at 7B.
