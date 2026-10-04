# Report 3 - Evaluation Strategy

Task: given a person's wave 1-3 persona, predict that same person's wave-4 answers. A good model must beat trivial baselines, avoid leakage, work across question types, and preserve population-level behaviour.

## 1. Dataset Features That Drive Evaluation

Wave 4 has **126 target columns** for the same 2,058 participants. All 126 targets appeared somewhere in waves 1-3, so the benchmark is mainly a repeated-item prediction task. That makes copy-forward an unusually strong baseline: use the person's earlier answer to the same item and score it against wave 4.

The dataset also mixes several response types:

| Example family | What the model predicts | Evaluation implication |
|---|---|---|
| binary choice | support/oppose, yes/no, option A/B | accuracy alone is fragile; use kappa and option-order robustness |
| short ordinal | Likert-style 1-5 or 1-7 answers | distance matters; use MAE and within-1 accuracy |
| continuous estimate | percentage estimates such as “what percentage of the public supports...” | exact match is too strict; use normalised MAE and correlation |
| product pricing | willingness-to-pay / product preference items | personalization signal is strong; compare against copy-forward and population mode |
| between-subject arms | only some participants saw each condition | NaN is often experimental design, not missingness; score per column on valid pairs |

The main leakage trap is `full_persona`: it contains wave-4 answers for repeated items. Legal model inputs must come from `wave_split`.

## 2. Metrics by Question Type

Metrics are chosen by observed answer support, not only by Qualtrics type.

| Question support | Columns | Primary metric | Secondary metrics |
|---|---:|---|---|
| binary | 65 | Cohen's kappa | accuracy, option-order robustness |
| ordinal short, <=11 values | 43 | MAE | within-1 accuracy, kappa |
| ordinal long | 2 | MAE | Spearman correlation |
| continuous | 14 | normalised MAE | Pearson correlation |
| numeric text entry | 2 | normalised MAE | exact agreement as diagnostic only |

Aggregation: compute each metric per column on valid pairs, then average columns with equal weight. Do not pool all person-question pairs, because the 40-column pricing block would dominate the score.

Uncertainty: use participant-level bootstrap with 1,000 resamples. Do not bootstrap individual person-question pairs; answers from the same person are correlated.

## 3. Binary Option-Order Robustness

Binary-choice prompts need an additional robustness test. The model may learn a position bias, such as preferring the first option, rather than using the persona.

For every binary item:

1. Score the original option order.
2. Randomly swap the two answer options.
3. Remap the swapped prediction back to the canonical answer label.
4. Compare original-order and swapped-order metrics.

Report:

| Robustness metric | Meaning |
|---|---|
| original accuracy / kappa | normal binary score |
| shuffled accuracy / kappa | score after option order is swapped and remapped |
| remapped consistency | share of examples where original and shuffled predictions match after remapping |
| position-flip rate | share of examples where the model follows position instead of semantic label |

Acceptance: shuffled-order kappa should remain within 0.02 of original-order kappa, and remapped consistency should be at least 0.95 for deterministic decoding. A larger gap means the model is using prompt-position bias.

## 4. Comparator Ladder

Every model is compared against this ladder:

| ID | Comparator | Measured value | Purpose |
|---|---|---:|---|
| B1 | population mode / mean per item | 0.471 | no-personalisation baseline |
| B2 | demographic-cell mode | 0.467 | demographic-only baseline |
| B3 | kNN on response vector | 0.486 | cheap similarity baseline |
| B4 | copy-forward from the person's earlier answer | 0.612 | strongest repeated-item baseline |
| C | human test-retest | 0.612 | human reliability reference |
| LLM | best shipped LLM, JSON Persona GPT-4.1 | 0.488 | published model reference |

Copy-forward and human test-retest are the same calculation here: `P(wave4 answer == same person's earlier answer)`. A model below population mode has not learned useful personalization. A model above copy-forward may be denoising, but only if leakage checks pass.

Normalised score:

```text
NS = (model - population_mode) / (copy_forward - population_mode)
```

NS = 0 means no personalization. NS = 1 means copy-forward level. NS > 1 suggests denoising.

## 5. Distribution Metrics

A Large Behavior Model should preserve population behaviour, not only individual point accuracy.

| Metric | Definition | Acceptance target |
|---|---|---|
| variance ratio | median over columns of var(predicted) / var(human) | 0.85-1.15 |
| marginal TV distance | per-item distance between predicted and human answer distributions | median < 0.10 |
| calibration | predicted vs observed frequency by confidence decile | within +/-0.05 |
| treatment-effect recovery | simulated experiment CI overlaps human experiment CI | >=80% of experiments |

Treatment-effect recovery is a full-build metric. The current prototype scorer focuses on point accuracy and distribution summaries.

## 6. Train/Test Protocol

| Split | Held out | Used for |
|---|---|---|
| S1 person split | 20% of participants | repeated-item denoising on the assignment target |
| S2 block-held-out split | held-out participants plus whole question blocks removed from persona and training targets | generalisation to unseen item families |
| S3 demographic cells | age x sex x education slices | fairness and failure analysis |

Protocol:

1. Split by participant id, never by person-question pair.
2. Use S2 for model selection and checkpoint choice.
3. Report S1 because it matches the assignment's wave-4 target.
4. Report S3 slices for any cell with n >= 100.
5. Serialize fixed splits to `data/derived/splits.json` for the full build.

Current prototype scope: the prototype uses a weaker held-out prompt construction. The target item is removed from the test person's prompt, but the same target item still appears in training examples for other people. The full build should implement the S2 dataset builder and write fixed splits to `data/derived/splits.json`.

## 7. Leakage Controls

Required controls:

1. Ban `full_persona` from all modelling loaders.
2. Use `wave_split` as the legal source of persona inputs.
3. Strip all `Answers` fields from `wave4_Q_wave4_A` before showing wave-4 questions to a model.
4. Treat `wave4_Q_wave1_3_A` as copy-forward baseline only, not model input.
5. Split on participant id.
6. Add string-match leakage tests for production prompts.
7. Treat repeated-item accuracy above 0.75 as suspected leakage until audited.

Current prototype scope: `tests/test_no_leakage.py` checks the prototype path for `full_persona` use, target-item prompt leakage, participant overlap, population-mode fitting, and copy-forward label direction. The full build should extend this to string-match every production prompt template.

## 8. Acceptance Criteria

Full LBM acceptance criteria:

| Criterion | Threshold |
|---|---|
| beats population mode on S2 | NS > 0.25 and CI excludes 0 |
| beats kNN / low-rank baseline on S2 | CI for delta excludes 0 |
| binary option-order robustness | shuffled kappa within 0.02 of original; remapped consistency >=0.95 |
| variance ratio | 0.85-1.15 |
| marginal TV distance | median < 0.10 |
| fairness | no n >= 100 demographic cell more than 0.05 below pooled score |
| leakage | tests pass; no unexplained repeated-item accuracy above 0.75 |
| S1 denoising target | NS > 1.0 against copy-forward, CI excludes 1.0 |

The prototype is not expected to satisfy these thresholds. It checks the data pipeline, training loop, and scoring code end to end. S2 block-held-out evaluation, treatment-effect recovery, binary option-order shuffling, and full prompt string-match leakage tests are full-build evaluation requirements, not claims made by the current prototype.
