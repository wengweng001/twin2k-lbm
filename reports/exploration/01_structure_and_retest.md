# 01 — Evaluation target, human test-retest ceiling, and trivial baselines

Generated from `wave1_3_response.csv` (2058x760) and `wave4_response.csv` (2058x126).

## 1. The evaluation target is 100% repeated items

- wave 1-3 columns: **760**
- wave 4 columns: **126**
- wave-4 columns also present in waves 1-3: **126**
- wave-4 columns that are new: **0**

Every wave-4 question was already asked in waves 1-3. There is no novel-question
generalisation set. This is structural, not incidental, and it determines the whole
evaluation design: see section 3.

### Blocks covered by wave 4

| Block | columns |
|---|---:|
| Product Preferences - Pricing | 40 |
| Non-experimental heuristics and biases  | 20 |
| Probability matching vs. maximizing - Problem 1 | 10 |
| False consensus  | 10 |
| Probability matching vs. maximizing - Problem 2 | 6 |
| Linda-conjunction | 3 |
| Linda -no conjunction  | 3 |
| Anchoring - African countries high | 2 |
| Anchoring - African countries low  | 2 |
| Anchoring - redwood high | 2 |
| Anchoring - redwood low | 2 |
| Proportion dominance 1C | 1 |
| Proportion dominance 1A  | 1 |
| Proportion dominance 1B | 1 |
| Absolute vs. relative - calculator | 1 |
| Proportion dominance 2A | 1 |
| Proportion dominance 2C | 1 |
| Sunk cost - no | 1 |
| Sunk cost - yes | 1 |
| WTA/WTP Thaler - WTP noncertainty | 1 |
| WTA/WTP Thaler problem - WTA certainty | 1 |
| Proportion dominance 2B  | 1 |
| Myside Ford | 1 |
| Outcome bias - success  | 1 |
| Outcome bias - failure | 1 |
| Myside German | 1 |
| Absolute vs. relative - jacket | 1 |
| Less is More Gamble C  | 1 |
| Less is More Gamble B  | 1 |
| Less is More Gamble A  | 1 |
| Disease-loss | 1 |
| Disease - gain  | 1 |
| Base-rate 70 engineers | 1 |
| Base-rate 30 engineers  | 1 |
| Allais Form 2 | 1 |
| Allais Form 1 | 1 |
| WTA/WTP Thaler problem - WTP certainty | 1 |

### Coverage is sparse by design (between-subject experiments)

- mean per-column response rate in wave 4: **76.2%**
- median: 100.0%, min: 31.6%, max: 100.0%
- mean answered columns per participant: **96.0** of 126

NaN here is mostly *not* missing data — each participant saw one arm of each
between-subject experiment. Any metric must be computed on per-column valid pairs,
and a naive `dropna()` across the frame would empty it.

## 2. Human test-retest reliability (the ceiling)

### scale = `binary` (65 columns, median n=2058)

- exact agreement: **0.809** (median 0.836, range 0.660-0.938)
- within-1 agreement: 1.000
- normalised MAE: 0.191
- test-retest correlation: 0.535
- Cohen's kappa: **0.534** (chance-corrected; this is the honest number)

### scale = `continuous` (14 columns, median n=2058)

- exact agreement: **0.136** (median 0.090, range 0.077-0.282)
- within-1 agreement: 0.175
- normalised MAE: 0.137
- test-retest correlation: 0.463

### scale = `ordinal_long` (2 columns, median n=1029)

- exact agreement: **0.328** (median 0.328, range 0.280-0.376)
- within-1 agreement: 0.384
- normalised MAE: 0.212
- test-retest correlation: 0.379
- Cohen's kappa: **0.211** (chance-corrected; this is the honest number)

### scale = `ordinal_short` (43 columns, median n=1043)

- exact agreement: **0.501** (median 0.489, range 0.360-0.674)
- within-1 agreement: 0.837
- normalised MAE: 0.149
- test-retest correlation: 0.572
- Cohen's kappa: **0.326** (chance-corrected; this is the honest number)

### scale = `text` (2 columns, median n=1029)

- exact agreement: **0.200** (median 0.200, range 0.180-0.219)
- within-1 agreement: 0.215
- normalised MAE: 0.025
- test-retest correlation: 0.476

### Overall

- mean exact agreement across all 126 evaluated columns: **0.612** (95% CI 0.608-0.615)
- mean chance-corrected kappa: **0.447** — defined on 110 of 126 columns; kappa is undefined for the 16 continuous-support columns

The gap between those two numbers is the point. Raw agreement flatters every model
on skewed items; kappa does not.

CIs throughout are percentile bootstrap, 1000 resamples of **participants**
(not of person-question pairs — answers within a person are correlated, and
resampling pairs would understate every interval).

## 2b. Question types and how each is scored

Qualtrics `QuestionType` is not sufficient to pick a metric — the same `Matrix`
type covers 2-point and 100-point items. Each column is therefore also classified
by its *observed support*, and the metric follows from that.

| Qualtrics type | observed scale | cols | metric used | why |
|---|---|---:|---|---|
| MC | `binary` | 49 | accuracy + **Cohen's kappa** | many items are 80/20 skewed; raw accuracy rewards constant predictors |
| MC | `ordinal_short` | 19 | MAE + within-1 accuracy + kappa | distance matters; off-by-one is not the same error as off-by-four |
| Matrix | `binary` | 16 | accuracy + **Cohen's kappa** | many items are 80/20 skewed; raw accuracy rewards constant predictors |
| Matrix | `ordinal_short` | 24 | MAE + within-1 accuracy + kappa | distance matters; off-by-one is not the same error as off-by-four |
| Slider | `continuous` | 12 | normalised MAE + Pearson | exact match is meaningless (retest exact agreement is 0.14) |
| TE | `continuous` | 2 | normalised MAE + Pearson | exact match is meaningless (retest exact agreement is 0.14) |
| TE | `ordinal_long` | 2 | MAE + Spearman | exact match is too strict to be informative |
| TE | `text` | 2 | normalised MAE on the numeric field | these are numeric text-entry, not free prose |

Aggregation rule: compute each metric **per column on its valid pairs**, then
average over columns with equal weight. Pooling raw pairs would let the 40-column
pricing block dominate the 1-column experiments, and would silently reweight the
benchmark toward whichever block has the most participants.

## 3. The ceiling and the trivial baseline are the same number

Test-retest reliability is `P(wave4 answer == the same person's earlier answer)`.
Copy-forward prediction is `predict wave4 with the same person's earlier answer`.
These are one computation. So on this dataset:

- copy-forward baseline accuracy = **0.612**
- human test-retest ceiling      = **0.612**

A model that 'reaches the human ceiling' has done exactly as well as pasting the
person's old answer. The headroom above copy-forward is not information about the
person - the persona already contains the answer - it is *denoising*: predicting the
person's stable disposition instead of one noisy draw of it. Any model can in
principle exceed the so-called ceiling on point-prediction metrics, because a human
re-test draw carries measurement noise that a conditional-mode predictor does not.
Calling test-retest an upper bound is therefore wrong; it is a *reference line*.

## 4. Baseline ladder

| Baseline | mean exact acc | 95% CI | mean normalised MAE |
|---|---:|---:|---:|
| Population mode / mean (no personalisation) | 0.471 | 0.468-0.475 | 0.321 |
| Copy-forward (= test-retest ceiling) | 0.612 | 0.608-0.615 | 0.169 |

The entire personalisation signal available on this benchmark is the **0.140**
absolute accuracy gap between those two rows (95% CI 0.136-0.144). A model scoring inside that band has learned
something about individuals; a model below the population mode has learned nothing;
a model above copy-forward is denoising.

The band is narrow but not fragile — its interval excludes zero comfortably, so the
personalisation signal is real. What it is not is *large*, and any reported gain
should be quoted against a band of this width rather than against 0-100%.

### Columns where personalisation matters most / least

**Largest gap (worth modelling)**

| col | block | retest | pop-mode | gap | question |
|---|---|---:|---:|---:|---|
| QID287_11 | False consensus  | 0.613 | 0.255 | +0.358 | Would you support or oppose... |
| QID9_3 | Product Preferences - Pricing | 0.848 | 0.543 | +0.305 | Please consider the following product category: Cat Food - Wet Type. S |
| QID9_6 | Product Preferences - Pricing | 0.852 | 0.551 | +0.302 | Please consider the following product category: Snacks - Potato Chips. |
| QID9_10 | Product Preferences - Pricing | 0.846 | 0.549 | +0.297 | Please consider the following product category: Lunchmeat - Sliced - R |
| QID9_21 | Product Preferences - Pricing | 0.841 | 0.544 | +0.297 | Please consider the following product category: Yogurt - Refrigerated. |
| QID9_11 | Product Preferences - Pricing | 0.853 | 0.558 | +0.295 | Please consider the following product category: Refrigerated Entrees.  |
| QID9_19 | Product Preferences - Pricing | 0.844 | 0.550 | +0.293 | Please consider the following product category: Ground and Whole Bean  |
| QID9_24 | Product Preferences - Pricing | 0.844 | 0.551 | +0.293 | Please consider the following product category: Paper Towels. Suppose  |

**Smallest / negative gap**

| col | block | retest | pop-mode | gap | question |
|---|---|---:|---:|---:|---|
| QID156 | Base-rate 70 engineers | 0.282 | 0.410 | -0.128 | A panel of psychologist have interviewed and administered personality  |
| QID154 | Base-rate 30 engineers  | 0.274 | 0.353 | -0.079 | A panel of psychologist have interviewed and administered personality  |
| QID198_4 | Probability matching vs. maximizing - Problem 1 | 0.727 | 0.793 | -0.066 | Consider the following hypothetical situation: A deck with 10 cards is |
| QID203_4 | Probability matching vs. maximizing - Problem 2 | 0.746 | 0.810 | -0.064 | Consider the following situation: A die with 4 red faces and 2 green f |
| QID198_5 | Probability matching vs. maximizing - Problem 1 | 0.738 | 0.795 | -0.056 | Consider the following hypothetical situation: A deck with 10 cards is |
| QID198_1 | Probability matching vs. maximizing - Problem 1 | 0.873 | 0.922 | -0.048 | Consider the following hypothetical situation: A deck with 10 cards is |
| QID203_2 | Probability matching vs. maximizing - Problem 2 | 0.821 | 0.866 | -0.046 | Consider the following situation: A die with 4 red faces and 2 green f |
| QID198_2 | Probability matching vs. maximizing - Problem 1 | 0.827 | 0.870 | -0.044 | Consider the following hypothetical situation: A deck with 10 cards is |

## 5. Worst-reliability items

These cap what any model can be scored on. Chasing accuracy here is chasing noise.

| col | block | retest exact | retest corr | n | question |
|---|---|---:|---:|---:|---|
| QID290_5 | Non-experimental heuristics and biases  | 0.077 | 0.450 | 2058 | What percentage of the public do you think supports the following poli |
| QID290_6 | Non-experimental heuristics and biases  | 0.082 | 0.462 | 2058 | What percentage of the public do you think supports the following poli |
| QID290_3 | Non-experimental heuristics and biases  | 0.082 | 0.583 | 2058 | What percentage of the public do you think supports the following poli |
| QID290_4 | Non-experimental heuristics and biases  | 0.083 | 0.549 | 2058 | What percentage of the public do you think supports the following poli |
| QID290_2 | Non-experimental heuristics and biases  | 0.084 | 0.505 | 2058 | What percentage of the public do you think supports the following poli |
| QID290_12 | Non-experimental heuristics and biases  | 0.084 | 0.522 | 2058 | What percentage of the public do you think supports the following poli |
| QID290_10 | Non-experimental heuristics and biases  | 0.087 | 0.545 | 2058 | What percentage of the public do you think supports the following poli |
| QID290_11 | Non-experimental heuristics and biases  | 0.093 | 0.530 | 2058 | What percentage of the public do you think supports the following poli |
| QID290_7 | Non-experimental heuristics and biases  | 0.098 | 0.548 | 2058 | What percentage of the public do you think supports the following poli |
| QID290_1 | Non-experimental heuristics and biases  | 0.113 | 0.573 | 2058 | What percentage of the public do you think supports the following poli |
