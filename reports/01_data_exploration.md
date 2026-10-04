# Deliverable 1 — Short Data Exploration Report

This short report shows what I checked before designing the model. The code that produced
the numbers is in `src/`; the generated evidence tables are in `reports/exploration/`.

## Dataset Structure

Twin-2K-500 has 2,058 US adults and two Hugging Face subsets that matter for this task:

| Subset | What it contains | Use in this assignment |
|---|---|---|
| `wave_split` | wave 1-3 persona fields, wave-4 questions, wave-4 answers, and the wave 1-3 answer to repeated wave-4 items | **Use this.** It separates legal inputs, targets, and copy-forward baselines. |
| `full_persona` | complete persona text/json with repeated wave-4 items filled using the wave-4 answer | **Do not use as input.** It contains the evaluation target. |

The relevant CSV files are `wave1_3_response.csv` (2,058 x 760) and
`wave4_response.csv` (2,058 x 126). Wave 4 covers behavioural-economics blocks such as
product pricing, heuristics and biases, probability matching, anchoring, base-rate tasks,
and false consensus.

## Target Structure

The most important structural finding is that **all 126 wave-4 columns already appear in
waves 1-3**. There are **zero new wave-4 questions**. That means the shipped benchmark
mainly tests repeated-item prediction, not generalisation to unseen question families.

Coverage is block-dependent: participants answered a mean of **96.0 of 126** wave-4
columns, and mean per-column response rate is **76.2%**. This is mostly experimental-arm
structure, not ordinary missingness, so metrics should be computed per column on valid
pairs and then averaged.

## Question Types and Scoring

Qualtrics question type alone is not enough to choose a metric, because the same type can
contain binary, ordinal, and continuous supports. I classified columns by observed support:

| Observed support | Columns | Scoring choice |
|---|---:|---|
| binary | 65 | accuracy plus Cohen's kappa |
| ordinal short | 43 | MAE, within-1 accuracy, kappa |
| ordinal long | 2 | MAE and rank/correlation-style checks |
| continuous | 14 | normalised MAE and Pearson correlation |
| numeric text entry | 2 | normalised MAE on the numeric value |

Exact-match accuracy is useful only with the right caveat: skewed binary items can make
constant predictors look strong. Kappa and MAE are needed to avoid over-crediting modal
answers.

## Human Test-Retest Reliability

The headline finding is that human test-retest agreement is **0.612 exact agreement**
across the 126 wave-4 columns, with 95% CI **0.608-0.615** from participant-level
bootstrap. Mean chance-corrected kappa is **0.447**, defined on 110 of 126 columns.

By support type:

| Support | Exact retest agreement | Kappa where defined |
|---|---:|---:|
| binary | 0.809 | 0.534 |
| ordinal short | 0.501 | 0.326 |
| ordinal long | 0.328 | 0.211 |
| continuous | 0.136 | n/a |
| numeric text entry | 0.200 | n/a |

Because every wave-4 item is repeated, the copy-forward baseline is numerically the same
computation as test-retest. Predicting a person's earlier answer gives **0.612**.
Population mode, which ignores the persona, gives **0.471**. So the observed
personalisation band is only **0.140 accuracy points**.

## Leakage and Biases

`full_persona` is the main leakage trap. In an audit of 18,900 question-person pairs,
`full_persona.persona_json` matched the wave-4 answer for **100.0%** of repeated wave-4
questions. Using it as a prompt input would evaluate target retrieval, not behaviour
prediction.

The sample is not a clean US-population proxy. It is well matched on sex and region, but
skewed on education and age:

| Variable | TV distance | Largest gap |
|---|---:|---|
| sex | 0.001 | Male 49.3% vs 49.4% |
| region | 0.028 | Midwest 18.2% vs 20.7% |
| age | 0.103 | 65+ 13.5% vs 22.0% |
| race | 0.111 | White 69.9% vs 58.7% |
| education | 0.235 | high school graduates 13.2% vs 27.5% |

The education skew matters because the wave-4 target is made of cognitive and
heuristics-and-biases tasks. A model trained here should be described as modelling this
online panel, not the US population.

## Code and Appendices

| Topic | Generated appendix | Code |
|---|---|---|
| target structure, scoring, retest | `reports/exploration/01_structure_and_retest.md` | `src/01_structure_and_retest.py` |
| leakage audit | `reports/exploration/02_leakage_audit.md` | `src/02_leakage_audit.py` |
| shipped LLM baselines | `reports/exploration/03_llm_baselines.md` | `src/03_llm_baselines.py` |
| persona length | `reports/exploration/04_persona_budget.md` | `src/04_persona_budget.py` |
| representativeness | `reports/exploration/05_representativeness.md` | `src/05_representativeness.py` |
| extra comparator baselines | `reports/exploration/06_comparator_baselines.md` | `src/07_comparator_baselines.py` |
