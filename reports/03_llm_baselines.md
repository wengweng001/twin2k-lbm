# 03 — Do the shipped LLM specifications beat the trivial baseline?

Each specification ships an aligned triple (LLM prediction / wave-4 truth /
wave-1-3 answer) over the same 126 wave-4 columns, so all three baselines are
computed on identical valid pairs. Exact-match accuracy, averaged over columns.

Important caveat: these are **re-scored shipped predictions**, not rerun API calls. I did
not regenerate GPT-4.1, GPT4.1-mini, or Gemini outputs from prompts. If a fresh API run
under pinned prompts/model versions differs from the shipped CSVs, the interpretation of
the comparison changes.

14 folders ship results; **13 are distinct systems**.
`GPT4.1-mini (default setup)` and `Text Persona - GPT4.1-mini` are byte-identical on every metric — the same run
published twice. Counting folders rather than runs would inflate the sample.

## Scoreboard

| Specification | LLM acc | copy-fwd | pop-mode | LLM − copy-fwd (95% CI) | band captured |
|---|---:|---:|---:|---:|---:|
| JSON Persona - GPT4.1 | **0.488** | 0.611 | 0.471 | -0.124 (-0.128, -0.120) | 12% |
| JSON Persona (Predicted Output) - GPT4.1 | **0.487** | 0.612 | 0.472 | -0.124 (-0.128, -0.121) | 11% |
| GPT4.1-mini (default setup) | **0.477** | 0.612 | 0.471 | -0.134 (-0.138, -0.131) | 4% |
| Text Persona - GPT4.1-mini | **0.477** | 0.612 | 0.471 | -0.134 (-0.138, -0.131) | 4% |
| Text Persona (Default Temperature) - GPT4.1-mini | **0.471** | 0.612 | 0.471 | -0.140 (-0.144, -0.136) | -0% |
| JSON Persona - GPT4.1-mini | **0.469** | 0.606 | 0.466 | -0.137 (-0.143, -0.131) | 2% |
| Text Persona (Repeating Questions) - GPT4.1-mini | **0.469** | 0.612 | 0.471 | -0.143 (-0.146, -0.139) | -2% |
| Text Persona (Reasoning) - GPT4.1-mini | **0.467** | 0.612 | 0.471 | -0.144 (-0.148, -0.140) | -3% |
| Text Persona - Gemini-Flash2.5 | **0.466** | 0.612 | 0.471 | -0.145 (-0.149, -0.142) | -4% |
| Persona Summary - GPT4.1-mini | **0.448** | 0.612 | 0.471 | -0.163 (-0.167, -0.159) | -16% |
| JSON Persona (Predicted Output) - GPT4.1-mini | **0.446** | 0.611 | 0.471 | -0.165 (-0.169, -0.160) | -18% |
| Demographics Only - GPT4.1-mini | **0.435** | 0.612 | 0.471 | -0.177 (-0.181, -0.173) | -26% |
| Persona Summary - JSON Persona - GPT4.1-mini | **0.419** | 0.612 | 0.471 | -0.193 (-0.197, -0.189) | -38% |
| LLM Finetuning (500 training samples) - GPT4.1-mini | **0.413** | 0.614 | 0.471 | -0.201 (-0.207, -0.196) | -41% |

CIs are percentile bootstrap over participants, 400 resamples.

## What this says

- Best specification: **JSON Persona - GPT4.1**, at 0.488 exact accuracy.
- Copy-forward on the same columns: 0.611.
- **Every one of the 13 distinct systems loses to copy-forward.** Best
  margin is -0.124, and its CI (-0.128, -0.120) excludes zero.
- **Only 4 of 13 beat the population mode at all**, and the best
  does so by +0.016 — 12% of the
  available band. 9 score below it, 5 of them by more than
  0.02. Fine-tuning is the worst, at -0.058 — materially worse than ignoring the persona.

The published approaches — including GPT-4.1 with a full JSON persona and a
fine-tuned GPT-4.1-mini — land between the no-personalisation floor and the
copy-forward line, and several fall through the floor. The band is ~14 accuracy
points wide in total and the best system captures about a tenth of it. Any
submission claiming a large win here should be read with suspicion: on this
benchmark the likeliest explanation for a high score is leakage, not skill.

And the ranking above is close to uninformative about distributional quality:
accuracy and variance fidelity turn out to be uncorrelated. See below.

## Variance ratio (the flattening check)

`var(LLM prediction) / var(human answer)`, median over columns. 1.0 means the
simulated population is as dispersed as the real one; below 1.0 means the model
is collapsing toward a consensus persona and erasing minority positions.

| Specification | variance ratio |
|---|---:|
| Text Persona - Gemini-Flash2.5 | 0.87 |
| JSON Persona - GPT4.1 | 0.60 ⚠ flattened |
| Text Persona (Default Temperature) - GPT4.1-mini | 0.60 ⚠ flattened |
| LLM Finetuning (500 training samples) - GPT4.1-mini | 0.56 ⚠ flattened |
| Persona Summary - JSON Persona - GPT4.1-mini | 0.54 ⚠ flattened |
| GPT4.1-mini (default setup) | 0.52 ⚠ flattened |
| Text Persona - GPT4.1-mini | 0.52 ⚠ flattened |
| Persona Summary - GPT4.1-mini | 0.48 ⚠ flattened |
| Text Persona (Repeating Questions) - GPT4.1-mini | 0.43 ⚠ flattened |
| Text Persona (Reasoning) - GPT4.1-mini | 0.41 ⚠ flattened |
| JSON Persona - GPT4.1-mini | 0.39 ⚠ flattened |
| JSON Persona (Predicted Output) - GPT4.1 | 0.24 ⚠ flattened |
| JSON Persona (Predicted Output) - GPT4.1-mini | 0.24 ⚠ flattened |
| Demographics Only - GPT4.1-mini | 0.24 ⚠ flattened |

Median across the 13 distinct systems: **0.48**,
range 0.24–0.87. Only **1 of 13** exceeds 0.85.

Correlation between accuracy and variance ratio: **r = +0.06** (n = 13) — effectively zero.

That null result is the important one. It is not that accuracy and fidelity trade
off in a predictable way; it is that **accuracy carries no information about
whether the simulated population is correctly dispersed**. A leaderboard ranked on
accuracy tells you nothing about the property that aggregate-level uses depend on.
Variance therefore has to be measured and gated separately, not inferred — which is
why `docs/03_evaluation.md` makes the variance ratio a ship blocker rather than a
secondary metric.

The mechanism is that predicting the modal answer is individually defensible and
collectively wrong: it costs little accuracy while erasing the distribution.
