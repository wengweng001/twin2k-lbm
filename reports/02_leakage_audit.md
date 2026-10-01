# 02 — Leakage audit

`full_persona`: 2058 rows. `wave_split`: 2058 rows.

## Trap 1 — `full_persona` contains the wave-4 ground truth

The dataset README states it plainly, in a line easy to skim past:

> `persona_text`: Complete survey responses in text format, including all questions
> and answers. **For questions that appear in both waves 1-3 and wave 4, the wave 4
> responses are used.**

`full_persona` is the first config listed and the one whose name implies 'everything
we know about this person'. It is the natural pick, but it is not a valid input for this task.

Audit over 300 randomly sampled participants, comparing each wave-4
QuestionID's answer payload in `full_persona.persona_json` against both possible
sources.

| `full_persona` answer matches... | share of wave-4 questions |
|---|---:|
| the **wave-4** answer (the eval target) | **100.0%** |
| the wave-1-3 answer | 68.2% |

(18,900 question-participant pairs compared.)

Per-participant, a median of **100.0%** of wave-4
questions are already answered — correctly — inside the persona.

Anyone prompting with `full_persona` and scoring on wave 4 is measuring string
retrieval. Expected accuracy approaches 100%, versus a
61% human test-retest reference. A result above the human ceiling is the
tell-tale that this trap has been stepped in.

## Trap 2 — the README's usage example leaks

```python
test_questions = wave_split["data"]["wave4_Q_wave4_A"]  # you want to remove the "Answers"
ground_truth   = wave_split["data"]["wave4_Q_wave4_A"]
```
Prompt input and ground truth are the *same field*. The fix is demoted to a trailing
comment. `wave4_Q_wave4_A` must be stripped of every `Answers` key before it is
shown to a model; `wave4_Q_wave1_3_A` is the copy-forward baseline, not an input.

## Trap 3 — even the clean persona contains the same items

`wave1_3_persona_text` legitimately includes the earlier administration of every
wave-4 question, because that is what makes copy-forward possible. That is not a bug,
but it means:

- the task is **not** 'generalise to unseen questions' — 0 of 126 wave-4 columns are new;
- a model can score well by copying, so *beating copy-forward* is the only meaningful
  success criterion;
- to test genuine generalisation you must hold out question *blocks* from the persona
  yourself. The dataset does not provide that split.

## Trap 4 — split granularity and contamination

- The natural fine-tuning unit is the (person, question) pair: ~197,580 of them.
  Splitting on pairs puts the same person in train and test. Split on **pid**.
- Splitting only on pid still lets the model memorise each item's population
  distribution, which is the no-personalisation baseline wearing a costume. Report a
  second split that holds out whole question blocks to separate the two abilities.
- The dataset has been public on HuggingFace since 2025 and ships with GPT-4.1 and
  Gemini-Flash-2.5 outputs. Any frontier model may have memorised it; pretraining
  contamination is a live confound for prompting baselines and cannot be ruled out
  from inside the benchmark.

## Trap 5 — between-subject arms are not missing data

- 56 of 126 wave-4 columns are answered by <90% of
  participants, because each person saw one arm of each between-subject experiment.
- Imputing or dropping these rows silently changes the estimand. Metrics must be
  computed per column on valid pairs, then aggregated with explicit weights.
