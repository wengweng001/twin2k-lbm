# Report 6 - Prototype

This prototype demonstrates the full loop: build a leakage-checked training slice, fine-tune
a lightweight model, evaluate it against baselines, and report failure modes. It is not the
proposed Large Behavior Model; the full build is described in `reports/02_modeling_plan.md`.

## 1. Purpose

The prototype is included to verify the mechanics of the project:

- construct legal persona prompts from `wave_split`;
- split by participant, not by person-question pair;
- prevent the wave-4 answer from entering the prompt;
- train a model end to end;
- compare against population mode, copy-forward, and the untuned base model;
- report distribution collapse as well as accuracy.

## 2. Design

| Choice | Reason |
|---|---|
| model: `HuggingFaceTB/SmolLM2-135M` | small enough to run on a free Colab T4; sufficient to validate the pipeline |
| target slice: binary and short-ordinal columns with <=9 options | answer can be represented as a single digit token |
| train size: 10,000 examples | keeps the run lightweight while exercising the full training code |
| test set: full held-out test slice | evaluates 33,329 test pairs over 411 held-out participants |
| arm: `held_out` | removes the target item and near-duplicates from the test person's prompt |
| loss | prompt tokens masked; loss applied only to the answer token |
| decoding | logits restricted to each item's valid answer digits |

The `held_out` arm is a **no-copy repeated-item prototype**. It is stricter than showing the
same earlier answer in the prompt, but it is not the full S2 block-held-out split: target
items still appear in training examples for other participants.

## 3. Files

| File | Role |
|---|---|
| `poc/build_dataset.py` | builds `same_item` and `held_out` JSONL train/test files |
| `poc/train.py` | fine-tunes SmolLM2-135M on the selected arm |
| `poc/evaluate.py` | scores baselines, base model, and fine-tuned model |
| `poc/results_held_out.json` | reported Colab T4 result |
| `tests/test_no_leakage.py` | leakage and split guardrails for the prototype path |

## 4. How to Run

```bash
.venv/bin/python poc/build_dataset.py
.venv/bin/python tests/test_no_leakage.py
.venv/bin/python -u poc/train.py --arm held_out --n-train 10000 --batch-size 16 --device cuda
.venv/bin/python -u poc/evaluate.py --arm held_out --n-test 0 --batch-size 32 --device cuda
```

The reported result was run on a Colab T4. CPU execution is possible but slow.

## 5. Reported Result

Configuration:

| Setting | Value |
|---|---:|
| model | SmolLM2-135M |
| train examples | 10,000 |
| test pairs | 33,329 |
| held-out test participants | 411 |
| epochs | 1 |
| target columns | 105 binary / short-ordinal columns |

| System | accuracy | var ratio | collapsed |
|---|---:|---:|---:|
| B1 population mode | 0.5397 | 0.00 | 100.0% |
| B4 copy-forward (= retest line) | 0.6936 | 1.00 | 0.0% |
| base model, untuned | 0.3633 | 0.00 | 96.2% |
| fine-tuned | 0.4854 | 0.26 | 19.0% |

Bootstrap CIs over participants, 400 resamples:

| Comparison | 95% CI |
|---|---:|
| fine-tuned vs base | [+0.111, +0.132] |
| fine-tuned vs B1 population mode | [-0.061, -0.048] |
| fine-tuned vs B4 copy-forward | [-0.216, -0.200] |

Interpretation: fine-tuning improves substantially over the untuned base model, so the
training loop is working. It still loses to population mode and copy-forward, so it has not
learned useful personalization. The variance ratio of 0.26 shows that distributional
fidelity remains poor even after fine-tuning.

## 6. Validation Checks

The prototype is checked by `tests/test_no_leakage.py`:

| Check | Purpose |
|---|---|
| `full_persona` not used outside audit scripts | prevents direct wave-4 answer leakage |
| held-out prompt excludes target item | prevents copying the target from the same person's persona |
| copy-forward field equals wave 1-3 answer | keeps baseline separate from wave-4 label |
| supervised label equals wave-4 answer | confirms the model trains on the intended target |
| train/test split by participant | prevents same person appearing on both sides |
| population mode fitted on train participants only | prevents test-set marginal leakage |

## 7. Scope

This prototype is evidence that the data, training, evaluation, and leakage checks run end
to end. It is not evidence that a 135M model is a competitive LBM. It also does not
implement S2 block-held-out evaluation, treatment-effect recovery, binary option-order
robustness, or full prompt string-match leakage checks; those are full-build evaluation
requirements in `reports/03_evaluation_strategy.md`.
