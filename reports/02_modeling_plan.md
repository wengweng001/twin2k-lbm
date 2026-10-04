# Report 2 - Modelling Plan

Goal: build a Large Behavior Model that takes a person's wave 1-3 answers as the persona
and predicts that same person's wave-4 answers.

## 1. What I Am Building

The system has two prediction modes, because the dataset supports two different jobs:

| Mode | Question type | Model used | Why |
|---|---|---|---|
| known item bank | one of the 126 repeated wave-4 columns | psychometric response model plus optional LLM reranker | the item already exists in wave 1-3, so a response-matrix model is the strongest baseline |
| new or edited item | question not previously asked | retrieval-augmented 7B/8B instruct model | this is where language understanding matters |

Proposed system: a **hybrid**.

1. A low-rank / IRT-style response model estimates the person's latent traits from the
   760 wave 1-3 response columns.
2. A retrieval component selects the most relevant prior answers for the target question.
3. A 7B/8B instruct model predicts an answer distribution from the retrieved evidence and
   target question.
4. A calibration layer adjusts the output distribution so simulated populations preserve
   human variance, not only point accuracy.

The prototype uses SmolLM2-135M only to exercise the pipeline; it is not the proposed LBM.

## 2. Model Choice

### Default model for the real LBM

Default model: **Qwen2.5-7B-Instruct**.

Selection reasons:

| Requirement | Why it matters here | Qwen2.5-7B-Instruct fit |
|---|---|---|
| open weights | reproducibility and lower contamination ambiguity than frontier APIs | yes |
| 7B/8B scale | large enough for instruction following and survey reasoning, trainable with QLoRA | yes |
| long enough context | retrieved persona plus question should fit in 1k-2k tokens | yes |
| good structured output | answers must be constrained to valid options | yes |
| affordable training | fits on a single A100/L4 class GPU with 4-bit QLoRA | yes |

Backup model: **Llama-3.1-8B-Instruct**. It is technically strong, but licensing and
deployment constraints can be more project-specific. GPT-4.1 is useful as a reference
baseline, but not as the main build target: the dataset already ships GPT-4.1 predictions,
the dataset is public, and a closed frontier model is harder to audit for contamination or
reproduce.

### Models not chosen first

| Model family | Why not first |
|---|---|
| SmolLM2-135M | useful for the prototype only; too small to be the proposed LBM |
| full GPT-4.1 prompting | strong reference baseline, but not reproducible enough for the main build |
| full fine-tuning of a 7B model | too much capacity for 2,058 people; LoRA is the right starting point |
| DPO/RLHF | no preference data exists; it would mostly sharpen already noisy hard labels |

## 3. Data Construction

Only `wave_split` is used for modelling. `full_persona` is banned because it contains
wave-4 answers for repeated items.

For each person and target question, construct:

```text
person_id
target_question_id
target_question_text
valid_answer_space
retrieved_persona_evidence
copy_forward_answer, if the same item appears in wave 1-3
wave4_answer_label
question_block
demographic_cell
```

The main training table has two sources:

| Source | Approximate size | Use |
|---|---:|---|
| wave-4 targets | about 198k valid person-question pairs | direct benchmark task |
| mined wave 1-3 leave-one-block targets | up to about 1.5M pairs | trains generalisation beyond the 126 wave-4 columns |

For the mined wave 1-3 examples, hide one block of a person's earlier answers and predict
those answers from the remaining persona. This creates the missing S2-style training
signal: answer questions that were not shown in that person's prompt.

## 4. Splits

Create and commit `data/derived/splits.json` with three fixed splits:

| Split | Held out | Purpose |
|---|---|---|
| S1 person split | 20% of participants | repeated-item denoising |
| S2 block-held-out split | held-out participants plus held-out question blocks removed from persona and training targets | genuine question-family generalisation |
| S3 demographic cells | age x sex x education slices | fairness and failure analysis |

S2 is the model-selection split. S1 is reported, but not used to choose checkpoints,
because S1 rewards copying repeated items. The current repo does not yet implement S2 end
to end; this remains a limitation.

## 5. Technical Route

### Stage 0 - response-matrix baselines

Before training an LM, fit models that use the response matrix directly:

1. Population mode / mean per item.
2. Demographic-cell mode.
3. kNN over the person's response vector.
4. Low-rank matrix factorisation or probabilistic PCA.
5. Graded-response IRT by question block where the item supports it.

Output: calibrated answer distributions for each person-item pair.

This stage comes first because the measured kNN baseline already reaches 0.486,
essentially tied with the best shipped GPT-4.1 result at 0.488. If low-rank / IRT beats the
LLM, the known-item engine should be the response model, not the language model.

### Stage 1 - retrieval-augmented prompting

Use the frozen 7B/8B model before fine-tuning:

1. Retrieve the person's same-block answers.
2. Add nearest items by item-text embedding and response-correlation similarity.
3. Render evidence as a compact table: question, earlier answer, scale, block.
4. Ask the model to return a valid option id or numeric value.
5. Sample 20 completions at temperature around 1.0 to estimate a distribution.

Ablations:

| Prompt input | What it tests |
|---|---|
| question only | item marginal knowledge |
| demographics only | demographic shortcut |
| retrieved persona | intended method |
| full legal persona | whether retrieval loses useful signal |

If retrieved persona matches full legal persona within the bootstrap CI, retrieval wins on
cost and deployability.

### Stage 2 - QLoRA fine-tuning

Fine-tune Qwen2.5-7B-Instruct with QLoRA, not full fine-tuning.

Starting recipe:

| Setting | Value |
|---|---|
| quantisation | 4-bit NF4 |
| LoRA target modules | attention projections and MLP projections |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| dropout | 0.05 |
| max length | 1024 first, 2048 if retrieval needs it |
| effective batch size | 64 |
| epochs | 2, early stop on S2 |
| learning rate | 1e-4 cosine schedule |
| warmup | 3% |
| precision | bf16 where available |

Prompt template:

```text
You are predicting one survey answer for a respondent.

Prior answers:
{retrieved_persona_table}

Target question:
{question_text}

Valid answers:
{answer_options_or_numeric_range}

Return only the answer id.
```

Loss:

1. Mask the prompt tokens.
2. Train only on answer tokens or on a small classification head over valid answers.
3. Prefer soft labels when available:
   - repeated item with multiple observations: empirical response distribution;
   - otherwise: label smoothing from the item's test-retest confusion matrix;
   - continuous items: discretise for classification plus report normalised MAE.

Soft targets matter because human retest is only 0.612, so a single hard wave-4 answer is
a noisy draw. Training the model to emit one deterministic answer encourages the variance
collapse seen in the shipped LLM systems.

### Stage 3 - calibration and ensembling

After training, calibrate outputs on the validation split:

1. Fit temperature per question type.
2. For known repeated items, ensemble the LM distribution with the low-rank / IRT
   distribution.
3. Choose the ensemble weight on S2, not S1.
4. Reject any checkpoint with variance ratio outside 0.85-1.15 unless the use case is
   explicitly point prediction only.

The expected final form is:

```text
prediction = w * psychometric_distribution + (1 - w) * LLM_distribution
```

For fixed-bank questions, `w` may be high. For genuinely new questions, `w` should drop
because the psychometric model has no item parameters yet.

## 6. Model Selection

Select checkpoints with this ordered gate:

1. No leakage tests fail.
2. Beats population mode on S2 with participant-bootstrap CI excluding zero.
3. Beats kNN / low-rank baseline on S2, or else the LM is not adding value.
4. Variance ratio is between 0.85 and 1.15.
5. No demographic cell with n >= 100 is more than 0.05 below pooled accuracy.
6. Among models passing those gates, choose the cheapest and simplest.

For S1 repeated items, report copy-forward/test-retest as the reference line. A model below
population mode has failed. A model between population mode and copy-forward has learned
some personalisation. A model above copy-forward may be denoising, but only if leakage
checks are clean.

## 7. Compute Plan

Prototype:

| Task | Hardware |
|---|---|
| data build, baselines, kNN | laptop CPU |
| SmolLM2-135M prototype | Colab T4 |
| QLoRA 7B training | Colab A100, L4, or rented single GPU |
| full evaluation | CPU/GPU batch inference, can be parallelised by person |

The laptop should not be expected to train the real LBM. It is used for data checks,
baselines, report generation, and leakage tests. Colab or a rented GPU is the right place
for 7B QLoRA.

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `full_persona` leaks wave-4 answers | ban it in loaders and test for it |
| S1 rewards copying | choose checkpoints on S2 block-held-out split |
| model collapses to modal answers | soft labels, sampling, temperature calibration, variance gate |
| LLM adds no value over matrix factorisation | keep the psychometric model as the main known-item engine |
| sample is not representative | report demographic-cell metrics and avoid US-population claims |
| frontier model contamination | prefer open-weight models and treat shipped GPT/Gemini results as reference artifacts |

## 9. What I Would Build First

The first two implementation days should be:

1. Implement `splits.json`, especially S2 block-held-out.
2. Fit low-rank / IRT baselines and compare against kNN, population mode, and copy-forward.
3. Build the retrieval table for each target question.
4. Run frozen Qwen2.5-7B-Instruct prompting with retrieved persona.
5. Fine-tune QLoRA only if the frozen model beats cheap baselines or shows value on S2.

This is the decision point: if low-rank / IRT wins on known repeated items, the correct
technical route is hybrid, not "more LLM". The LLM earns its place only by improving S2
generalisation or by handling genuinely new question text.
