# Deliverable 2 — Plan to build the behaviour model

Every design choice below follows from a measured fact in `reports/`. The numbers are
restated where they bind.

---

## 1. Problem framing

The assignment's natural task is the right starting point: use a person's wave 1-3 answers
as their persona and predict that same person's held-out wave-4 answers. The design choice
is what it means to do that credibly. Three measurements from the exploration determine the
model and evaluation plan:

| Measurement | Source | Consequence |
|---|---|---|
| 0 of 126 wave-4 columns are new questions | `reports/01` | The benchmark cannot test generalisation to unseen questions |
| copy-forward accuracy 0.612 = the test-retest "ceiling" | `reports/01` | The trivial baseline and the ceiling are one number |
| personalisation band 0.140, 95% CI [0.137, 0.144] | `reports/01` | Total available headroom is 14 accuracy points |
| all 13 shipped LLM systems score below copy-forward | `reports/03` | Published SOTA has not cleared the trivial baseline |

So the LBM objective is decomposed into three separable capabilities, each with its own
target:

- **C1 — Denoise.** Given the persona, predict the person's *stable disposition*, not the
  noisy single draw that copy-forward reproduces. Success = beating copy-forward on
  repeated items. This is the only capability the shipped benchmark measures.
- **C2 — Generalise.** Answer a question the person has never been asked, from their other
  answers. This is what "behaviour model" means commercially, and the dataset does **not**
  provide a split for it. We construct one (§3.2).
- **C3 — Preserve the distribution.** Reproduce population variance and inter-item
  correlation, not just per-person point accuracy. Median variance ratio across shipped
  systems is 0.48 (`reports/03`): simulated populations are half as dispersed as real ones.

C2 and C3 are where the product value is. C1 is where the benchmark is. Conflating them
is the central evaluation risk in this assignment.

---

## 2. What the target actually is

Formally, for person $i$ and item $q$, the human response is

$$y_{iq} = f(\theta_i, q) + \varepsilon_{iq}$$

where $\theta_i$ is a stable latent disposition and $\varepsilon$ is occasion noise.
Copy-forward predicts $f(\theta_i,q) + \varepsilon_{iq}^{(1)}$ — signal plus *an
independent draw of noise*. A model predicting $\mathbb{E}[y_{iq}]$ carries signal and no
noise, and therefore **can exceed the so-called ceiling** on point metrics. Test-retest is
a reference line, not an upper bound.

This also sets the loss: we want the conditional distribution $p(y_{iq}\mid\theta_i,q)$,
not an argmax. Training on hard labels with cross-entropy and decoding greedily is what
produces the 0.48 variance ratio — the model learns the mode and emits it every time.

---

## 3. Data pipeline

### 3.1 Sources and the hard rule

```
wave_split/            <- the ONLY legal source for personas
  wave1_3_persona_text    input
  wave1_3_persona_json    input (structured; preferred for retrieval)
  wave4_Q_wave1_3_A       copy-forward BASELINE — never an input
  wave4_Q_wave4_A         ground truth — strip every `Answers` key before use
full_persona/          <- BANNED. Contains wave-4 answers (100.0% verified, reports/02)
```

The POC enforces the subset it implements with assertions, not convention: tests check that
`full_persona` is not used, that the held-out prompt does not include the target item, that
copy-forward stores the wave1-3 answer rather than the wave-4 answer, and that participant
splits do not overlap. A production S2 pipeline should extend this to full string-level
checks across every prompt template. See `docs/03_evaluation.md` §5.

### 3.2 Splits

Three splits, because they answer different questions:

| Split | Unit held out | Tests | Notes |
|---|---|---|---|
| **S1 person** | 20% of pids | C1 denoising | Standard. Never split on (person, question) pairs — same person leaks. |
| **S2 block** | 20% of pids × whole question *blocks* removed from the persona and from model selection/training targets | C2 generalisation | The block's items are absent from the persona and held out as target families. This is the split the dataset does not ship and the one that matters. |
| **S3 cell** | stratified by demographic cell | fairness | Report per-cell, not pooled (`reports/05`). |

For the full build, fixed splits should be written to `data/derived/splits.json` and
committed, so every experiment scores on identical people and item blocks. This repo does
not yet include that S2 split file; the bonus POC implements only a person split plus
per-example target-item removal.

### 3.3 Example construction

One training example = (compressed persona, question, answer). ~2,058 people × 126
evaluable columns ≈ 259k pairs at the ceiling, minus between-subject arms the person never
saw — mean 96 answered columns per person, so ≈ 198k real pairs. For C2 we can additionally
mine the 760 wave-1-3 columns as targets, giving ~1.5M pairs. That is ample for LoRA.

Answers are normalised to a canonical form per question type (`reports/01` §2b): integer
index for MC, integer for Likert, float for continuous. The prompt states the valid range
explicitly so decoding can be constrained.

---

## 4. Handling the long persona

A legal persona is **27,484 tokens** median, measured with a real BPE tokenizer rather than
estimated (`reports/04`; the chars/4 rule of thumb understates it by 15%, because this text
is structured and repetitive rather than prose). Four options, with the evidence for each:

| Option | Cost | Evidence |
|---|---|---|
| Full persona in context | 27.5k tok/call; fits GPT-4.1 class, impossible for <0.5B | Best shipped system does this and still loses to copy-forward |
| `persona_summary` (3.2k tok) | 7x cheaper | **Scores worse** — 0.448 vs 0.477 (`reports/03`). Compression is not free. |
| **Retrieval by construct** | ~300–800 tok/question | Recommended. Not yet tried by the authors. |
| Learned trait vector | fixed d-dim, no text | Recommended for the fine-tuned path. |

**Recommended: retrieval.** For a target item, retrieve the person's answers to items
measuring the same construct — same `BlockName`, same psychological scale, plus a
k-nearest set by item-embedding similarity. The 126 wave-4 columns map to 85 QuestionIDs
and a handful of constructs, so a few hundred tokens of *relevant* persona beats 27.5k tokens
of mostly-irrelevant persona, and it is the only option that fits a <0.5B context window.

**Recommended for fine-tuning: a two-stage encoder.** Compress the 760-dim response vector
to a low-dimensional $\hat\theta_i$ (§5.1), and condition the LM on
`[trait_vector] + [question text]`. This sidesteps the context limit entirely and makes
the persona a dense prefix rather than 27.5k tokens of prose.

---

## 5. Modelling ladder

Ordered by cost. The cheap baselines are first-class candidates, not ceremonial checks:
13 LLM systems lose to a one-line baseline on the shipped repeated-item task, so the
burden of proof is on the expensive methods.

### 5.1 Tier 0 — classical, hours of work, likely to win

The response matrix is 2,058 × 760, dense-ish, mostly ordinal. This is a matrix-completion
problem and the field has solved it:

- **Low-rank factorisation / probabilistic PCA** on the z-scored response matrix. Predict
  held-out entries from $U_i V_q^\top$. Rank chosen by CV on S1.
- **Item response theory (IRT)**, graded-response model: one latent trait per construct,
  per-item discrimination and thresholds. This *is* the generative model psychometrics
  uses for exactly this data, and it produces calibrated per-item distributions for free —
  which directly addresses C3.
- **kNN on the response vector**: predict from the 50 most similar people's answers to
  that item.

Why this tier is likely to beat the LLMs: the signal being exploited is cross-person
correlation in a fixed item bank, which is precisely what factorisation is optimal for and
precisely what an LLM has to rediscover from text. A simple kNN response-vector baseline
already reaches **0.486** (`reports/06`), essentially tying the best shipped LLM at 0.488.
IRT / low-rank factorisation is the better version of that idea, and it also gives the
trait vector $\hat\theta_i$ that Tier 2 needs. If it captures more of the 14-point band,
that is the headline result and the LLM becomes the thing you use only for C2 on genuinely
novel questions.

### 5.2 Tier 1 — prompting with retrieval

Frozen instruct model. Per target item: retrieve same-construct answers, render as a
compact table, ask for an answer constrained to the valid options.

- Batch all items for one person into a single call and cache the persona prefix —
  otherwise a full sweep costs ~27.5k × 126 × 2,058 ≈ 7.1B input tokens.
- **Sample, don't argmax.** Draw $n$=20 completions at $T$≈1.0 and keep the empirical
  distribution. Report the mode for point accuracy *and* the distribution for C3. Greedy
  decoding is the direct cause of variance collapse.
- Ablate: no persona / demographics only / retrieved persona / full persona. The shipped
  "Demographics Only" run scores 0.435, so the ablation floor is already known.

### 5.3 Tier 2 — fine-tuning (the main build)

- **Base model.** Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct for the real system.
  The reported POC uses SmolLM2-135M only because it is a tiny, cheap prototype that fits
  a free Colab T4 run. It is a pipeline validation artifact, not the proposed LBM.
- **Method.** LoRA, r=16, α=32, dropout 0.05, on attention + MLP projections. Full
  fine-tuning is unjustified at this data scale and would erase the base model's
  instruction following.
- **Objective.** Cross-entropy on *answer tokens only* (mask the prompt).
  Critically: **use soft targets.** For items where the person answered the same question
  in multiple waves, the empirical distribution over their answers is the label. Where only
  one observation exists, label-smooth by the item's test-retest confusion matrix —
  i.e. teach the model how unstable this item is. This is the single most important
  departure from the shipped fine-tuning run, which scored 0.413, *below a constant
  predictor*, almost certainly by overfitting hard labels on 500 samples.
  The POC (deliverable 6) trains on hard labels deliberately. On the held-out arm it moves
  well above the untuned base, but still lands below the population-mode baseline
  (0.485 vs 0.540) and has a variance ratio of only 0.26. That is not proof that hard labels
  are the whole problem, but it is enough evidence to make soft targets the first thing I
  would change in the real build.
- **Head option.** For the fixed 126-item bank, a classification head per item over the
  valid options is cheaper and better-calibrated than free generation. Keep generation only
  for C2 / novel items.
- **Hyperparameters (starting point).** lr 1e-4 cosine, warmup 3%, effective batch 64,
  2 epochs, bf16, max_len 1024 with retrieval. Early stop on S2 (generalisation), not S1 —
  optimising S1 rewards copying.
- **Calibration stage.** After SFT, fit a temperature per question type on a held-out
  slice to match predicted entropy to human entropy. Report variance ratio before/after.

### 5.4 Tier 3 — preference tuning (only if justified)

DPO/GRPO needs a preference signal. There is no human preference data here; constructing
pairs from "correct vs incorrect answer" turns DPO into an awkward reimplementation of
cross-entropy, and sharpens the distribution — making C3 *worse*. **Recommendation: skip.**
If pursued, the reward must be a distributional score (e.g. negative CRPS against the
person's observed answer distribution), not accuracy. State this explicitly rather than
reaching for RLHF because it is fashionable.

---

## 6. Risks and mitigations

| Risk | Why it is live here | Mitigation |
|---|---|---|
| **Leakage via `full_persona`** | 100.0% of wave-4 answers present (`reports/02`) | Banned in the loader; automated string-match test on every built prompt |
| **Leakage via the README snippet** | Ships input == ground truth | Strip `Answers` in the loader, not in the caller |
| **Variance collapse** | Median ratio 0.48 across the 13 distinct systems | Sampled decoding, soft targets, temperature calibration, variance ratio as a *blocking* metric |
| **Beating the benchmark without being useful** | Benchmark has 0 novel items | S2 block-held-out split is the primary reporting split |
| **Pretraining contamination** | Public on HF since 2025; GPT-4.1/Gemini outputs shipped | Report open-weight models with known cutoffs; treat frontier-model prompting numbers as upper bounds of unknown validity |
| **Overfitting a 14-point band** | Band CI is [0.137, 0.144] | Every reported delta carries a participant-level bootstrap CI; differences under ~0.01 are not claims |
| **Sample non-representativeness** | Education TV distance 0.235 (`reports/05`) | Per-cell reporting; no claims about groups with thin support |
| **Between-subject NaN treated as missing** | 76.2% mean coverage is structural | Per-column valid-pair metrics; never impute across arms |

---

## 7. What I would build first

1. Tier 0 IRT / low-rank. Half a day. Likely beats every shipped LLM.
2. S2 block-held-out split + the leakage assertion. Half a day. Without it nothing else is
   interpretable.
3. Tier 1 retrieval prompting with sampled decoding, as the LLM reference point.
4. Tier 2 LoRA with soft targets, only if 1–3 show the LLM adding something over Tier 0.

The honest expected outcome: **Tier 0 wins on C1, the LLM earns its place only on C2**, and
the deliverable worth shipping is a hybrid — factorisation for known items, LM for novel
ones. I have not verified this; it is the hypothesis the build is designed to test, and
`docs/03_evaluation.md` §6 states the criterion that would falsify it.
