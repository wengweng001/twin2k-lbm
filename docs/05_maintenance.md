# Deliverable 5 — Long-run maintenance

Behaviour changes, panels age, base models get silently replaced. This is how the system
stays trustworthy over months and years.

---

## 1. What actually drifts, ranked

Not all drift is equal. Ranked by how fast it moves and how much damage it does:

| # | Drift source | Timescale | Damage | Detectable from inside? |
|---|---|---|---|---|
| 1 | **Base model version** | days — silent | Severe: all historical results become incomparable | No — must be pinned externally |
| 2 | **Prompt/template changes** | per deploy | Severe, and usually invisible | Yes, with versioned artefacts |
| 3 | **Persona staleness** — a 2025 persona predicting 2027 behaviour | months | Moderate, compounding | Only with fresh ground truth |
| 4 | **Population drift** — panel composition vs the real population | months | Moderate; biases aggregates | Yes, against census updates |
| 5 | **Item drift** — a question's meaning changes (prices, current events) | months–years | Moderate, item-specific | Partially |
| 6 | **Genuine attitude change** in the population | years | This is signal, not noise | Only with fresh waves |

The top two are the ones teams actually get wrong. A model that scored 0.49 last quarter and
0.46 this quarter has usually not drifted — the provider swapped the endpoint.

---

## 2. Versioning

Nothing is reproducible unless all five of these are pinned together and recorded with every
result:

```yaml
run_id:            2026-09-25-tier2-lora-r16
base_model:        Qwen2.5-7B-Instruct@<commit-sha>   # never a floating alias
adapter:           s3://.../lora-r16/<sha256>
prompt_template:   templates/retrieval_v3.jinja@<sha>
dataset_snapshot:  Twin-2K-500@f883165a3026fde855dfd448e0cd16443ab257b6
split_file:        data/derived/splits.json@<sha>
decode:            {n: 20, temperature: 1.0, seed: 0}
```

Production rules:
- **Never a floating model alias.** `gpt-4.1-mini` without a dated suffix is not a version.
- The HF dataset revision is pinned by commit sha, because HF datasets are mutable.
- Evaluation splits, including the S2 block-held-out split proposed in `docs/03_evaluation.md`,
  are committed once implemented, not regenerated. A regenerated split silently changes the
  test set.
- Every metric in every report carries its `run_id`. A number without one is deleted, not
  debated.

---

## 3. Monitoring

### 3.1 Always-on (no fresh ground truth needed)

These run on every batch and need no new human data, which is what makes them viable:

| Signal | Alert | Why |
|---|---|---|
| **Variance ratio** | outside 0.85–1.15 | The known failure mode; shipped systems sit at a median of 0.48 |
| Predicted marginal TV vs training marginal | > 0.15 | Detects distribution shift and template regressions |
| Refusal / out-of-bank rate | changes > 5pp | Detects input drift |
| Answer-format violation rate | > 1% | Usually the first symptom of a base-model swap |
| Latency and token count per call | > 20% change | Cheap proxy for a silent model change |
| Per-cell prediction distribution | any cell shifting > 0.10 TV | Fairness regression |

**Canary set.** 200 frozen (person, question) pairs re-scored on every deploy. Output must
match the recorded baseline within a bootstrap CI. This is the single most effective
control: it catches base-model swaps, template regressions and decoding changes in one
check, within minutes, for near-zero cost.

### 3.2 Periodic (needs fresh ground truth)

Everything above monitors *self-consistency*. Only new human data measures *correctness*.

- **Quarterly mini-wave.** Re-administer ~40 items to a 300-person subsample of the
  original panel. Cost is roughly $3–6k; it is the only way to separate "the model got
  worse" from "people changed". It also refreshes the test-retest reference line, which
  will itself drift.
- **Annual refresh wave.** Full re-administration to a 500-person subsample plus new
  recruits to correct composition.
- **Continuous census re-benchmark.** When ACS/CPS updates, recompute `reports/05` and
  re-check the panel's TV distances.

---

## 4. Retraining triggers

Retrain on any of these, not on a calendar:

| Trigger | Threshold | Action |
|---|---|---|
| Canary regression | outside CI on any deploy | **Block the deploy**; investigate before anything else |
| Normalised score on quarterly wave | drops > 0.10 | Retrain on pooled old + new waves |
| Variance ratio | outside 0.85–1.15 for 2 consecutive batches | Recalibrate temperature; retrain if that fails |
| Per-cell fairness | any n≥100 cell > 0.05 below pooled | Blocking; reweight or restrict claims for that cell |
| Panel composition | TV distance to census worsens > 0.05 | Recruit to correct composition, then retrain |
| Base model deprecation | provider notice | Full re-evaluation against the frozen comparator ladder before switching |
| New data | a full wave lands | Retrain and re-baseline |

**Time-aware evaluation.** Once more than one wave-year exists, the primary split becomes
*temporal*: train on older waves, test on the newest. Random splits will overstate
performance once behaviour drift is real, and the transition to temporal splitting should
happen before it is obviously necessary.

**Persona expiry.** A persona has a shelf life. Until it is measured, treat 12 months as the
default and attach an age to every persona at serving time. The quarterly wave measures the
true decay curve — that curve, once known, is a genuinely publishable result and should
replace the guess.

---

## 5. Governance

**Ownership.** A named owner per model version. A model without an owner is decommissioned,
not inherited.

**Model card**, updated per release: intended use, the §2 prohibitions from
`docs/04_business_applications.md`, comparator-ladder scores, per-cell fairness table, panel
composition, known failure modes, expiry date.

**Change review.** Prompt templates, splits and the comparator ladder are code-reviewed like
code. The most dangerous change in this system is a one-line prompt edit with no review.

**Immutable result log.** Append-only. Results are never silently restated; corrections are
new entries referencing the old `run_id`.

**Consent and deletion.** Deletion requests propagate to the retrieval index within 30 days
and to model weights at the next retrain, with the retrain scheduled rather than
best-effort. Re-identification testing — prompt for a known pid and check for regurgitation
— runs each release.

**Ethics review** on any new application, against the §2 prohibited list. The default answer
to "can we use this for individual-level decisions?" is no, and overriding it requires an
evidence package, not a product argument.

---

## 6. The failure this system will actually have

Not a dramatic one. The realistic failure is quiet:

> Someone uses the model for screening, it works, they stop validating on humans because it
> keeps agreeing with them, the panel drifts, the base model is silently updated twice, and
> eighteen months later a decision is made on simulated data nobody remembers is simulated.

Every control above is aimed at that story, in order of effectiveness:
**(1)** provenance tagging so synthetic data can never be mistaken for real,
**(2)** the canary set so silent model changes surface in minutes,
**(3)** the quarterly wave so "is it still true?" has an actual answer,
**(4)** persona expiry dates so staleness is visible at the point of use.

The accuracy metrics matter less than these. A model that is 3 points worse and correctly
labelled is safe; a model that is 3 points better and quietly wrong is not.
