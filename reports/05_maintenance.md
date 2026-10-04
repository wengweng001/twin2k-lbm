# Report 5 - Long-Run Maintenance

Maintain the model as an end-to-end system: data, splits, prompts, retrieval, model
weights, evaluation code, and governance all need versions. A model score is trustworthy
only if the full path that produced it is reproducible.

## 1. End-to-End Maintenance Structure

```text
fresh human data
  -> data QA + leakage checks
  -> versioned dataset snapshot
  -> fixed train/test splits
  -> model training or recalibration
  -> evaluation against frozen baselines
  -> versioned release
  -> production monitoring
  -> drift trigger: recalibrate, retrain, rollback, or restrict use
```

## 2. What Must Be Versioned

| Component | Version record | Why it matters |
|---|---|---|
| raw data | dataset snapshot / HF commit / collection date | human behaviour and dataset files change over time |
| derived data | script version + output hash | preprocessing changes can move metrics |
| train/test split | `splits.json` hash | changing splits changes the benchmark |
| prompt template | git hash | small prompt edits can change behaviour |
| retrieval index | index hash + embedding model version | retrieved persona evidence is part of the model input |
| base model | exact model commit, not alias | provider or checkpoint changes break comparability |
| adapter / fine-tune | adapter hash + training config | weights must be reproducible |
| decoding config | temperature, samples, seed | distribution metrics depend on decoding |
| evaluation code | metric version / git hash | metric changes can look like model changes |
| output data | `run_id`, timestamp, synthetic tag | synthetic rows must not be confused with human data |

Minimum run record:

```yaml
run_id: 2026-09-25-lbm-v1
data_snapshot: Twin-2K-500@<hf-commit-sha>
split_file: data/derived/splits.json@<git-sha>
prompt_template: retrieval_v3@<git-sha>
retrieval_index: <index-sha>
base_model: Qwen2.5-7B-Instruct@<commit-sha>
adapter: <adapter-sha>
decode: {n: 20, temperature: 1.0, seed: 0}
metrics_version: eval_v3@<git-sha>
```

## 3. Maintenance Problems and Strategies

| Problem | Strategy |
|---|---|
| human behaviour changes | run quarterly mini-waves; recompute accuracy, calibration, and test-retest |
| personas become stale | attach persona age; treat personas older than 12 months as stale unless revalidated |
| panel composition drifts | recompute representativeness; reweight, recruit, or restrict population claims |
| question meaning changes | flag time-sensitive items; refresh item bank and rerun human validation |
| base model changes | pin model commit; rerun canary and comparator ladder before switching |
| prompt or retrieval changes | code-review prompt/index diffs; require canary pass |
| model output collapses | monitor variance ratio; recalibrate or retrain if outside 0.85-1.15 |
| subgroup performance drops | monitor demographic cells; collect more data or restrict claims for weak cells |
| synthetic data mixes with real data | tag all generated rows with `synthetic=true`, `run_id`, and model version |
| evaluation changes | version evaluator; rerun previous model when metrics change |

## 4. Monitoring and Retraining Triggers

| Signal | Trigger | Response |
|---|---|---|
| canary set | outside prior-release CI | block deploy or roll back |
| quarterly mini-wave score | NS drops >0.10 | retrain or recalibrate on pooled waves |
| variance ratio | outside 0.85-1.15 for two batches | recalibrate temperature; retrain if unresolved |
| marginal TV | >0.15 vs training distribution | inspect input drift; recalibrate or retrain |
| answer format | invalid rate >1% | block deploy until fixed |
| demographic cell | n >= 100 cell >0.05 below pooled score | collect data, reweight, or restrict claims |
| panel representativeness | TV worsens >0.05 | update population caveat; rebalance before broad claims |
| new full wave | new ground truth available | create temporal split, retrain, publish new model card |

Retraining is not automatic promotion. A candidate replaces production only after passing
the same leakage, baseline, calibration, variance, and fairness gates.

## 5. Governance and Ethics

| Risk | Control |
|---|---|
| use outside intended scope | review new use cases against `reports/04_business_applications.md` |
| individual-level misuse | aggregate outputs by default; prohibit credit, hiring, insurance, medical, policing, and personalised pricing decisions |
| privacy / memorisation | test whether prompts can recover a known participant's answer set before release |
| deletion requests | remove persona from retrieval immediately; remove from model weights at next retrain |
| no accountable owner | assign owner and expiry date to every model version |
| stale model remains available | sunset models that fail validation or pass expiry without reapproval |
