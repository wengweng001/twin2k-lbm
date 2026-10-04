# Report 4 - Business Applications and Guardrails

The most appropriate product shape is a **research accelerator**: a synthetic respondent
system that helps teams decide what to test with real people. It should be used for early
screening, survey pre-testing, message testing, and experiment design. It should not be
used as a replacement for real respondents in high-stakes or confirmatory decisions.

## 1. Capability Boundary

Given a person's wave 1-3 persona, the model predicts a distribution over how similar
respondents may answer behavioural, product-preference, or survey questions. The useful
output is an aggregate response distribution, not a claim about a named individual.

Best-fit jobs:

- identify weak or ambiguous survey items before fielding;
- compare candidate messages, prices, product descriptions, or experimental arms;
- estimate which variants are worth testing with real respondents;
- flag where additional human sampling is needed.

## 2. Products and Organizations

| Product / organization | Job the model does | Concrete use |
|---|---|---|
| Market research and consumer insights teams | Early concept, message, and pricing screening | Simulate response distributions for draft concepts or price points, then advance the strongest candidates to real panel testing. |
| Survey platforms and panel providers | Survey quality control | Pre-test draft surveys, flag unstable questions, wording-sensitive items, and questions likely to produce low test-retest reliability. |
| E-commerce and marketplaces | Product-page and offer screening | Compare product titles, descriptions, bundles, prices, shipping offers, or promotional wording before running live A/B tests. |
| CPG and retail research teams | Product and category research | Narrow product concepts, packaging claims, or promotion mechanics before commissioning consumer research. |
| Product and UX research teams | Preference and wording tests | Compare onboarding copy, feature descriptions, survey wording, or preference questions before user studies. |
| Academic and behavioural science labs | Experiment piloting | Pre-test framing, anchoring, loss/gain, and choice-task stimuli before recruiting human subjects. |
| Policy and public-communication researchers | Message pre-testing | Check whether health, climate, civic, or policy communication materials are confusing or produce large subgroup differences. |

## 3. Required Guardrails

| Guardrail | Requirement |
|---|---|
| Aggregate by default | Report distributions and segment-level summaries, not individual-level predictions. |
| Human validation | Treat synthetic results as screening evidence; validate final claims with real respondents. |
| Provenance | Label synthetic responses clearly in every export so they are not mixed with human data. |
| Uncertainty | Show confidence intervals, variance ratio, and calibration diagnostics where applicable. |
| Population caveat | Attach the panel composition and note under-represented groups before making population claims. |
| Out-of-bank refusal | Refuse or mark low confidence for questions far from the training item bank. |
| Leakage control | Use only legal `wave_split` persona inputs; never use `full_persona` for model input. |
| Audit trail | Log prompt template, model version, data snapshot, and split file for every result. |

## 4. Should Not Be Used For

This model should not be used for:

- credit, hiring, insurance, medical, policing, or other decisions about named individuals;
- individual targeting, personalised pricing, or discriminatory offer selection;
- replacing human respondents in confirmatory research, regulatory submissions, or launch decisions;
- official population estimates, election polling, or high-stakes public-opinion forecasting;
- claims about under-represented groups without additional human validation;
- impersonating a real respondent or presenting simulated answers as that person's views;
- clinical, emotional, or cognitive diagnosis.

## 5. Appropriate Positioning

The model is useful when the business action is **screening and prioritisation before human
validation**. It helps decide which surveys, product concepts, messages, prices, or
experimental conditions are worth testing with real people. It is not evidence that those
variants will succeed in the market, and it is not a substitute for human-subject research.
