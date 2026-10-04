# 04 — Persona size, context budget, and sample shape

## Persona length

| Field | median chars | p95 chars | median ~tokens | p95 ~tokens |
|---|---:|---:|---:|---:|
| wave1_3_persona_text (the legal input) | 95,204 | 96,256 | 23,801 | 24,064 |
| wave1_3_persona_json | 121,425 | 122,530 | 30,356 | 30,633 |
| wave4_Q_wave4_A (target, must be stripped) | 45,451 | 45,797 | 11,363 | 11,449 |
| full_persona.persona_text (LEAKS) | 128,665 | 129,716 | 32,166 | 32,429 |
| full_persona.persona_summary | 12,993 | 13,944 | 3,248 | 3,486 |

(~tokens = chars / 4, the usual English rule of thumb.)

Measured on 20 sampled personas with a real BPE tokenizer: **27,484 tokens** median, i.e. 3.46 characters per
token. The rule of thumb understates the true count by roughly 15%, because
this text is structured and repetitive rather than prose. Budget against the
measured figure.

## What this implies for the model

- A legal persona is ~**27,484 tokens** at the median (measured, not
  estimated). That fits a modern long-context model but is far too large for the
  <0.5B models the prototype is restricted to (typically 2k-8k usable context).
- Naive prompting cost: 27,484 tokens x 126 questions x
  2,058 people ≈ 7.1B input tokens per full
  evaluation sweep if the persona is re-sent per question. Batch all 126 questions into
  one call per person and it drops by ~126x; cache the persona prefix and it drops again.
- `persona_summary` is ~3,248
  tokens — a ~7x compression. Section 03 shows the summary specifications score
  *worse*, so compression here is not free.
- Retrieval is the obvious lever: for each target question, pull only the items
  measuring the same construct. The 126 wave-4 columns map to 85 QuestionIDs, so a
  per-question context of a few hundred tokens is achievable.

## Sample shape

- N = 2,058 participants; 760 wave-1-3 columns, 126 wave-4 columns.
- Demographic columns in the catalog: 14.

**QID11** — Which part of the United States do you currently live in?

| value | share |
|---|---:|
| South (TX, OK, AR, LA, KY, TN, MS, AL, WV, DC, MD, DE, VA, N | 40.5% |
| West (WA, OR, ID, MT, WY, CA, NV, UT, CO, AZ, NM) | 24.0% |
| Midwest (ND, SD, NE, KS, MN, IA, MO, WI, IL, MI, IN, OH) | 18.1% |
| Northeast (PA, NY, NJ, RI, CT, MA, VT, NH, ME) | 16.6% |
| Pacific (HI, AK) | 0.8% |

**QID12** — What is the sex that you were assigned at birth?

| value | share |
|---|---:|
| Female | 50.7% |
| Male | 49.3% |

**QID13** — How old are you?

| value | share |
|---|---:|
| 30-49 | 35.7% |
| 50-64 | 32.0% |
| 18-29 | 18.9% |
| 65+ | 13.5% |

**QID14** — What is the highest level of schooling or degree that you have completed?

| value | share |
|---|---:|
| College graduate/some postgrad | 35.7% |
| Some college, no degree | 22.7% |
| Postgraduate | 15.2% |
| High school graduate | 13.2% |
| Associate's degree | 12.3% |
| Less than high school | 0.8% |

**QID15** — What is your race or origin?

| value | share |
|---|---:|
| White | 66.1% |
| Black | 12.2% |
| Hispanic | 9.4% |
| Asian | 6.8% |
| Other | 5.4% |

**QID16** — Are you a citizen of the United States?

| value | share |
|---|---:|
| Yes | 99.8% |
| No | 0.2% |

## Limitations that bound every claim built on this data

- US adults only, surveyed in 2024-25. No cross-national or longitudinal coverage
  beyond the four waves, so nothing here supports claims about behaviour drift.
- Self-report throughout: social-desirability and satisficing are baked in, and the
  61% test-retest figure is itself partly a measure of survey noise, not of
  genuine preference instability.
- Wave 4 covers only heuristics-and-biases plus pricing. Personality, cognitive and
  demographic blocks are **never** evaluated, so 'predicts a person' really means
  'predicts their behavioural-economics answers'.
- Attrition across waves and panel self-selection are not corrected for here.
