# 05 — Representativeness

Panel: N = 2,058 US adults. Compared against US adult population
marginals (ACS 2023 / CPS 2023; educational attainment is for age 25+). The
benchmarks are approximate and are used to show the direction and rough size of
skew, not to reweight anything.

Total-variation distance (half the sum of absolute differences) summarises each
variable: 0 means the marginal matches, 0.10 means a tenth of the sample would
have to be reassigned to match.

| Variable | TV distance | largest gap |
|---|---:|---|
| age | 0.103 | 65+: 13.5% vs 22.0% (-8.5pp) |
| education | 0.235 | High school graduate: 13.2% vs 27.5% (-14.3pp) |
| race | 0.111 | White: 69.9% vs 58.7% (+11.2pp) |
| region | 0.028 | Midwest: 18.2% vs 20.7% (-2.5pp) |
| sex | 0.001 | Male: 49.3% vs 49.4% (-0.1pp) |

### age

| category | sample | US adults | diff (pp) | ratio |
|---|---:|---:|---:|---:|
| 18-29 | 18.9% | 20.6% | -1.7 | 0.92x |
| 30-49 | 35.7% | 33.1% | +2.6 | 1.08x |
| 50-64 | 32.0% | 24.3% | +7.7 | 1.32x |
| 65+ | 13.5% | 22.0% | -8.5 | 0.61x |

### education

| category | sample | US adults | diff (pp) | ratio |
|---|---:|---:|---:|---:|
| Less than high school | 0.8% | 9.0% | -8.2 | 0.09x |
| High school graduate | 13.2% | 27.5% | -14.3 | 0.48x |
| Some college, no degree | 22.7% | 14.8% | +7.9 | 1.54x |
| Associate's degree | 12.3% | 8.7% | +3.6 | 1.41x |
| College graduate/some postgrad | 35.7% | 23.4% | +12.3 | 1.53x |
| Postgraduate | 15.2% | 14.6% | +0.6 | 1.04x |

### race

| category | sample | US adults | diff (pp) | ratio |
|---|---:|---:|---:|---:|
| White | 69.9% | 58.7% | +11.2 | 1.19x |
| Hispanic | 10.0% | 19.4% | -9.4 | 0.51x |
| Black | 12.9% | 12.3% | +0.6 | 1.05x |
| Asian | 7.2% | 6.2% | +1.0 | 1.16x |

112 responses did not map to a benchmark category and are excluded from this variable's shares.

### region

| category | sample | US adults | diff (pp) | ratio |
|---|---:|---:|---:|---:|
| South | 40.8% | 38.4% | +2.4 | 1.06x |
| West | 24.2% | 23.8% | +0.4 | 1.02x |
| Midwest | 18.2% | 20.7% | -2.5 | 0.88x |
| Northeast | 16.7% | 17.1% | -0.4 | 0.98x |

16 responses did not map to a benchmark category and are excluded from this variable's shares.

### sex

| category | sample | US adults | diff (pp) | ratio |
|---|---:|---:|---:|---:|
| Male | 49.3% | 49.4% | -0.1 | 1.00x |
| Female | 50.7% | 50.6% | +0.1 | 1.00x |

## What this means for the model

Two skews are large enough to change how results should be read:

- **Education.** 'Less than high school' is 0.8% of the panel against 9.0% of US adults — a 0.09x
  under-representation. The least-educated decile of the country is effectively
  absent. Cognitive-task and heuristics-and-biases results are exactly the
  measures most sensitive to this, and they are the entire wave-4 evaluation set.
- **Age.** 65+ is 13.5% against 22.0% (-8.5pp). Older adults are under-sampled.

Consequences, stated plainly:

1. A behaviour model fit here is a model of an online-panel population that is
   more educated and younger than the US, not of the US.
2. Post-stratification weights can repair *aggregate* estimates for these marginals.
   They cannot repair individual-level prediction for a demographic cell that
   contains almost no training examples.
3. Combined with the flattening documented in report 03 — simulated populations at
   roughly half the real variance — minority positions are compressed twice: once by
   who was sampled, once by the model's pull toward the modal answer. Any use of
   this system to speak for an under-represented group is unsupported.

These marginals are also the only handle available for the fairness slice of the
evaluation: report accuracy *per demographic cell*, not just pooled, and treat a
cell with a materially worse score as a blocking result rather than a footnote.
