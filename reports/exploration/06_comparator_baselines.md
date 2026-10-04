# 06 — Additional comparator baselines

These baselines are not model proposals; they are comparators. The point is to make
the evaluation ladder concrete before spending more compute on LLMs.

## Headline exact-match accuracy

| Comparator | Mean exact accuracy | Mean normalised MAE |
|---|---:|---:|
| B1 population mode / mean | 0.471 | 0.321 |
| B2 demographic-cell mode | 0.467 | 0.292 |
| B3 kNN response vector (k=50) | 0.486 | 0.278 |
| B4 copy-forward / test-retest | 0.612 | 0.169 |

B2 uses age x sex x education cells and excludes the participant's own answer.
B3 uses the participant's other wave1-3 answers to find similar respondents, then
predicts the modal earlier answer among the 50 nearest neighbours. The target column
is removed from the similarity calculation, so this is not copy-forward.

## By answer scale

| Scale | B1 pop | B2 demo | B3 kNN | B4 copy-forward |
|---|---:|---:|---:|---:|
| binary | 0.633 | 0.631 | 0.650 | 0.809 |
| continuous | 0.124 | 0.113 | 0.125 | 0.136 |
| ordinal_long | 0.222 | 0.224 | 0.216 | 0.328 |
| ordinal_short | 0.369 | 0.363 | 0.386 | 0.501 |
| text | 0.087 | 0.090 | 0.095 | 0.200 |

## Where kNN adds the most over population mode

| col | block | pop | kNN | copy-forward | question |
|---|---|---:|---:|---:|---|
| QID287_11 | False consensus  | 0.255 | 0.378 | 0.613 | Would you support or oppose... |
| QID287_3 | False consensus  | 0.345 | 0.449 | 0.611 | Would you support or oppose... |
| QID287_6 | False consensus  | 0.387 | 0.481 | 0.653 | Would you support or oppose... |
| QID172 | Less is More Gamble B  | 0.189 | 0.277 | 0.440 | Please rate your level of disagreement or agreement with the following statement |
| QID164_TEXT | Anchoring - African countries low  | 0.078 | 0.148 | 0.241 | How many African countries do you think are in the United Nations? |
| QID287_5 | False consensus  | 0.389 | 0.445 | 0.566 | Would you support or oppose... |
| QID9_17 | Product Preferences - Pricing | 0.555 | 0.602 | 0.845 | Please consider the following product category: Toilet Tissue. Suppose you are i |
| QID287_4 | False consensus  | 0.481 | 0.527 | 0.672 | Would you support or oppose... |

## Interpretation

The response-vector baseline beats the demographic-cell baseline, which means
there is useful behavioural signal in the persona beyond demographics. That is
the signal an LBM should exploit.
Neither cheap baseline approaches copy-forward, so the repeated-item benchmark still
leaves a large denoising gap. These rows should be reported next to any LLM result.