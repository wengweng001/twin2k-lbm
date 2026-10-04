# Deliverable 4 — Business applications and guardrails

The measured capability is narrower than "digital twin" suggests, so the applications are
scoped to it. What this system can currently do, stated honestly:

> Given a person's ~500 prior survey answers, produce a *distribution* over how they would
> answer a behavioural-economics or product-pricing question, at roughly the reliability
> with which that person agrees with themselves two weeks later — which is 61%.

It cannot do personality, cognitive ability, or demographic inference, because wave 4 never
evaluates those blocks. That limitation is structural, not a matter of more training.

---

## 1. Where this does a real job

### 1.1 Survey and questionnaire pre-testing — strongest fit

**The job:** before fielding an expensive instrument to 2,000 real people, run it against
simulated respondents to find items that are broken.

**Why it fits:** this only needs *aggregate* signal, and it needs it for items similar to
ones in the training bank. It is also the one use where the model's weaknesses are
diagnostic rather than disqualifying — an item on which the model is wildly uncertain is
usually an item that is ambiguously worded.

**Who buys it:** market-research agencies, academic survey labs, government statistical
offices, anyone running a panel.

**Value:** a wave of 2,000 respondents costs roughly $10–40k and 2–3 weeks. Catching three
broken items before fielding pays for the system.

**Acceptance bar:** T3 (treatment-effect recovery ≥ 80%) and T4 (TV < 0.10) from
`reports/03_evaluation_strategy.md`. Below that bar it is not fit for this use.

### 1.2 Experiment power analysis and design

**The job:** estimate the likely effect size and variance of a planned experiment to size
the sample, choose arms, and decide whether to run it at all.

**Why it fits:** the dataset's 11 between-subject experiments are exactly this shape, so it
is the use case with the most direct evidence. Note it requires the *variance* to be right,
not the mean — and the shipped systems get variance wrong by 2x. This is buildable, not
built.

### 1.3 Product concept and pricing screening

**The job:** narrow 40 product concepts to 8 worth testing with real people.

**Why it fits:** the pricing block is 40 of the 126 evaluated columns, and it is where
personalisation signal is strongest — the largest B1→B4 gaps in
`reports/exploration/01_structure_and_retest.md` are pricing items (+0.29 to +0.31 versus
+0.14 overall).

**Framing that keeps it honest:** a *screening* tool that reduces the candidate set, with
the survivors always validated on humans. Never a replacement for the confirmatory test.

### 1.4 Longitudinal panel gap-filling

**The job:** a panel member skips wave 3; impute their answers so the panel stays balanced.

**Why it fits:** this is literally the training task, and copy-forward already gives a
strong, auditable baseline to fall back on.

**Guardrail:** imputed values must be flagged in the data model and excluded from any
published estimate without an explicit sensitivity analysis. The failure mode is imputed
data quietly becoming "data".

---

## 2. Where it must not be used

These are not risk-management boilerplate; each maps to a specific measurement.

| Prohibited use | The measurement that forbids it |
|---|---|
| **Any decision about a real named individual** — credit, hiring, insurance, medical triage, policing | Per-person accuracy is 0.49 at best, below the 0.61 rate at which people agree with themselves. A coin-flip-grade prediction attached to a person's name is an unacceptable basis for consequential decisions, and in many jurisdictions unlawful. |
| **Replacing human subjects in published research or regulatory submissions** | 13 published systems fail to beat copy-forward. There is no evidence base for substituting simulated for real respondents in a confirmatory claim. |
| **Speaking for under-represented groups** | Education TV distance 0.235; "less than high school" is 0.8% of the panel vs ~9% of US adults. Combined with the 0.48 median variance ratio, minority positions are compressed twice — once by sampling, once by modelling. |
| **Political or public-opinion forecasting** | Sample is a 2024–25 online panel, skewed young and educated; wave 4 covers heuristics and pricing, not political attitudes. Nothing here supports opinion estimates. |
| **Impersonating a specific real person** | The persona is that person's real survey answers. Generating new statements "as them" and presenting those as their views is misrepresentation regardless of accuracy, and the source data is consented for research, not for synthetic voice. |
| **Emotional or clinical inference** | No clinical instruments in the evaluated blocks, no clinical validation, no duty-of-care framework. |

---

## 3. Guardrails that ship with the product

1. **Distribution, never a point.** The API returns a distribution with an uncertainty
   estimate. There is no endpoint that returns a bare predicted answer, because a bare
   answer invites exactly the individual-level misuse in §2.
2. **Refuse out-of-bank questions.** Questions far from the training item bank get a
   low-confidence flag and a refusal rather than a confident guess. Measured by embedding
   distance to the nearest catalog item, thresholded on the S2 split.
3. **Aggregate-only outputs by default.** Individual-level responses are available only
   behind an explicit flag documenting the §2 prohibitions, and are rate-limited and
   audit-logged.
4. **Population card on every output.** Every result carries the panel's composition and
   its deviation from the population the customer thinks they are studying — the table from
   `reports/exploration/05_representativeness.md`, attached automatically.
5. **Provenance.** Synthetic responses are tagged in every export so they cannot be
   silently merged into a real dataset. This is the single highest-value guardrail: the
   realistic catastrophic failure here is not a bad prediction, it is simulated data
   entering a real analysis pipeline undetected.
6. **Human-validation clause in the contract.** Screening outputs must be confirmed on real
   respondents before any launch or publication decision. Written into terms, not just docs.

---

## 4. Consent and data protection

The underlying panel consented to research participation, not to having a persistent model
of their responses operated commercially. Even anonymised, ~500 answers per person is
plausibly re-identifying by combination. Concretely:

- Do not ship models whose weights or retrieval index can regurgitate an individual's
  answer set; test for memorisation by prompting for a known pid's persona.
- Operate on the aggregate, and treat individual personas as restricted data under the
  original research ethics approval.
- Honour deletion requests through to retraining, not just from the serving index — which
  means the retraining cadence in `reports/05_maintenance.md` is a compliance requirement, not
  only an accuracy one.

---

## 5. Honest positioning

The commercially attractive story is "replace your survey panel". The evidence does not
support it and probably will not: humans only agree with themselves 61% of the time, so
there is a hard ceiling on how faithfully any system can reproduce an individual response.

The defensible story is narrower and still valuable: **a cheap, fast, always-available
pre-test that tells you which questions to ask real people, and which experiments are worth
running.** That framing survives contact with the measurements in `reports/`, and it does
not require the model to beat a human at being themselves.
