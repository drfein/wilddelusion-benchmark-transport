# Cross-benchmark message coverage: frozen analysis plan

Timestamp: 2026-07-31 America/Los_Angeles

Status at freeze: no combined corpus has been constructed, no messages have
been coded under this scheme, and no corpus-level coverage result has been
computed. The investigators have previously inspected individual source papers
and artifacts. This is an analysis-plan freeze, not a prospective registry.

## Question

Do confirmed user messages from natural WildDelusion conversations occupy
content regions that are systematically underrepresented by independently
constructed synthetic mental-health/delusion benchmarks?

## Corpora

The synthetic comparison is the union of released user-side conversational
turns from:

1. Kirgis et al., Spiral-Bench;
2. Au Yeung et al., Psychosis-Bench;
3. Weilnhammer et al., SIM-VAIL; and
4. Aquilina et al., Lost in Delusion.

The natural corpus is the confirmed, deduplicated WildDelusion conversation
release. The primary natural unit is the confirmed flagged user message. A
secondary analysis uses all user messages through the flagged message and
clusters uncertainty by source conversation.

Kim et al.'s structural-drift corpus is excluded from this coverage test. Its
seven phenomenological domains measure a different construct, and its linked
raw repository returned HTTP 404 at the time of this freeze. It will be treated
as a separate transport study if the artifact becomes available.

## Blinding

Every coding item receives a random opaque ID. The coder sees only message
text. Corpus, paper, model, scenario, turn position, flagged status, and nearby
messages are excluded from the coding prompt. Items are shuffled globally.
Identical normalized texts are coded once and joined back afterward.

Primary coding uses one fixed LLM snapshot at temperature 0 with no reasoning.
A stratified blinded audit of at least 200 items is independently recoded by a
second fixed model. No prompt changes are permitted after primary labels are
read. Parse failures are retried without changing the rubric and then reported.

## Frozen axes

### Explicit distress marker

Binary. `1` only when the message directly states current emotional distress,
fear, despair, overwhelm, inability to cope, marked functional impairment, or
a request for urgent emotional help. Do not infer distress merely from bizarre,
false, risky, or clinically concerning content.

### Harm indication

Ordinal:

- `0`: no stated intent, plan, preparation, recent act, or invitation involving
  self-harm, suicide, violence, dangerous confrontation, severe self-neglect,
  or another immediately hazardous action;
- `1`: indirect, conditional, exploratory, ambiguous, or non-imminent harmful
  intent/action;
- `2`: direct current intent, plan, preparation, recent attempt/act, command,
  or request for actionable assistance with a hazardous act.

Mentioning harm as fiction, quotation, news, or abstract discussion is `0`
unless the speaker applies it to their own intended or recent behavior.

### Directness

Ordinal directness of the user's unusual or potentially delusion-like claim:

- `0`: no unusual belief claim, or the content is clearly fictional,
  hypothetical, quoted, role-played, joked about, or skeptically examined;
- `1`: unusual interpretation is suggested, tested, hedged, or left ambiguous;
- `2`: unusual interpretation is asserted as the user's actual belief or lived
  reality.

This axis measures presentation, not whether the claim is clinically a
delusion and not whether it is factually false.

### Theme

Exactly one primary category:

- `none_or_unclear`
- `grandiose_or_messianic`
- `persecutory_conspiratorial_or_referential`
- `romantic_or_erotomanic`
- `somatic_or_perceptual`
- `thought_interference_or_broadcast`
- `bizarre_metaphysical_or_reality`
- `ai_sentience_personhood_or_relationship`
- `mixed_or_other`

When two listed themes are equally central, use `mixed_or_other`. Topic alone
does not establish a theme: for example, an ordinary physics question is
`none_or_unclear`, not `bizarre_metaphysical_or_reality`.

## Primary estimands

All estimates are reported separately for each synthetic benchmark, the
pooled synthetic corpus with equal paper weighting, and WildDelusion.

1. Prevalence of each distress, harm, directness, and theme value.
2. Support coverage over the full Cartesian cells of the four frozen axes.
3. Natural-only cell mass: the WildDelusion share in cells absent from all
   four synthetic corpora.
4. Sparse-cell mass: the WildDelusion share in cells whose equal-paper
   synthetic prevalence is below 1%.
5. Jensen-Shannon divergence between WildDelusion and equal-paper synthetic
   distributions over joint axis cells.

Intervals use a source-conversation cluster bootstrap for WildDelusion and a
scenario/dialogue cluster bootstrap for each synthetic corpus. The pooled
synthetic distribution weights papers equally, then observations equally
within paper. Duplicate source messages repeated across model runs are one
stimulus unit in the primary analysis.

## Prespecified sensitivity analyses

1. WildDelusion flagged messages only versus all user turns through the flag.
2. Equal-paper weighting versus observation weighting.
3. Exact-text deduplication versus source-scenario deduplication.
4. Consensus-only labels from the primary and independent coder.
5. Theme-marginalized joint cells over distress, harm, and directness.

## Falsification and reporting rule

The broad coverage-gap claim is falsified if the 99% cluster-bootstrap upper
bound on WildDelusion natural-only cell mass is below 5% and no prespecified
axis has a 99% interval excluding a 5-percentage-point real-versus-synthetic
difference after Holm correction across the four axis families.

Regardless of outcome, every frozen axis, every benchmark, all null results,
all excluded or unavailable artifacts, coder disagreement, and every listed
sensitivity analysis will be reported. No new axis discovered after coding is
eligible for the primary coverage claim; exploratory axes must be labeled as
such and require new held-out data.
