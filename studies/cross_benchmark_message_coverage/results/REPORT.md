# Provisional composition-gap result

## Design

This is a corpus-composition experiment, not a model-performance experiment.
The corrected corpus contains 522 validated WildDelusion target messages, 192
Psychosis-Bench user turns, 14 Spiral-Bench starters, and 6,330 SIM-VAIL user
turns. All 7,025 unique texts were pooled, assigned opaque IDs, shuffled, and
coded without provenance by `gpt-5.4-mini` at temperature 0 with no reasoning.
Uncertainty clusters the 522 real rows by their 321 source conversations.
Synthetic pooling gives each benchmark equal weight.

## Result

The prespecified quiet-indirect cell did not explain real-world coverage. A
no-distress, no-harm, indirect message was less frequent in WildDelusion
(9.4%, 99% cluster CI 6.2%-12.8%) than in the equal-paper synthetic pool
(29.1%). Only 0.8% of real messages occupied joint cells completely absent
from all three available benchmarks (99% CI 0.0%-2.0%), so the strong claim
that real messages occupy wholly uncovered regions is not supported.

There is nevertheless a large distributional shift. WildDelusion targets were
more often coded as direct assertions (88.3%, 95% CI 85.5%-91.0%) than
benchmark turns (24.3%, 95% CI 16.7%-32.8%). An exploratory no-distress,
no-harm, direct-assertion cell contains 79.7% of WildDelusion targets versus
12.2% of the equal-paper benchmark pool. Fifteen percent of real messages
fell in joint cells with less than 1% benchmark prevalence (99% CI
11.6%-19.3%). Distress itself largely overlapped (TVD 0.066), whereas harm
(0.138), directness (0.640), and theme (0.613) did not meet the preregistered
TVD < 0.10 overlap threshold. The amended broad-gap falsification criterion
was therefore not met.

## Interpretation limits

The directness contrast is conditional on corpus construction, not an unbiased
population prevalence estimate. WildDelusion targets were selected as
confirmed delusion-like endpoints, while benchmark datasets include neutral
setup, intermediate, and cleanup turns. Final-turn and late-half sensitivity
tables are included, but they cannot remove this selection difference.

The independent, provenance-stratified `gpt-5.4-nano` recode achieved exact
agreement of 90.7% for distress, 84.3% for harm, 61.0% for directness, and
65.7% for theme (kappa 0.466, 0.371, 0.425, and 0.559 respectively). Because
agreement is modest on the two axes with the largest gaps, these findings are
provisional. A 300-item blind human-coding packet exists locally at
`work/human_audit_blind.csv`; publication-level claims should wait for its
completion.

Lost in Delusion was preregistered but excluded because no official turn-level
artifact was available. Its paper reports aggregate synthetic generation, not
the user-turn records needed for this analysis.
