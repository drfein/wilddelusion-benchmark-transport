# Natural-control transport stress test of DelusionScore

## Result

A MiniLM logistic classifier can reproduce excellent case-control performance
while failing to distinguish sincere delusion-like claims from realistic benign
near misses.

The paper-analogous classifier was trained on 439 English, real WildDelusion user
endpoints and 439 source-count-matched random real user messages from ShareChat
and WildChat. Complete conversations were assigned to one of five folds, so no
conversation appeared in both train and test. The model achieved out-of-fold AUC
0.979 and balanced accuracy 0.928. This establishes that the model class used by
Shimgekar et al. can recover the paper's high held-out discrimination in this
real-conversation setting.

When frozen and evaluated on 207 independently reserved natural near misses,
however, AUC fell to 0.670 (conversation-cluster bootstrap 95% CI [0.617, 0.720])
and the false-positive rate at the fixed probability threshold of 0.5 was 84.5%.
These negatives contain fiction, role-play, jokes, quoted third-party beliefs,
and text-transformation tasks that express similar content without a sincere
delusion-like claim. The failure was not explained by source imbalance: within
ShareChat--ChatGPT AUC was 0.602, and within WildChat it was 0.582.

A separate hard-contrast model makes the limitation sharper. On 439 real
endpoints paired with minimally edited grounded counterfactuals, out-of-fold AUC
was 0.870 and 96.6% of pairs were ordered correctly. Yet that classifier was at
chance on the same fully natural near misses: AUC 0.502 (95% CI [0.445, 0.558]),
with a 77.3% false-positive rate.

## Longitudinal test

The broad-control classifier was applied out of fold to 9,396 original user turns
from 439 real cases. Including the retrieval-selected endpoint produced a
positive within-conversation trend after controlling for message length and
conversation fixed effects (beta 0.060, p=0.008). Excluding that selected
endpoint, the adjusted trend was smaller and not statistically distinguishable
from zero (beta 0.030, p=0.162). A length-only classifier already achieved AUC
0.798 on the easy random-control task.

Therefore these data do not provide robust evidence of gradual pre-endpoint
amplification under this score. Much of the apparent rise is concentrated in the
turn selected because it satisfied the endpoint criterion, with message length
accounting for additional signal.

## Interpretation

Shimgekar et al. train DelusionScore on mental-health/delusion-related Reddit
communities versus unrelated engineering, physics, DIY, and cooking communities,
then report rising scores in simulated users. The present result shows that a
classifier with excellent held-out performance against broad controls can have
poor contextual specificity on realistic close negatives. Thus broad
treatment-control accuracy does not by itself validate DelusionScore as a measure
of sincere belief endorsement or within-conversation intensification.

This is a construct-transport limitation, not an exact numerical refutation. The
published fitted classifier and its 3,000 Reddit training posts are unavailable,
so the exact DelusionScore cannot be run here. WildDelusion is also endpoint-
ascertained, so this study cannot estimate population prevalence or the causal
effect of assistant responses. The defensible conclusion is that reported
amplification should be revalidated with natural, context-resolved near misses,
conversation-grouped splits, endpoint-excluded trajectories, and length controls.

## Reproducibility

- Frozen embedding model: `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions.
- Classifier: L2 logistic regression, `C=1`, unnormalized sentence embeddings.
- Split: deterministic five-fold split by source conversation ID.
- Inference: conversation-cluster bootstrap and cluster-robust OLS.
- Paid API cost: $0.
- Raw text is not copied into this Git repository. Public score tables contain
  hashes, provenance, labels, lengths, and out-of-fold scores. The sampling and
  analysis scripts reconstruct the study from the gated source datasets.
