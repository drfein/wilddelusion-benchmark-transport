# Context-token attribution

This study asks which parts of a complete, naturally occurring conversation
change an open model's probability of endorsing a delusion-like user claim.
It treats token saliency as a hypothesis generator and requires behavioral
deletion or replacement tests before calling an attribution causal.

The private sampling frame is the outcome-independent 216-conversation cohort
from `studies/agreement_context_scaled`. Histories are never truncated: exact
model-tokenized prompts that exceed the frozen limit are excluded and counted.

## Planned evidence hierarchy

1. Held-out endorsement propensity predicted from pre-response activations.
2. Message and clause leave-one-out effects on a frozen probe and on fixed
   endorsing-response log likelihood (AttriCoT-style local attribution).
3. Gradient-times-input and integrated-gradient token rankings.
4. FlashTrace multi-token information-flow attribution.
5. Regeneration after deleting the top assistant span versus a matched random
   assistant span, judged with the frozen SPIRALS endorsement rubric.
6. Output-token uncertainty trajectories to test whether endorsement is an
   early committed failure or persistent uncertainty process.

Raw conversations, generations, judgments, token scores, and activations stay
under `artifacts/private/` and are excluded from git.

