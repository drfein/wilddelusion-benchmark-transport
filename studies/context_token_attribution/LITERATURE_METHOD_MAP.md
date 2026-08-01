# Method map

## How Language Models Fail (arXiv:2606.06635)

The paper extracts entropy, top-two margin, top-token negative log likelihood,
0.9-nucleus size, and near-tie rate over successive output prefixes. A grouped
logistic classifier's precision-recall curve distinguishes an early
"committed" failure from persistent uncertainty. This is not a context
attribution method. Here it is a secondary, explicitly adapted test of *when*
endorsement becomes externally detectable. It cannot identify which context
caused endorsement.

Paper: https://arxiv.org/abs/2606.06635
Code: https://github.com/sisl/LMTwoFailureModeFramework

## AttriCoT (arXiv:2606.21821)

AttriCoT deletes input/output units, measures the log probability of a fixed
later unit, and fits a linear structural equation in unit-presence indicators.
Its default leave-one-out design costs O(U) forward passes. This is the most
direct black-box method for asking which prior conversational message supports
a *specific generated response*. With a baseline plus one deletion per unit,
the saturated linear coefficient is exactly baseline log probability minus the
deleted-context log probability.

Limitation: the effect is local to the fixed response. It does not by itself
show that deleting the unit changes the distribution of newly generated
responses, so this study adds regeneration as the causal endpoint.

Paper: https://arxiv.org/abs/2606.21821

## FlashTrace (arXiv:2602.01914)

FlashTrace aggregates target spans and recursively propagates information-flow
scores through transformer layers. It is far cheaper than target-token-wise
integrated gradients for long outputs and is released with Qwen3-8B support.
We use it on the same fixed-response subset as AttriCoT and compare each
method's ranking against actual unit-deletion effects.

Limitation: information flow and attention mixing are not interventions. A
high score is treated as a candidate cause until deletion changes behavior.

Paper: https://arxiv.org/abs/2602.01914
Code: https://github.com/wbopan/flashtrace
Pinned commit: `e15e117a5fbbe6e8ad6ea6d0f1314fc7835e5784`

## SAE feature patching (arXiv:2507.22928)

The paper combines sparse autoencoding with activation patching to test
feature-level causal relevance in Pythia chain-of-thought. The transferable
idea is feature patching, not its Pythia features or CoT conclusions. A
pretrained Qwen3-8B SAE is preferable to fitting an SAE on 187 prompts.

This study uses only an SAE trained on the exact instruction model revision and
reports reconstruction/feature-activity diagnostics before interpretation.
Patching is performed at a layer where earlier-token states can still affect
later computation; patching final-layer post-residual states of prior tokens
would be causally inert and is therefore not used.

Paper: https://arxiv.org/abs/2507.22928
Compatible SAE: https://huggingface.co/sammyliu/qwen3-8b-sae-l20-topk64

## Evidence rule

The methods answer different questions:

| Method | Target | Evidence |
|---|---|---|
| Output uncertainty | Generated prefixes | Detection timing |
| Activation probe gradient | Endorsement propensity | Local sensitivity |
| FlashTrace | Fixed response span | Internal information flow |
| AttriCoT/LOO | Fixed response or probe score | Unit intervention |
| SAE patching | Sparse internal feature | Mechanistic intervention |
| Regenerate after deletion | New response distribution | Behavioral causal effect |

Only the final row licenses the paper's primary causal language. Agreement
between the other rows strengthens localization but does not replace it.

