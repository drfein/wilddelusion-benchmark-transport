# Model-specific synthetic trajectory probes

This experiment asks whether a model's pre-response residual-stream activation
predicts its own imminent or future delusion endorsement on conversations it
has not seen during probe training.

## Design

- Stimuli: Psychosis-Bench commit
  `73966f95be2e40f1ceee76dfe08eebe939ad6c21`, 16 scripted 12-turn cases.
- Target models: Llama-3.1-8B-Instruct, Qwen3-8B, and Gemma-3-12B-IT at pinned
  revisions in `experiment.json`. The gated Llama and Gemma repositories are
  loaded from pinned full-precision Unsloth mirrors, also recorded there.
- Each target model generates every assistant message, so both context and
  activation trajectories are model-specific.
- Activations are captured immediately before each assistant response at four
  depth fractions.
- Independent trajectories are batched at the same turn. Responses may use up
  to 1,536 tokens, and cap hits are recorded rather than silently truncated.
- Explicit and implicit variants of one underlying scenario remain in the same
  outer and inner folds. Repetitions never cross folds independently.
- Outcomes: endorsement on the current turn and at fixed future horizons.
- Labeling: the exact released Psychosis-Bench 0/1/2 DCS rubric, evaluated from
  the complete generated trajectory through the target response.
- Judges: Qwen3-8B is the primary judge and Gemma-3-12B-IT is an independent
  sensitivity judge. Three-class and binary endorsement agreement are saved;
  Qwen labels remain primary for Gemma, avoiding a self-judge primary endpoint.
- Baselines: turn/condition metadata and user-text features. A probe is useful
  only if it improves on these scripted-trajectory baselines out of group.

Psychosis-Bench has only eight independent scenario pairs. Results are a
mechanistic pilot and require replication on the larger Lost in Delusion
persona set before a broad synthetic-data claim.
