# Rebuilding generated examples

The reported analyses use immutable validated examples in `frozen/`.
Regenerating them is a robustness analysis, not a bitwise reproduction, because
provider-side LLM inference can change despite identical code and model names.

## Grounded controls

The exact scripts are:

1. `full_history_522/generate_high_stakes_controls.py` and the general
   `generate_controls.py` generation path;
2. `validate_controls.py` for blinded pair metrics;
3. `repair_controls.py` and
   `full_history_522/repair_controls_messagewise.py` for failed pairs;
4. `full_history_522/audit_controls_blind.py` for independent absence of the
   delusion-like belief; and
5. `full_history_522/prepare_model_inputs.py` for the final 101-pair freeze.

The frozen files preserve the final control messages, pair validation, and blind
audit so the accepted cohort does not drift between reproductions.

## Explicit/implicit pairs

The exact sequence is:

```bash
python psychogenic_machine_transport_250910970/full522/label_safety.py
python psychogenic_machine_transport_250910970/generate_pairs.py
python psychogenic_machine_transport_250910970/validate_pairs.py
python psychogenic_machine_transport_250910970/full522/repair_pairs.py ...
python psychogenic_machine_transport_250910970/validate_pairs.py ...
python psychogenic_machine_transport_250910970/full522/merge_pair_rounds.py ...
python psychogenic_machine_transport_250910970/prepare_inputs.py
```

The accepted final set contains 327 pairs: 139 from the initial pass, 113 from
repair round one, and 75 from repair round two.

## Natural near-misses

`scripts/build_natural_near_miss_controls.py` deterministically mines verifier
exclusions for fiction, role-play, jokes, dreams, quotation/third-party claims,
and text tasks. The study then runs:

- `adjudicate_natural_controls.py` over full transcripts;
- `audit_control_context_retention.py` to compute the complete-message suffix
  visible to every classifier tokenizer; and
- `adjudicate_retained_control_context.py` to require high-confidence,
  verbatim exclusion evidence in that suffix.

The headline set is the 66-row explicit-evidence intersection. Five rows use
LMSYS-Chat-1M source text and are intentionally not distributed here.
