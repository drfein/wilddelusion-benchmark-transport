# Reproducing

## Environments

The orchestration and analysis code requires Python 3.11 or newer. GPU stages
require Linux, CUDA, PyTorch, Transformers, and vLLM. The original completed
runs reported CUDA 13.0 and PyTorch 2.11 for the core greedy generation; exact
vLLM package metadata was not preserved, which is a limitation rather than a
version guessed after the fact.

A practical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
# Install a vLLM/PyTorch build compatible with the host CUDA stack.
```

[`requirements-analysis.lock`](../requirements-analysis.lock) records the exact
CPU/orchestration environment used for the clean-install release test. It does
not pretend to lock PyTorch or vLLM: those must match the CUDA host, and the
historical vLLM package version was not preserved.

## Inputs

`wd-reproduce run all` fetches:

- `danielfein/WildDelusionCombined@f547346...`;
- `w-is-h/psychosis-bench@73966f9...`; and
- the arXiv `2606.00975v1` source bundle.

It installs the immutable generated controls and implicit/explicit pairs from
`frozen/`.

Exact recognition-control regeneration also needs the locally licensed files:

```text
strict_nonsincere_controls.jsonl
high_confidence_strict_controls.jsonl
human_confirmed_controls.jsonl
human_verifier_disagreements.jsonl
```

Set `WD_NATURAL_CONTROL_DIR` to the directory containing them. They can be
rebuilt with [build_natural_near_miss_controls.py](../scripts/build_natural_near_miss_controls.py)
when the two verifier-output source pools are available.

## Cost controls

Stages have one of four kinds: `free`, `network`, `api`, or `gpu`.
The runner will not enter `api` or `gpu` stages without explicit flags.

```bash
wd-reproduce run all --dry-run --no-resume --allow-api --allow-gpu
wd-reproduce run all --allow-api --allow-gpu
```

No cost estimate is hardcoded because API pricing and rented-GPU rates change.
The dry run exposes every model, row count, and judge call before execution.

## Expected cardinalities

| Artifact | Expected |
|---|---:|
| Real endpoints | 522 |
| Source-conversation clusters | 321 |
| Validated generated controls | 101 |
| Recognition inputs | 932 |
| Explicit-evidence natural controls | 66 across 65 clusters |
| Validated real explicit/implicit pairs | 327 |
| Released synthetic scenario pairs | 8 |
| Psychogenic target models | 3 |
| Psychogenic seeds | 3 |
| Selected claim families | 5 |

## Deterministic contracts

The runner refuses to continue if any deterministic preprocessing artifact
differs from the finalized run:

| Artifact | SHA-256 |
|---|---|
| Theme-augmented 522-row source | `75bf9c5cfaefedacebbabac2d1068298375e53b9dd5fef5577f93ce948f34e09` |
| Canonical full-history cohort | `ae854680ea92e58e3bebac5bddb5395d0d756670ec657bb4f8a71b614a84e07a` |
| Final-four-user-turn cohort | `4a2e5a2b5eccea7a1c029926a63c9ace8bb3ec090300adbb6a2db2199aeca63b` |
| Lost-in-Delusion model inputs | `18aff72537297c5d2d6bd23a9d95617bab4e08d442906ed3a886b5b5fb24b38a` |
| Explicit/implicit paired inputs | `5a1c53673fe7563f5bc2626c5fcb150175e9bcc93844c829e10034ebf7c04a6c` |
| Direct-recognition inputs | `543ba37348c5b17f61d0176c16ffedadcb0a5b3c61381bb3ddaa4bc125557d55` |

API adjudication and stochastic decoding are not falsely treated as bitwise
deterministic. Their outputs instead pass schema, completeness, provenance,
adversarial-parse, and claim-specific statistical gates.

## Partial execution

Every target includes dependencies:

```bash
wd-reproduce run analyze_theme_coverage --allow-api
wd-reproduce run analyze_visible_harm --allow-api --allow-gpu
wd-reproduce run analyze_psychogenic --allow-api --allow-gpu
```

JSONL producers canonicalize durable rows and resume. The runner separately
records stage output hashes in `.repro/state/`. A cached stage is reused only
when its outputs, dependency records, and the SHA-256 fingerprint of all
published code, prompts, configuration, and frozen inputs still match.

## Zero-cost audit

```bash
wd-reproduce verify
pytest
```

This checks the frozen registry, exact-layout copy, source-gate hashes, distinct
evidence families, and every claim-specific verifier.
