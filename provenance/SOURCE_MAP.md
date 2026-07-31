# Source map

| Result family | Example construction / validation | Model execution | Analysis and gate |
|---|---|---|---|
| Visible Mod+Harm coverage | `full_history_522/label_mod_harm_equivalent.py`; `label_mod_harm_visible_context.py` | `run_vllm_models.py`; `judge_responses_vllm.py` | `analyze_visible_context_mod_harm.py`; public primary-only wrapper `scripts/analyze_visible_primary.py` |
| Control realism | `scripts/build_natural_near_miss_controls.py`; `adjudicate_natural_controls.py`; `audit_control_context_retention.py`; `adjudicate_retained_control_context.py` | `run_vllm_classifier.py` | `analyze_control_realism.py` |
| Implicit wording | `generate_pairs.py`; `validate_pairs.py`; `repair_pairs.py`; `merge_pair_rounds.py` | `run_psychosis_bench_stochastic.py`; `run_psychosis_bench_local_contrast.py`; `run_real_pairs_stochastic.py`; `judge_dcs_exact.py` | `analyze_psychogenic_robustness.py` |
| Ontology coverage | `adjudicate_paper_theme_fit.py` rubrics A/B | OpenAI Responses API | `analyze_theme_coverage.py` |
| Recognition model gap | Same dual theme adjudication; exact published classifier prompt | `run_vllm_classifier.py` | `analyze_recognition_model_gap.py` |
| Final selection | All five claim gates | n/a | `build_claim_registry.py`; `verify_abstract_claims.py` |
