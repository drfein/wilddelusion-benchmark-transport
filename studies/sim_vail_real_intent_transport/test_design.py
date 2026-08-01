from design import CUE_PAIRS, PRIMARY_DIMENSION, template_index


def test_cues_are_pairwise_word_matched_and_distinct() -> None:
    for intent, pairs in CUE_PAIRS.items():
        assert len(pairs) == 3
        for target, control in pairs:
            assert target != control
            assert len(target.split()) == len(control.split())


def test_template_assignment_is_stable_and_in_range() -> None:
    for intent in CUE_PAIRS:
        first = template_index("source::conversation", intent)
        second = template_index("source::conversation", intent)
        assert first == second
        assert 0 <= first < 3


def test_each_intent_has_one_primary_dimension() -> None:
    assert set(PRIMARY_DIMENSION) == set(CUE_PAIRS)

