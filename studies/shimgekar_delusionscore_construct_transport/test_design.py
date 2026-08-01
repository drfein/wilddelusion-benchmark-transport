from design import N_FOLDS, fold_for_group


def test_fold_is_deterministic_and_grouped():
    assert fold_for_group("conversation-a") == fold_for_group("conversation-a")
    assert 0 <= fold_for_group("conversation-a") < N_FOLDS


def test_different_conditions_share_fold():
    conversation = "same-source-conversation"
    assert fold_for_group(conversation) == fold_for_group(conversation)
