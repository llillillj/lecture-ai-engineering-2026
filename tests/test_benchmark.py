from types import SimpleNamespace

from app.benchmark import _extract_token_counts


def test_extract_token_counts_uses_reported_thought_tokens():
    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=20,
        thoughts_token_count=7,
        total_token_count=37,
    )

    assert _extract_token_counts(usage) == (10, 20, 7, 37)


def test_extract_token_counts_recovers_missing_thought_tokens():
    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=20,
        thoughts_token_count=0,
        total_token_count=37,
        tool_use_prompt_token_count=0,
    )

    assert _extract_token_counts(usage) == (10, 20, 7, 37)


def test_extract_token_counts_does_not_infer_when_total_is_unavailable():
    usage = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=20,
        thoughts_token_count=None,
        total_token_count=None,
    )

    assert _extract_token_counts(usage) == (10, 20, 0, 0)
