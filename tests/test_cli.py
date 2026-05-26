from wiki_census_eval.cli import (
    _default_model_for_provider,
    _parse_provider_model_pairs,
    _resolve_model_alias,
)


def test_anthropic_model_alias_resolves_to_api_model():
    assert (
        _resolve_model_alias("anthropic", "sonnet")
        == "claude-sonnet-4-20250514"
    )


def test_anthropic_cli_model_alias_stays_cli_model():
    assert _resolve_model_alias("anthropic-cli", "sonnet") == "sonnet"


def test_skip_provider_model_pairs_resolve_provider_scoped_aliases():
    assert _parse_provider_model_pairs(
        ["anthropic-cli:sonnet", "anthropic:sonnet"]
    ) == [
        ("anthropic-cli", "sonnet"),
        ("anthropic", "claude-sonnet-4-20250514"),
    ]


def test_default_anthropic_model_uses_canonical_api_model(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_EVAL_MODEL", raising=False)
    monkeypatch.delenv("CLAUDE_EVAL_MODEL", raising=False)

    assert _default_model_for_provider("anthropic") == "claude-sonnet-4-20250514"
