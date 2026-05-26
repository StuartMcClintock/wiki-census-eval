from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from .case_lists import write_case_list_from_evaluations
from .clients import (
    AnthropicCliJudgeClient,
    AnthropicJudgeClient,
    CodexJudgeClient,
    OpenAIJudgeClient,
)
from .pipeline import EvaluationConfig, run_evaluation
from .states import parse_state_filters


ANTHROPIC_API_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_MODEL_ALIASES_BY_PROVIDER = {
    "anthropic": {
        "sonnet": ANTHROPIC_API_DEFAULT_MODEL,
    },
}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="wiki-census-eval",
        description="Evaluate proposed Wikipedia census demographics edits.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Run the evaluation pipeline.")
    evaluate.add_argument("--before-root", type=Path, default=Path("precomputed-before"))
    evaluate.add_argument("--results-dir", type=Path, default=Path("results"))
    evaluate.add_argument(
        "--db-path",
        type=Path,
        help="SQLite database path. Defaults to <results-dir>/evaluations.sqlite.",
    )
    evaluate.add_argument(
        "--provider",
        choices=("openai", "anthropic", "codex", "anthropic-cli"),
        default="openai",
        help="Model runner to use for judging.",
    )
    evaluate.add_argument("--model")
    evaluate.add_argument("--codex-bin", default="codex")
    evaluate.add_argument("--claude-bin", default="claude")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--case-id")
    evaluate.add_argument(
        "--case-list",
        type=Path,
        help=(
            "Evaluate only cases listed in a handoff JSON file, such as "
            "wikipedia-census-cyrus .poster-runs/refresh-stale-cache/latest.json."
        ),
    )
    evaluate.add_argument(
        "--states",
        action="append",
        help=(
            "Restrict evaluation to one or more states. Accepts postal "
            "abbreviations or FIPS codes, comma-separated or repeated "
            "(e.g. --states AL,GA or --states 01 --states 13)."
        ),
    )
    evaluate.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip any case id already present in SQLite, regardless of result.",
    )
    evaluate.add_argument(
        "--skip-evaluated-articles",
        action="store_true",
        help="Skip any article title already present in SQLite, regardless of result.",
    )
    evaluate.add_argument(
        "--skip-evaluated-by-model",
        action="store_true",
        help=(
            "Skip articles already evaluated by a specific provider/model. "
            "Defaults to the current --provider and selected --model."
        ),
    )
    evaluate.add_argument(
        "--skip-evaluated-by-provider",
        help="Provider to use with --skip-evaluated-by-model.",
    )
    evaluate.add_argument(
        "--skip-evaluated-by-model-name",
        help="Model name to use with --skip-evaluated-by-model.",
    )
    evaluate.add_argument(
        "--skip-evaluated-by-provider-model",
        action="append",
        help=(
            "Skip articles already evaluated by a provider/model pair, formatted "
            "as provider:model. Can be repeated and implies --skip-evaluated-by-model."
        ),
    )
    evaluate.add_argument(
        "--skip-passed",
        action="store_true",
        help=(
            "Skip cases whose current after-hash already has a pass from an "
            "equal-or-stronger model in SQLite."
        ),
    )
    evaluate.add_argument(
        "--min-model-strength",
        type=int,
        help=(
            "Override the required model strength for --skip-passed. "
            "Higher means stricter; defaults to the selected model's strength."
        ),
    )
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.add_argument("--save-prompts", action="store_true")
    evaluate.add_argument("--max-input-chars", type=int, default=60_000)
    evaluate.add_argument("--max-output-tokens", type=int, default=1200)
    evaluate.add_argument("--timeout", type=float)
    evaluate.add_argument(
        "--codex-limit-retries",
        type=int,
        default=5,
        help=(
            "Number of one-case retries when the Codex CLI reports a usage "
            "or rate limit. Use 0 to disable."
        ),
    )
    evaluate.add_argument(
        "--codex-limit-retry-delay",
        type=float,
        default=3600.0,
        help="Seconds to wait before retrying a Codex usage/rate-limit failure.",
    )
    evaluate.add_argument(
        "--wait-for-anthropic-limit-reset",
        action="store_true",
        help=(
            "When Anthropic CLI reports a usage limit, wait until the parsed "
            "reset time and retry. Falls back to --anthropic-limit-retry-delay "
            "when no reset time is reported."
        ),
    )
    evaluate.add_argument(
        "--anthropic-limit-retry-delay",
        type=float,
        default=3600.0,
        help=(
            "Fallback seconds to wait before retrying an Anthropic CLI usage "
            "limit when no reset time can be parsed."
        ),
    )

    case_list = subparsers.add_parser(
        "case-list",
        help="Build evaluator case-list JSON files.",
    )
    case_list_subparsers = case_list.add_subparsers(
        dest="case_list_command",
        required=True,
    )
    from_evaluations = case_list_subparsers.add_parser(
        "from-evaluations",
        help="Build a case list from prior SQLite evaluations.",
    )
    from_evaluations.add_argument("--results-dir", type=Path, default=Path("results"))
    from_evaluations.add_argument(
        "--db-path",
        type=Path,
        help="SQLite database path. Defaults to <results-dir>/evaluations.sqlite.",
    )
    from_evaluations.add_argument("--provider")
    from_evaluations.add_argument("--model")
    from_evaluations.add_argument(
        "--verdict",
        action="append",
        help="Include only this verdict. Can be comma-separated or repeated.",
    )
    from_evaluations.add_argument(
        "--exclude-verdict",
        action="append",
        help="Exclude this verdict. Can be comma-separated or repeated.",
    )
    from_evaluations.add_argument(
        "--include-historical",
        action="store_true",
        help=(
            "Include all matching historical rows. By default, only the latest "
            "evaluation per case id for the selected provider/model is considered."
        ),
    )
    from_evaluations.add_argument("--limit", type=int)
    from_evaluations.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "evaluate":
        model = _resolve_model_alias(
            args.provider,
            args.model or _default_model_for_provider(args.provider),
        )
        try:
            state_fips_filter = parse_state_filters(args.states)
            skip_provider_model_pairs = _parse_provider_model_pairs(
                args.skip_evaluated_by_provider_model
            )
        except ValueError as exc:
            parser.error(str(exc))
        config = EvaluationConfig(
            before_root=args.before_root,
            results_dir=args.results_dir,
            db_path=args.db_path,
            case_list_path=args.case_list,
            state_fips_filter=state_fips_filter or None,
            limit=args.limit,
            case_id=args.case_id,
            skip_existing=args.skip_existing,
            skip_evaluated_articles=args.skip_evaluated_articles,
            skip_evaluated_by_model=args.skip_evaluated_by_model,
            skip_evaluated_by_provider=args.skip_evaluated_by_provider,
            skip_evaluated_by_model_name=_resolve_model_alias(
                args.skip_evaluated_by_provider or args.provider,
                args.skip_evaluated_by_model_name,
            ),
            skip_evaluated_by_provider_models=skip_provider_model_pairs or None,
            skip_passed=args.skip_passed,
            requested_model=model,
            min_model_strength=args.min_model_strength,
            dry_run=args.dry_run,
            save_prompts=args.save_prompts,
            max_input_chars=args.max_input_chars,
        )
        client = None
        if not args.dry_run:
            if args.provider == "anthropic":
                client = AnthropicJudgeClient(
                    model=model,
                    max_output_tokens=args.max_output_tokens,
                    timeout=args.timeout,
                )
            elif args.provider == "anthropic-cli":
                client = AnthropicCliJudgeClient(
                    model=model,
                    claude_bin=args.claude_bin,
                    timeout=args.timeout,
                    cwd=Path.cwd(),
                    wait_for_limit_reset=args.wait_for_anthropic_limit_reset,
                    limit_retry_delay_seconds=args.anthropic_limit_retry_delay,
                )
            elif args.provider == "codex":
                client = CodexJudgeClient(
                    model=model,
                    codex_bin=args.codex_bin,
                    timeout=args.timeout,
                    cwd=Path.cwd(),
                    limit_retry_attempts=args.codex_limit_retries,
                    limit_retry_delay_seconds=args.codex_limit_retry_delay,
                )
            else:
                client = OpenAIJudgeClient(
                    model=model,
                    max_output_tokens=args.max_output_tokens,
                    timeout=args.timeout,
                )
        summary = run_evaluation(config, client=client)
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "case-list":
        if args.case_list_command == "from-evaluations":
            db_path = args.db_path or (args.results_dir / "evaluations.sqlite")
            if not db_path.exists():
                parser.error(f"SQLite database does not exist: {db_path}")
            try:
                payload = write_case_list_from_evaluations(
                    db_path=db_path,
                    output_path=args.output,
                    provider=args.provider,
                    model=_resolve_model_alias(args.provider, args.model),
                    verdicts=_split_repeated_csv(args.verdict),
                    exclude_verdicts=_split_repeated_csv(args.exclude_verdict),
                    include_historical=args.include_historical,
                    limit=args.limit,
                )
            except ValueError as exc:
                parser.error(str(exc))
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "cases": len(payload["refreshed"]),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )


def _default_model_for_provider(provider: str) -> str:
    if provider in {"anthropic", "anthropic-cli"}:
        model = (
            os.getenv("ANTHROPIC_EVAL_MODEL")
            or os.getenv("CLAUDE_EVAL_MODEL")
            or ("sonnet" if provider == "anthropic" else "sonnet")
        )
        return _resolve_model_alias(provider, model) or model
    return os.getenv("OPENAI_EVAL_MODEL", "gpt-4.1-mini")


def _resolve_model_alias(provider: Optional[str], model: Optional[str]) -> Optional[str]:
    if provider is None or model is None:
        return model
    aliases = _MODEL_ALIASES_BY_PROVIDER.get(provider)
    if not aliases:
        return model
    return aliases.get(model.strip().lower(), model)


def _split_repeated_csv(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    parsed: List[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


def _parse_provider_model_pairs(
    values: Optional[List[str]],
) -> List[Tuple[str, str]]:
    pairs = []
    for value in values or []:
        provider, separator, model = value.partition(":")
        if not separator or not provider.strip() or not model.strip():
            raise ValueError(
                "--skip-evaluated-by-provider-model must be formatted as "
                "provider:model"
            )
        provider = provider.strip()
        model = model.strip()
        pairs.append((provider, _resolve_model_alias(provider, model) or model))
    return pairs


if __name__ == "__main__":
    main()
