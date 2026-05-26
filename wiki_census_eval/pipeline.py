from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Set, Tuple

from .artifacts import (
    ArtifactError,
    discover_before_manifests,
    load_case,
    load_case_list_manifests,
)
from .clients import JudgeClient, JudgeClientError, JudgeRequest
from .history import model_strength
from .prompts import build_prompt
from .schema import EvaluationRecord, PipelineErrorRecord
from .storage import EvaluationStore


@dataclass(frozen=True)
class EvaluationConfig:
    before_root: Path
    results_dir: Path
    db_path: Optional[Path] = None
    case_list_path: Optional[Path] = None
    state_fips_filter: Optional[Set[str]] = None
    limit: Optional[int] = None
    case_id: Optional[str] = None
    skip_existing: bool = False
    skip_evaluated_articles: bool = False
    skip_evaluated_by_model: bool = False
    skip_evaluated_by_provider: Optional[str] = None
    skip_evaluated_by_model_name: Optional[str] = None
    skip_evaluated_by_provider_models: Optional[Tuple[Tuple[str, str], ...]] = None
    skip_passed: bool = False
    requested_model: Optional[str] = None
    min_model_strength: Optional[int] = None
    dry_run: bool = False
    save_prompts: bool = False
    max_input_chars: int = 60_000


def run_evaluation(config: EvaluationConfig, client: Optional[JudgeClient] = None) -> dict:
    if not config.dry_run and client is None:
        raise ValueError("client is required unless dry_run is enabled")

    config.results_dir.mkdir(parents=True, exist_ok=True)
    db_path = config.db_path or (config.results_dir / "evaluations.sqlite")
    prompt_dir = config.results_dir / "prompts"
    if config.save_prompts:
        prompt_dir.mkdir(parents=True, exist_ok=True)

    requested_model = _requested_model(config, client)
    requested_model_strength = (
        config.min_model_strength
        if config.min_model_strength is not None
        else model_strength(requested_model)
    )
    provider = client.provider if client is not None else "dry-run"
    skip_provider_model_pairs = _skip_provider_model_pairs(
        config,
        provider=provider,
        requested_model=requested_model,
    )
    store = EvaluationStore(db_path)
    run_id = store.start_run(
        before_root=config.before_root,
        provider=provider,
        model=requested_model,
        requested_model_strength=requested_model_strength,
        dry_run=config.dry_run,
        skip_passed=config.skip_passed,
    )

    counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    processed = 0
    attempted = 0

    try:
        for manifest_path in _iter_manifest_paths(config):
            if config.limit is not None and attempted >= config.limit:
                break
            attempted += 1
            case = None

            try:
                case = load_case(manifest_path)
                if (
                    config.state_fips_filter is not None
                    and case.metadata.state_fips not in config.state_fips_filter
                ):
                    continue
                if config.case_id is not None and case.metadata.case_id != config.case_id:
                    continue
                if config.skip_existing and store.has_evaluation_for_case(
                    case.metadata.case_id
                ):
                    counts["skipped_existing"] += 1
                    continue
                if (
                    config.skip_evaluated_articles
                    and store.has_evaluation_for_article(case.metadata.article)
                ):
                    counts["skipped_evaluated_articles"] += 1
                    continue
                if (
                    skip_provider_model_pairs
                    and _has_evaluation_for_any_provider_model(
                        store,
                        article=case.metadata.article,
                        provider_model_pairs=skip_provider_model_pairs,
                    )
                ):
                    counts["skipped_evaluated_by_model"] += 1
                    continue
                if config.skip_passed:
                    history_match = store.find_passing_match(
                        case_id=case.metadata.case_id,
                        article=case.metadata.article,
                        after_hash=case.metadata.after_hash,
                        requested_model=requested_model,
                        min_model_strength=config.min_model_strength,
                    )
                    if history_match is not None:
                        counts["skipped_passed"] += 1
                        print(
                            "SKIP PASSED "
                            f"{case.metadata.case_id}: after hash already passed "
                            f"with {history_match.model} at {history_match.evaluated_at}"
                        )
                        continue

                prompt = build_prompt(case, max_chars=config.max_input_chars)
                if config.save_prompts:
                    _write_prompt(prompt_dir, case.metadata.case_id, prompt)

                if config.dry_run:
                    print(
                        f"DRY RUN {case.metadata.case_id}: "
                        f"{case.metadata.article} ({len(prompt)} prompt chars)"
                    )
                    counts["dry_run"] += 1
                    processed += 1
                    continue

                assert client is not None
                judge_response = client.judge(
                    JudgeRequest(
                        case_id=case.metadata.case_id,
                        article=case.metadata.article,
                        prompt=prompt,
                    )
                )
                record = EvaluationRecord(
                    evaluated_at=datetime.now(timezone.utc),
                    metadata=case.metadata,
                    judge=judge_response,
                )
                store.insert_evaluation(run_id, record)
                counts[judge_response.result.verdict] += 1
                for issue in judge_response.result.issues:
                    issue_counts[issue.code] += 1
                processed += 1
                print(
                    f"{processed}: {judge_response.result.emoji} "
                    f"{case.metadata.case_id} {judge_response.result.verdict}"
                )
                if judge_response.result.verdict != "pass":
                    issue_codes = [
                        issue.code for issue in judge_response.result.issues
                    ] or ["no_issue_codes"]
                    print(f"  Summary: {judge_response.result.summary}")
                    print(f"  Issue codes: {', '.join(issue_codes)}")
            except JudgeClientError as exc:
                error_record = PipelineErrorRecord(
                    evaluated_at=datetime.now(timezone.utc),
                    case_id=case.metadata.case_id if case is not None else None,
                    article=case.metadata.article if case is not None else None,
                    manifest_path=str(manifest_path),
                    provider=exc.provider,
                    model=exc.model,
                    response_id=exc.response_id,
                    error=str(exc),
                    raw_output_text=exc.raw_output_text,
                    expected_schema=exc.expected_schema,
                )
                store.insert_error(run_id, error_record)
                counts["judge_client_error"] += 1
                print(f"JUDGE ERROR {manifest_path}: {exc}")
                if exc.raw_output_text:
                    print("Raw model output:")
                    print(exc.raw_output_text)
                if exc.expected_schema:
                    print("Expected schema:")
                    print(json.dumps(exc.expected_schema, indent=2, sort_keys=True))
            except ArtifactError as exc:
                error_record = PipelineErrorRecord(
                    evaluated_at=datetime.now(timezone.utc),
                    manifest_path=str(manifest_path),
                    error=str(exc),
                )
                if not config.dry_run:
                    store.insert_error(run_id, error_record)
                counts["pipeline_error"] += 1
                print(f"ERROR {manifest_path}: {exc}")
    finally:
        store.finish_run(
            run_id,
            attempted=attempted,
            processed=processed,
            counts=dict(counts),
            issue_counts=dict(issue_counts),
        )
        store.close()

    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "before_root": str(config.before_root),
        "case_list_path": (
            str(config.case_list_path) if config.case_list_path is not None else None
        ),
        "db_path": str(db_path),
        "run_id": run_id,
        "attempted": attempted,
        "processed": processed,
        "counts": dict(counts),
        "issue_counts": dict(issue_counts),
        "skip_evaluated_articles": config.skip_evaluated_articles,
        "skip_evaluated_by_model": bool(skip_provider_model_pairs),
        "skip_evaluated_by_provider": (
            skip_provider_model_pairs[0][0]
            if len(skip_provider_model_pairs) == 1
            else None
        ),
        "skip_evaluated_by_model_name": (
            skip_provider_model_pairs[0][1]
            if len(skip_provider_model_pairs) == 1
            else None
        ),
        "skip_evaluated_by_provider_models": [
            {"provider": pair[0], "model": pair[1]}
            for pair in skip_provider_model_pairs
        ],
        "skip_passed": config.skip_passed,
        "requested_model": requested_model,
        "requested_model_strength": requested_model_strength,
        "dry_run": config.dry_run,
    }
    return summary


def _skip_provider_model_pairs(
    config: EvaluationConfig,
    *,
    provider: str,
    requested_model: str,
) -> Tuple[Tuple[str, str], ...]:
    pairs = list(config.skip_evaluated_by_provider_models or ())
    if config.skip_evaluated_by_model:
        pairs.append(
            (
                config.skip_evaluated_by_provider or provider,
                config.skip_evaluated_by_model_name or requested_model,
            )
        )
    return tuple(dict.fromkeys(pairs))


def _has_evaluation_for_any_provider_model(
    store: EvaluationStore,
    *,
    article: str,
    provider_model_pairs: Sequence[Tuple[str, str]],
) -> bool:
    return any(
        store.has_evaluation_for_article_by_model(
            article=article,
            provider=provider,
            model=model,
        )
        for provider, model in provider_model_pairs
    )


def _write_prompt(prompt_dir: Path, case_id: str, prompt: str) -> None:
    path = prompt_dir / f"{_safe_filename(case_id)}.txt"
    path.write_text(prompt, encoding="utf-8")


def _safe_filename(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def _iter_manifest_paths(config: EvaluationConfig):
    if config.case_list_path is not None:
        yield from load_case_list_manifests(config.case_list_path, config.before_root)
        return
    yield from discover_before_manifests(
        config.before_root,
        state_fips_filter=config.state_fips_filter,
    )


def _requested_model(config: EvaluationConfig, client: Optional[JudgeClient]) -> str:
    if config.requested_model:
        return config.requested_model
    if client is not None:
        return client.model
    return "dry-run"
