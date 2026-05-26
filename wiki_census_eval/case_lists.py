from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence


def write_case_list_from_evaluations(
    *,
    db_path: Path,
    output_path: Path,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    verdicts: Optional[Iterable[str]] = None,
    exclude_verdicts: Optional[Iterable[str]] = None,
    include_historical: bool = False,
    limit: Optional[int] = None,
) -> dict:
    cases = select_cases_from_evaluations(
        db_path=db_path,
        provider=provider,
        model=model,
        verdicts=verdicts,
        exclude_verdicts=exclude_verdicts,
        include_historical=include_historical,
        limit=limit,
    )
    payload = {
        "version": 1,
        "source": "wiki-census-eval evaluations",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "filters": {
            "provider": provider,
            "model": model,
            "verdicts": sorted(_normalize_set(verdicts)),
            "exclude_verdicts": sorted(_normalize_set(exclude_verdicts)),
            "include_historical": include_historical,
            "limit": limit,
        },
        "refreshed": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def select_cases_from_evaluations(
    *,
    db_path: Path,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    verdicts: Optional[Iterable[str]] = None,
    exclude_verdicts: Optional[Iterable[str]] = None,
    include_historical: bool = False,
    limit: Optional[int] = None,
) -> list:
    include = _normalize_set(verdicts)
    exclude = _normalize_set(exclude_verdicts)
    if include and exclude:
        overlap = include & exclude
        if overlap:
            raise ValueError(
                "verdict filters cannot both include and exclude: "
                + ", ".join(sorted(overlap))
            )

    rows = _fetch_evaluation_rows(db_path, provider=provider, model=model)
    cases = []
    seen_case_ids = set()
    for row in rows:
        case_id = row["case_id"]
        if not include_historical:
            if case_id in seen_case_ids:
                continue
            seen_case_ids.add(case_id)

        verdict = row["verdict"]
        if include and verdict not in include:
            continue
        if verdict in exclude:
            continue

        cases.append(_case_from_row(row))
        if limit is not None and len(cases) >= limit:
            break
    return cases


def _fetch_evaluation_rows(
    db_path: Path,
    *,
    provider: Optional[str],
    model: Optional[str],
) -> Sequence[sqlite3.Row]:
    clauses = []
    params = []
    if provider:
        clauses.append("provider = ?")
        params.append(provider)
    if model:
        clauses.append("model = ?")
        params.append(model)

    where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(
            f"""
            SELECT
              id, evaluated_at, case_id, article, location_kind, state_fips,
              target_fips, before_manifest_path, before_section_path,
              after_manifest_path, after_section_path, before_hash, after_hash,
              freshness_status, freshness_reason, provider, model, verdict,
              summary, confidence
            FROM evaluations
            {where_sql}
            ORDER BY evaluated_at DESC, id DESC
            """,
            params,
        ).fetchall()
    finally:
        connection.close()


def _case_from_row(row: sqlite3.Row) -> dict:
    return {
        "evaluation_id": row["id"],
        "evaluated_at": row["evaluated_at"],
        "case_id": row["case_id"],
        "article": row["article"],
        "location_kind": row["location_kind"],
        "state_fips": row["state_fips"],
        "target_fips": row["target_fips"],
        "before_manifest_path": row["before_manifest_path"],
        "before_section_path": row["before_section_path"],
        "after_manifest_path": row["after_manifest_path"],
        "after_section_path": row["after_section_path"],
        "before_hash": row["before_hash"],
        "after_hash": row["after_hash"],
        "freshness_status": row["freshness_status"],
        "freshness_reason": row["freshness_reason"],
        "provider": row["provider"],
        "model": row["model"],
        "verdict": row["verdict"],
        "summary": row["summary"],
        "confidence": row["confidence"],
    }


def _normalize_set(values: Optional[Iterable[str]]) -> set:
    return {
        value.strip()
        for value in (values or [])
        if isinstance(value, str) and value.strip()
    }
