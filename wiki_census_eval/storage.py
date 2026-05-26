from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .history import HistoryMatch, model_strength
from .schema import EvaluationRecord, PipelineErrorRecord


class EvaluationStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(db_path), timeout=30.0)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def close(self) -> None:
        self._connection.close()

    def start_run(
        self,
        *,
        before_root: Path,
        provider: str,
        model: str,
        requested_model_strength: int,
        dry_run: bool,
        skip_passed: bool,
    ) -> int:
        cursor = self._connection.execute(
            """
            INSERT INTO runs (
              started_at, before_root, provider, model, model_strength,
              dry_run, skip_passed
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now_iso(),
                str(before_root),
                provider,
                model,
                requested_model_strength,
                int(dry_run),
                int(skip_passed),
            ),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def finish_run(
        self,
        run_id: int,
        *,
        attempted: int,
        processed: int,
        counts: dict,
        issue_counts: dict,
    ) -> None:
        self._connection.execute(
            """
            UPDATE runs
            SET completed_at = ?, attempted = ?, processed = ?,
                counts_json = ?, issue_counts_json = ?
            WHERE id = ?
            """,
            (
                _now_iso(),
                attempted,
                processed,
                json.dumps(counts, sort_keys=True),
                json.dumps(issue_counts, sort_keys=True),
                run_id,
            ),
        )
        self._connection.commit()

    def has_evaluation_for_case(self, case_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM evaluations WHERE case_id = ? LIMIT 1",
            (case_id,),
        ).fetchone()
        return row is not None

    def has_evaluation_for_article(self, article: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM evaluations WHERE article = ? LIMIT 1",
            (article,),
        ).fetchone()
        return row is not None

    def has_evaluation_for_article_by_model(
        self,
        *,
        article: str,
        provider: str,
        model: str,
    ) -> bool:
        row = self._connection.execute(
            """
            SELECT 1 FROM evaluations
            WHERE article = ?
              AND provider = ?
              AND model = ?
            LIMIT 1
            """,
            (article, provider, model),
        ).fetchone()
        return row is not None

    def find_passing_match(
        self,
        *,
        case_id: str,
        article: str,
        after_hash: Optional[str],
        requested_model: str,
        min_model_strength: Optional[int] = None,
    ) -> Optional[HistoryMatch]:
        if not after_hash:
            return None
        requested_strength = (
            min_model_strength
            if min_model_strength is not None
            else model_strength(requested_model)
        )
        row = self._connection.execute(
            """
            SELECT case_id, article, evaluated_at, verdict, model,
                   model_strength, after_hash
            FROM evaluations
            WHERE case_id = ?
              AND article = ?
              AND after_hash = ?
              AND verdict = 'pass'
              AND model_strength >= ?
            ORDER BY evaluated_at DESC, id DESC
            LIMIT 1
            """,
            (case_id, article, after_hash, requested_strength),
        ).fetchone()
        if row is None:
            return None
        return HistoryMatch(
            case_id=row["case_id"],
            article=row["article"],
            evaluated_at=row["evaluated_at"],
            verdict=row["verdict"],
            model=row["model"],
            model_strength=int(row["model_strength"]),
            requested_model_strength=requested_strength,
            after_hash=row["after_hash"],
        )

    def insert_evaluation(self, run_id: int, record: EvaluationRecord) -> int:
        metadata = record.metadata
        judge = record.judge
        result = judge.result
        cursor = self._connection.execute(
            """
            INSERT INTO evaluations (
              run_id, evaluated_at, case_id, article, location_kind, state_fips,
              target_fips, before_manifest_path, before_section_path,
              after_manifest_path, after_section_path, before_hash, after_hash,
              freshness_status, freshness_reason, current_has_demographics_section,
              provider, model, model_strength, response_id, raw_output_text,
              verdict, summary, confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                record.evaluated_at.isoformat(),
                metadata.case_id,
                metadata.article,
                metadata.location_kind,
                metadata.state_fips,
                metadata.target_fips,
                metadata.before_manifest_path,
                metadata.before_section_path,
                metadata.after_manifest_path,
                metadata.after_section_path,
                metadata.before_hash,
                metadata.after_hash,
                metadata.freshness_status,
                metadata.freshness_reason,
                _optional_bool(metadata.current_has_demographics_section),
                judge.provider,
                judge.model,
                model_strength(judge.model),
                judge.response_id,
                judge.raw_output_text,
                result.verdict,
                result.summary,
                result.confidence,
            ),
        )
        evaluation_id = int(cursor.lastrowid)
        for issue in result.issues:
            self._connection.execute(
                """
                INSERT INTO issues (
                  evaluation_id, code, severity, explanation, evidence
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    issue.code,
                    issue.severity,
                    issue.explanation,
                    issue.evidence,
                ),
            )
        self._connection.commit()
        return evaluation_id

    def insert_error(self, run_id: int, record: PipelineErrorRecord) -> int:
        cursor = self._connection.execute(
            """
            INSERT INTO pipeline_errors (
              run_id, evaluated_at, case_id, article, manifest_path, provider,
              model, response_id, error, raw_output_text, expected_schema_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                record.evaluated_at.isoformat(),
                record.case_id,
                record.article,
                record.manifest_path,
                record.provider,
                record.model,
                record.response_id,
                record.error,
                record.raw_output_text,
                (
                    json.dumps(record.expected_schema, sort_keys=True)
                    if record.expected_schema is not None
                    else None
                ),
            ),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              started_at TEXT NOT NULL,
              completed_at TEXT,
              before_root TEXT NOT NULL,
              provider TEXT NOT NULL,
              model TEXT NOT NULL,
              model_strength INTEGER NOT NULL,
              dry_run INTEGER NOT NULL,
              skip_passed INTEGER NOT NULL,
              attempted INTEGER,
              processed INTEGER,
              counts_json TEXT,
              issue_counts_json TEXT
            );

            CREATE TABLE IF NOT EXISTS evaluations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
              evaluated_at TEXT NOT NULL,
              case_id TEXT NOT NULL,
              article TEXT NOT NULL,
              location_kind TEXT,
              state_fips TEXT,
              target_fips TEXT,
              before_manifest_path TEXT NOT NULL,
              before_section_path TEXT NOT NULL,
              after_manifest_path TEXT NOT NULL,
              after_section_path TEXT NOT NULL,
              before_hash TEXT,
              after_hash TEXT,
              freshness_status TEXT,
              freshness_reason TEXT,
              current_has_demographics_section INTEGER,
              provider TEXT NOT NULL,
              model TEXT NOT NULL,
              model_strength INTEGER NOT NULL,
              response_id TEXT,
              raw_output_text TEXT,
              verdict TEXT NOT NULL,
              summary TEXT NOT NULL,
              confidence REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS issues (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              evaluation_id INTEGER NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
              code TEXT NOT NULL,
              severity TEXT NOT NULL,
              explanation TEXT NOT NULL,
              evidence TEXT
            );

            CREATE TABLE IF NOT EXISTS pipeline_errors (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
              evaluated_at TEXT NOT NULL,
              case_id TEXT,
              article TEXT,
              manifest_path TEXT,
              provider TEXT,
              model TEXT,
              response_id TEXT,
              error TEXT NOT NULL,
              raw_output_text TEXT,
              expected_schema_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_evaluations_case_after
              ON evaluations(case_id, after_hash);
            CREATE INDEX IF NOT EXISTS idx_evaluations_pass_strength
              ON evaluations(case_id, article, after_hash, verdict, model_strength);
            CREATE INDEX IF NOT EXISTS idx_evaluations_article
              ON evaluations(article);
            CREATE INDEX IF NOT EXISTS idx_evaluations_model_time
              ON evaluations(model, evaluated_at);
            CREATE INDEX IF NOT EXISTS idx_issues_code
              ON issues(code);
            CREATE INDEX IF NOT EXISTS idx_errors_case
              ON pipeline_errors(case_id);
            """
        )
        self._connection.commit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _optional_bool(value: Optional[bool]) -> Optional[int]:
    if value is None:
        return None
    return int(value)
