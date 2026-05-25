from pathlib import Path
import json
import sqlite3

from wiki_census_eval.clients import JudgeRequest
from wiki_census_eval.clients import JudgeClientError
from wiki_census_eval.pipeline import EvaluationConfig, run_evaluation
from wiki_census_eval.schema import JudgeResponse, JudgeResult


class FakeJudgeClient:
    provider = "fake"
    model = "fake-model"

    def judge(self, request: JudgeRequest) -> JudgeResponse:
        return JudgeResponse(
            provider=self.provider,
            model=self.model,
            response_id="fake-response",
            raw_output_text=None,
            result=JudgeResult(
                article=request.article,
                verdict="pass",
                summary=f"**{request.article}** looks safe.",
                issues=[],
                confidence=0.9,
            ),
        )


class BadJudgeClient:
    provider = "fake"
    model = "bad-model"

    def judge(self, request: JudgeRequest) -> JudgeResponse:
        raise JudgeClientError(
            "fake structured output parse failure",
            provider=self.provider,
            model=self.model,
            response_id="bad-response",
            raw_output_text="not json",
            expected_schema={"type": "object"},
        )


class WarningJudgeClient:
    provider = "fake"
    model = "fake-model"

    def judge(self, request: JudgeRequest) -> JudgeResponse:
        return JudgeResponse(
            provider=self.provider,
            model=self.model,
            response_id="fake-warning",
            raw_output_text=None,
            result=JudgeResult(
                article=request.article,
                verdict="warning",
                summary=f"**{request.article}** needs review for a heading issue.",
                issues=[
                    {
                        "code": "bad_heading_level",
                        "severity": "warning",
                        "explanation": "A heading level may be wrong.",
                    }
                ],
                confidence=0.7,
            ),
        )


def test_run_evaluation_writes_jsonl_and_summary(tmp_path: Path):
    before_root = _write_fixture(tmp_path)
    results_dir = tmp_path / "results"

    summary = run_evaluation(
        EvaluationConfig(
            before_root=before_root,
            results_dir=results_dir,
            limit=1,
            save_prompts=True,
        ),
        client=FakeJudgeClient(),
    )

    assert summary["processed"] == 1
    assert summary["counts"]["pass"] == 1
    db_path = results_dir / "evaluations.sqlite"
    assert db_path.exists()
    rows = _fetch_all(db_path, "SELECT case_id, verdict FROM evaluations")
    assert rows == [
        {
            "case_id": "municipality/01/00100/Sample__Alabama",
            "verdict": "pass",
        }
    ]
    run = _fetch_one(db_path, "SELECT attempted, processed, counts_json FROM runs")
    assert run["attempted"] == 1
    assert run["processed"] == 1
    assert json.loads(run["counts_json"]) == {"pass": 1}
    assert list((results_dir / "prompts").glob("*.txt"))


def test_warning_prints_summary_and_issue_codes(tmp_path: Path, capsys):
    before_root = _write_fixture(tmp_path)
    results_dir = tmp_path / "results"

    summary = run_evaluation(
        EvaluationConfig(
            before_root=before_root,
            results_dir=results_dir,
            limit=1,
        ),
        client=WarningJudgeClient(),
    )

    captured = capsys.readouterr()
    assert summary["counts"]["warning"] == 1
    assert "Summary: **Sample,_Alabama** needs review for a heading issue." in captured.out
    assert "Issue codes: bad_heading_level" in captured.out


def test_run_evaluation_filters_by_state(tmp_path: Path):
    _write_fixture(tmp_path, state_fips="01", target_fips="00100", slug="Alabama")
    before_root = _write_fixture(
        tmp_path,
        state_fips="13",
        target_fips="00100",
        slug="Georgia",
        article="Sample,_Georgia",
    )
    results_dir = tmp_path / "results"

    summary = run_evaluation(
        EvaluationConfig(
            before_root=before_root,
            results_dir=results_dir,
            state_fips_filter={"13"},
        ),
        client=FakeJudgeClient(),
    )

    rows = _fetch_all(
        results_dir / "evaluations.sqlite",
        "SELECT article, state_fips FROM evaluations",
    )
    assert summary["processed"] == 1
    assert rows == [{"article": "Sample,_Georgia", "state_fips": "13"}]


def test_run_evaluation_records_judge_client_diagnostics(tmp_path: Path):
    before_root = _write_fixture(tmp_path)
    results_dir = tmp_path / "results"

    summary = run_evaluation(
        EvaluationConfig(
            before_root=before_root,
            results_dir=results_dir,
            limit=1,
        ),
        client=BadJudgeClient(),
    )

    assert summary["counts"]["judge_client_error"] == 1
    record = _fetch_one(results_dir / "evaluations.sqlite", "SELECT * FROM pipeline_errors")
    assert record["case_id"] == "municipality/01/00100/Sample__Alabama"
    assert record["article"] == "Sample,_Alabama"
    assert record["provider"] == "fake"
    assert record["model"] == "bad-model"
    assert record["response_id"] == "bad-response"
    assert record["raw_output_text"] == "not json"
    assert json.loads(record["expected_schema_json"]) == {"type": "object"}


def test_skip_passed_reuses_equal_or_stronger_model_for_same_after_hash(tmp_path: Path):
    before_root = _write_fixture(tmp_path)
    results_dir = tmp_path / "results"

    first = run_evaluation(
        EvaluationConfig(
            before_root=before_root,
            results_dir=results_dir,
            limit=1,
            requested_model="fake-model",
        ),
        client=FakeJudgeClient(),
    )
    second = run_evaluation(
        EvaluationConfig(
            before_root=before_root,
            results_dir=results_dir,
            limit=1,
            skip_passed=True,
            requested_model="fake-model",
        ),
        client=FakeJudgeClient(),
    )

    assert first["counts"]["pass"] == 1
    assert second["counts"]["skipped_passed"] == 1
    rows = _fetch_all(results_dir / "evaluations.sqlite", "SELECT id FROM evaluations")
    assert len(rows) == 1


def test_skip_passed_does_not_reuse_when_after_hash_changes(tmp_path: Path):
    before_root = _write_fixture(tmp_path)
    results_dir = tmp_path / "results"

    run_evaluation(
        EvaluationConfig(
            before_root=before_root,
            results_dir=results_dir,
            limit=1,
            requested_model="fake-model",
        ),
        client=FakeJudgeClient(),
    )
    after_section = (
        tmp_path
        / "precomputed"
        / "municipality"
        / "01"
        / "00100"
        / "Sample__Alabama"
        / "demographics_section.wikitext"
    )
    after_section.write_text(
        "==Demographics==\n===2020 census===\nChanged text.\n",
        encoding="utf-8",
    )
    second = run_evaluation(
        EvaluationConfig(
            before_root=before_root,
            results_dir=results_dir,
            limit=1,
            skip_passed=True,
            requested_model="fake-model",
        ),
        client=FakeJudgeClient(),
    )

    assert second["counts"]["pass"] == 1
    rows = _fetch_all(results_dir / "evaluations.sqlite", "SELECT id FROM evaluations")
    assert len(rows) == 2


def test_skip_passed_does_not_reuse_weaker_prior_model(tmp_path: Path):
    before_root = _write_fixture(tmp_path)
    results_dir = tmp_path / "results"

    run_evaluation(
        EvaluationConfig(
            before_root=before_root,
            results_dir=results_dir,
            limit=1,
            requested_model="fake-model",
        ),
        client=FakeJudgeClient(),
    )
    second = run_evaluation(
        EvaluationConfig(
            before_root=before_root,
            results_dir=results_dir,
            limit=1,
            skip_passed=True,
            requested_model="gpt-5.4",
        ),
        client=FakeJudgeClient(),
    )

    assert second["counts"]["pass"] == 1
    rows = _fetch_all(results_dir / "evaluations.sqlite", "SELECT id FROM evaluations")
    assert len(rows) == 2


def _write_fixture(
    tmp_path: Path,
    state_fips: str = "01",
    target_fips: str = "00100",
    slug: str = "Sample__Alabama",
    article: str = "Sample,_Alabama",
) -> Path:
    before_dir = (
        tmp_path
        / "precomputed-before"
        / "municipality"
        / state_fips
        / target_fips
        / slug
    )
    after_dir = (
        tmp_path
        / "precomputed"
        / "municipality"
        / state_fips
        / target_fips
        / slug
    )
    before_dir.mkdir(parents=True)
    after_dir.mkdir(parents=True)

    before_section = before_dir / "before_demographics_section.wikitext"
    after_section = after_dir / "demographics_section.wikitext"
    after_manifest = after_dir / "manifest.json"
    before_section.write_text("==Demographics==\nOld text.\n", encoding="utf-8")
    after_section.write_text("==Demographics==\n===2020 census===\nNew text.\n", encoding="utf-8")
    after_manifest.write_text(
        json.dumps({"section_path": str(after_section)}) + "\n",
        encoding="utf-8",
    )
    (before_dir / "before_manifest.json").write_text(
        json.dumps(
            {
                "article": article,
                "location_kind": "municipality",
                "state_fips": state_fips,
                "target_fips": target_fips,
                "before_section_path": str(before_section),
                "source_precomputed_manifest_path": str(after_manifest),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path / "precomputed-before"


def _fetch_all(db_path: Path, query: str):
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(query).fetchall()]
    finally:
        connection.close()


def _fetch_one(db_path: Path, query: str):
    rows = _fetch_all(db_path, query)
    assert len(rows) == 1
    return rows[0]
