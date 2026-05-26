from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import subprocess

import pytest

from wiki_census_eval.clients import AnthropicCliJudgeClient, CodexJudgeClient
from wiki_census_eval.clients import JudgeClientError, JudgeRequest
from wiki_census_eval.schema import strict_judge_result_json_schema


def test_codex_judge_client_parses_output_last_message(tmp_path: Path, monkeypatch):
    calls = []
    schema_texts = []

    def fake_run(cmd, input, text, capture_output, cwd, timeout, check):
        schema_path = Path(cmd[cmd.index("--output-schema") + 1])
        schema_texts.append(schema_path.read_text(encoding="utf-8"))
        calls.append(
            {
                "cmd": cmd,
                "input": input,
                "text": text,
                "capture_output": capture_output,
                "cwd": cwd,
                "timeout": timeout,
                "check": check,
            }
        )
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        output_path.write_text(
            (
                '{"article":"Sample,_Alabama","verdict":"pass",'
                '"summary":"**Sample,_Alabama** looks safe.",'
                '"issues":[],"confidence":0.9}'
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = CodexJudgeClient(model="gpt-5.4-mini", codex_bin="codex-test", cwd=tmp_path)

    response = client.judge(
        JudgeRequest(
            case_id="case",
            article="Sample,_Alabama",
            prompt="Evaluate this.",
        )
    )

    assert response.provider == "codex"
    assert response.model == "gpt-5.4-mini"
    assert response.result.verdict == "pass"
    assert response.result.emoji == "✅"
    assert calls[0]["cmd"][:4] == ["codex-test", "--ask-for-approval", "never", "exec"]
    assert "--output-schema" in calls[0]["cmd"]
    assert "--output-last-message" in calls[0]["cmd"]
    assert calls[0]["cmd"][-1] == "-"
    assert "Evaluate this." in calls[0]["input"]
    assert '"additionalProperties": false' in schema_texts[0]


def test_codex_judge_client_reports_schema_failures(tmp_path: Path, monkeypatch):
    def fake_run(cmd, input, text, capture_output, cwd, timeout, check):
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        output_path.write_text("not json", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = CodexJudgeClient(model="gpt-5.4-mini", cwd=tmp_path)

    with pytest.raises(JudgeClientError) as exc_info:
        client.judge(
            JudgeRequest(
                case_id="case",
                article="Sample,_Alabama",
                prompt="Evaluate this.",
            )
        )

    assert "did not match JudgeResult schema" in str(exc_info.value)
    assert exc_info.value.provider == "codex"
    assert exc_info.value.raw_output_text == "not json"
    assert exc_info.value.expected_schema is not None


def test_codex_judge_client_retries_usage_limit_failure(tmp_path: Path, monkeypatch):
    calls = []
    sleeps = []

    def fake_run(cmd, input, text, capture_output, cwd, timeout, check):
        calls.append(cmd)
        if len(calls) == 1:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="Usage limit reached. Try again later.",
            )
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        output_path.write_text(
            (
                '{"article":"Sample,_Alabama","verdict":"pass",'
                '"summary":"**Sample,_Alabama** looks safe.",'
                '"issues":[],"confidence":0.9}'
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = CodexJudgeClient(
        model="gpt-5.4-mini",
        cwd=tmp_path,
        limit_retry_attempts=1,
        limit_retry_delay_seconds=0,
        sleeper=sleeps.append,
    )

    response = client.judge(
        JudgeRequest(
            case_id="case",
            article="Sample,_Alabama",
            prompt="Evaluate this.",
        )
    )

    assert response.result.verdict == "pass"
    assert len(calls) == 2
    assert sleeps == [0]


def test_codex_judge_client_reports_usage_limit_after_retries(
    tmp_path: Path,
    monkeypatch,
):
    calls = []
    sleeps = []

    def fake_run(cmd, input, text, capture_output, cwd, timeout, check):
        calls.append(cmd)
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Rate limit exceeded. Try again later.",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = CodexJudgeClient(
        model="gpt-5.4-mini",
        cwd=tmp_path,
        limit_retry_attempts=1,
        limit_retry_delay_seconds=0,
        sleeper=sleeps.append,
    )

    with pytest.raises(JudgeClientError) as exc_info:
        client.judge(
            JudgeRequest(
                case_id="case",
                article="Sample,_Alabama",
                prompt="Evaluate this.",
            )
        )

    assert "failed with exit code 1" in str(exc_info.value)
    assert len(calls) == 2
    assert sleeps == [0]


def test_anthropic_cli_judge_client_parses_json_output(tmp_path: Path, monkeypatch):
    calls = []

    def fake_run(cmd, text, capture_output, cwd, timeout, check):
        calls.append(
            {
                "cmd": cmd,
                "text": text,
                "capture_output": capture_output,
                "cwd": cwd,
                "timeout": timeout,
                "check": check,
            }
        )
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{"type":"result","session_id":"session-1",'
                '"result":"{\\"article\\":\\"Sample,_Alabama\\",'
                '\\"verdict\\":\\"pass\\",'
                '\\"summary\\":\\"**Sample,_Alabama** looks safe.\\",'
                '\\"issues\\":[],\\"confidence\\":0.9}"}'
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = AnthropicCliJudgeClient(
        model="sonnet",
        claude_bin="claude-test",
        cwd=tmp_path,
    )

    response = client.judge(
        JudgeRequest(
            case_id="case",
            article="Sample,_Alabama",
            prompt="Evaluate this.",
        )
    )

    assert response.provider == "anthropic-cli"
    assert response.model == "sonnet"
    assert response.response_id == "session-1"
    assert response.result.verdict == "pass"
    assert response.result.emoji == "✅"
    assert calls[0]["cmd"][:2] == ["claude-test", "--print"]
    assert "--json-schema" in calls[0]["cmd"]
    assert "--output-format" in calls[0]["cmd"]
    assert "Evaluate this." in calls[0]["cmd"][-1]


def test_anthropic_cli_judge_client_prefers_structured_output(
    tmp_path: Path,
    monkeypatch,
):
    def fake_run(cmd, text, capture_output, cwd, timeout, check):
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{"type":"result","session_id":"session-1",'
                '"result":"**Verdict: Warning**\\nHuman readable text.",'
                '"structured_output":{'
                '"article":"Cloverleaf_Colony,_South_Dakota",'
                '"verdict":"warning",'
                '"summary":"**Cloverleaf Colony, South Dakota** needs review.",'
                '"issues":[{'
                '"code":"incorrect_or_suspicious_census_content",'
                '"severity":"warning",'
                '"explanation":"The age and household claims contradict each other.",'
                '"evidence":"0.0% under 18 and households with children"'
                '}],'
                '"confidence":0.82'
                '}}'
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = AnthropicCliJudgeClient(
        model="sonnet",
        claude_bin="claude-test",
        cwd=tmp_path,
    )

    response = client.judge(
        JudgeRequest(
            case_id="case",
            article="Cloverleaf_Colony,_South_Dakota",
            prompt="Evaluate this.",
        )
    )

    assert response.result.verdict == "warning"
    assert response.result.article == "Cloverleaf_Colony,_South_Dakota"
    assert response.result.issues[0].code == "incorrect_or_suspicious_census_content"
    assert response.raw_output_text is not None


def test_anthropic_cli_judge_client_reports_schema_failures(tmp_path: Path, monkeypatch):
    def fake_run(cmd, text, capture_output, cwd, timeout, check):
        return SimpleNamespace(
            returncode=0,
            stdout='{"type":"result","result":"not json"}',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = AnthropicCliJudgeClient(model="sonnet", cwd=tmp_path)

    with pytest.raises(JudgeClientError) as exc_info:
        client.judge(
            JudgeRequest(
                case_id="case",
                article="Sample,_Alabama",
                prompt="Evaluate this.",
            )
        )

    assert "did not match JudgeResult schema" in str(exc_info.value)
    assert exc_info.value.provider == "anthropic-cli"
    assert exc_info.value.expected_schema is not None


def test_anthropic_cli_judge_client_waits_for_limit_reset(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    calls = []
    sleeps = []

    def fake_run(cmd, text, capture_output, cwd, timeout, check):
        calls.append(cmd)
        if len(calls) == 1:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="You've hit your limit · resets 12pm (America/New_York)",
            )
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{"type":"result","session_id":"session-1",'
                '"result":"{\\"article\\":\\"Sample,_Alabama\\",'
                '\\"verdict\\":\\"pass\\",'
                '\\"summary\\":\\"**Sample,_Alabama** looks safe.\\",'
                '\\"issues\\":[],\\"confidence\\":0.9}"}'
            ),
            stderr="",
        )

    def fake_now(tz):
        value = datetime(2026, 4, 25, 11, 30)
        localize = getattr(tz, "localize", None)
        if callable(localize):
            return localize(value)
        return value.replace(tzinfo=tz)

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = AnthropicCliJudgeClient(
        model="sonnet",
        cwd=tmp_path,
        wait_for_limit_reset=True,
        sleeper=sleeps.append,
        now=fake_now,
    )

    response = client.judge(
        JudgeRequest(
            case_id="case",
            article="Sample,_Alabama",
            prompt="Evaluate this.",
        )
    )

    assert response.result.verdict == "pass"
    assert len(calls) == 2
    assert sleeps == [31 * 60]
    captured = capsys.readouterr()
    assert "retrying at 2026-04-25 12:01:00" in captured.out
    assert "retrying in" not in captured.out
    assert "seconds" not in captured.out


def test_anthropic_cli_judge_client_waits_for_json_session_limit(
    tmp_path: Path,
    monkeypatch,
):
    calls = []
    sleeps = []

    def fake_run(cmd, text, capture_output, cwd, timeout, check):
        calls.append(cmd)
        if len(calls) == 1:
            return SimpleNamespace(
                returncode=1,
                stdout=(
                    '{"type":"result","subtype":"success","is_error":true,'
                    '"api_error_status":429,'
                    '"result":"You\'ve hit your session limit · '
                    'resets 3:10am (America/Los_Angeles)"}'
                ),
                stderr="",
            )
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{"type":"result","session_id":"session-1",'
                '"result":"{\\"article\\":\\"Sample,_Alabama\\",'
                '\\"verdict\\":\\"pass\\",'
                '\\"summary\\":\\"**Sample,_Alabama** looks safe.\\",'
                '\\"issues\\":[],\\"confidence\\":0.9}"}'
            ),
            stderr="",
        )

    def fake_now(tz):
        value = datetime(2026, 5, 25, 2, 30)
        localize = getattr(tz, "localize", None)
        if callable(localize):
            return localize(value)
        return value.replace(tzinfo=tz)

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = AnthropicCliJudgeClient(
        model="sonnet",
        cwd=tmp_path,
        wait_for_limit_reset=True,
        sleeper=sleeps.append,
        now=fake_now,
    )

    response = client.judge(
        JudgeRequest(
            case_id="case",
            article="Sample,_Alabama",
            prompt="Evaluate this.",
        )
    )

    assert response.result.verdict == "pass"
    assert len(calls) == 2
    assert sleeps == [41 * 60]


def test_anthropic_cli_judge_client_uses_fallback_limit_delay(
    tmp_path: Path,
    monkeypatch,
):
    calls = []
    sleeps = []

    def fake_run(cmd, text, capture_output, cwd, timeout, check):
        calls.append(cmd)
        if len(calls) == 1:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="Usage limit reached. Try again later.",
            )
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{"type":"result","session_id":"session-1",'
                '"result":"{\\"article\\":\\"Sample,_Alabama\\",'
                '\\"verdict\\":\\"pass\\",'
                '\\"summary\\":\\"**Sample,_Alabama** looks safe.\\",'
                '\\"issues\\":[],\\"confidence\\":0.9}"}'
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = AnthropicCliJudgeClient(
        model="sonnet",
        cwd=tmp_path,
        wait_for_limit_reset=True,
        limit_retry_delay_seconds=42,
        sleeper=sleeps.append,
    )

    response = client.judge(
        JudgeRequest(
            case_id="case",
            article="Sample,_Alabama",
            prompt="Evaluate this.",
        )
    )

    assert response.result.verdict == "pass"
    assert len(calls) == 2
    assert sleeps == [42]


def test_anthropic_cli_judge_client_does_not_wait_without_flag(
    tmp_path: Path,
    monkeypatch,
):
    def fake_run(cmd, text, capture_output, cwd, timeout, check):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Usage limit reached. Try again later.",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = AnthropicCliJudgeClient(
        model="sonnet",
        cwd=tmp_path,
        wait_for_limit_reset=False,
        sleeper=lambda seconds: (_ for _ in ()).throw(AssertionError("slept")),
    )

    with pytest.raises(JudgeClientError) as exc_info:
        client.judge(
            JudgeRequest(
                case_id="case",
                article="Sample,_Alabama",
                prompt="Evaluate this.",
            )
        )

    assert "failed with exit code 1" in str(exc_info.value)


def test_strict_judge_schema_disallows_additional_properties():
    schema = strict_judge_result_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["JudgeIssue"]["additionalProperties"] is False
    assert "issues" in schema["required"]
    assert "evidence" in schema["$defs"]["JudgeIssue"]["required"]
