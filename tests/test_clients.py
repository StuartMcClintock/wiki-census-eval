from pathlib import Path
from types import SimpleNamespace
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


def test_strict_judge_schema_disallows_additional_properties():
    schema = strict_judge_result_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["JudgeIssue"]["additionalProperties"] is False
    assert "issues" in schema["required"]
    assert "evidence" in schema["$defs"]["JudgeIssue"]["required"]
