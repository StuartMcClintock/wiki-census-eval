from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as datetime_timezone
from pathlib import Path
from typing import Callable, Optional, Protocol

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover - Python < 3.9 fallback
    ZoneInfo = None

    class ZoneInfoNotFoundError(Exception):
        pass

try:
    import pytz
except ImportError:  # pragma: no cover - optional Python < 3.9 helper
    pytz = None

from openai import OpenAI

from .prompts import SYSTEM_PROMPT
from .schema import JudgeResponse, JudgeResult, strict_judge_result_json_schema


@dataclass(frozen=True)
class JudgeRequest:
    case_id: str
    article: str
    prompt: str


class JudgeClient(Protocol):
    """Small adapter boundary so LangChain can replace OpenAI later."""

    provider: str
    model: str

    def judge(self, request: JudgeRequest) -> JudgeResponse:
        ...


class JudgeClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str,
        response_id: Optional[str] = None,
        raw_output_text: Optional[str] = None,
        expected_schema: Optional[dict] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.response_id = response_id
        self.raw_output_text = raw_output_text
        self.expected_schema = expected_schema


class OpenAIJudgeClient:
    provider = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        max_output_tokens: int = 1200,
        timeout: Optional[float] = None,
    ):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self._client = OpenAI(api_key=api_key, timeout=timeout)

    def judge(self, request: JudgeRequest) -> JudgeResponse:
        response = self._client.responses.parse(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=request.prompt,
            text_format=JudgeResult,
            max_output_tokens=self.max_output_tokens,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise JudgeClientError(
                "OpenAI response did not include parsed structured output",
                provider=self.provider,
                model=self.model,
                response_id=getattr(response, "id", None),
                raw_output_text=getattr(response, "output_text", None),
                expected_schema=strict_judge_result_json_schema(),
            )
        return JudgeResponse(
            result=parsed,
            provider=self.provider,
            model=self.model,
            response_id=getattr(response, "id", None),
            raw_output_text=getattr(response, "output_text", None),
        )


class CodexJudgeClient:
    provider = "codex"

    def __init__(
        self,
        *,
        model: str,
        codex_bin: str = "codex",
        timeout: Optional[float] = None,
        cwd: Optional[Path] = None,
        limit_retry_attempts: int = 5,
        limit_retry_delay_seconds: float = 3600.0,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if limit_retry_attempts < 0:
            raise ValueError("limit_retry_attempts must be non-negative")
        self.model = model
        self.codex_bin = codex_bin
        self.timeout = timeout
        self.cwd = Path(cwd).resolve() if cwd is not None else Path.cwd()
        self.limit_retry_attempts = limit_retry_attempts
        self.limit_retry_delay_seconds = limit_retry_delay_seconds
        self._sleeper = sleeper

    def judge(self, request: JudgeRequest) -> JudgeResponse:
        expected_schema = strict_judge_result_json_schema()
        prompt = (
            SYSTEM_PROMPT
            + "\n\n"
            + request.prompt
            + "\n\nReturn only a JSON object matching the supplied output schema."
        )
        attempt = 0
        while True:
            try:
                return self._judge_once(request, expected_schema, prompt)
            except JudgeClientError as exc:
                if (
                    attempt >= self.limit_retry_attempts
                    or not _is_codex_usage_limit_error(exc.raw_output_text)
                ):
                    raise
                attempt += 1
                print(
                    "Codex usage limit appears to be reached for "
                    f"{request.case_id}; retrying in "
                    f"{self.limit_retry_delay_seconds:g} seconds "
                    f"({attempt}/{self.limit_retry_attempts})."
                )
                self._sleeper(self.limit_retry_delay_seconds)

    def _judge_once(
        self,
        request: JudgeRequest,
        expected_schema: dict,
        prompt: str,
    ) -> JudgeResponse:
        with tempfile.TemporaryDirectory(prefix="wiki-census-eval-codex-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            schema_path = tmpdir_path / "judge_result_schema.json"
            output_path = tmpdir_path / "judge_result.json"
            schema_path.write_text(
                json.dumps(expected_schema, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            cmd = [
                self.codex_bin,
                "--ask-for-approval",
                "never",
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "--model",
                self.model,
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "-",
            ]
            try:
                completed = subprocess.run(
                    cmd,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    cwd=str(self.cwd),
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise JudgeClientError(
                    "Codex evaluation timed out",
                    provider=self.provider,
                    model=self.model,
                    raw_output_text=_join_process_output(exc.stdout, exc.stderr),
                    expected_schema=expected_schema,
                ) from exc

            raw_output_text = _join_process_output(completed.stdout, completed.stderr)
            if completed.returncode != 0:
                raise JudgeClientError(
                    f"Codex evaluation failed with exit code {completed.returncode}",
                    provider=self.provider,
                    model=self.model,
                    raw_output_text=raw_output_text,
                    expected_schema=expected_schema,
                )
            if not output_path.exists():
                raise JudgeClientError(
                    "Codex did not write an output-last-message file",
                    provider=self.provider,
                    model=self.model,
                    raw_output_text=raw_output_text,
                    expected_schema=expected_schema,
                )

            raw_last_message = output_path.read_text(encoding="utf-8")
            try:
                parsed = JudgeResult.model_validate_json(raw_last_message)
            except Exception as exc:
                raise JudgeClientError(
                    "Codex output did not match JudgeResult schema",
                    provider=self.provider,
                    model=self.model,
                    raw_output_text=raw_last_message or raw_output_text,
                    expected_schema=expected_schema,
                ) from exc

            return JudgeResponse(
                result=parsed,
                provider=self.provider,
                model=self.model,
                response_id=None,
                raw_output_text=raw_last_message,
            )


class AnthropicCliJudgeClient:
    provider = "anthropic-cli"

    def __init__(
        self,
        *,
        model: str,
        claude_bin: str = "claude",
        timeout: Optional[float] = None,
        cwd: Optional[Path] = None,
        wait_for_limit_reset: bool = False,
        limit_retry_delay_seconds: float = 3600.0,
        sleeper: Callable[[float], None] = time.sleep,
        now: Callable[[object], datetime] = datetime.now,
    ):
        self.model = model
        self.claude_bin = claude_bin
        self.timeout = timeout
        self.cwd = Path(cwd).resolve() if cwd is not None else Path.cwd()
        self.wait_for_limit_reset = wait_for_limit_reset
        self.limit_retry_delay_seconds = limit_retry_delay_seconds
        self._sleeper = sleeper
        self._now = now

    def judge(self, request: JudgeRequest) -> JudgeResponse:
        expected_schema = strict_judge_result_json_schema()
        schema_text = json.dumps(expected_schema, separators=(",", ":"), sort_keys=True)
        prompt = (
            request.prompt
            + "\n\nReturn only a JSON object matching the supplied output schema."
        )
        cmd = [
            self.claude_bin,
            "--print",
            "--model",
            self.model,
            "--system-prompt",
            SYSTEM_PROMPT,
            "--permission-mode",
            "dontAsk",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--json-schema",
            schema_text,
            prompt,
        ]
        while True:
            try:
                completed = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=True,
                    cwd=str(self.cwd),
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise JudgeClientError(
                    "Anthropic CLI evaluation timed out",
                    provider=self.provider,
                    model=self.model,
                    raw_output_text=_join_process_output(exc.stdout, exc.stderr),
                    expected_schema=expected_schema,
                ) from exc

            raw_output_text = _join_process_output(completed.stdout, completed.stderr)
            if completed.returncode != 0 and self.wait_for_limit_reset:
                retry_plan = _anthropic_limit_retry_plan(
                    raw_output_text,
                    fallback_seconds=self.limit_retry_delay_seconds,
                    now=self._now,
                )
                if retry_plan is not None:
                    print(
                        "Anthropic CLI usage limit appears to be reached for "
                        f"{request.case_id}; retrying at "
                        f"{_format_retry_at(retry_plan.retry_at)}."
                    )
                    self._sleeper(retry_plan.wait_seconds)
                    continue
            break

        if completed.returncode != 0:
            raise JudgeClientError(
                f"Anthropic CLI evaluation failed with exit code {completed.returncode}",
                provider=self.provider,
                model=self.model,
                raw_output_text=raw_output_text,
                expected_schema=expected_schema,
            )

        try:
            result_text = _extract_claude_result_text(completed.stdout or "")
            parsed = JudgeResult.model_validate_json(result_text)
        except Exception as exc:
            raise JudgeClientError(
                "Anthropic CLI output did not match JudgeResult schema",
                provider=self.provider,
                model=self.model,
                raw_output_text=raw_output_text,
                expected_schema=expected_schema,
            ) from exc

        response_id = _extract_claude_response_id(completed.stdout or "")
        return JudgeResponse(
            result=parsed,
            provider=self.provider,
            model=self.model,
            response_id=response_id,
            raw_output_text=raw_output_text,
        )


def _join_process_output(stdout, stderr) -> str:
    parts = []
    for value in (stdout, stderr):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if value:
            parts.append(str(value))
    return "\n".join(parts)


def _is_codex_usage_limit_error(raw_output_text: Optional[str]) -> bool:
    if not raw_output_text:
        return False
    normalized = raw_output_text.lower()
    return any(
        marker in normalized
        for marker in (
            "usage limit",
            "rate limit",
            "rate_limit",
            "quota",
            "too many requests",
            "429",
        )
    )


_ANTHROPIC_USAGE_LIMIT_MARKERS = (
    "you've hit your limit",
    "you've hit your session limit",
    "you’ve hit your limit",
    "you’ve hit your session limit",
    "usage limit reached",
    "rate limit exceeded",
    "you have exceeded",
)
_ANTHROPIC_LIMIT_RESET_PATTERN = re.compile(
    r"you['’]?ve hit your (?:session\s+)?limit.*?resets\s+"
    r"(\d{1,2}(?::\d{2})?\s*[ap]m)\s*\(([^)]+)\)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class _RetryPlan:
    wait_seconds: float
    retry_at: datetime


def _anthropic_limit_wait_seconds(
    raw_output_text: Optional[str],
    *,
    fallback_seconds: float,
    now: Callable[[object], datetime] = datetime.now,
) -> Optional[float]:
    retry_plan = _anthropic_limit_retry_plan(
        raw_output_text,
        fallback_seconds=fallback_seconds,
        now=now,
    )
    if retry_plan is None:
        return None
    return retry_plan.wait_seconds


def _anthropic_limit_retry_plan(
    raw_output_text: Optional[str],
    *,
    fallback_seconds: float,
    now: Callable[[object], datetime] = datetime.now,
) -> Optional[_RetryPlan]:
    if not raw_output_text:
        return None
    lowered = raw_output_text.lower()
    if not any(marker in lowered for marker in _ANTHROPIC_USAGE_LIMIT_MARKERS):
        return None
    retry_at = _extract_anthropic_limit_reset_time(raw_output_text, now=now)
    if retry_at is None:
        wait_seconds = max(0.0, fallback_seconds)
        current = now(None)
        return _RetryPlan(
            wait_seconds=wait_seconds,
            retry_at=current + timedelta(seconds=wait_seconds),
        )
    current = _now_in_timezone(retry_at.tzinfo, now)
    wait_seconds = max(0.0, (retry_at - current).total_seconds())
    return _RetryPlan(wait_seconds=wait_seconds, retry_at=retry_at)


def _extract_anthropic_limit_reset_time(
    raw_output_text: str,
    *,
    now: Callable[[object], datetime] = datetime.now,
) -> Optional[datetime]:
    match = _ANTHROPIC_LIMIT_RESET_PATTERN.search(raw_output_text)
    if match is None:
        return None
    time_text, timezone_name = match.groups()
    timezone = _get_timezone(timezone_name.strip())
    if timezone is None:
        return None

    normalized = re.sub(r"\s+", "", time_text).upper()
    fmt = "%I:%M%p" if ":" in normalized else "%I%p"
    try:
        reset_time = datetime.strptime(normalized, fmt).time()
    except ValueError:
        return None

    current = _now_in_timezone(timezone, now)
    retry_at = current.replace(
        hour=reset_time.hour,
        minute=reset_time.minute,
        second=0,
        microsecond=0,
    )
    if retry_at <= current:
        retry_at += timedelta(days=1)
    return retry_at + timedelta(minutes=1)


def _now_in_timezone(
    tzinfo,
    now: Callable[[object], datetime] = datetime.now,
) -> datetime:
    current = now(tzinfo)
    if current.tzinfo is None:
        return _localize_datetime(tzinfo, current)
    return current.astimezone(tzinfo)


def _format_retry_at(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value.strftime("%Y-%m-%d %H:%M:%S %Z").strip()


def _get_timezone(timezone_name: str):
    if timezone_name.upper() in {"UTC", "GMT", "Z"}:
        return datetime_timezone.utc
    if ZoneInfo is not None:
        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            pass
    if pytz is not None:
        try:
            return pytz.timezone(timezone_name)
        except pytz.UnknownTimeZoneError:
            return None
    return None


def _localize_datetime(tzinfo, value: datetime) -> datetime:
    localize = getattr(tzinfo, "localize", None)
    if callable(localize):
        return localize(value)
    return value.replace(tzinfo=tzinfo)


def _extract_claude_result_text(stdout: str) -> str:
    raw = stdout.strip()
    if not raw:
        raise ValueError("empty Claude output")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(data, dict):
        if _looks_like_judge_result(data):
            return json.dumps(data)
        structured_output = data.get("structured_output")
        if isinstance(structured_output, dict) and _looks_like_judge_result(
            structured_output
        ):
            return json.dumps(structured_output)
        for key in ("result", "response", "text", "content"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return _extract_json_object_text(value)
            if isinstance(value, list):
                text = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in value
                )
                if text.strip():
                    return _extract_json_object_text(text)
    return raw


def _extract_claude_response_id(stdout: str) -> Optional[str]:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    for key in ("message_id", "response_id", "id", "session_id"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _looks_like_judge_result(value: dict) -> bool:
    return {"article", "verdict", "summary", "confidence"}.issubset(value.keys())


def _extract_json_object_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped
