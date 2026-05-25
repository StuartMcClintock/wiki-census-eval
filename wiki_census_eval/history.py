from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from .schema import EvaluationRecord


_MODEL_STRENGTH_BY_ID = {
    "gpt-5.5": 500,
    "gpt-5.4": 400,
    "gpt-5.4-mini": 300,
    "gpt-5.4-nano": 200,
    "gpt-4.1": 250,
    "gpt-4.1-mini": 150,
    "gpt-4.1-nano": 100,
    "opus": 450,
    "sonnet": 300,
    "haiku": 150,
}


@dataclass(frozen=True)
class HistoryMatch:
    case_id: str
    article: str
    evaluated_at: str
    verdict: str
    model: str
    model_strength: int
    requested_model_strength: int
    after_hash: str


class EvaluationHistory:
    def __init__(self, records: Dict[Tuple[str, str], EvaluationRecord]):
        self._records = records

    @classmethod
    def load(cls, paths: Iterable[Path]) -> "EvaluationHistory":
        records: Dict[Tuple[str, str], EvaluationRecord] = {}
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if raw.get("record_type") != "evaluation":
                        continue
                    try:
                        record = EvaluationRecord.model_validate(raw)
                    except Exception:
                        continue
                    after_hash = record.metadata.after_hash
                    if not after_hash:
                        continue
                    key = (record.metadata.case_id, after_hash)
                    previous = records.get(key)
                    if previous is None or record.evaluated_at > previous.evaluated_at:
                        records[key] = record
        return cls(records)

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
        record = self._records.get((case_id, after_hash))
        if record is None:
            return None
        if record.metadata.article != article:
            return None
        if record.judge.result.verdict != "pass":
            return None

        requested_strength = (
            min_model_strength
            if min_model_strength is not None
            else model_strength(requested_model)
        )
        prior_strength = model_strength(record.judge.model)
        if prior_strength < requested_strength:
            return None

        return HistoryMatch(
            case_id=case_id,
            article=article,
            evaluated_at=_format_datetime(record.evaluated_at),
            verdict=record.judge.result.verdict,
            model=record.judge.model,
            model_strength=prior_strength,
            requested_model_strength=requested_strength,
            after_hash=after_hash,
        )


def model_strength(model: str) -> int:
    normalized = normalize_model_id(model)
    if normalized in _MODEL_STRENGTH_BY_ID:
        return _MODEL_STRENGTH_BY_ID[normalized]
    if "nano" in normalized:
        return 100
    if "mini" in normalized:
        return 200
    if re.search(r"gpt-5(?:[.-]|$)", normalized):
        return 400
    if re.search(r"gpt-4(?:[.-]|$)", normalized):
        return 250
    if "opus" in normalized:
        return 450
    if "sonnet" in normalized:
        return 300
    if "haiku" in normalized:
        return 150
    return 0


def normalize_model_id(model: str) -> str:
    return model.strip().lower()


def _format_datetime(value: datetime) -> str:
    return value.isoformat()
