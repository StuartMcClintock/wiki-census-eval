from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


Verdict = Literal["pass", "warning", "fail"]
IssueSeverity = Literal["warning", "fail"]
IssueCode = Literal[
    "bad_heading_level",
    "missing_demographics_heading",
    "removed_prior_census_heading",
    "unjustified_removal",
    "duplicated_content",
    "stale_update_banner",
    "malformed_wikitext",
    "bad_table_structure",
    "incorrect_or_suspicious_census_content",
    "citation_problem",
    "scope_or_location_mismatch",
    "other",
]


class JudgeIssue(BaseModel):
    code: IssueCode
    severity: IssueSeverity
    explanation: str = Field(..., min_length=1)
    evidence: Optional[str] = Field(
        default=None,
        description="Short quote or paraphrase of the relevant before/after evidence.",
    )


class JudgeResult(BaseModel):
    article: str = Field(..., description="Wikipedia article title being evaluated.")
    verdict: Verdict
    summary: str = Field(
        ...,
        min_length=1,
        description="One to three sentences. Mention the municipality/article name in bold.",
    )
    issues: List[JudgeIssue] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @property
    def emoji(self) -> str:
        return emoji_for_verdict(self.verdict)


def emoji_for_verdict(verdict: Verdict) -> str:
    return {"pass": "✅", "warning": "⚠️", "fail": "❌"}[verdict]


def strict_judge_result_json_schema() -> dict:
    schema = JudgeResult.model_json_schema()
    _make_objects_strict(schema)
    _require_all_properties(schema)
    return schema


def _make_objects_strict(value) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            value["additionalProperties"] = False
        for child in value.values():
            _make_objects_strict(child)
    elif isinstance(value, list):
        for child in value:
            _make_objects_strict(child)


def _require_all_properties(value) -> None:
    if isinstance(value, dict):
        properties = value.get("properties")
        if value.get("type") == "object" and isinstance(properties, dict):
            value["required"] = list(properties.keys())
        for child in value.values():
            _require_all_properties(child)
    elif isinstance(value, list):
        for child in value:
            _require_all_properties(child)


class EvaluationMetadata(BaseModel):
    case_id: str
    article: str
    location_kind: Optional[str] = None
    state_fips: Optional[str] = None
    target_fips: Optional[str] = None
    before_manifest_path: str
    before_section_path: str
    after_manifest_path: str
    after_section_path: str
    before_hash: Optional[str] = None
    after_hash: Optional[str] = None
    freshness_status: Optional[str] = None
    freshness_reason: Optional[str] = None
    current_has_demographics_section: Optional[bool] = None


class JudgeResponse(BaseModel):
    result: JudgeResult
    provider: str
    model: str
    response_id: Optional[str] = None
    raw_output_text: Optional[str] = None


class EvaluationRecord(BaseModel):
    record_type: Literal["evaluation"] = "evaluation"
    evaluated_at: datetime
    metadata: EvaluationMetadata
    judge: JudgeResponse


class PipelineErrorRecord(BaseModel):
    record_type: Literal["pipeline_error"] = "pipeline_error"
    evaluated_at: datetime
    case_id: Optional[str] = None
    article: Optional[str] = None
    manifest_path: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    response_id: Optional[str] = None
    error: str
    raw_output_text: Optional[str] = None
    expected_schema: Optional[dict] = None
