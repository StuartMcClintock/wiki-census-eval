from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Set

from .schema import EvaluationMetadata


class ArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvaluationCase:
    metadata: EvaluationMetadata
    before_text: str
    after_text: str

    @property
    def diff_text(self) -> str:
        return build_unified_diff(self.before_text, self.after_text)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.rstrip().encode("utf-8")).hexdigest()


def build_unified_diff(before_text: str, after_text: str) -> str:
    return "".join(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile="before_demographics_section.wikitext",
            tofile="after_demographics_section.wikitext",
            lineterm="",
        )
    )


def discover_before_manifests(
    before_root: Path,
    state_fips_filter: Optional[Set[str]] = None,
) -> Iterator[Path]:
    filters = set(state_fips_filter or [])
    if not filters:
        yield from sorted(before_root.rglob("before_manifest.json"))
        return
    for location_kind_dir in sorted(path for path in before_root.iterdir() if path.is_dir()):
        for state_fips in sorted(filters):
            state_dir = location_kind_dir / state_fips
            if state_dir.exists():
                yield from sorted(state_dir.rglob("before_manifest.json"))


def load_case(before_manifest_path: Path) -> EvaluationCase:
    before_manifest_path = before_manifest_path.resolve()
    try:
        before_manifest = json.loads(before_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read before manifest: {exc}") from exc

    article = before_manifest.get("article")
    if not article:
        raise ArtifactError("before manifest is missing article")

    before_section_path = _resolve_existing_path(
        before_manifest.get("before_section_path"),
        before_manifest_path.parent / "before_demographics_section.wikitext",
        label="before section",
    )
    after_manifest_path = _resolve_existing_path(
        before_manifest.get("source_precomputed_manifest_path"),
        None,
        label="source precomputed manifest",
    )

    try:
        after_manifest = json.loads(after_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read source precomputed manifest: {exc}") from exc

    after_section_path = _resolve_existing_path(
        after_manifest.get("section_path"),
        after_manifest_path.parent / "demographics_section.wikitext",
        label="after section",
    )

    try:
        before_text = before_section_path.read_text(encoding="utf-8")
        after_text = after_section_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"cannot read before/after section text: {exc}") from exc

    case_id = "/".join(before_manifest_path.parent.relative_to(_before_root(before_manifest_path)).parts)
    metadata = EvaluationMetadata(
        case_id=case_id,
        article=article,
        location_kind=before_manifest.get("location_kind"),
        state_fips=before_manifest.get("state_fips"),
        target_fips=before_manifest.get("target_fips"),
        before_manifest_path=str(before_manifest_path),
        before_section_path=str(before_section_path),
        after_manifest_path=str(after_manifest_path),
        after_section_path=str(after_section_path),
        before_hash=hash_text(before_text),
        after_hash=hash_text(after_text),
        freshness_status=before_manifest.get("status"),
        freshness_reason=before_manifest.get("reason"),
        current_has_demographics_section=before_manifest.get(
            "current_has_demographics_section"
        ),
    )
    return EvaluationCase(metadata=metadata, before_text=before_text, after_text=after_text)


def iter_cases(
    before_root: Path,
    limit: Optional[int] = None,
    state_fips_filter: Optional[Set[str]] = None,
) -> Iterator[EvaluationCase]:
    count = 0
    for manifest_path in discover_before_manifests(
        before_root,
        state_fips_filter=state_fips_filter,
    ):
        yield load_case(manifest_path)
        count += 1
        if limit is not None and count >= limit:
            return


def _resolve_existing_path(raw_path, fallback: Optional[Path], *, label: str) -> Path:
    candidates: List[Path] = []
    if isinstance(raw_path, str) and raw_path.strip():
        candidates.append(Path(raw_path).expanduser())
    if fallback is not None:
        candidates.append(fallback)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    attempted = ", ".join(str(path) for path in candidates) or "<none>"
    raise ArtifactError(f"missing {label}; tried {attempted}")


def _before_root(before_manifest_path: Path) -> Path:
    for parent in before_manifest_path.parents:
        if parent.name == "precomputed-before":
            return parent
    if len(before_manifest_path.parents) >= 5:
        return before_manifest_path.parents[4]
    return before_manifest_path.parent
