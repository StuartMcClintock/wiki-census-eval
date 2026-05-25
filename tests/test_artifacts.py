from pathlib import Path
import json

from wiki_census_eval.artifacts import build_unified_diff, discover_before_manifests
from wiki_census_eval.artifacts import load_case


def test_load_case_pairs_before_and_after(tmp_path: Path):
    before_dir = (
        tmp_path
        / "precomputed-before"
        / "municipality"
        / "01"
        / "00100"
        / "Sample__Alabama"
    )
    after_dir = (
        tmp_path
        / "precomputed"
        / "municipality"
        / "01"
        / "00100"
        / "Sample__Alabama"
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
    before_manifest = before_dir / "before_manifest.json"
    before_manifest.write_text(
        json.dumps(
            {
                "article": "Sample,_Alabama",
                "location_kind": "municipality",
                "state_fips": "01",
                "target_fips": "00100",
                "before_section_path": str(before_section),
                "source_precomputed_manifest_path": str(after_manifest),
                "status": "match",
                "reason": "demographics section unchanged since precompute",
                "current_has_demographics_section": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    case = load_case(before_manifest)

    assert case.metadata.case_id == "municipality/01/00100/Sample__Alabama"
    assert case.metadata.article == "Sample,_Alabama"
    assert case.before_text == "==Demographics==\nOld text.\n"
    assert "New text." in case.after_text


def test_build_unified_diff_marks_before_and_after():
    diff = build_unified_diff("old\n", "new\n")

    assert "--- before_demographics_section.wikitext" in diff
    assert "+++ after_demographics_section.wikitext" in diff
    assert "-old" in diff
    assert "+new" in diff


def test_discover_before_manifests_filters_by_state_fips(tmp_path: Path):
    al_manifest = _write_minimal_before_manifest(tmp_path, "01", "00100", "Alabama")
    _write_minimal_before_manifest(tmp_path, "13", "00100", "Georgia")

    manifests = list(
        discover_before_manifests(
            tmp_path / "precomputed-before",
            state_fips_filter={"01"},
        )
    )

    assert manifests == [al_manifest.resolve()]


def _write_minimal_before_manifest(
    tmp_path: Path,
    state_fips: str,
    target_fips: str,
    article_slug: str,
) -> Path:
    before_dir = (
        tmp_path
        / "precomputed-before"
        / "municipality"
        / state_fips
        / target_fips
        / article_slug
    )
    before_dir.mkdir(parents=True)
    manifest = before_dir / "before_manifest.json"
    manifest.write_text('{"article":"Sample"}\n', encoding="utf-8")
    return manifest
