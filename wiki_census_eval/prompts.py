from __future__ import annotations

import json

from .artifacts import EvaluationCase


SYSTEM_PROMPT = """You are evaluating proposed Wikipedia edits made by a bot.

The bot updates municipality demographics sections by adding 2020 census data.
Evaluate only issues introduced by the proposed after text. Ignore issues that
already existed in the before text unless the after text makes them worse.
"""


RUBRIC = """Return a structured judgment for this proposed demographics edit.

Use verdict "pass" when the edit appears safe and useful.
Use verdict "warning" when there is a plausible issue that needs human review but the edit is not clearly bad.
Use verdict "fail" when the edit clearly damages structure, removes important content, or introduces materially wrong/malformed content.

Evaluation rules:
- Focus on structural regressions such as removing a needed "===2010 census===" heading above retained 2010 census prose.
- Do not flag a race/ethnicity mismatch caused solely by differences between non-Hispanic race tables and all-residents race breakdowns.
- Do flag race/ethnicity content if the article explicitly mislabels definitions or mixes incompatible definitions in the same sentence/table.
- Be careful before claiming duplication. It is duplication only if content appears twice in the after text, not merely once in before and once in after.
- Pay attention to stale update-needed banners that should have been removed after the 2020 update.
- Pay particular attention to content/information unjustifiably removed from before to after.
- If before text is empty, evaluate whether after is a valid new ==Demographics== section containing only appropriate generated 2020 census content.
- The summary must be 1-3 sentences and mention the article name in bold.
- Do not require perfect style; classify only meaningful issues.
"""


def build_prompt(case: EvaluationCase, max_chars: int = 60_000) -> str:
    payload = {
        "metadata": case.metadata.model_dump(),
        "before_demographics_section": _truncate(case.before_text, max_chars // 3),
        "after_demographics_section": _truncate(case.after_text, max_chars // 3),
        "unified_diff": _truncate(case.diff_text, max_chars // 3),
    }
    return RUBRIC + "\n\nEvaluation input:\n" + json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    )


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return (
        text[:head]
        + "\n\n[...TRUNCATED FOR EVALUATION INPUT...]\n\n"
        + text[-tail:]
    )
