# wiki-census-eval

MVP evaluation pipeline for proposed Wikipedia demographics-section edits.

The pipeline pairs:

- `precomputed-before/.../before_demographics_section.wikitext`
- the generated `demographics_section.wikitext` referenced by `before_manifest.json`

It builds a deterministic before/after diff, sends it to an LLM judge via a small
`JudgeClient` interface, and writes results to SQLite. The
default implementation uses the OpenAI SDK directly; swapping in LangChain later
should only require adding another `JudgeClient` adapter.

## Run

```bash
python3 -m wiki_census_eval evaluate --limit 20 --model gpt-5.4-mini
```

To run the same evaluation through Codex CLI instead of the OpenAI SDK:

```bash
python3 -m wiki_census_eval evaluate \
  --provider codex \
  --model gpt-5.4-mini \
  --limit 20
```

The Codex provider uses `codex exec --output-schema` and writes the final
structured response through `--output-last-message`, then validates it against
the same `JudgeResult` schema used by the SDK provider.

If Codex reports a usage or rate limit, the provider waits one hour and retries
the same case, up to 5 times by default. Use `--codex-limit-retries 0` to disable
this, or `--codex-limit-retry-delay <seconds>` to change the wait.

To run through the Anthropic API SDK:

```bash
python3 -m wiki_census_eval evaluate \
  --provider anthropic \
  --model sonnet \
  --limit 20
```

The Anthropic API provider uses the Messages API with a forced tool call, then
validates the tool input against the same `JudgeResult` schema. For the
Anthropic API provider, `--model sonnet` resolves to
`claude-sonnet-4-20250514` before the request and database write.

To run through Claude Code / Anthropic CLI instead of the Anthropic API SDK:

```bash
python3 -m wiki_census_eval evaluate \
  --provider anthropic-cli \
  --model sonnet \
  --limit 20
```

The Anthropic CLI provider uses `claude --print --output-format json
--json-schema ...` and validates the returned result against the same
`JudgeResult` schema.

Results are stored in `results/evaluations.sqlite` by default. Use `--db-path`
to write to a different SQLite database.

Useful options:

```bash
python3 -m wiki_census_eval evaluate \
  --before-root precomputed-before \
  --results-dir results \
  --db-path results/evaluations.sqlite \
  --states AL,GA \
  --limit 100 \
  --skip-passed
```

To evaluate only articles refreshed by a specific
`wikipedia-census-cyrus --refresh-stale-cache` run, use that run's handoff file:

```bash
python3 -m wiki_census_eval evaluate \
  --case-list ../wikipedia-census-cyrus/.poster-runs/refresh-stale-cache/latest.json \
  --skip-evaluated-articles
```

To re-evaluate cases that a prior model marked non-pass, first build a case list
from SQLite history. By default this considers only the latest evaluation per
case id for the selected provider/model:

```bash
python3 -m wiki_census_eval case-list from-evaluations \
  --provider codex \
  --model gpt-5.5 \
  --exclude-verdict pass \
  --output results/gpt-5.5-codex-nonpass-case-list.json
```

Then pass that file to another evaluator:

```bash
python3 -m wiki_census_eval evaluate \
  --provider anthropic-cli \
  --model sonnet \
  --case-list results/gpt-5.5-codex-nonpass-case-list.json
```

To skip articles already evaluated by either Claude Code Sonnet or Anthropic API
Sonnet, repeat `--skip-evaluated-by-provider-model`:

```bash
python3 -m wiki_census_eval evaluate \
  --provider anthropic \
  --model sonnet \
  --case-list results/gpt-5.5-codex-nonpass-case-list.json \
  --skip-evaluated-by-provider-model anthropic-cli:sonnet \
  --skip-evaluated-by-provider-model anthropic:sonnet
```

`--states` restricts the artifact scan to one or more states. It accepts postal
abbreviations or FIPS codes, comma-separated or repeated:

```bash
python3 -m wiki_census_eval evaluate --states AL,GA
python3 -m wiki_census_eval evaluate --states 01 --states 13
```

`--skip-passed` uses the SQLite database as history. It skips an
article only when the current generated after-section hash already passed and
the prior pass was produced by a model with equal-or-greater configured strength.
Use `--skip-existing` only when you want to skip any previously evaluated case id
regardless of verdict or whether the generated text changed. Use
`--skip-evaluated-articles` when you want to skip any previously evaluated
article title, even if it appears under a different case id.

Set `OPENAI_API_KEY` in the environment before running live evaluations.
Set `ANTHROPIC_API_KEY` before using `--provider anthropic`.
The Codex provider uses your local Codex CLI authentication instead.
The Anthropic CLI provider uses your local Claude Code authentication instead.

## Expected Model Output

The OpenAI adapter requests structured output matching `JudgeResult` in
`wiki_census_eval/schema.py`: `article`, `verdict`, `summary`, `issues`, and
`confidence`. The pipeline derives the display emoji from `verdict`. If the
model response cannot be parsed into that
schema, the pipeline writes a `pipeline_errors` row containing the raw
model output, response id, model, provider, and expected schema, then continues
with the next article.

## SQLite Tables

The database contains:

- `runs`: one row per invocation, including aggregate counts.
- `evaluations`: one row per judged article.
- `issues`: one row per issue attached to an evaluation.
- `pipeline_errors`: schema/API/artifact errors that did not produce a judgment.

## Dry Run

Inspect paired artifacts and prompts without calling the model:

```bash
python3 -m wiki_census_eval evaluate --limit 3 --dry-run
```
