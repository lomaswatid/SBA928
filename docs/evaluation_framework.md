# Evaluation framework — Phases 1, 2 and 3

The Phase 1 primitives do not load models or access data. The Phase 2 dataset
loader reads only explicit development files, the original source manifest,
and optionally the training reference file. Existing step scripts and artifacts
remain separate. No extra dependencies are required. The Phase 1 schema remains
unchanged at `sba928_evaluation_v1`. Phase 3 scores saved responses offline; no
module in this package loads or executes models.

## Minimal schema

`EvaluationCase.from_mapping` validates a JSON object with exactly these fields:

```json
{
  "schema_version": "sba928_evaluation_v1",
  "case_id": "synthetic-example",
  "instruction": "Compare the supplied conversion rates.",
  "context": "A: 22.5%; B: 17.0%.",
  "expected_facts": [
    {"group": "A minus B", "metric": "conversion_difference",
     "value": "5.5", "unit": "percentage_point", "decimal_places": 1}
  ]
}
```

IDs, instruction, context, groups and metrics must be nonempty strings.
`validate_cases` rejects an empty collection and duplicate case IDs. Expected
facts must be nonempty with unique `(group, metric, unit)` keys. This is schema
validation, not a provenance or leakage guarantee.

Facts accept plain decimal strings, integers or Decimal objects; booleans,
binary floats, commas, exponent notation, NaN and infinity are rejected.
Inputs are limited to 28 significant digits and 28 fractional places. Decimal
objects are normalized to plain notation; exponent strings are rejected. JSON
exports use strings for exact numbers. `decimal_places` defaults to 1 and must
be an integer from 0 through 12. Units are:

- `count`: nonnegative integer.
- `fraction`: 0 through 1.
- `percent`: 0 through 100.
- `percentage_point`: signed difference from -100 through 100.

Group/metric labels match exactly. There is no implicit alias, capitalization,
boolean-label or fraction-to-percent conversion. Agree on canonical labels
before assigning facts. `format_input` includes only instruction and context,
using the existing `instruction + "\n\nContext:\n" + context` convention.

## Numerical rules

`conversion_rate(purchases, sessions, decimal_places=1)` computes
`100 * purchases / sessions`. Counts must be integers, sessions positive, and
purchases between zero and sessions. Zero-session rates are undefined and raise
ValueError. `percentage_point_difference(left, right, decimal_places=1)` takes
percent operands and returns **left minus right**. It does not calculate
relative percentage change or infer purchases from rounded rates.

Calculations use a private 128-digit Decimal context and round once with
`ROUND_HALF_UP` (halfway values round away from zero). Results do not depend on
the caller's decimal precision or rounding traps.

Examples from synthetic inputs:

```python
from sba_928.evaluation.numerical import (
    conversion_rate, percentage_point_difference, score_fact,
)
from sba_928.evaluation.schema import Fact

assert str(conversion_rate(1, 16)) == "6.3"
assert str(percentage_point_difference("22.5", "17.0")) == "5.5"

expected = Fact("A", "conversion_rate", "22.5", "percent")
score_fact(expected, Fact("A", "conversion_rate", "22.5", "percent"))
# {"status": "correct", "reason": "matched"}
score_fact(expected, Fact("B", "conversion_rate", "22.5", "percent"))
# {"status": "incorrect", "reason": "group_mismatch"}
score_fact(expected, None, attribution_confirmed=False)
# {"status": "needs_review", "reason": "ambiguous_attribution"}
```

`score_fact` compares one explicitly paired expected/observed fact. It checks
group, metric and unit before value. Counts match exactly. Other values match
after rounding both to the **expected** precision; observed precision cannot
relax the scoring rule. A percentage-point result labeled `percent` fails the
unit check even when its number is correct. Validated facts cannot represent
impossible rates such as 954%; retain such response text for manual review
rather than silently correcting it or treating it as missing.

`observed=None` with confirmed attribution means the required fact is absent
(`missing`). Uncertain extraction or attribution requires
`attribution_confirmed=False` (`needs_review`), including when no fact could be
parsed. The function does not parse arbitrary prose, align multiple claims,
compute a composite accuracy score, or establish whether arithmetic reasoning
was actually used. Do not mark prose attribution confirmed merely because a
number appears somewhere in the answer. Invalid values raise ValueError;
unsupported numeric/text input types raise TypeError. These validation errors
are not model-quality scores.

## Repetition rules

`repetition_report(text, n=3)` casefolds Unicode word tokens and reports:

- Word count and the number of overlapping word n-grams.
- Repeated n-gram occurrences beyond the first, divided by total occurrences.
- Sentence count and exact duplicate sentence occurrences beyond the first.

For `"a b c a b c"`, there are four trigrams, one repeated occurrence, and a
repeated fraction of `0.25`. Empty/short texts have zero trigram repetition.
Sentences split after `.`, `!` or `?` followed by whitespace; decimals remain
intact. Sentence matching ignores case and repeated whitespace but retains
punctuation. Abbreviations and unusual formatting may be segmented incorrectly.
Tables, standard phrases, and legitimate comparisons may repeat. No automatic
pass/fail threshold is assigned. Word repetition differs from the earlier
model-token diagnostic; the fractions are not interchangeable. EOS and token
cap detection require generation metadata and are outside this phase.

## Verification and remaining scope

New tests use synthetic values/text only, without dataset reads, downloads or
GPU access. Run with the existing environment:

```powershell
.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider tests
.venv/Scripts/ruff.exe check --no-cache src/sba_928/evaluation tests/test_evaluation_schema.py tests/test_evaluation_numerical.py tests/test_evaluation_repetition.py tests/test_evaluation_isolation.py
```

General prose claim extraction, completed manual-review storage,
and an offline generation runner remain future work. Semantic instruction
compliance, statistical reasoning, unsupported assertions, fairness, and
business recommendations remain manual judgments. Phase 2 authorizes only
the small pilot below, not the proposed 96-case dataset or any model evaluation.

## Phase 2: provenance and development pilot

`evaluation/dataset.py` introduces `sba928_evaluation_v2` without changing the
strict v1 schema. Every v2 record contains the v1 input/expected-fact fields
plus `scenario_id`, `family`, `split`, `paraphrase_of`, `calculations`,
`response_requirements`, `review_criteria`, and `provenance`.

The split must be `development_validation`. There must be exactly one canonical
case (`paraphrase_of: null`) per scenario. Other variants reference that canonical
case ID and preserve context, factual expectations, calculations, family,
requirements, review criteria, provenance kind and source-group membership.
They may vary the instruction and case ID. Score/report scenarios separately
from prompts: variants are not independent observations.

Calculations use named operations, exactly two decimal operands, and a typed
`result` fact. Only `conversion_rate` and `percentage_point_difference` are
supported. Their results are recomputed with Phase 1 arithmetic and must occur
in `expected_facts` with the correct unit, attribution and value. Empty calculation
lists are valid for interpretation-only tasks. Requirements and review criteria
are nonempty lists of unique text strings, intended for human review; they are
not keyword-based semantic scores.

Provenance contains exactly:

- `kind`: `synthetic` or `validation_source`.
- `source_group_ids`: empty for synthetic cases; otherwise a nonempty unique
  subset of the source manifest's validation groups.
- `source_manifest_sha256`: hash of the exact original manifest bytes.
- `evidence_sha256`: hash of the exact UTF-8 context string.
- `description`: nonempty authoring/derivation description.

Synthetic contexts must explicitly include the synthetic label. They cannot
claim source-group membership. Validation-source cases reject training, test,
unknown, empty or duplicate source groups. The source manifest itself must have
unique, disjoint source-group IDs and supervised record IDs across all three
splits. Reading these test **IDs** does not open test examples or predictions.

Hashes bind files/metadata but do not authenticate an author or prove that
context facts were correctly derived. Source-group membership is a provenance
check, not a factual-grounding proof. Future source-derived cases need a reviewed
derivation against permitted validation rows. Synthetic origin is an author
declaration; lexical screening cannot prove independent invention.

### Duplicate checks and access boundaries

Exact equality of normalized `instruction + context` is forbidden even when
case IDs, scenario IDs or expectations differ. Normalization casefolds and
collapses whitespace; it does not strip numbers. Declared valid paraphrases
are retained and explicitly reported. Sibling paraphrases share the canonical
parent. Exact prompt copies never receive a paraphrase exemption.

Other pairs are flagged `needs_review` when their normalized contexts match,
or when the maximum SequenceMatcher similarity of the full inputs or contexts
is at least **0.90**. The threshold is a conservative screen, fixed without
examining test examples or predictions. It is not proof of leakage. Duplicate
reports are stable under case ordering. Automatic pass means no candidates
under these rules, not absence of all semantic overlap.

`load_development_suite(cases_path, suite_manifest_path, source_manifest_path)`
requires `cases.jsonl` and its `manifest.json` in a separate development directory.
It verifies hashes, schema versions, declared counts and family coverage. It
does not scan directories or read original test examples, validation examples,
the source CSV, model files, or historical predictions. With
`check_training_references=True`, it reads only `train.jsonl` beside the source
manifest, verifies its manifest hash and record IDs, then screens training
instructions/contexts against the pilot. There is no held-out reference option.

### Pilot contents

The pilot is in `data/evaluation/dev_pilot_v1/`:

- `cases.jsonl`: 16 hand-authored synthetic development cases, 15 scenarios.
- `manifest.json`: suite version, source-manifest binding, case-file hash,
  family/scenario counts, authoring description and limitations.
- `validation_report.json`: provenance, calculation and duplicate-check results,
  including screening against 18 permitted training references; no predictions.

| Family | Two pilot cases |
|---|---|
| Conversion | 43/250 = 17.2%; fraction 0.064 corresponds to 6.4% |
| Percentage points | Positive 5.5-point difference; negative 4.5-point difference |
| Region/traffic | Eligible coded-region extrema; coded traffic comparison |
| Boolean groups | Promotion True/False comparison; near-holiday True/False comparison |
| Sample thresholds | Inclusive 100-session boundary; no eligible group |
| Association/causation | Confounded banner comparison; observational device comparison |
| Bias/fairness | Outcome imbalance without protected attributes; unequal segment sizes |
| Recommendations | Controlled layout-test recommendation and intentional paraphrase |

All 16 cases have empty source-group lists. All include expected factual anchors,
response requirements and manual criteria. Nine calculations are declared and
checked. The two recommendation cases use `scenario_id: pilot-layout-test` with
`pilot-recommendation-02` referencing `pilot-recommendation-01`.

The initial validation found no exact duplicates or unplanned near-duplicates
within the pilot or against the 18 training references. The one intentional
paraphrase pair was preserved. Training and validation JSONL hashes still match
the original source manifest. Original supervised records and historical results
were not written. Held-out content was not opened or screened, so no claim of
content-level disjointness from it is made.

### Verification and limitations

Run the full pytest suite and relevant Ruff checks in the existing environment:

```powershell
.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider tests
.venv/Scripts/ruff.exe check --no-cache src/sba_928/evaluation tests/test_evaluation_*.py
.venv/Scripts/ruff.exe format --no-cache --check src/sba_928/evaluation tests/test_evaluation_*.py
```

Tests cover permitted/forbidden groups, overlapping manifest splits, hash
tampering, duplicate identities/prompts, near-duplicate review, paraphrase
exemptions, calculation consistency, suite integrity and allowed file access.
Fixtures are synthetic apart from reading this pilot and source-manifest metadata.
The isolation test verifies that the new module imports without model libraries
or opening data. Temporary test files never replace project artifacts.

This is a schema/safeguard pilot, not a model-performance benchmark. No model
inference, retraining, cached-tokenizer length checks,
protected-group metric, or independent human annotation study has been performed.
The pilot does not exercise real validation-row derivation and contains only two
prompts per family. Evidence grounding, semantic duplicate review and review-rule
calibration remain human responsibilities. Broader case coverage and any model
run require a later approval.

## Phase 3: offline saved-response scoring

`scoring.py` uses the numerical and repetition primitives. `reporting.py` loads
a validated development suite, validates saved response records, scores them,
and writes JSON plus Markdown into a fresh UUID run directory. Imports are inert;
there are no model, tokenizer, training, network or generation calls.

### Saved-response contract

Each JSONL record has exactly `schema_version: sba928_saved_response_v1`,
`case_id`, `model_id`, `response` (a string, including empty responses), and
`generation`. Every declared model must supply exactly one response for every
development case. Models are explicitly declared by the caller; missing models
cannot silently disappear from a report. Missing, duplicate or unexpected
model/case pairs are rejected before writing outputs.

Generation metadata has exactly these fields:

| Field | Rule |
|---|---|
| `origin` | `saved_model` or `synthetic_fixture` |
| `model_revision`, `tokenizer_revision`, `device` | Nonempty identifiers |
| `input_sha256` | Matches the exact UTF-8 `instruction + "\n\nContext:\n" + context` input |
| `response_format` | `fact_lines` or `free_text` |
| `seed` | Nonnegative integer |
| `settings` | JSON-compatible object with boolean `do_sample`, positive integer `num_beams`, positive integer `max_new_tokens`; extra settings retained |
| `generated_tokens` | Nonnegative integer, no larger than the declared cap |
| `termination` | `eos`, `length_cap`, or `unknown`; a declared cap termination must reach the cap |

NaN/infinity metadata is rejected. Generation metadata is a declaration from
the response producer, not independent proof of model execution or identity.
Do not guess missing revisions, token counts or settings for historical outputs.
Synthetic test fixtures explicitly use `synthetic_fixture`, a fixture model ID,
zero generated tokens, and `unknown` termination because no tokens were generated.

### Deterministic scope

`fact_lines` means the **actual saved response** uses one explicit claim per line:

```text
Store A | sessions | 250 | count
Store A | purchases | 43 | count
Store A | conversion_rate | 17.2 | percent
```

There is no header row. Labels and units must match the case's canonical labels.
Rows must have exactly four nonempty pipe-separated fields. Value precision is
judged using the expected fact, never a precision supplied by the response.
Correct labels with a wrong unit fail unit attribution; impossible values such
as 954 percent fail validation and remain visible as incorrect claims. Different
groups do not receive credit merely for mentioning the expected number.

The scorer retains line numbers and exact evidence text. Duplicate/conflicting
claims for a fact require review. Unmatched claims are preserved for unsupported-
assertion review. When every nonblank line follows the declared fact-line
structure, absent required facts count as `missing`. If some lines are unparsed,
absence is uncertain and unobserved facts become `needs_review` instead.

For `free_text`, no numbers are extracted automatically, even when correct values
appear in the prose. Expected facts remain `needs_review`. An empty response
is a deterministic omission and counts as `missing`. Do not reformat prose into
fact lines or mark it structured merely to obtain automatic scores; future human
annotations need their own auditable attribution contract. The original pilot
does not request this fact-line format, so saved prose answers normally require
manual numerical review. This phase does not change any pilot prompt.

Arithmetic scores reuse the extraction result for each declared calculation's
expected result, after Phase 2 has recomputed and validated that result. They
measure the **stated numerical result**, not whether the model used the formula,
correct operands, or sound reasoning. The same result may appear in both
numerical and arithmetic dimensions; there is no combined overall score.

Repetition uses Phase 1 word-trigram and duplicate-sentence signals. Declared
length-cap responses are counted separately. No automatic quality threshold,
causal judgment, statistical-significance judgment or fairness finding is made.

### Reports and denominators

`sba928_scoring_report_v1` retains every raw response, generation record, scenario
ID, family, expected fact, evidence span, numerical status, arithmetic status,
repetition signal, response requirement, and review criterion.

Numerical and arithmetic summaries, per case and model, preserve:

- `applicable`: all expected facts/results in that dimension.
- `correct`, `incorrect`, `missing`, `needs_review`: disjoint status counts.
- `deterministically_scored`: correct + incorrect + missing.
- `accuracy_on_scored`: correct/scored, or null when nothing is scored.
- `coverage`: scored/applicable, or null when the dimension is not applicable.

Pending facts are not silently treated as wrong or correct. Never interpret a
high accuracy with low coverage as strong overall performance. Cases without
calculations have arithmetic status `not_applicable`, with null accuracy and
coverage. Repetition denominators count responses; sentence/n-gram denominators
remain available individually. Origin counts distinguish synthetic fixtures.

Instruction compliance, statistical reasoning, fairness and unsupported
assertions each have manual fields: `status: pending`, null rating/reviewer/
rationale, and an empty evidence list. Their per-model summaries report pending
and reviewed denominators. This phase provides review placeholders, not a
review-import or adjudication workflow. Overall review remains pending even
when all deterministic facts are correct.

Reports flag whether declared generation settings match; this does not establish
full comparability across devices, revisions, formats or seeds. Aggregation is
by case, with scenario IDs and the 15-scenario count retained. Intentional
paraphrases are correlated and are not independent benchmark observations.

### Running and preservation

The named input file must be `saved_responses.jsonl`, explicitly prepared for
development cases. The workflow does not discover historical prediction files
or open held-out examples. It rejects unresolved development duplicate candidates
before scoring. Its default suite validation reads no training references; the
Phase 2 pilot's training-reference screen remains separate historical evidence.

```powershell
.venv/Scripts/python.exe -B -m sba_928.evaluation.reporting --predictions PATH/saved_responses.jsonl --suite-dir data/evaluation/dev_pilot_v1 --source-manifest data/processed/manifest.json --model MODEL_ID --output-root artifacts/evaluation_v3
```

Repeat `--model` for every model in the complete saved-response matrix. The
workflow records prediction, suite/source-manifest and implementation hashes,
Python version, and `models_executed: false`. No wall-clock field affects scoring;
reports are stable under input record ordering. Output directories are named
`run-<uuid>`, created with `exist_ok=False`; `report.json` and `summary.md` use
exclusive creation. A UUID collision fails without replacing files. Input or
validation failures create no output directory. An I/O failure during writing
may leave a partial new directory; existing runs remain untouched.

Synthetic saved-response tests cover complete ID matrices, metadata, input
binding, correct/incorrect/missing/pending facts, invalid units/values,
conflicting claims, repetition, deterministic order, access boundaries, and
unique output preservation. They do not measure either FLAN-T5 model. Any saved
synthetic demonstration is explicitly labeled as workflow evidence, not model
performance. Future work includes audited prose annotations, completed manual
reviews and scenario-level summaries; inference still requires separate approval.

The Phase 3 synthetic demonstration is saved in
`artifacts/evaluation_v3/run-33c778a357c34d508dc464f03b09ff52/` (`report.json` and
`summary.md`). Its hand-authored input is in
`artifacts/evaluation_v3/run-input-317857169d9a4715b5a8c6143fa8fc69/saved_responses.jsonl`.
All 16 responses are labeled `synthetic_fixture`; they deliberately include a
numeric error, an empty response, free-text review cases and a repeated sentence.
This demonstration has 56 applicable numeric facts (52 scored, 4 pending) and
9 arithmetic results (8 scored, 1 pending). These are workflow fixture results,
not estimates of either model's performance.
