"""Synthetic saved-response tests; no historical predictions or model execution."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sba_928.evaluation.dataset import load_development_suite
from sba_928.evaluation.reporting import render_summary, score_saved_file, write_report
from sba_928.evaluation.scoring import (
    MANUAL_DIMENSIONS,
    RESPONSE_VERSION,
    input_hash,
    score_predictions,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data/evaluation/dev_pilot_v1"


@pytest.fixture
def cases():
    records, _ = load_development_suite(
        PILOT / "cases.jsonl",
        PILOT / "manifest.json",
        ROOT / "data/processed/manifest.json",
    )
    return records


def prediction(case, model="fixture-A", response=None, response_format="fact_lines"):
    if response is None:
        response = "\n".join(
            f"{f['group']} | {f['metric']} | {f['value']} | {f['unit']}"
            for f in case["expected_facts"]
        )
    return {
        "schema_version": RESPONSE_VERSION,
        "case_id": case["case_id"],
        "model_id": model,
        "response": response,
        "generation": {
            "origin": "synthetic_fixture",
            "model_revision": "fixture-v1",
            "tokenizer_revision": "not-used-fixture",
            "input_sha256": input_hash(case),
            "response_format": response_format,
            "seed": 42,
            "device": "not-executed",
            "settings": {"do_sample": False, "num_beams": 1, "max_new_tokens": 100},
            "generated_tokens": 0,
            "termination": "unknown",
        },
    }


def conversion(cases):
    return next(case for case in cases if case["case_id"] == "pilot-conversion-01")


def one_case_report(cases, text=None, response_format="fact_lines"):
    case = conversion(cases)
    return score_predictions(
        [case],
        [prediction(case, response=text, response_format=response_format)],
        ["fixture-A"],
    )


def test_complete_matrix_and_dimensions(cases):
    predictions = [
        prediction(case, model)
        for model in ("fixture-A", "fixture-B")
        for case in cases
    ]
    report = score_predictions(cases, predictions, ["fixture-A", "fixture-B"])
    assert report["response_count"] == 32
    assert report["scenario_count"] == 15
    for summary in report["model_summaries"].values():
        assert summary["numerical_extraction"]["coverage"] == 1
        assert summary["numerical_extraction"]["accuracy_on_scored"] == 1
        assert summary["arithmetic"]["applicable"] == 9
        assert summary["arithmetic"]["correct"] == 9
        assert summary["manual_review"]["fairness"]["pending"] == 16
    assert report["review_status"] == "pending"


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "unexpected_case", "unexpected_model", "missing_model"],
)
def test_rejects_incomplete_or_ambiguous_matrix(cases, change):
    predictions = [prediction(case) for case in cases]
    models = ["fixture-A"]
    if change == "missing":
        predictions.pop()
    elif change == "duplicate":
        predictions.append(deepcopy(predictions[0]))
    elif change == "unexpected_case":
        predictions[0]["case_id"] = "not-a-development-case"
    elif change == "unexpected_model":
        predictions[0]["model_id"] = "not-declared"
    else:
        models.append("fixture-B")
    with pytest.raises(ValueError):
        score_predictions(cases, predictions, models)


@pytest.mark.parametrize(
    "field", ["schema_version", "case_id", "model_id", "response", "generation"]
)
def test_required_response_fields(cases, field):
    case = conversion(cases)
    record = prediction(case)
    del record[field]
    with pytest.raises(ValueError):
        score_predictions([case], [record], ["fixture-A"])


@pytest.mark.parametrize(
    "field",
    [
        "origin",
        "model_revision",
        "tokenizer_revision",
        "input_sha256",
        "response_format",
        "seed",
        "device",
        "settings",
        "generated_tokens",
        "termination",
    ],
)
def test_generation_metadata_required(cases, field):
    case = conversion(cases)
    record = prediction(case)
    del record["generation"][field]
    with pytest.raises(ValueError, match="generation metadata"):
        score_predictions([case], [record], ["fixture-A"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_sha256", "bad"),
        ("origin", "unknown"),
        ("model_revision", ""),
        ("tokenizer_revision", ""),
        ("device", ""),
        ("seed", True),
        ("seed", -1),
        ("response_format", "guess"),
        ("generated_tokens", 101),
        ("termination", "length_cap"),
    ],
)
def test_invalid_generation_metadata(cases, field, value):
    case = conversion(cases)
    record = prediction(case)
    record["generation"][field] = value
    with pytest.raises(ValueError):
        score_predictions([case], [record], ["fixture-A"])


def test_nonfinite_settings_rejected(cases):
    case = conversion(cases)
    record = prediction(case)
    record["generation"]["settings"]["temperature"] = float("nan")
    with pytest.raises(ValueError):
        score_predictions([case], [record], ["fixture-A"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_beams", 0),
        ("num_beams", True),
        ("max_new_tokens", 0),
        ("do_sample", "false"),
    ],
)
def test_invalid_generation_settings(cases, field, value):
    case = conversion(cases)
    record = prediction(case)
    record["generation"]["settings"][field] = value
    with pytest.raises(ValueError):
        score_predictions([case], [record], ["fixture-A"])


def test_response_text_must_be_preserved_as_string(cases):
    case = conversion(cases)
    record = prediction(case)
    record["response"] = None
    with pytest.raises(TypeError):
        score_predictions([case], [record], ["fixture-A"])


def test_unexpected_fields_rejected(cases):
    case = conversion(cases)
    record = prediction(case)
    record["unverified_score"] = 1
    with pytest.raises(ValueError):
        score_predictions([case], [record], ["fixture-A"])


def test_numerical_error_and_arithmetic_result(cases):
    report = one_case_report(
        cases,
        "Store A | sessions | 250 | count\nStore A | purchases | 43 | count\nStore A | conversion_rate | 18.0 | percent",
    )
    result = report["results"][0]
    assert result["dimension_summaries"]["numerical_extraction"]["correct"] == 2
    assert result["dimension_summaries"]["numerical_extraction"]["incorrect"] == 1
    assert result["arithmetic"][0]["status"] == "incorrect"
    assert result["arithmetic"][0]["scope"].startswith("stated_result_only")


def test_wrong_unit_and_impossible_value_are_visible(cases):
    report = one_case_report(
        cases,
        "Store A | sessions | 250 | count\nStore A | purchases | 43 | count\nStore A | conversion_rate | 17.2 | count",
    )
    assert report["results"][0]["numerical_extraction"][2]["status"] == "incorrect"
    report = one_case_report(cases, "Store A | conversion_rate | 954 | percent")
    claim = report["results"][0]["numerical_extraction"][2]
    assert claim["reason"] == "invalid_fact_value_or_unit"
    assert claim["evidence"][0]["validation_error"]


def test_wrong_group_does_not_receive_credit(cases):
    report = one_case_report(cases, "Store B | conversion_rate | 17.2 | percent")
    result = report["results"][0]
    assert result["numerical_extraction"][2]["status"] == "missing"
    assert (
        result["extraction_notes"]["unmatched_claims"][0]["raw"]["group"] == "Store B"
    )
    assert result["manual_review"]["unsupported_assertions"]["status"] == "pending"


def test_free_text_is_pending_even_when_it_contains_correct_numbers(cases):
    report = one_case_report(
        cases,
        "Store A had 250 sessions, 43 purchases and 17.2% conversion.",
        "free_text",
    )
    summary = report["model_summaries"]["fixture-A"]["numerical_extraction"]
    assert summary["applicable"] == summary["needs_review"] == 3
    assert summary["deterministically_scored"] == 0
    assert summary["accuracy_on_scored"] is None
    assert summary["coverage"] == 0


def test_empty_response_is_missing_not_excluded(cases):
    report = one_case_report(cases, "", "free_text")
    summary = report["model_summaries"]["fixture-A"]["numerical_extraction"]
    assert summary["missing"] == summary["applicable"] == 3
    assert summary["accuracy_on_scored"] == 0


def test_duplicate_claims_need_review(cases):
    report = one_case_report(
        cases,
        "Store A | conversion_rate | 17.2 | percent\nStore A | conversion_rate | 19.0 | percent",
    )
    claim = report["results"][0]["numerical_extraction"][2]
    assert claim["status"] == "needs_review"
    assert len(claim["evidence"]) == 2


def test_mixed_unparsed_prose_prevents_false_missing(cases):
    report = one_case_report(
        cases,
        "Store A | sessions | 250 | count\nOther quantities appear in this prose: 43 and 17.2%.",
    )
    summary = report["results"][0]["dimension_summaries"]["numerical_extraction"]
    assert summary["correct"] == 1
    assert summary["needs_review"] == 2
    assert summary["missing"] == 0


def test_arithmetic_not_applicable_and_manual_fields(cases):
    case = next(case for case in cases if case["case_id"] == "pilot-recommendation-01")
    report = score_predictions([case], [prediction(case)], ["fixture-A"])
    assert (
        report["model_summaries"]["fixture-A"]["arithmetic"]["status"]
        == "not_applicable"
    )
    assert (
        report["model_summaries"]["fixture-A"]["arithmetic"]["accuracy_on_scored"]
        is None
    )
    for dimension in MANUAL_DIMENSIONS:
        assert report["results"][0]["manual_review"][dimension]["rating"] is None


def test_repetition_and_declared_length_cap(cases):
    case = conversion(cases)
    record = prediction(
        case,
        response="Repeat this sentence. Repeat this sentence.",
        response_format="free_text",
    )
    record["generation"].update(generated_tokens=100, termination="length_cap")
    report = score_predictions([case], [record], ["fixture-A"])
    assert report["results"][0]["repetition"]["duplicate_sentence_count"] == 1
    assert (
        report["model_summaries"]["fixture-A"]["repetition"]["length_cap_responses"]
        == 1
    )


def test_order_independence_and_no_input_mutation(cases):
    predictions = [prediction(case) for case in cases]
    before = deepcopy((cases, predictions))
    first = score_predictions(cases, predictions, ["fixture-A"])
    second = score_predictions(
        list(reversed(cases)), list(reversed(predictions)), ["fixture-A"]
    )
    assert first == second
    assert (cases, predictions) == before
    json.dumps(first, allow_nan=False)


def test_mismatched_settings_recorded_without_comparative_claim(cases):
    selected = cases[:2]
    predictions = [prediction(case) for case in selected]
    predictions[1]["generation"]["settings"]["num_beams"] = 2
    report = score_predictions(selected, predictions, ["fixture-A"])
    assert report["generation_settings_match"] is False


def test_unique_outputs_and_no_overwrite(cases, tmp_path):
    report = one_case_report(cases)
    first = write_report(report, tmp_path)
    snapshot = (first / "report.json").read_bytes()
    second = write_report(report, tmp_path)
    assert first != second
    assert (first / "report.json").read_bytes() == snapshot
    assert json.loads(snapshot)["case_count"] == 1
    assert "Individual cases" in (second / "summary.md").read_text()


def test_uuid_collision_refuses_existing_run(cases, tmp_path, monkeypatch):
    from sba_928.evaluation import reporting

    class FixedUUID:
        hex = "fixed"

    monkeypatch.setattr(reporting, "uuid4", FixedUUID)
    report = one_case_report(cases)
    first = write_report(report, tmp_path)
    with pytest.raises(FileExistsError):
        write_report(report, tmp_path)
    assert (first / "report.json").is_file()


def test_saved_workflow_reads_only_allowed_inputs(cases, tmp_path, monkeypatch):
    saved = tmp_path / "saved_responses.jsonl"
    predictions = [prediction(case) for case in cases]
    saved.write_text(
        "".join(json.dumps(record) + "\n" for record in predictions), encoding="utf-8"
    )
    module_dir = ROOT / "src/sba_928/evaluation"
    allowed = {
        saved,
        PILOT / "cases.jsonl",
        PILOT / "manifest.json",
        ROOT / "data/processed/manifest.json",
    }
    allowed.update(
        module_dir / name
        for name in (
            "schema.py",
            "numerical.py",
            "repetition.py",
            "dataset.py",
            "scoring.py",
            "reporting.py",
        )
    )
    original = Path.read_bytes

    def guarded(path):
        assert path in allowed, f"Unexpected read: {path}"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    run = score_saved_file(
        saved,
        PILOT / "cases.jsonl",
        PILOT / "manifest.json",
        ROOT / "data/processed/manifest.json",
        ["fixture-A"],
        tmp_path / "outputs",
    )
    # Reading the result is outside the workflow's input guard.
    monkeypatch.setattr(Path, "read_bytes", original)
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert report["provenance"]["models_executed"] is False
    assert report["response_count"] == 16


def test_bad_matrix_writes_nothing(cases, tmp_path):
    saved = tmp_path / "saved_responses.jsonl"
    saved.write_text(json.dumps(prediction(cases[0])) + "\n", encoding="utf-8")
    output = tmp_path / "outputs"
    with pytest.raises(ValueError, match="Missing"):
        score_saved_file(
            saved,
            PILOT / "cases.jsonl",
            PILOT / "manifest.json",
            ROOT / "data/processed/manifest.json",
            ["fixture-A"],
            output,
        )
    assert not output.exists()


def test_historical_input_name_rejected_before_read(tmp_path, monkeypatch):
    def forbidden(path):
        raise AssertionError("Historical content must not be read")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(ValueError, match="saved_responses.jsonl"):
        score_saved_file(
            tmp_path / "model_comparison.jsonl",
            PILOT / "cases.jsonl",
            PILOT / "manifest.json",
            ROOT / "data/processed/manifest.json",
            ["fixture-A"],
            tmp_path / "outputs",
        )


def test_markdown_model_labels_escape_table_separators(cases):
    case = conversion(cases)
    report = score_predictions(
        [case], [prediction(case, "fixture|model")], ["fixture|model"]
    )
    assert "fixture\\|model" in render_summary(report)
