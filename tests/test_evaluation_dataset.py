"""Synthetic provenance/duplicate cases and the separately versioned pilot."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sba_928.evaluation.dataset import (
    CASE_VERSION,
    FAMILIES,
    SUITE_VERSION,
    duplicate_report,
    load_development_suite,
    sha256,
    validate_development_cases,
    validate_source_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source_manifest():
    return {
        "schema_version": "supervised_uci468_v1",
        "source_sha256": "a" * 64,
        "source_group_ids": {
            "train": ["1" * 64],
            "validation": ["2" * 64],
            "test": ["3" * 64],
        },
        "record_ids": {
            "train": ["original-train"],
            "validation": ["original-validation"],
            "test": ["original-test"],
        },
    }


@pytest.fixture
def case():
    context = "Synthetic Store X: 200 sessions and 25 purchases."
    result = {
        "group": "Store X",
        "metric": "conversion_rate",
        "value": "12.5",
        "unit": "percent",
        "decimal_places": 1,
    }
    return {
        "schema_version": CASE_VERSION,
        "case_id": "dev-1",
        "scenario_id": "scenario-1",
        "family": "conversion",
        "split": "development_validation",
        "paraphrase_of": None,
        "instruction": "Calculate Store X conversion.",
        "context": context,
        "expected_facts": [result],
        "calculations": [
            {
                "operation": "conversion_rate",
                "operands": ["25", "200"],
                "result": result,
            }
        ],
        "response_requirements": ["Report conversion as a percentage."],
        "review_criteria": ["Check the denominator and group attribution."],
        "provenance": {
            "kind": "synthetic",
            "source_group_ids": [],
            "source_manifest_sha256": "b" * 64,
            "evidence_sha256": sha256(context.encode()),
            "description": "Hand-authored synthetic unit fixture.",
        },
    }


def validate(records, source_manifest):
    return validate_development_cases(records, source_manifest, "b" * 64)


def test_valid_synthetic_provenance(case, source_manifest):
    report = validate([case], source_manifest)
    assert report["status"] == "passed"
    assert report["provenance_counts"] == {"synthetic": 1}


def test_validation_groups_accepted(case, source_manifest):
    case["provenance"]["kind"] = "validation_source"
    case["provenance"]["source_group_ids"] = source_manifest["source_group_ids"][
        "validation"
    ]
    assert validate([case], source_manifest)["source_group_checks_passed"]


@pytest.mark.parametrize("group", ["1" * 64, "3" * 64, "4" * 64])
def test_train_test_and_unknown_groups_rejected(case, source_manifest, group):
    case["provenance"]["kind"] = "validation_source"
    case["provenance"]["source_group_ids"] = [group]
    with pytest.raises(ValueError, match="validation source groups only"):
        validate([case], source_manifest)


@pytest.mark.parametrize("groups", [[], ["2" * 64, "2" * 64]])
def test_empty_or_duplicate_validation_groups(case, source_manifest, groups):
    case["provenance"].update(kind="validation_source", source_group_ids=groups)
    with pytest.raises(ValueError):
        validate([case], source_manifest)


def test_synthetic_cannot_hide_source_membership(case, source_manifest):
    case["provenance"]["source_group_ids"] = ["1" * 64]
    with pytest.raises(ValueError, match="Synthetic"):
        validate([case], source_manifest)


def test_synthetic_label_required_in_context(case, source_manifest):
    case["context"] = "Store X has 200 sessions and 25 purchases."
    case["provenance"]["evidence_sha256"] = sha256(case["context"].encode())
    with pytest.raises(ValueError, match="labeled"):
        validate([case], source_manifest)


@pytest.mark.parametrize("field", ["source_manifest_sha256", "evidence_sha256"])
def test_manifest_and_evidence_tampering(case, source_manifest, field):
    case["provenance"][field] = "c" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        validate([case], source_manifest)


@pytest.mark.parametrize("field", ["source_group_ids", "record_ids"])
def test_manifest_overlap_rejected(source_manifest, field):
    source_manifest[field]["validation"] = source_manifest[field]["test"]
    with pytest.raises(ValueError, match="overlapping"):
        validate_source_manifest(source_manifest)


@pytest.mark.parametrize("change", ["split", "version", "source", "group", "empty"])
def test_invalid_manifest_metadata(source_manifest, change):
    if change == "split":
        del source_manifest["source_group_ids"]["test"]
    elif change == "version":
        source_manifest["schema_version"] = "v2"
    elif change == "source":
        source_manifest["source_sha256"] = "not a hash"
    elif change == "group":
        source_manifest["source_group_ids"]["train"] = ["not a hash"]
    else:
        source_manifest["record_ids"]["train"] = []
    with pytest.raises(ValueError):
        validate_source_manifest(source_manifest)


def test_case_ids_cannot_reuse_originals(case, source_manifest):
    case["case_id"] = "original-test"
    with pytest.raises(ValueError, match="collides"):
        validate([case], source_manifest)


def test_renamed_normalized_exact_duplicate_rejected(case, source_manifest):
    duplicate = deepcopy(case)
    duplicate.update(case_id="dev-2", scenario_id="scenario-2")
    duplicate["instruction"] = "  CALCULATE   Store X conversion. "
    with pytest.raises(ValueError, match="Exact duplicate"):
        validate([case, duplicate], source_manifest)


def test_near_duplicate_requires_review(case, source_manifest):
    candidate = deepcopy(case)
    candidate.update(
        case_id="dev-2",
        scenario_id="scenario-2",
        instruction="Calculate Store X conversion now.",
    )
    report = validate([case, candidate], source_manifest)
    assert report["status"] == "needs_review"
    assert (
        report["duplicate_checks"]["near_duplicates"][0]["reason"] == "shared_evidence"
    )


def test_intentional_paraphrase_preserves_scenario(case, source_manifest):
    paraphrase = deepcopy(case)
    paraphrase.update(
        case_id="dev-2",
        paraphrase_of="dev-1",
        instruction="What proportion of sessions purchased?",
    )
    report = validate([case, paraphrase], source_manifest)
    assert report["scenario_count"] == 1
    assert report["status"] == "passed"
    assert len(report["duplicate_checks"]["intentional_paraphrases"]) == 1
    assert validate([paraphrase, case], source_manifest) == report


def test_multiple_paraphrases_share_one_scenario(case, source_manifest):
    first, second = deepcopy(case), deepcopy(case)
    first.update(
        case_id="dev-2", paraphrase_of="dev-1", instruction="Interpret this rate."
    )
    second.update(
        case_id="dev-3",
        paraphrase_of="dev-1",
        instruction="State the purchase percentage.",
    )
    report = validate([case, first, second], source_manifest)
    assert report["status"] == "passed"
    assert report["scenario_count"] == 1
    assert len(report["duplicate_checks"]["intentional_paraphrases"]) == 3


def test_text_similarity_flag_without_identical_evidence(case, source_manifest):
    candidate = deepcopy(case)
    candidate.update(case_id="dev-2", scenario_id="scenario-2")
    candidate["context"] += " Today."
    candidate["provenance"]["evidence_sha256"] = sha256(candidate["context"].encode())
    report = validate([case, candidate], source_manifest)
    assert report["status"] == "needs_review"
    assert (
        report["duplicate_checks"]["near_duplicates"][0]["reason"] == "text_similarity"
    )


@pytest.mark.parametrize(
    "change", ["case_id", "split", "version", "requirements", "review"]
)
def test_case_identity_and_required_metadata(case, source_manifest, change):
    if change == "case_id":
        with pytest.raises(ValueError, match="Duplicate case IDs"):
            validate([case, deepcopy(case)], source_manifest)
        return
    if change == "split":
        case["split"] = "test"
    elif change == "version":
        case["schema_version"] = "sba928_evaluation_v1"
    elif change == "requirements":
        case["response_requirements"] = []
    else:
        case["review_criteria"] = []
    with pytest.raises(ValueError):
        validate([case], source_manifest)


@pytest.mark.parametrize("change", ["context", "requirements", "parent", "canonical"])
def test_paraphrase_exemption_cannot_hide_changed_evidence(
    case, source_manifest, change
):
    paraphrase = deepcopy(case)
    paraphrase.update(
        case_id="dev-2", paraphrase_of="dev-1", instruction="Interpret conversion."
    )
    if change == "context":
        paraphrase["context"] += " Different population."
        paraphrase["provenance"]["evidence_sha256"] = sha256(
            paraphrase["context"].encode()
        )
    elif change == "requirements":
        paraphrase["response_requirements"] = ["A different requirement."]
    elif change == "parent":
        paraphrase["paraphrase_of"] = "missing-parent"
    else:
        paraphrase["paraphrase_of"] = None
    with pytest.raises(ValueError):
        validate([case, paraphrase], source_manifest)


def test_exact_training_reference_collision(case, source_manifest):
    references = [
        {
            "case_id": "training:known",
            "instruction": case["instruction"],
            "context": case["context"],
        }
    ]
    with pytest.raises(ValueError, match="Exact duplicate"):
        validate_development_cases([case], source_manifest, "b" * 64, references)


@pytest.mark.parametrize(
    "change", ["operation", "value", "unit", "operands", "expectation"]
)
def test_calculation_errors_rejected(case, source_manifest, change):
    calculation = deepcopy(case["calculations"][0])
    case["calculations"] = [calculation]
    if change == "operation":
        calculation["operation"] = "execute_python"
    elif change == "value":
        calculation["result"]["value"] = "99"
    elif change == "unit":
        calculation["result"]["unit"] = "count"
    elif change == "operands":
        calculation["operands"] = ["25", "0"]
    else:
        calculation["result"]["group"] = "Wrong group"
    with pytest.raises(ValueError):
        validate([case], source_manifest)


@pytest.mark.parametrize("threshold", [0, 1.1, True, "0.9"])
def test_invalid_duplicate_threshold(threshold):
    with pytest.raises(ValueError):
        duplicate_report([], threshold=threshold)


def write_suite(tmp_path, source_manifest, case):
    source_dir = tmp_path / "processed"
    suite_dir = tmp_path / "development"
    source_dir.mkdir()
    suite_dir.mkdir()
    source_bytes = json.dumps(source_manifest).encode()
    (source_dir / "manifest.json").write_bytes(source_bytes)
    case["provenance"]["source_manifest_sha256"] = sha256(source_bytes)
    case_bytes = (json.dumps(case) + "\n").encode()
    (suite_dir / "cases.jsonl").write_bytes(case_bytes)
    suite = {
        "schema_version": SUITE_VERSION,
        "case_schema_version": CASE_VERSION,
        "split": "development_validation",
        "source_manifest_sha256": sha256(source_bytes),
        "cases_sha256": sha256(case_bytes),
        "case_count": 1,
        "scenario_count": 1,
        "family_counts": {"conversion": 1},
    }
    (suite_dir / "manifest.json").write_text(json.dumps(suite), encoding="utf-8")
    return suite_dir, source_dir


def test_loader_reads_only_explicit_permitted_files(
    tmp_path, source_manifest, case, monkeypatch
):
    suite_dir, source_dir = write_suite(tmp_path, source_manifest, case)
    allowed = {
        suite_dir / "cases.jsonl",
        suite_dir / "manifest.json",
        source_dir / "manifest.json",
    }
    original = Path.read_bytes
    opened = []

    def guarded(path):
        assert path in allowed, f"Unexpected file access: {path}"
        opened.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    _, report = load_development_suite(
        suite_dir / "cases.jsonl",
        suite_dir / "manifest.json",
        source_dir / "manifest.json",
    )
    assert report["status"] == "passed"
    assert set(opened) == allowed
    assert report["training_reference_count"] == 0


@pytest.mark.parametrize(
    "field",
    [
        "cases_sha256",
        "source_manifest_sha256",
        "case_count",
        "scenario_count",
        "family_counts",
    ],
)
def test_loader_rejects_suite_tampering(tmp_path, source_manifest, case, field):
    suite_dir, source_dir = write_suite(tmp_path, source_manifest, case)
    path = suite_dir / "manifest.json"
    suite = json.loads(path.read_text())
    suite[field] = None
    path.write_text(json.dumps(suite))
    with pytest.raises(ValueError):
        load_development_suite(
            suite_dir / "cases.jsonl", path, source_dir / "manifest.json"
        )


@pytest.mark.parametrize("change", ["none", "hash", "ids"])
def test_training_references_are_hash_and_id_checked(
    tmp_path, source_manifest, case, change
):
    training_record = {
        "record_id": "original-train",
        "instruction": "Describe a different fixture.",
        "context": "Separate fictional field definitions with no rates.",
    }
    if change == "ids":
        training_record["record_id"] = "wrong-id"
    payload = (json.dumps(training_record) + "\n").encode()
    source_manifest["output_file_sha256"] = {
        "data/processed/train.jsonl": sha256(payload)
    }
    suite_dir, source_dir = write_suite(tmp_path, source_manifest, case)
    (source_dir / "train.jsonl").write_bytes(
        payload + (b" " if change == "hash" else b"")
    )
    if change != "none":
        with pytest.raises(ValueError, match="reference"):
            load_development_suite(
                suite_dir / "cases.jsonl",
                suite_dir / "manifest.json",
                source_dir / "manifest.json",
                check_training_references=True,
            )
    else:
        _, report = load_development_suite(
            suite_dir / "cases.jsonl",
            suite_dir / "manifest.json",
            source_dir / "manifest.json",
            check_training_references=True,
        )
        assert report["training_reference_count"] == 1
        assert report["status"] == "passed"


def test_original_test_path_is_rejected_before_any_read(tmp_path, monkeypatch):
    def forbidden(path):
        raise AssertionError("No input should be read for an invalid development path")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(ValueError, match="cases.jsonl"):
        load_development_suite(
            tmp_path / "test.jsonl",
            tmp_path / "manifest.json",
            tmp_path / "processed/manifest.json",
        )


def test_pilot_has_two_cases_per_family_and_no_source_reuse():
    pilot = ROOT / "data/evaluation/dev_pilot_v1"
    records, report = load_development_suite(
        pilot / "cases.jsonl",
        pilot / "manifest.json",
        ROOT / "data/processed/manifest.json",
    )
    assert len(records) == 16
    assert report["family_counts"] == dict.fromkeys(FAMILIES, 2)
    assert report["scenario_count"] == 15
    assert report["provenance_counts"] == {"synthetic": 16}
    assert all(record["provenance"]["source_group_ids"] == [] for record in records)
    assert report["status"] == "passed"
    assert len(report["duplicate_checks"]["intentional_paraphrases"]) == 1
    # A known answer independently anchors recomputation instead of only
    # comparing generated values to another use of the same calculation code.
    conversion = next(
        record for record in records if record["case_id"] == "pilot-conversion-01"
    )
    assert conversion["calculations"][0]["result"]["value"] == "17.2"
