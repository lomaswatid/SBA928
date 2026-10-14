"""Development-case provenance and duplicate checks; no model dependencies.

Only explicit suite files, the source manifest, and optional training references
are read. Held-out examples and prediction files are never discovered or opened.
"""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

from .numerical import conversion_rate, percentage_point_difference
from .schema import SCHEMA_VERSION, EvaluationCase, Fact, nonempty

CASE_VERSION = "sba928_evaluation_v2"
SUITE_VERSION = "sba928_development_suite_v1"
FAMILIES = (
    "conversion",
    "percentage_points",
    "region_traffic",
    "boolean_groups",
    "sample_thresholds",
    "association_causation",
    "bias_fairness",
    "recommendations",
)


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def normalized(text):
    return " ".join(text.casefold().split())


def _fields(data, fields, label):
    if not isinstance(data, Mapping) or set(data) != set(fields):
        raise ValueError(f"{label} must contain exactly its versioned fields")


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA-256 digest")


def validate_source_manifest(manifest):
    """Validate split metadata, without opening any source/example files."""
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != "supervised_uci468_v1"
    ):
        raise ValueError("Unsupported source manifest")
    _digest(manifest.get("source_sha256"))
    for field in ("source_group_ids", "record_ids"):
        split_values = manifest.get(field)
        if not isinstance(split_values, Mapping) or set(split_values) != {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(f"Invalid {field} splits")
        seen = set()
        for values in split_values.values():
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(v, str) or not v.strip() for v in values)
            ):
                raise ValueError(f"Invalid {field} values")
            if len(values) != len(set(values)) or seen.intersection(values):
                raise ValueError(f"Duplicate or overlapping {field}")
            if field == "source_group_ids":
                for value in values:
                    _digest(value)
            seen.update(values)


def _text_list(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a nonempty list")
    for value in values:
        nonempty(value, label)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label}")


def validate_development_case(record, source_manifest, manifest_digest):
    """Validate v2 metadata and independently recompute declared calculations."""
    _fields(
        record,
        {
            "schema_version",
            "case_id",
            "scenario_id",
            "family",
            "split",
            "paraphrase_of",
            "instruction",
            "context",
            "expected_facts",
            "calculations",
            "response_requirements",
            "review_criteria",
            "provenance",
        },
        "Development case",
    )
    if (
        record["schema_version"] != CASE_VERSION
        or record["split"] != "development_validation"
    ):
        raise ValueError("Cases must use v2 and development_validation")
    nonempty(record["scenario_id"], "scenario_id")
    if record["family"] not in FAMILIES:
        raise ValueError("Unknown scenario family")
    if record["paraphrase_of"] is not None:
        nonempty(record["paraphrase_of"], "paraphrase_of")
    # Preserve the strict v1 contract rather than weakening its field checks.
    case = EvaluationCase.from_mapping(
        {
            key: record[key]
            for key in ("case_id", "instruction", "context", "expected_facts")
        }
        | {"schema_version": SCHEMA_VERSION}
    )
    _text_list(record["response_requirements"], "response_requirements")
    _text_list(record["review_criteria"], "review_criteria")
    provenance = record["provenance"]
    _fields(
        provenance,
        {
            "kind",
            "source_group_ids",
            "source_manifest_sha256",
            "evidence_sha256",
            "description",
        },
        "Provenance",
    )
    nonempty(provenance["description"], "provenance description")
    _digest(manifest_digest)
    if provenance["source_manifest_sha256"] != manifest_digest:
        raise ValueError("Source manifest hash mismatch")
    if provenance["evidence_sha256"] != sha256(record["context"].encode("utf-8")):
        raise ValueError("Evidence hash mismatch")
    groups = provenance["source_group_ids"]
    if not isinstance(groups, list) or any(not isinstance(g, str) for g in groups):
        raise ValueError("Invalid source groups")
    if len(groups) != len(set(groups)):
        raise ValueError("Duplicate source groups")
    if provenance["kind"] == "synthetic":
        if groups:
            raise ValueError("Synthetic cases cannot claim source groups")
        if "synthetic" not in record["context"].casefold():
            raise ValueError("Synthetic evidence must be labeled in the model context")
    elif provenance["kind"] == "validation_source":
        allowed = set(source_manifest["source_group_ids"]["validation"])
        if not groups or not set(groups) <= allowed:
            raise ValueError(
                "Development evidence must use validation source groups only"
            )
    else:
        raise ValueError("Unsupported provenance kind")
    if not isinstance(record["calculations"], list):
        raise TypeError("calculations must be a list")
    results = set()
    for calculation in record["calculations"]:
        _fields(calculation, {"operation", "operands", "result"}, "Calculation")
        result = Fact.from_mapping(calculation["result"])
        operands = calculation["operands"]
        if not isinstance(operands, list) or len(operands) != 2:
            raise ValueError("Calculations need exactly two operands")
        operation = calculation["operation"]
        if operation == "conversion_rate":
            computed = conversion_rate(*operands, decimal_places=result.decimal_places)
            unit = "percent"
        elif operation == "percentage_point_difference":
            computed = percentage_point_difference(
                *operands, decimal_places=result.decimal_places
            )
            unit = "percentage_point"
        else:
            raise ValueError("Unsupported calculation operation")
        if (
            result.unit != unit
            or result.value != computed
            or result not in case.expected_facts
        ):
            raise ValueError(
                "Calculation result must match its operation and expected fact"
            )
        if result in results:
            raise ValueError("Duplicate calculation result")
        results.add(result)
    return case


def _scenario_signature(record):
    """Paraphrases may change only IDs/instruction, not evidence or expectations."""
    value = {
        key: record[key]
        for key in (
            "family",
            "expected_facts",
            "calculations",
            "response_requirements",
            "review_criteria",
        )
    }
    value["context"] = normalized(record["context"])
    value["provenance_kind"] = record["provenance"]["kind"]
    value["source_group_ids"] = sorted(record["provenance"]["source_group_ids"])
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def duplicate_report(records, references=(), threshold=0.90):
    """Screen normalized prompts/evidence. Similarity is a manual-review signal."""
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 < threshold <= 1
    ):
        raise ValueError("threshold must be greater than zero and at most one")
    exact, near, intentional = [], [], []

    def compare(left, right, external=False):
        ids = [left["case_id"], right["case_id"]]
        input_left = normalized(left["instruction"] + "\n" + left["context"])
        input_right = normalized(right["instruction"] + "\n" + right["context"])
        if input_left == input_right:
            exact.append({"case_ids": ids, "external_reference": external})
            return
        if not external and left["scenario_id"] == right["scenario_id"]:
            linked = (
                left["paraphrase_of"] == right["case_id"]
                or right["paraphrase_of"] == left["case_id"]
                or (
                    left["paraphrase_of"] is not None
                    and left["paraphrase_of"] == right["paraphrase_of"]
                )
            )
            if linked and _scenario_signature(left) == _scenario_signature(right):
                intentional.append(
                    {"case_ids": ids, "scenario_id": left["scenario_id"]}
                )
                return
        evidence_equal = normalized(left["context"]) == normalized(right["context"])
        similarity = max(
            SequenceMatcher(None, input_left, input_right, autojunk=False).ratio(),
            SequenceMatcher(
                None,
                normalized(left["context"]),
                normalized(right["context"]),
                autojunk=False,
            ).ratio(),
        )
        if evidence_equal or similarity >= threshold:
            near.append(
                {
                    "case_ids": ids,
                    "external_reference": external,
                    "similarity": round(similarity, 4),
                    "reason": "shared_evidence"
                    if evidence_equal
                    else "text_similarity",
                    "status": "needs_review",
                }
            )

    ordered = sorted(records, key=lambda record: record["case_id"])
    for left, right in combinations(ordered, 2):
        compare(left, right)
    for record in ordered:
        for reference in sorted(references, key=lambda reference: reference["case_id"]):
            compare(record, reference, external=True)
    return {
        "exact_duplicates": exact,
        "near_duplicates": near,
        "intentional_paraphrases": intentional,
        "threshold": threshold,
    }


def validate_development_cases(
    records, source_manifest, manifest_digest, references=()
):
    validate_source_manifest(source_manifest)
    if not isinstance(records, list) or not records:
        raise ValueError("Development cases must be a nonempty list")
    ids = set()
    for record in records:
        validate_development_case(record, source_manifest, manifest_digest)
        if record["case_id"] in ids:
            raise ValueError("Duplicate case IDs")
        if record["case_id"] in {
            v for values in source_manifest["record_ids"].values() for v in values
        }:
            raise ValueError("Case ID collides with original supervised records")
        ids.add(record["case_id"])
    scenarios = {}
    for record in records:
        scenarios.setdefault(record["scenario_id"], []).append(record)
    for members in scenarios.values():
        originals = [record for record in members if record["paraphrase_of"] is None]
        if len(originals) != 1:
            raise ValueError("Each scenario needs exactly one canonical case")
        original = originals[0]
        for record in members:
            if record is original:
                continue
            if record["paraphrase_of"] != original["case_id"] or _scenario_signature(
                record
            ) != _scenario_signature(original):
                raise ValueError(
                    "Paraphrase must reference its canonical case and preserve evidence/rules"
                )
    duplicates = duplicate_report(records, references)
    if duplicates["exact_duplicates"]:
        raise ValueError(
            "Exact duplicate prompts are forbidden, including renamed cases"
        )
    return {
        "case_count": len(records),
        "scenario_count": len(scenarios),
        "family_counts": dict(sorted(Counter(r["family"] for r in records).items())),
        "provenance_counts": dict(
            sorted(Counter(r["provenance"]["kind"] for r in records).items())
        ),
        "source_group_checks_passed": True,
        "calculation_checks_passed": True,
        "duplicate_checks": duplicates,
        "status": "needs_review" if duplicates["near_duplicates"] else "passed",
        "test_content_screening": "not performed; held-out content is not accessed",
    }


def load_development_suite(
    cases_path,
    suite_manifest_path,
    source_manifest_path,
    *,
    check_training_references=False,
):
    """Read named inputs only; optional reference access is fixed to train.jsonl."""
    source_path = Path(source_manifest_path)
    if source_path.name != "manifest.json":
        raise ValueError("Expected the original source manifest.json")
    # Reject original supervised records and historical prediction paths up front.
    cases_path = Path(cases_path)
    if cases_path.name != "cases.jsonl":
        raise ValueError("Expected a separate development cases.jsonl")
    suite_path = Path(suite_manifest_path)
    if (
        suite_path.name != "manifest.json"
        or suite_path.parent.resolve() != cases_path.parent.resolve()
        or cases_path.parent.resolve() == source_path.parent.resolve()
    ):
        raise ValueError(
            "Cases and suite manifest must share a separate development directory"
        )
    payload = source_path.read_bytes()
    source_manifest = json.loads(payload)
    validate_source_manifest(source_manifest)
    manifest_digest = sha256(payload)
    suite = json.loads(suite_path.read_bytes())
    if (
        suite.get("schema_version") != SUITE_VERSION
        or suite.get("case_schema_version") != CASE_VERSION
        or suite.get("split") != "development_validation"
        or suite.get("source_manifest_sha256") != manifest_digest
    ):
        raise ValueError("Invalid suite manifest or source-manifest binding")
    case_payload = cases_path.read_bytes()
    if suite.get("cases_sha256") != sha256(case_payload):
        raise ValueError("Development cases hash mismatch")
    records = [json.loads(line) for line in case_payload.decode("utf-8").splitlines()]
    references = []
    if check_training_references:
        train_payload = (source_path.parent / "train.jsonl").read_bytes()
        expected = source_manifest.get("output_file_sha256", {}).get(
            "data/processed/train.jsonl"
        )
        if sha256(train_payload) != expected:
            raise ValueError("Training reference hash mismatch")
        references = [
            {
                "case_id": "training:" + r["record_id"],
                "instruction": r["instruction"],
                "context": r["context"],
            }
            for r in map(json.loads, train_payload.decode("utf-8").splitlines())
        ]
        if sorted(r["case_id"][9:] for r in references) != sorted(
            source_manifest["record_ids"]["train"]
        ):
            raise ValueError("Training reference IDs mismatch")
    report = validate_development_cases(
        records, source_manifest, manifest_digest, references
    )
    for key in ("case_count", "scenario_count", "family_counts"):
        if suite.get(key) != report[key]:
            raise ValueError(f"Suite manifest {key} mismatch")
    report["training_reference_count"] = len(references)
    report["source_manifest_sha256"] = manifest_digest
    report["cases_sha256"] = sha256(case_payload)
    return records, report
