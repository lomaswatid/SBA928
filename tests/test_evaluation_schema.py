"""Synthetic schema checks; no project datasets or models are opened."""

import json
from copy import deepcopy
from decimal import Decimal

import pytest

from sba_928.evaluation.schema import (
    SCHEMA_VERSION,
    EvaluationCase,
    Fact,
    validate_cases,
)


@pytest.fixture
def record():
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": "synthetic-1",
        "instruction": "Compare A and B.",
        "context": "A: 20%; B: 15%.",
        "expected_facts": [
            {
                "group": "A",
                "metric": "conversion_rate",
                "value": "20",
                "unit": "percent",
            }
        ],
    }


def test_case_roundtrip_and_input_excludes_expectations(record):
    case = EvaluationCase.from_mapping(record)
    restored = EvaluationCase.from_mapping(json.loads(json.dumps(case.to_dict())))
    assert restored == case
    assert case.expected_facts[0].value == Decimal(20)
    assert case.format_input() == "Compare A and B.\n\nContext:\nA: 20%; B: 15%."
    assert "expected_facts" not in case.format_input()


@pytest.mark.parametrize(
    "field", ["schema_version", "case_id", "instruction", "context", "expected_facts"]
)
def test_required_case_fields(record, field):
    del record[field]
    with pytest.raises((ValueError, TypeError)):
        EvaluationCase.from_mapping(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "v2"),
        ("case_id", " "),
        ("instruction", None),
        ("context", ""),
        ("expected_facts", []),
        ("expected_facts", {}),
    ],
)
def test_invalid_case_fields(record, field, value):
    record[field] = value
    with pytest.raises((ValueError, TypeError)):
        EvaluationCase.from_mapping(record)


def test_unknown_fields_and_duplicate_attributions(record):
    extra = {**record, "unexpected": True}
    with pytest.raises(ValueError):
        EvaluationCase.from_mapping(extra)
    record["expected_facts"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        EvaluationCase.from_mapping(record)


def test_duplicate_case_ids_and_empty_collection(record):
    for records in ([], [record, deepcopy(record)]):
        with pytest.raises(ValueError):
            validate_cases(records)
    assert len(validate_cases([record])) == 1


@pytest.mark.parametrize(
    "value,unit",
    [
        (True, "count"),
        (0.1, "fraction"),
        ("NaN", "percent"),
        ("Infinity", "percent"),
        ("1e2", "percent"),
        ("1,000", "count"),
        ("-1", "count"),
        ("1.5", "count"),
        ("1.1", "fraction"),
        ("101", "percent"),
        ("-101", "percentage_point"),
        ("3", "dollars"),
        ("9" * 29, "count"),
    ],
)
def test_invalid_facts(value, unit):
    with pytest.raises((ValueError, TypeError)):
        Fact("A", "metric", value, unit)


@pytest.mark.parametrize("places", [True, -1, 13, 1.5])
def test_invalid_precision(places):
    with pytest.raises(ValueError):
        Fact("A", "rate", "20", "percent", places)


def test_invalid_fact_mapping():
    with pytest.raises(ValueError):
        Fact.from_mapping({"group": "A"})
    with pytest.raises(ValueError):
        Fact.from_mapping(
            {
                "group": "A",
                "metric": "rate",
                "value": "20",
                "unit": "percent",
                "extra": 1,
            }
        )


def test_small_decimal_object_roundtrip():
    fact = Fact("A", "rate", Decimal("0.0000001"), "fraction", 7)
    assert fact.to_dict()["value"] == "0.0000001"
    # JSON serialization must remain valid for the same schema.
    assert Fact.from_mapping(fact.to_dict()) == fact
