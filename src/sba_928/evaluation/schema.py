"""Strict v1 factual expectations; no prose interpretation or model imports."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

SCHEMA_VERSION = "sba928_evaluation_v1"
UNITS = frozenset({"count", "fraction", "percent", "percentage_point"})


def decimal_value(value):
    """Accept exact decimal strings, integers or Decimal; reject floats/bools."""
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise TypeError("Value must be a decimal string, integer or Decimal")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Value must be finite")
        if (
            len(value.as_tuple().digits) > 28
            or abs(value.as_tuple().exponent) > 28
            or value.adjusted() > 27
        ):
            raise ValueError("Decimal exceeds v1's 28-digit limit")
    text = format(value, "f") if isinstance(value, Decimal) else str(value)
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text, flags=re.ASCII):
        raise ValueError(
            "Use a finite plain decimal without commas or exponent notation"
        )
    number = Decimal(text)
    if len(number.as_tuple().digits) > 28 or abs(number.as_tuple().exponent) > 28:
        raise ValueError("Decimal exceeds v1's 28-digit limit")
    return number


def nonempty(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


@dataclass(frozen=True)
class Fact:
    """A quantity attributed to an explicit group, metric and unit."""

    group: str
    metric: str
    value: Decimal
    unit: str
    decimal_places: int = 1

    def __post_init__(self):
        nonempty(self.group, "group")
        nonempty(self.metric, "metric")
        if not isinstance(self.unit, str) or self.unit not in UNITS:
            raise ValueError("Unsupported unit")
        value = decimal_value(self.value)
        object.__setattr__(self, "value", value)
        if type(self.decimal_places) is not int or not 0 <= self.decimal_places <= 12:
            raise ValueError("decimal_places must be an integer from 0 to 12")
        if self.unit == "count" and (value < 0 or value != value.to_integral_value()):
            raise ValueError("Counts must be nonnegative integers")
        if self.unit == "fraction" and not 0 <= value <= 1:
            raise ValueError("Fractions must be between 0 and 1")
        if self.unit == "percent" and not 0 <= value <= 100:
            raise ValueError("Percentages must be between 0 and 100")
        if self.unit == "percentage_point" and not -100 <= value <= 100:
            raise ValueError(
                "Percentage-point differences must be between -100 and 100"
            )

    @classmethod
    def from_mapping(cls, data):
        required = {"group", "metric", "value", "unit"}
        if not isinstance(data, Mapping) or not required <= data.keys():
            raise ValueError("Fact is missing required fields")
        if set(data) - required - {"decimal_places"}:
            raise ValueError("Unknown fact fields")
        return cls(**data)

    def to_dict(self):
        return {
            "group": self.group,
            "metric": self.metric,
            "value": format(self.value, "f"),
            "unit": self.unit,
            "decimal_places": self.decimal_places,
        }


@dataclass(frozen=True)
class EvaluationCase:
    """Minimal development-case schema; references never enter model inputs."""

    case_id: str
    instruction: str
    context: str
    expected_facts: tuple[Fact, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported schema version")
        for field in ("case_id", "instruction", "context"):
            nonempty(getattr(self, field), field)
        if not isinstance(self.expected_facts, tuple) or not self.expected_facts:
            raise ValueError("expected_facts must be a nonempty tuple")
        if any(not isinstance(fact, Fact) for fact in self.expected_facts):
            raise ValueError("expected_facts must contain Facts")
        keys = [(f.group, f.metric, f.unit) for f in self.expected_facts]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate expected attribution")

    @classmethod
    def from_mapping(cls, data):
        fields = {
            "schema_version",
            "case_id",
            "instruction",
            "context",
            "expected_facts",
        }
        if not isinstance(data, Mapping) or set(data) != fields:
            raise ValueError("Case must contain exactly the v1 fields")
        if not isinstance(data["expected_facts"], list):
            raise TypeError("expected_facts must be a JSON list")
        return cls(
            **{
                **data,
                "expected_facts": tuple(
                    Fact.from_mapping(fact) for fact in data["expected_facts"]
                ),
            }
        )

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "instruction": self.instruction,
            "context": self.context,
            "expected_facts": [fact.to_dict() for fact in self.expected_facts],
        }

    def format_input(self):
        return self.instruction + "\n\nContext:\n" + self.context


def validate_cases(records):
    """Validate JSON-shaped records and reject duplicate case IDs."""
    cases = [EvaluationCase.from_mapping(record) for record in records]
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and have unique IDs")
    return cases
