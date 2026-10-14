"""Independent arithmetic and attribution examples; no held-out fixtures."""

from decimal import Decimal, Inexact, localcontext

import pytest

from sba_928.evaluation.numerical import (
    conversion_rate,
    percentage_point_difference,
    score_fact,
)
from sba_928.evaluation.schema import Fact


@pytest.mark.parametrize(
    "purchases,sessions,expected",
    [(1, 4, "25.0"), (1, 3, "33.3"), (0, 7, "0.0"), (7, 7, "100.0"), (1, 16, "6.3")],
)
def test_conversion_known_results(purchases, sessions, expected):
    assert conversion_rate(purchases, sessions) == Decimal(expected)


@pytest.mark.parametrize(
    "purchases,sessions",
    [(1, 0), (-1, 5), (6, 5), (1, -5), ("1.5", 5), (1, "5.5"), (True, 5), (1.0, 5)],
)
def test_invalid_conversion_counts(purchases, sessions):
    with pytest.raises((ValueError, TypeError)):
        conversion_rate(purchases, sessions)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("22.5", "17.0", "5.5"),
        ("17.0", "22.5", "-5.5"),
        ("20", "20", "0.0"),
        ("100", "0", "100.0"),
        ("12.25", "10", "2.3"),
        ("10", "12.25", "-2.3"),
    ],
)
def test_signed_percentage_point_difference(left, right, expected):
    assert percentage_point_difference(left, right) == Decimal(expected)


@pytest.mark.parametrize("left,right", [("101", "10"), ("10", "-1"), (0.2, "10")])
def test_invalid_percentage_operands(left, right):
    with pytest.raises((ValueError, TypeError)):
        percentage_point_difference(left, right)


def test_reproducible_under_hostile_decimal_context():
    with localcontext() as context:
        context.prec = 2
        context.traps[Inexact] = True
        assert conversion_rate(1, 3) == Decimal("33.3")
        assert percentage_point_difference("22.55", "17.0") == Decimal("5.6")


@pytest.mark.parametrize(
    "observed,reason",
    [
        (Fact("B", "rate", "20", "percent"), "group_mismatch"),
        (Fact("A", "sessions", "20", "percent"), "metric_mismatch"),
        (Fact("A", "rate", "20", "count"), "unit_mismatch"),
        (Fact("A", "rate", "0.2", "fraction"), "unit_mismatch"),
        (Fact("A", "rate", "21", "percent"), "value_mismatch"),
    ],
)
def test_wrong_attribution_and_value(observed, reason):
    assert score_fact(Fact("A", "rate", "20", "percent"), observed) == {
        "status": "incorrect",
        "reason": reason,
    }


def test_rounding_uses_expected_precision_and_counts_are_exact():
    expected = Fact("A", "rate", "6.25", "percent", 1)
    assert (
        score_fact(expected, Fact("A", "rate", "6.3", "percent", 0))["status"]
        == "correct"
    )
    assert (
        score_fact(expected, Fact("A", "rate", "6.2", "percent"))["status"]
        == "incorrect"
    )
    assert (
        score_fact(
            Fact("A", "sessions", 100, "count", 0),
            Fact("A", "sessions", 101, "count", 0),
        )["status"]
        == "incorrect"
    )


def test_missing_and_ambiguous_are_distinct():
    expected = Fact("A", "rate", "20", "percent")
    assert score_fact(expected, None)["status"] == "missing"
    assert (
        score_fact(expected, None, attribution_confirmed=False)["status"]
        == "needs_review"
    )
    assert (
        score_fact(expected, expected, attribution_confirmed=False)["status"]
        == "needs_review"
    )


def test_percentage_points_are_not_percent():
    expected = Fact("A minus B", "difference", "5.5", "percentage_point")
    assert (
        score_fact(expected, Fact("A minus B", "difference", "5.5", "percent"))[
            "reason"
        ]
        == "unit_mismatch"
    )


def test_scoring_repeatable_and_does_not_mutate_facts():
    fact = Fact("A", "rate", "20", "percent")
    before = fact.to_dict()
    assert score_fact(fact, fact) == score_fact(fact, fact)
    assert fact.to_dict() == before
