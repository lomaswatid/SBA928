"""Exact-input arithmetic and explicit attribution scoring."""

from decimal import ROUND_HALF_UP, Context, Decimal, localcontext

from .schema import Fact, decimal_value


def rounded(value, decimal_places=1):
    if type(decimal_places) is not int or not 0 <= decimal_places <= 12:
        raise ValueError("decimal_places must be an integer from 0 to 12")
    value = value if isinstance(value, Decimal) else decimal_value(value)
    if not value.is_finite():
        raise ValueError("Value must be finite")
    with localcontext(Context(prec=128)):
        return value.quantize(
            Decimal(1).scaleb(-decimal_places), rounding=ROUND_HALF_UP
        )


def conversion_rate(purchases, sessions, decimal_places=1):
    """Return percent, rounding once; zero-session rates are undefined."""
    purchases, sessions = decimal_value(purchases), decimal_value(sessions)
    if (
        sessions <= 0
        or purchases < 0
        or purchases > sessions
        or any(v != v.to_integral_value() for v in (purchases, sessions))
    ):
        raise ValueError(
            "Require integer counts with 0 <= purchases <= sessions and sessions > 0"
        )
    with localcontext(Context(prec=128)):
        value = purchases / sessions * 100
        return rounded(value, decimal_places)


def percentage_point_difference(left, right, decimal_places=1):
    """Return left minus right; both operands must be percent values."""
    left, right = decimal_value(left), decimal_value(right)
    if not 0 <= left <= 100 or not 0 <= right <= 100:
        raise ValueError("Operands must be percent values between 0 and 100")
    with localcontext(Context(prec=128)):
        return rounded(left - right, decimal_places)


def score_fact(expected, observed, *, attribution_confirmed=True):
    """Score an explicitly matched fact; do not extract or judge arbitrary prose.

    observed=None means confirmed absence. Uncertain parsing/attribution must
    instead set attribution_confirmed=False, yielding needs_review.
    """
    if not isinstance(expected, Fact) or (
        observed is not None and not isinstance(observed, Fact)
    ):
        raise ValueError("Expected and observed quantities must be Facts")
    if type(attribution_confirmed) is not bool:
        raise ValueError("attribution_confirmed must be bool")
    if not attribution_confirmed:
        return {"status": "needs_review", "reason": "ambiguous_attribution"}
    if observed is None:
        return {"status": "missing", "reason": "required_fact_absent"}
    for field in ("group", "metric", "unit"):
        if getattr(expected, field) != getattr(observed, field):
            return {"status": "incorrect", "reason": f"{field}_mismatch"}
    # Counts are exact; rates use the expectation's declared display precision.
    equal = (
        expected.value == observed.value
        if expected.unit == "count"
        else rounded(expected.value, expected.decimal_places)
        == rounded(observed.value, expected.decimal_places)
    )
    return {
        "status": "correct" if equal else "incorrect",
        "reason": "matched" if equal else "value_mismatch",
    }
