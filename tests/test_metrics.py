"""Known metric results using synthetic inputs; no evaluation entry points."""

import pandas as pd
import pytest

from sba_928.common import pct
from sba_928.step_01_generate_dataset import row_stats
from sba_928.step_04_compare_models import token_f1


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [
        ("same words", "same words", 1.0),
        ("Hello, WORLD!", "hello world", 1.0),
        ("a b", "a c", 0.5),
        ("a a b", "a b b", 2 / 3),
        ("a b", "a", 2 / 3),
        ("a", "z", 0.0),
        ("", "a", 0.0),
        ("a", "", 0.0),
        ("", "", 0.0),
    ],
)
def test_token_f1_known_results(reference, prediction, expected):
    assert token_f1(reference, prediction) == pytest.approx(expected)
    assert token_f1(prediction, reference) == pytest.approx(expected)


def test_conversion_statistics_known_results():
    frame = pd.DataFrame({
        "group": ["A", "A", "A", "A", "B", "B"],
        "Revenue": [True, False, False, False, True, False],
    })
    statistics = row_stats(frame, "group").set_index("group")
    assert statistics.loc["A", "sessions"] == 4
    assert statistics.loc["A", "conversion"] == pytest.approx(0.25)
    assert statistics.loc["B", "sessions"] == 2
    assert statistics.loc["B", "conversion"] == pytest.approx(0.5)


@pytest.mark.parametrize("value, expected", [(0, "0.0%"), (0.25, "25.0%"), (1, "100.0%")])
def test_percentage_format(value, expected):
    assert pct(value) == expected
