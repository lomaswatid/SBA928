"""Mechanical repetition signals on synthetic text."""

import pytest

from sba_928.evaluation.repetition import repetition_report


@pytest.mark.parametrize("text", ["", "one", "one two"])
def test_short_text_has_no_trigrams(text):
    report = repetition_report(text)
    assert report["ngram_count"] == report["repeated_ngram_count"] == 0
    assert report["repeated_ngram_fraction"] == 0.0


def test_known_loop_fraction():
    report = repetition_report("a b c a b c")
    assert report["ngram_count"] == 4
    assert report["repeated_ngram_count"] == 1
    assert report["repeated_ngram_fraction"] == 0.25


def test_duplicate_sentences_ignore_case_and_whitespace():
    report = repetition_report("Rate is 12.5%. RATE  is 12.5%.")
    assert report["sentence_count"] == 2
    assert report["duplicate_sentence_count"] == 1


def test_table_repetition_is_only_a_signal():
    report = repetition_report("A has 100 sessions. B has 100 sessions.")
    assert report["repeated_ngram_count"] > 0
    assert report["duplicate_sentence_count"] == 0
    assert "status" not in report


@pytest.mark.parametrize(
    "text,n", [(None, 3), ("text", 0), ("text", True), ("text", 1.5)]
)
def test_invalid_inputs(text, n):
    with pytest.raises((ValueError, TypeError)):
        repetition_report(text, n)


def test_repeatable_report():
    text = "Repeat this. Repeat this."
    assert repetition_report(text) == repetition_report(text)
