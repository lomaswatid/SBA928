"""Mechanical word/sentence repetition signals, not semantic quality scores."""

import re


def repetition_report(text, n=3):
    """Casefold words; count duplicate n-gram occurrences beyond the first."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    words = re.findall(r"\b\w+\b", text.casefold())
    grams = [tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))]
    repeated = len(grams) - len(set(grams))
    # Only split after sentence punctuation followed by whitespace: decimal
    # points are retained. This is a conservative heuristic, not an NLP parser.
    sentences = [
        " ".join(s.strip().casefold().split())
        for s in re.split(r"(?<=[.!?])\s+", text.strip())
        if s.strip()
    ]
    return {
        "word_count": len(words),
        "ngram_size": n,
        "ngram_count": len(grams),
        "repeated_ngram_count": repeated,
        "repeated_ngram_fraction": repeated / len(grams) if grams else 0.0,
        "sentence_count": len(sentences),
        "duplicate_sentence_count": len(sentences) - len(set(sentences)),
    }
