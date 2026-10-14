"""Offline scoring of explicitly identified saved development responses."""

import json
from collections import Counter
from collections.abc import Mapping

from .dataset import CASE_VERSION, sha256
from .numerical import score_fact
from .repetition import repetition_report
from .schema import SCHEMA_VERSION, EvaluationCase, Fact, nonempty

RESPONSE_VERSION = "sba928_saved_response_v1"
REPORT_VERSION = "sba928_scoring_report_v1"
MANUAL_DIMENSIONS = (
    "instruction_compliance",
    "statistical_reasoning",
    "fairness",
    "unsupported_assertions",
)


def input_hash(case):
    return sha256(
        (case["instruction"] + "\n\nContext:\n" + case["context"]).encode("utf-8")
    )


def _integer(value, minimum, label):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def validate_prediction(prediction, case):
    fields = {"schema_version", "case_id", "model_id", "response", "generation"}
    if not isinstance(prediction, Mapping) or set(prediction) != fields:
        raise ValueError("Saved response must contain exactly the v1 response fields")
    if prediction["schema_version"] != RESPONSE_VERSION:
        raise ValueError("Unsupported saved-response version")
    nonempty(prediction["case_id"], "case_id")
    nonempty(prediction["model_id"], "model_id")
    if not isinstance(prediction["response"], str):
        raise TypeError(
            "response must be a string; an empty string is a scoreable failure"
        )
    generation = prediction["generation"]
    fields = {
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
    }
    if not isinstance(generation, Mapping) or set(generation) != fields:
        raise ValueError("Missing or unexpected generation metadata")
    if generation["origin"] not in ("saved_model", "synthetic_fixture"):
        raise ValueError("Unknown response origin")
    for field in ("model_revision", "tokenizer_revision", "device"):
        nonempty(generation[field], field)
    if generation["input_sha256"] != input_hash(case):
        raise ValueError("Response input hash does not match the development case")
    if generation["response_format"] not in ("fact_lines", "free_text"):
        raise ValueError("Unknown response format")
    _integer(generation["seed"], 0, "seed")
    _integer(generation["generated_tokens"], 0, "generated_tokens")
    settings = generation["settings"]
    if (
        not isinstance(settings, Mapping)
        or not {"do_sample", "num_beams", "max_new_tokens"} <= settings.keys()
    ):
        raise ValueError(
            "Generation settings need do_sample, num_beams and max_new_tokens"
        )
    if type(settings["do_sample"]) is not bool:
        raise ValueError("do_sample must be bool")
    _integer(settings["num_beams"], 1, "num_beams")
    _integer(settings["max_new_tokens"], 1, "max_new_tokens")
    if generation["generated_tokens"] > settings["max_new_tokens"]:
        raise ValueError("Generated tokens exceed the declared cap")
    if generation["termination"] not in ("eos", "length_cap", "unknown"):
        raise ValueError("Unknown termination reason")
    if (
        generation["termination"] == "length_cap"
        and generation["generated_tokens"] != settings["max_new_tokens"]
    ):
        raise ValueError("length_cap termination must reach the declared cap")
    # Reject NaN/infinity and other non-JSON metadata before any output is written.
    json.dumps(prediction, allow_nan=False)


def _fact_key(fact):
    return fact.group, fact.metric, fact.unit


def numerical_results(case, response, response_format):
    """Only declared `group | metric | value | unit` lines permit attribution.

    Prose is never mined for numbers. Unparsed lines make absent facts uncertain.
    Extra/invalid/duplicated claims remain visible rather than being discarded.
    """
    expected = [Fact.from_mapping(value) for value in case["expected_facts"]]
    parsed, unparsed = [], []
    if response_format == "fact_lines":
        for line_number, line in enumerate(response.splitlines(), 1):
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) != 4 or any(not part for part in parts):
                unparsed.append({"line": line_number, "text": line})
                continue
            group, metric, value, unit = parts
            raw = {"group": group, "metric": metric, "value": value, "unit": unit}
            try:
                fact = Fact.from_mapping(raw)
            except (ValueError, TypeError) as error:
                fact, error_text = None, str(error)
            else:
                error_text = None
            parsed.append(
                {
                    "line": line_number,
                    "text": line,
                    "raw": raw,
                    "fact": fact,
                    "validation_error": error_text,
                }
            )
    complete = not response.strip() or (
        response_format == "fact_lines" and not unparsed
    )
    results, used = [], set()
    for index, fact in enumerate(expected):
        exact = [
            p
            for p in parsed
            if (p["raw"]["group"], p["raw"]["metric"], p["raw"]["unit"])
            == _fact_key(fact)
        ]
        candidates = exact
        if not exact:
            same_pair = [
                f for f in expected if (f.group, f.metric) == (fact.group, fact.metric)
            ]
            if len(same_pair) == 1:
                candidates = [
                    p
                    for p in parsed
                    if (p["raw"]["group"], p["raw"]["metric"])
                    == (fact.group, fact.metric)
                ]
        used.update(p["line"] for p in candidates)
        if len(candidates) > 1:
            result = {
                "status": "needs_review",
                "reason": "duplicate_or_conflicting_claims",
            }
        elif candidates:
            candidate = candidates[0]
            if candidate["fact"] is None:
                result = {"status": "incorrect", "reason": "invalid_fact_value_or_unit"}
            else:
                result = score_fact(fact, candidate["fact"])
        else:
            result = score_fact(fact, None, attribution_confirmed=complete)
        results.append(
            {
                "expected_fact_index": index,
                "expected": fact.to_dict(),
                **result,
                "evidence": [
                    {
                        "line": p["line"],
                        "text": p["text"],
                        "validation_error": p["validation_error"],
                    }
                    for p in candidates
                ],
            }
        )
    extras = [
        {key: p[key] for key in ("line", "text", "raw", "validation_error")}
        for p in parsed
        if p["line"] not in used
    ]
    return results, {
        "unparsed_lines": unparsed,
        "unmatched_claims": extras,
        "attribution_mode": response_format,
    }


def dimension_summary(results):
    counts = Counter(result["status"] for result in results)
    scored = sum(counts[status] for status in ("correct", "incorrect", "missing"))
    return {
        "applicable": len(results),
        "deterministically_scored": scored,
        **{
            status: counts[status]
            for status in ("correct", "incorrect", "missing", "needs_review")
        },
        "accuracy_on_scored": counts["correct"] / scored if scored else None,
        "coverage": scored / len(results) if results else None,
        "status": "not_applicable"
        if not results
        else "needs_review"
        if counts["needs_review"]
        else "scored",
    }


def score_response(case, prediction):
    validate_prediction(prediction, case)
    facts, extraction = numerical_results(
        case, prediction["response"], prediction["generation"]["response_format"]
    )
    calculations = []
    for calculation in case["calculations"]:
        expected = Fact.from_mapping(calculation["result"])
        matches = [
            result
            for result in facts
            if Fact.from_mapping(result["expected"]) == expected
        ]
        if len(matches) != 1:
            raise ValueError("Calculation result must identify one expected fact")
        result = matches[0]
        calculations.append(
            {
                "calculation": calculation,
                "status": result["status"],
                "reason": result["reason"],
                "evidence": result["evidence"],
                "scope": "stated_result_only; reasoning is not verified",
            }
        )
    manual = {
        dimension: {
            "status": "pending",
            "rating": None,
            "reviewer": None,
            "evidence": [],
            "rationale": None,
        }
        for dimension in MANUAL_DIMENSIONS
    }
    return {
        "case_id": case["case_id"],
        "scenario_id": case["scenario_id"],
        "family": case["family"],
        "model_id": prediction["model_id"],
        "response": prediction["response"],
        "generation": prediction["generation"],
        "numerical_extraction": facts,
        "arithmetic": calculations,
        "extraction_notes": extraction,
        "repetition": repetition_report(prediction["response"]),
        "manual_review": manual,
        "response_requirements": case["response_requirements"],
        "review_criteria": case["review_criteria"],
        "dimension_summaries": {
            "numerical_extraction": dimension_summary(facts),
            "arithmetic": dimension_summary(calculations),
        },
        "review_status": "pending",
    }


def score_predictions(cases, predictions, model_ids):
    """Score a complete model-by-case matrix from already validated dev cases."""
    if not isinstance(model_ids, list) or not model_ids:
        raise ValueError("Declare a nonempty unique model_ids list")
    for model_id in model_ids:
        nonempty(model_id, "model_id")
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Declare a nonempty unique model_ids list")
    if not isinstance(cases, list) or not isinstance(predictions, list):
        raise TypeError("Cases and saved predictions must be lists")
    lookup = {}
    for case in cases:
        if (
            case.get("schema_version") != CASE_VERSION
            or case.get("split") != "development_validation"
        ):
            raise ValueError("Only v2 development cases can be scored")
        EvaluationCase.from_mapping(
            {
                key: case[key]
                for key in ("case_id", "instruction", "context", "expected_facts")
            }
            | {"schema_version": SCHEMA_VERSION}
        )
        if case["case_id"] in lookup:
            raise ValueError("Duplicate development case IDs")
        lookup[case["case_id"]] = case
    if not lookup:
        raise ValueError("No development cases")
    matrix = {}
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            raise TypeError("Prediction must be an object")
        case_id, model_id = prediction.get("case_id"), prediction.get("model_id")
        nonempty(case_id, "case_id")
        nonempty(model_id, "model_id")
        if case_id not in lookup or model_id not in model_ids:
            raise ValueError("Unexpected case ID or model identifier")
        key = model_id, case_id
        if key in matrix:
            raise ValueError("Duplicate model/case prediction")
        validate_prediction(prediction, lookup[case_id])
        matrix[key] = prediction
    required = {(model_id, case_id) for model_id in model_ids for case_id in lookup}
    if set(matrix) != required:
        raise ValueError(
            f"Missing model/case predictions: {sorted(required - matrix.keys())}"
        )
    results = [
        score_response(lookup[case_id], matrix[model_id, case_id])
        for model_id, case_id in sorted(matrix)
    ]
    summaries = {}
    for model_id in sorted(model_ids):
        selected = [result for result in results if result["model_id"] == model_id]
        summaries[model_id] = {
            dimension: dimension_summary(
                [item for result in selected for item in result[dimension]]
            )
            for dimension in ("numerical_extraction", "arithmetic")
        }
        summaries[model_id]["repetition"] = {
            "applicable_responses": len(selected),
            "mean_repeated_ngram_fraction": sum(
                r["repetition"]["repeated_ngram_fraction"] for r in selected
            )
            / len(selected),
            "responses_with_duplicate_sentences": sum(
                r["repetition"]["duplicate_sentence_count"] > 0 for r in selected
            ),
            "length_cap_responses": sum(
                r["generation"]["termination"] == "length_cap" for r in selected
            ),
            "status": "signal_only",
        }
        summaries[model_id]["manual_review"] = {
            dimension: {
                "applicable_responses": len(selected),
                "reviewed": 0,
                "pending": len(selected),
            }
            for dimension in MANUAL_DIMENSIONS
        }
    settings = {
        json.dumps(p["generation"]["settings"], sort_keys=True, allow_nan=False)
        for p in predictions
    }
    return {
        "schema_version": REPORT_VERSION,
        "response_schema_version": RESPONSE_VERSION,
        "case_count": len(lookup),
        "scenario_count": len({c["scenario_id"] for c in cases}),
        "model_ids": sorted(model_ids),
        "response_count": len(results),
        "response_origins": dict(
            sorted(Counter(p["generation"]["origin"] for p in predictions).items())
        ),
        "results": results,
        "model_summaries": summaries,
        "generation_settings_match": len(settings) == 1,
        "review_status": "pending",
        "aggregation_unit": "case; paraphrases are correlated",
        "limitations": [
            "Fact-line attribution only; prose needs manual review.",
            "Arithmetic checks the stated result, not reasoning or formula use.",
            "Accuracy denominators exclude pending facts; coverage is always reported.",
            "Repetition signals and declared generation metadata do not establish quality or provenance.",
            "No overall quality score or independent-scenario performance claim is produced.",
        ],
    }
