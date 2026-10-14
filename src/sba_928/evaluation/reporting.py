"""Offline CLI: validated development suite + saved predictions -> new run."""

import argparse
import json
import platform
from pathlib import Path
from uuid import uuid4

from .dataset import load_development_suite, sha256
from .scoring import score_predictions


def render_summary(report):
    lines = [
        "# Saved-response scoring",
        "",
        "Offline development analysis; no models executed.",
        "",
        f"Cases: {report['case_count']}; scenarios: {report['scenario_count']}; responses: {report['response_count']}.",
        f"Manual review: {report['review_status']}. Matching generation settings: {report['generation_settings_match']}.",
        f"Response origins: {report['response_origins']}.",
        "Synthetic fixtures, when present, verify the workflow and are not model-performance results.",
        "Accuracy uses scored facts only; inspect applicable and pending counts alongside it.",
        "",
        "| Model | Dimension | Correct | Scored | Applicable | Needs review | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    repetition_lines = []
    for model_id, dimensions in report["model_summaries"].items():
        label = model_id.replace("|", "\\|").replace("\n", " ")
        for dimension in ("numerical_extraction", "arithmetic"):
            summary = dimensions[dimension]
            lines.append(
                f"| {label} | {dimension} | {summary['correct']} | {summary['deterministically_scored']} | {summary['applicable']} | {summary['needs_review']} | {summary['status']} |"
            )
        repetition = dimensions["repetition"]
        repetition_lines.extend(
            [
                "",
                (
                    f"{label}: mean repeated word-trigram fraction {repetition['mean_repeated_ngram_fraction']:.3f}; "
                    f"length-cap responses {repetition['length_cap_responses']}. Repetition is a signal, not a pass/fail score."
                ),
            ]
        )
    lines.extend(repetition_lines)
    lines.extend(
        [
            "",
            "## Individual cases",
            "",
            "| Model | Case | Numerical correct/scored | Pending numerical | Arithmetic correct/scored | Review |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for result in report["results"]:
        numerical = result["dimension_summaries"]["numerical_extraction"]
        arithmetic = result["dimension_summaries"]["arithmetic"]
        model = result["model_id"].replace("|", "\\|").replace("\n", " ")
        case_id = result["case_id"].replace("|", "\\|").replace("\n", " ")
        arithmetic_display = (
            f"{arithmetic['correct']}/{arithmetic['deterministically_scored']}"
            if arithmetic["applicable"]
            else "N/A"
        )
        lines.append(
            f"| {model} | {case_id} | {numerical['correct']}/{numerical['deterministically_scored']} | "
            f"{numerical['needs_review']} | {arithmetic_display} | pending |"
        )
    lines.extend(
        ["", "## Limitations", "", *[f"- {item}" for item in report["limitations"]], ""]
    )
    return "\n".join(lines)


def write_report(report, output_root):
    """Create a fresh UUID directory; never reuse a run or replace files."""
    json_payload = (
        json.dumps(
            report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    )
    summary = render_summary(report)
    run = Path(output_root) / ("run-" + uuid4().hex)
    run.mkdir(parents=True, exist_ok=False)
    with (run / "report.json").open("x", encoding="utf-8") as stream:
        stream.write(json_payload)
    with (run / "summary.md").open("x", encoding="utf-8") as stream:
        stream.write(summary)
    return run


def score_saved_file(
    predictions_path,
    cases_path,
    suite_manifest_path,
    source_manifest_path,
    model_ids,
    output_root,
):
    """Read explicit saved responses only after validating the development suite."""
    # Deliberate filename contract prevents accidentally opening historical
    # comparison predictions or original held-out records through this CLI.
    predictions_path = Path(predictions_path)
    if predictions_path.name != "saved_responses.jsonl":
        raise ValueError("Expected explicitly prepared saved_responses.jsonl")
    cases, validation = load_development_suite(
        cases_path, suite_manifest_path, source_manifest_path
    )
    if validation["status"] != "passed":
        raise ValueError(
            "Development duplicate candidates require review before scoring"
        )
    payload = predictions_path.read_bytes()
    predictions = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    report = score_predictions(cases, predictions, model_ids)
    report["provenance"] = {
        "development_validation": validation,
        "predictions_sha256": sha256(payload),
        "suite_manifest_sha256": sha256(Path(suite_manifest_path).read_bytes()),
        "implementation_sha256": {
            name: sha256(Path(__file__).with_name(name).read_bytes())
            for name in (
                "schema.py",
                "numerical.py",
                "repetition.py",
                "dataset.py",
                "scoring.py",
                "reporting.py",
            )
        },
        "python_version": platform.python_version(),
        "models_executed": False,
    }
    return write_report(report, output_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/evaluation_v3")
    )
    args = parser.parse_args()
    run = score_saved_file(
        args.predictions,
        args.suite_dir / "cases.jsonl",
        args.suite_dir / "manifest.json",
        args.source_manifest,
        args.model,
        args.output_root,
    )
    print(run)


if __name__ == "__main__":
    main()
