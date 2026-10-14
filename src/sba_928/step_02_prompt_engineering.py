"""Export prompt variations and compare their pretrained-model responses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
from importlib.metadata import version
from pathlib import Path

from .common import DATA_PATH, PROMPTS_DIR, ROOT, load_data

LOGGER = logging.getLogger(__name__)
CATALOG = PROMPTS_DIR / "prompt_catalog.json"
OUTPUT = PROMPTS_DIR / "prompt_variations.md"
ARTIFACT_DIR = ROOT / "artifacts" / "prompt_baseline"
MODEL_NAME = "google/flan-t5-small"
INPUT_TOKEN_LIMIT = 512
GENERATION_SETTINGS = {"max_new_tokens": 180, "do_sample": False, "num_beams": 2}
VARIANTS = {
    "basic": "base_prompt",
    "specific": "specific_prompt",
    "role + context": "role_prompt",
    "grounded + constrained": "grounded_prompt",
}
RUBRIC = {
    "relevance": "Answers the topic and requested task.",
    "completeness": "Covers the shared core expected facts for the case.",
    "grounding": "Claims match context; proposals are distinguished from findings.",
    "specificity": "Uses concrete groups, metrics, and relevant details.",
    "format_adherence": "Follows variant-specific output constraints; use N/A if absent.",
}


def export_prompt_variations(catalog: list[dict]) -> None:
    """Preserve the existing Markdown export of all catalog variants."""
    prompts = catalog
    lines = [
        "# Requirement 2 - Prompt Variations",
        "",
        ("Each research question is expressed at four levels. The progression adds "
         "specificity, analyst context, output constraints, and grounding rules."),
        "",
    ]

    for item in prompts:
        lines.extend([
            f"## {item['id']}. {item['category']}",
            "",
            f"**Basic:** {item['base_prompt']}",
            "",
            f"**Specific:** {item['specific_prompt']}",
            "",
            f"**Role + context:** {item['role_prompt']}",
            "",
            f"**Grounded + constrained:** {item['grounded_prompt']}",
            "",
        ])

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {OUTPUT}")


def summarize_groups(df, column: str) -> list[dict]:
    """Return deterministically ordered group counts and conversion rates."""
    table = df.groupby(column, dropna=False)["Revenue"].agg(
        sessions="count", conversion_rate="mean"
    ).reset_index()
    rows = []
    for _, row in table.iterrows():
        group = row[column]
        if hasattr(group, "item"):
            group = group.item()
        rows.append({"group": group, "sessions": int(row["sessions"]),
                     "conversion_rate": float(row["conversion_rate"])})
    return sorted(rows, key=lambda row: str(row["group"]))


def build_prompt_cases(df, catalog: list[dict], source_hash: str) -> list[dict]:
    """Build twelve dedicated cases; references remain separate from context."""
    required = {"Revenue", "VisitorType", "Month", "TrafficType", "Weekend",
                "Browser", "Region", "ProductRelated", "ProductRelated_Duration",
                "BounceRates", "ExitRates", "PageValues"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Source dataset is missing columns: {sorted(missing)}")
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError("Source dataset is empty or contains missing required values")
    if set(df["Revenue"].unique()) != {True, False}:
        raise ValueError("Both purchasing and non-purchasing sessions are required")
    if len(catalog) != 12 or {item["id"] for item in catalog} != set(range(1, 13)):
        raise ValueError("Expected twelve unique catalog IDs, numbered 1 through 12")
    metrics = ["ProductRelated", "ProductRelated_Duration", "BounceRates",
               "ExitRates", "PageValues"]
    engagement = []
    for outcome in [False, True]:
        subset = df[df["Revenue"] == outcome]
        engagement.append({"purchase": outcome, "sessions": len(subset),
                           **{metric: float(subset[metric].mean()) for metric in metrics}})
    groups = {column: summarize_groups(df, column) for column in
              ["VisitorType", "Month", "TrafficType", "Weekend", "Browser", "Region"]}
    visitor = []
    for row in groups["VisitorType"]:
        subset = df[df["VisitorType"] == row["group"]]
        visitor.append({**row, "mean_product_pages": float(subset["ProductRelated"].mean()),
                        "mean_product_duration": float(subset["ProductRelated_Duration"].mean())})

    def compact_engagement(selected):
        return [{"purchase": row["purchase"], "sessions": row["sessions"],
                 **{metric: row[metric] for metric in selected}} for row in engagement]

    strategy = {"engagement": compact_engagement(["BounceRates", "ExitRates", "ProductRelated"]),
                "visitor": groups["VisitorType"], "timing": groups["Weekend"],
                "traffic": groups["TrafficType"]}
    evidence = {
        1: {"engagement": engagement},
        2: {"visitor": visitor},
        3: {"monthly": groups["Month"]},
        4: {"traffic": groups["TrafficType"]},
        5: {"engagement": compact_engagement(["BounceRates"])},
        6: {"engagement": compact_engagement(["ExitRates"])},
        7: {"timing": groups["Weekend"]},
        8: {"browser": groups["Browser"]},
        9: {"region": groups["Region"]},
        10: {"engagement": compact_engagement(["PageValues"])},
        11: strategy,
        12: {"engagement": engagement, "visitor": groups["VisitorType"],
             "available_columns": sorted(df.columns.tolist()),
             "competitor_performance_fields": []},
    }
    cases = []
    for item in sorted(catalog, key=lambda item: item["id"]):
        facts = evidence[item["id"]]
        # Rounded context and references share the same displayed precision.
        displayed = json.loads(json.dumps(facts), parse_float=lambda value: round(float(value), 6))
        if item["id"] == 4:
            # Case 004: share column labels once without dropping any traffic facts.
            context = (
                "Conversion rates are purchase fractions, not percentages. "
                "These are descriptive statistics. "
                "TrafficType codes have no verified channel-name mapping.\n\n"
                "Traffic code | Session count | Conversion rate\n"
                + "\n".join(
                    f"{row['group']} | {row['sessions']:,} | {row['conversion_rate']}"
                    for row in displayed["traffic"]
                )
            )
        elif item["id"] == 11:
            # Case 011: preserve all four evidence sections with shared column labels.
            lines = [
                ("Purchasing=true; non-purchasing=false. Rates are fractions. "
                 "Conversion is the fraction of sessions purchasing. Descriptive statistics. "
                 "TrafficType codes have no verified channel-name mapping."),
                "",
                "Engagement: purchase | Sessions | Mean bounce rate | Mean exit rate | Mean product-related pages",
            ]
            lines.extend(
                f"{row['purchase']} | {row['sessions']:,} | {row['BounceRates']} | "
                f"{row['ExitRates']} | {row['ProductRelated']}"
                for row in displayed["engagement"]
            )
            headings = {"visitor": "Visitor type", "timing": "Weekend", "traffic": "Traffic code"}
            for section, heading in headings.items():
                lines.extend(["", heading + " | Sessions | Conversion rate"])
                lines.extend(
                    f"{row['group']} | {row['sessions']:,} | {row['conversion_rate']}"
                    for row in displayed[section]
                )
            context = "\n".join(lines)
        else:
            context = format_factual_context(displayed)
        cases.append({"example_id": f"PE-UCI468-{item['id']:03d}",
                      "catalog_id": item["id"], "category": item["category"],
                      "context": context, "expected_facts": displayed,
                      "source_file": DATA_PATH.relative_to(ROOT).as_posix(),
                      "source_sha256": source_hash})
    return cases


def format_factual_context(facts: dict) -> str:
    """Render source observations with readable labels and relevant qualifications."""
    labels = {
        "sessions": "Session count",
        "ProductRelated": "Mean product-related page count",
        "ProductRelated_Duration": "Mean product-related duration",
        "BounceRates": "Mean bounce rate",
        "ExitRates": "Mean exit rate",
        "PageValues": "Mean page value",
        "conversion_rate": "Conversion rate",
        "mean_product_pages": "Mean product-related page count",
        "mean_product_duration": "Mean product-related duration",
    }
    rows = [row for value in facts.values() if isinstance(value, list)
            for row in value if isinstance(row, dict)]
    keys = {key for row in rows for key in row}
    lines = []
    if "purchase" in keys:
        lines.extend([
            "Revenue/purchase true means a purchasing session.",
            "Revenue/purchase false means a non-purchasing session.",
        ])
    if keys & {"BounceRates", "ExitRates", "conversion_rate"}:
        lines.append("Rates are fractions, not percentages.")
    if "conversion_rate" in keys:
        lines.append("Conversion rate is the fraction of sessions ending in a purchase.")
    lines.append("The values below are descriptive statistics.")
    if keys & {"ProductRelated_Duration", "mean_product_duration"}:
        lines.append("Duration units are not specified in the available dataset information.")
    if "traffic" in facts:
        lines.append("Traffic groups use TrafficType codes; no channel-name mapping is supplied.")
    if "region" in facts:
        lines.append("Region groups use Region codes; no geographic-name mapping is supplied.")
    if "browser" in facts:
        lines.append("Browser groups use source Browser codes.")
    headings = {
        "visitor": "Visitor type", "monthly": "Month", "traffic": "Traffic code",
        "browser": "Browser code", "region": "Region code",
    }
    for section, values in facts.items():
        if section == "available_columns":
            lines.extend(["", "Available source columns:", ", ".join(values)])
            continue
        if section == "competitor_performance_fields":
            lines.extend(["", "Competitor performance fields: " +
                          (", ".join(values) if values else "none.")])
            continue
        for row in values:
            if section == "engagement":
                heading = "Purchasing sessions" if row["purchase"] else "Non-purchasing sessions"
            elif section == "timing":
                heading = "Weekend sessions" if row["group"] else "Non-weekend sessions"
            else:
                heading = f"{headings[section]} {row['group']}"
            lines.extend(["", heading + ":"])
            for key, value in row.items():
                if key in {"purchase", "group"}:
                    continue
                # Only counts gain thousands separators; numeric precision is unchanged.
                rendered = f"{value:,}" if key == "sessions" else str(value)
                lines.append(f"- {labels[key]}: {rendered}")
    return "\n".join(lines)


def build_prompt(variant_text: str, context: str) -> str:
    """Append factual context only, never the separate reference object."""
    return f"{variant_text}\n\nSource facts:\n{context}"


def resolve_device(device: str) -> str:
    """Resolve automatic device selection and validate explicit CUDA requests."""
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def load_base_model(device: str):
    """Load pretrained model weights after all input lengths pass preflight."""
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return model

def prepare_inputs(tokenizer, cases: list[dict], catalog: list[dict]) -> list[dict]:
    """Preflight every variant without truncation before any generation."""
    by_id = {item["id"]: item for item in catalog}
    prepared = []
    for case in cases:
        for name, field in VARIANTS.items():
            wording = by_id[case["catalog_id"]][field]
            prompt = build_prompt(wording, case["context"])
            encoded = tokenizer(prompt, return_tensors="pt", truncation=False,
                                add_special_tokens=True)
            count = int(encoded["input_ids"].shape[-1])
            if count > INPUT_TOKEN_LIMIT:
                raise ValueError(
                    f"Input too long: {case['example_id']} / {name}: "
                    f"{count} tokens exceeds {INPUT_TOKEN_LIMIT}; "
                    "shorten this case consistently across all variants before retrying."
                )
            prepared.append({"case": case, "variant_name": name, "variant_text": wording,
                             "exact_prompt": prompt, "encoded": encoded,
                             "input_token_count": count})
    return prepared


def generate_response(model, tokenizer, encoded_input, device: str) -> str:
    """Generate with the same deterministic settings for every input."""
    import torch

    with torch.inference_mode():
        output = model.generate(**encoded_input.to(device), **GENERATION_SETTINGS)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Save JSON records as UTF-8 JSONL."""
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def write_evaluation_sheet(path: Path, predictions: list[dict]) -> None:
    """Write responses and blank human-review scores and evidence notes."""
    fields = ["example_id", "catalog_id", "category", "variant_name", "exact_prompt",
              "context", "expected_facts", "model_response", *RUBRIC, "evaluation_notes"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for prediction in predictions:
            row = {field: prediction.get(field, "") for field in fields}
            row["expected_facts"] = json.dumps(prediction["expected_facts"], ensure_ascii=False)
            writer.writerow(row)


def run_experiment(catalog: list[dict], device: str = "auto", limit: int | None = None) -> None:
    """Run all four styles on each selected case and save separate evidence."""
    source_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    df = load_data()
    if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Source file changed while loading; retry with a stable snapshot")
    cases = build_prompt_cases(df, catalog, source_hash)
    if limit is not None:
        if not 1 <= limit <= len(cases):
            raise ValueError("--limit must be between 1 and 12")
        cases = cases[:limit]
    from transformers import AutoTokenizer

    device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prepared = prepare_inputs(tokenizer, cases, catalog)
    model = load_base_model(device)
    revision = getattr(model.config, "_commit_hash", None)
    effective_config = model.generation_config.to_dict()
    effective_config.update(GENERATION_SETTINGS)
    predictions = []
    for item in prepared:
        LOGGER.info("Generating %s / %s", item["case"]["example_id"], item["variant_name"])
        response = generate_response(model, tokenizer, item["encoded"], device)
        predictions.append({**item["case"],
                            **{key: item[key] for key in ["variant_name", "variant_text",
                                                        "exact_prompt", "input_token_count"]},
                            "model_response": response, "model_name": MODEL_NAME,
                            "model_revision": revision,
                            "generation_settings": dict(GENERATION_SETTINGS),
                            "input_token_limit": INPUT_TOKEN_LIMIT})
    summary = {
        "model_name": MODEL_NAME, "model_revision": revision,
        "generation_settings": GENERATION_SETTINGS,
        "effective_generation_configuration": effective_config,
        "device": device, "python_version": platform.python_version(),
        "library_versions": {name: version(name) for name in ["torch", "transformers", "pandas"]},
        "case_count": len(cases), "variant_count": len(VARIANTS),
        "response_count": len(predictions), "input_token_limit": INPUT_TOKEN_LIMIT,
        "input_token_counts": [{"example_id": row["example_id"],
                                "variant_name": row["variant_name"],
                                "count": row["input_token_count"]} for row in predictions],
        "source_file": DATA_PATH.relative_to(ROOT).as_posix(),
        "source_sha256": source_hash, "source_row_count": len(df),
        "catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        "evaluation_rubric": RUBRIC,
        "evaluation_scale": "Human scores: 0 (not met), 1 (partly met), 2 (met); format may be N/A.",
        "human_evaluation_status": "not evaluated", "evaluated_response_count": 0,
        "evaluation_summary": None,
        "limitations": ["Existing styles vary requested scope as well as wording.",
                        "Cases are prompt-engineering cases, not held-out fine-tuning test data.",
                        "Deterministic decoding does not ensure identical results across hardware/versions."],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(ARTIFACT_DIR / "prompt_examples.jsonl", cases)
    write_jsonl(ARTIFACT_DIR / "prompt_predictions.jsonl", predictions)
    write_evaluation_sheet(ARTIFACT_DIR / "prompt_evaluation.csv", predictions)
    (ARTIFACT_DIR / "prompt_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    LOGGER.info("Saved %s responses to %s", len(predictions), ARTIFACT_DIR)


def main() -> None:
    """Export prompt variations and run the controlled base-model experiment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Number of complete cases (1-12); always four styles")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    LOGGER.info("Starting controlled prompt-engineering experiment")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    export_prompt_variations(catalog)
    run_experiment(catalog, args.device, args.limit)


if __name__ == "__main__":
    main()
