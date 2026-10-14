"""Authorized held-out comparison; importing this module never runs evaluation.

The saved fine-tuned artifact has a known embedding weight-tying inconsistency.
Its config declares tie_word_embeddings=True; loading may reconcile embedding
aliases differently from the saved tensors. This script does not repair weights,
call tie_weights(), or change model configuration. Interpret results with this
limitation, and retain any loader warnings during the authorized run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import re
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "google/flan-t5-small"
FINE_TUNED = ROOT / "artifacts/models/flan-t5-sba-928/final"
DATA_DIR = ROOT / "data/processed"
EVALUATION_DIR = ROOT / "artifacts/evaluation"
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 96
SEED = 42
GENERATION_SETTINGS = {"do_sample": False, "num_beams": 2, "max_new_tokens": 180}
TYING_LIMITATION = (
    "Known embedding weight-tying inconsistency: the saved fine-tuned configuration "
    "declares tie_word_embeddings=True, while saved embedding tying is inconsistent. "
    "Standard model loading may reconcile aliases; loader warnings must be retained. "
    "No manual weight repair, tie_weights call, or model configuration change is made."
)
METRIC_LIMITATION = (
    "Token-F1 measures lexical overlap only and is supplemental, not a factual-grounding "
    "score. Human review is required; no improvement claim follows from token-F1 alone."
)
REVIEW_FIELDS = (
    "relevance", "factual_accuracy", "factual_grounding", "market_research_terminology",
    "specificity", "actionability", "unsupported_claims",
)


def load_test_data(*, authorized: bool = False) -> tuple[list[dict], dict]:
    """Read test bytes only after explicit final-comparison authorization."""
    if authorized is not True:
        raise PermissionError("Final comparison must be explicitly authorized before test access")
    manifest_bytes = (DATA_DIR / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    path = DATA_DIR / "test.jsonl"
    relative = path.relative_to(ROOT).as_posix()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest["output_file_sha256"][relative]:
        raise ValueError("Test file hash does not match manifest")
    records = []
    required = {"record_id", "category", "instruction", "context", "target",
                "source_split", "evidence_group", "split"}
    for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank test JSONL line: {number}")
        record = json.loads(line)
        if not isinstance(record, dict) or not required <= record.keys():
            raise ValueError(f"Missing test record fields: line {number}")
        if any(not isinstance(record[key], str) or not record[key].strip() for key in required):
            raise ValueError(f"Invalid test record fields: line {number}")
        if record["split"] != "test" or record["source_split"] != "test":
            raise ValueError(f"Incorrect test split: {record['record_id']}")
        groups = record.get("source_group_ids")
        if (not isinstance(groups, list) or not groups
                or any(not isinstance(group, str) for group in groups)
                or not set(groups) <= set(manifest["source_group_ids"]["test"])):
            raise ValueError(f"Invalid test provenance: {record['record_id']}")
        if record.get("source_sha256") != manifest["source_sha256"]:
            raise ValueError("Test source hash metadata mismatch")
        records.append(record)
    if not records or len(records) != manifest["generated_record_counts"]["test"]:
        raise ValueError("Test record count does not match manifest")
    ids = [record["record_id"] for record in records]
    if len(set(ids)) != len(ids) or sorted(ids) != sorted(manifest["record_ids"]["test"]):
        raise ValueError("Test IDs are duplicated or do not match manifest")
    for split in ("train", "validation"):
        if set(ids) & set(manifest["record_ids"][split]):
            raise ValueError(f"Test IDs overlap {split}")
        if set(manifest["source_group_ids"]["test"]) & set(manifest["source_group_ids"][split]):
            raise ValueError(f"Test source groups overlap {split}")
    return records, {"test_sha256": digest, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                     "record_count": len(records), "record_ids": ids, "split": "test",
                     "source_sha256": manifest["source_sha256"], "validation_passed": True}


def format_input(record: dict) -> str:
    return record["instruction"] + "\n\nContext:\n" + record["context"]


def preflight_records(records, base_tokenizer, tuned_tokenizer):
    """Check both tokenizers on every source and target before loading weights."""
    features, lengths = [], []
    options = {"truncation": False, "padding": False, "add_special_tokens": True,
               "return_attention_mask": True}
    for record in records:
        source = format_input(record)
        base = base_tokenizer(source, **options)
        tuned = tuned_tokenizer(source, **options)
        base_target = base_tokenizer(text_target=record["target"], **options)
        tuned_target = tuned_tokenizer(text_target=record["target"], **options)
        for label, left, right in (("source", base, tuned), ("target", base_target, tuned_target)):
            for key in ("input_ids", "attention_mask"):
                if left[key] != right[key]:
                    raise ValueError(f"Tokenizer {label} {key} mismatch: {record['record_id']}")
        source_count, target_count = len(base["input_ids"]), len(base_target["input_ids"])
        if source_count > MAX_SOURCE_LENGTH or target_count > MAX_TARGET_LENGTH:
            raise ValueError(f"Token limit exceeded before model loading: {record['record_id']} "
                             f"source={source_count}, target={target_count}")
        features.append({key: base[key] for key in ("input_ids", "attention_mask")})
        lengths.append({"record_id": record["record_id"], "source_tokens": source_count,
                        "target_tokens": target_count, "token_ids_match": True,
                        "attention_masks_match": True})
    return features, lengths


def token_f1(reference, prediction):
    """Supplemental lexical overlap; never a factual-grounding score."""
    ref = Counter(re.findall(r"\b\w+\b", reference.lower()))
    pred = Counter(re.findall(r"\b\w+\b", prediction.lower()))
    overlap = sum((ref & pred).values())
    if not ref or not pred or not overlap:
        return 0.0
    precision, recall = overlap / sum(pred.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def shared_generation_config(base_model, tuned_model, tokenizers):
    """Use fresh shared defaults rather than either model's generation defaults."""
    special = {}
    for key in ("pad_token_id", "eos_token_id", "decoder_start_token_id", "bos_token_id"):
        left, right = getattr(base_model.config, key, None), getattr(tuned_model.config, key, None)
        if left != right:
            raise ValueError(f"Model generation token mismatch: {key}")
        special[key] = left
    for tokenizer in tokenizers:
        for key in ("pad_token_id", "eos_token_id", "bos_token_id"):
            if getattr(tokenizer, key, None) != special[key]:
                raise ValueError(f"Tokenizer/model special token mismatch: {key}")
    # T5 begins with one decoder-start token; the effective total cap is 181.
    config = GenerationConfig(**GENERATION_SETTINGS, **special, max_length=181, use_cache=False)
    config.validate()
    return config


def generate(model, tokenizer, feature, device, generation_config):
    encoded = {key: torch.tensor([value], dtype=torch.long, device=device)
               for key, value in feature.items()}
    with torch.inference_mode():
        output = model.generate(**encoded, generation_config=generation_config)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-final-comparison", action="store_true",
                        help="Explicitly authorize opening held-out test data and generating predictions")
    args = parser.parse_args()
    if not args.authorize_final_comparison:
        parser.error("Final comparison is gated; explicit --authorize-final-comparison is required")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not FINE_TUNED.is_dir():
        raise FileNotFoundError(f"Fine-tuned final model missing: {FINE_TUNED}")
    records, provenance = load_test_data(authorized=args.authorize_final_comparison)
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tuned_tokenizer = AutoTokenizer.from_pretrained(str(FINE_TUNED), local_files_only=True)
    features, lengths = preflight_records(records, base_tokenizer, tuned_tokenizer)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.warning(TYING_LIMITATION)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device).eval()
    tuned_model = AutoModelForSeq2SeqLM.from_pretrained(str(FINE_TUNED), local_files_only=True).to(device).eval()
    config = shared_generation_config(base_model, tuned_model, (base_tokenizer, tuned_tokenizer))
    effective_settings = config.to_dict()
    rows = []
    for record, feature, length in zip(records, features, lengths):
        base_output = generate(base_model, base_tokenizer, feature, device, config)
        tuned_output = generate(tuned_model, tuned_tokenizer, feature, device, config)
        row = {"record_id": record["record_id"], "split": "test", "category": record["category"],
               "instruction": record["instruction"], "context": record["context"],
               "input": format_input(record), "reference_response": record["target"],
               "original_model_output": base_output, "fine_tuned_model_output": tuned_output,
               "original_token_f1": token_f1(record["target"], base_output),
               "fine_tuned_token_f1": token_f1(record["target"], tuned_output),
               "source_tokens": length["source_tokens"], "target_tokens": length["target_tokens"],
               "effective_generation_settings": effective_settings}
        for model in ("original", "fine_tuned"):
            row.update({f"{model}_{field}_score": "" for field in REVIEW_FIELDS})
        row["human_review_notes"] = ""
        rows.append(row)
    summary = {
        "models": {"original": BASE_MODEL, "fine_tuned": str(FINE_TUNED)},
        "data_provenance": provenance, "final_comparison_authorized": True,
        "input_format": 'instruction + "\\n\\nContext:\\n" + context',
        "tokenization": {"truncation": False, "padding": False, "add_special_tokens": True,
                         "max_source_tokens": 256, "max_target_tokens": 96,
                         "matching_token_ids_and_attention_masks": True, "preflight": lengths},
        "effective_generation_settings": effective_settings,
        "generation_length_semantics": "180 new tokens plus one T5 decoder-start token; total cap 181",
        "supplemental_token_f1": {model: sum(row[f"{model}_token_f1"] for row in rows) / len(rows)
                                  for model in ("original", "fine_tuned")},
        "metric_limitation": METRIC_LIMITATION, "embedding_weight_tying_limitation": TYING_LIMITATION,
        "evaluation_limitation": "Only three held-out records in the current manifest; limited-sample evaluation.",
        "human_review_status": "pending; score fields intentionally blank",
        "seed": SEED, "device": str(device), "python_version": platform.python_version(),
        "library_versions": {name: version(name) for name in ("torch", "transformers")},
        "reproducibility_limitation": "Deterministic decoding and eval mode; bitwise identity across devices "
                                      "and library versions is not guaranteed.",
        "model_configurations": {"original": base_model.config.to_dict(),
                                 "fine_tuned": tuned_model.config.to_dict()},
    }
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    (EVALUATION_DIR / "model_comparison.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    (EVALUATION_DIR / "model_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (EVALUATION_DIR / "model_comparison_evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "effective_generation_settings": json.dumps(effective_settings, sort_keys=True)})
    LOGGER.info("Saved paired comparison and pending human review to %s", EVALUATION_DIR)


if __name__ == "__main__":
    main()
