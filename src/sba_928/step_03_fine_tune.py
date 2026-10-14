"""Fine-tune FLAN-T5 on validated training data with independent validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from .common import ROOT

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "google/flan-t5-small"
OUTPUT_DIR = ROOT / "artifacts" / "models" / "flan-t5-sba-928"
DATA_DIR = ROOT / "data" / "processed"
SEED = 42
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 96


def load_jsonl(path: Path) -> list[dict]:
    """Read supervised records with line-numbered parse errors."""
    records = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line: {path.name}:{number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL: {path.name}:{number}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"Expected object: {path.name}:{number}")
            records.append(record)
    return records


def load_training_data() -> tuple[dict[str, list[dict]], dict]:
    """Verify only train/validation artifacts; never open held-out test data."""
    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "source_file": manifest["source_file"], "source_sha256": manifest["source_sha256"],
              "dataset_files": {}}
    required = {"record_id", "category", "instruction", "context", "target",
                "source_split", "evidence_group", "split"}
    datasets = {}
    seen_ids = set()
    provenance_sets = {}
    for split in ("train", "validation"):
        path = DATA_DIR / f"{split}.jsonl"
        relative = path.relative_to(ROOT).as_posix()
        before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if before_hash != manifest["output_file_sha256"][relative]:
            raise ValueError(f"Manifest hash mismatch: {relative}")
        records = load_jsonl(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != before_hash:
            raise ValueError(f"Dataset changed while reading: {relative}")
        if not records or len(records) != manifest["generated_record_counts"][split]:
            raise ValueError(f"Invalid record count: {split}")
        ids = []
        source_groups = set()
        allowed_groups = set(manifest["source_group_ids"][split])
        for record in records:
            if not required <= record.keys():
                raise ValueError(f"Missing required fields in {split}")
            if any(not isinstance(record[field], str) or not record[field].strip()
                   for field in required):
                raise ValueError(f"Empty or invalid required field: {record.get('record_id')}")
            if record["split"] != split or record["source_split"] != split:
                raise ValueError(f"Incorrect declared split: {record['record_id']}")
            identifier = record["record_id"]
            if identifier in seen_ids:
                raise ValueError(f"Duplicate record ID: {identifier}")
            seen_ids.add(identifier)
            ids.append(identifier)
            groups = record.get("source_group_ids")
            if not isinstance(groups, list) or not groups or any(not isinstance(g, str) for g in groups):
                raise ValueError(f"Invalid provenance groups: {identifier}")
            if not set(groups) <= allowed_groups:
                raise ValueError(f"Foreign source groups: {identifier}")
            if record.get("source_sha256") != manifest["source_sha256"]:
                raise ValueError(f"Source hash metadata mismatch: {identifier}")
            source_groups.update(groups)
        if sorted(ids) != sorted(manifest["record_ids"][split]):
            raise ValueError(f"Manifest ID mismatch: {split}")
        datasets[split] = records
        provenance_sets[split] = source_groups
        hashes["dataset_files"][relative] = before_hash
    if provenance_sets["train"] & provenance_sets["validation"]:
        raise ValueError("Train and validation source groups overlap")
    return datasets, hashes


def format_training_input(record: dict) -> str:
    """Exclude all provenance metadata from the model source text."""
    return record["instruction"] + "\n\nContext:\n" + record["context"]


def preflight_records(tokenizer, datasets: dict) -> tuple[dict, list[dict]]:
    """Tokenize once without truncation and reject all overlong examples."""
    encoded = {}
    lengths = []
    failures = []
    for split, records in datasets.items():
        features = []
        for record in records:
            inputs = tokenizer(format_training_input(record), truncation=False,
                               add_special_tokens=True, return_attention_mask=True)
            labels = tokenizer(text_target=record["target"], truncation=False,
                               add_special_tokens=True)["input_ids"]
            source_count = len(inputs["input_ids"])
            target_count = len(labels)
            lengths.append({"record_id": record["record_id"], "split": split,
                            "source_tokens": source_count, "target_tokens": target_count})
            if source_count > MAX_SOURCE_LENGTH or target_count > MAX_TARGET_LENGTH:
                failures.append(lengths[-1])
            features.append({"input_ids": inputs["input_ids"],
                             "attention_mask": inputs["attention_mask"], "labels": labels})
        encoded[split] = features
    if failures:
        raise ValueError(f"Token limits exceeded; no model loaded: {failures}")
    return encoded, lengths


def main() -> None:
    """Preflight training/validation, then train and save reproducible evidence."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.info("Starting FLAN-T5-small fine-tuning")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()
    if (not math.isfinite(args.epochs) or args.epochs <= 0 or args.batch_size <= 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("Epochs, batch size and learning rate must be positive and finite")
    set_seed(SEED)
    datasets, hashes = load_training_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    features, lengths = preflight_records(tokenizer, datasets)
    # Model weights are loaded only after every train/validation record passes.
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                    padding=True, label_pad_token_id=-100)
    use_cuda = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR), eval_strategy="epoch", save_strategy="epoch",
        learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, num_train_epochs=args.epochs,
        weight_decay=0.01, predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH, logging_steps=2,
        save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", fp16=use_cuda, seed=SEED, data_seed=SEED,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=Dataset.from_list(features["train"]),
        eval_dataset=Dataset.from_list(features["validation"]),
        data_collator=collator, processing_class=tokenizer,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configuration = {
        "model_name": MODEL_NAME, "model_revision": getattr(model.config, "_commit_hash", None),
        "training_arguments": training_args.to_dict(), "data_provenance": hashes,
        "input_format": 'instruction + "\\n\\nContext:\\n" + context',
        "max_source_length": MAX_SOURCE_LENGTH, "max_target_length": MAX_TARGET_LENGTH,
        "truncation": False, "label_pad_token_id": -100, "seed": SEED, "data_seed": SEED,
        "record_counts": {s: len(r) for s, r in datasets.items()}, "token_preflight": lengths,
        "python_version": platform.python_version(),
        "library_versions": {name: version(name) for name in ["torch", "transformers", "datasets"]},
        "test_policy": "Held-out test records are never opened or used by this script.",
        "nondeterminism_limitations": "Seeds initialize Python, NumPy and PyTorch via set_seed. "
        "Bitwise reproducibility is not guaranteed across devices, library versions, "
        "CUDA kernels or mixed precision; deterministic algorithms are not forced.",
    }
    (OUTPUT_DIR / "training_config_resolved.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    train_result = trainer.train()
    validation_metrics = trainer.evaluate(metric_key_prefix="validation")
    final_dir = OUTPUT_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    trainer.save_state()
    summary = {"training_metrics": train_result.metrics, "validation_metrics": validation_metrics,
               "best_checkpoint": trainer.state.best_model_checkpoint,
               "best_validation_loss": trainer.state.best_metric,
               "full_log_history": trainer.state.log_history, "data_provenance": hashes,
               "final_model_path": str(final_dir),
               "evaluation_limitation": "Only three validation examples; improvement is not guaranteed."}
    (OUTPUT_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    LOGGER.info("Saved final model and training evidence to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
