"""Run unreviewed custom FLAN-T5 inference without accessing project datasets.

Use --instruction and exactly one of --context / --context-file. Optional
--compare-original produces independent evidence, never Step 04 results.
Transformers 5.18.0 resolves generation defaults; checkpoint tying is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "artifacts/models/flan-t5-sba-928/final"
OUTPUT_ROOT = ROOT / "artifacts/custom_inference"
BASE_MODEL = "google/flan-t5-small"
MAX_SOURCE_TOKENS = 256
SEED = 42
TRANSFORMERS_VERSION = "5.18.0"
GENERATION_SETTINGS = {"do_sample": False, "num_beams": 2, "max_new_tokens": 180}
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "model.safetensors",
)
TYING_LIMITATION = (
    "Known embedding weight-tying inconsistency: configuration may declare tying "
    "while shared.weight and lm_head.weight differ. Transformers 5.18.0 leaves "
    "differing checkpoint tensors untied and warns. No manual tying, configuration "
    "changes or checkpoint writes are performed."
)
LOGGER = logging.getLogger(__name__)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--context")
    group.add_argument("--context-file", type=Path)
    parser.add_argument("--compare-original", action="store_true")
    return parser, parser.parse_args(argv)


def read_inputs(args):
    """Preserve supplied whitespace, including UTF-8 context-file line endings."""
    source = {"kind": "inline"}
    if args.context_file is not None:
        try:
            path = args.context_file.resolve(strict=True)
            payload = path.read_bytes()
            context = payload.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(
                f"Cannot read UTF-8 context file {args.context_file}: {exc}"
            ) from exc
        source = {
            "kind": "utf8_file",
            "path": str(path),
            "file_sha256": digest(payload),
        }
    else:
        context = args.context
    for label, text in (("instruction", args.instruction), ("context", context)):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{label.capitalize()} must be nonempty")
    return args.instruction, context, source


def format_input(instruction: str, context: str) -> str:
    return instruction + "\n\nContext:\n" + context


def validate_model_files(model_dir=MODEL_DIR):
    """Check filenames only; no weights are read or loaded by this precheck."""
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Saved fine-tuned model files missing in {model_dir}: {', '.join(missing)}"
        )
    empty = [name for name in MODEL_FILES if (model_dir / name).stat().st_size == 0]
    if empty:
        raise ValueError(f"Saved fine-tuned model files are empty: {', '.join(empty)}")


def preflight_input(instruction, context, tokenizers):
    """Check the full formatted source, with special tokens, before weight loading."""
    text = format_input(instruction, context)
    features, token_counts = {}, {}
    for name, tokenizer in tokenizers.items():
        encoded = tokenizer(
            text,
            truncation=False,
            padding=False,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        if not ids or len(ids) != len(mask):
            raise ValueError(f"Invalid token IDs/attention mask: {name}")
        count = len(ids)
        if count > MAX_SOURCE_TOKENS:
            raise ValueError(
                f"Input has {count} tokens for {name}; limit is {MAX_SOURCE_TOKENS} "
                "including special tokens. No truncation or model loading performed."
            )
        features[name] = {"input_ids": ids, "attention_mask": mask}
        token_counts[name] = count
    if not features:
        raise ValueError("At least one tokenizer is required")
    first = next(iter(features.values()))
    if any(feature != first for feature in features.values()):
        raise ValueError(
            "Original/fine-tuned token IDs or attention masks differ; no model loaded"
        )
    evidence = {
        "source_token_counts": token_counts,
        "max_source_tokens": MAX_SOURCE_TOKENS,
        "add_special_tokens": True,
        "truncation": False,
        "padding": False,
        "matching_token_ids_and_attention_masks": True if len(features) > 1 else None,
        "encoded_input_sha256": digest(
            json.dumps(first, sort_keys=True).encode("utf-8")
        ),
    }
    return text, features, evidence


def resolve_settings(models, tokenizers):
    """Resolve actual 5.18.0 defaults and compare before any prediction."""
    first_model = next(iter(models.values()))
    special = {
        key: getattr(first_model.config, key, None)
        for key in (
            "pad_token_id",
            "eos_token_id",
            "bos_token_id",
            "decoder_start_token_id",
        )
    }
    if special["decoder_start_token_id"] is None:
        raise ValueError("Saved T5 model lacks a decoder-start token")
    for name, model in models.items():
        for key, value in special.items():
            if getattr(model.config, key, None) != value:
                raise ValueError(f"Special generation token mismatch: {name}/{key}")
        for key in ("pad_token_id", "eos_token_id", "bos_token_id"):
            if getattr(tokenizers[name], key, None) != special[key]:
                raise ValueError(
                    f"Tokenizer/model special token mismatch: {name}/{key}"
                )
    requested = GenerationConfig(
        **{
            **GenerationConfig._get_default_generation_params(),
            **special,
            **GENERATION_SETTINGS,
            "max_length": 181,
            "use_cache": False,
        }
    )
    resolved = {}
    for name, model in models.items():
        config, forward_kwargs = model._prepare_generation_config(deepcopy(requested))
        if set(forward_kwargs) - {"output_attentions", "output_hidden_states"}:
            raise ValueError(f"Unexpected generation kwargs: {name}")
        config.validate()
        resolved[name] = config
    first = next(iter(resolved.values()))
    if any(config.to_dict() != first.to_dict() for config in resolved.values()):
        raise ValueError(
            "Resolved effective generation settings differ; no prediction generated"
        )
    if any(getattr(first, key) != value for key, value in GENERATION_SETTINGS.items()):
        raise ValueError(
            "Required deterministic generation settings did not resolve correctly"
        )
    return first, {
        "requested": requested.to_dict(),
        "effective_by_model": {
            name: config.to_dict() for name, config in resolved.items()
        },
        "comparison_verified": len(models) > 1,
        "length_semantics": "180 new tokens plus one decoder-start token; total cap 181.",
    }


def generate_response(model, tokenizer, feature, device, config):
    if model.training:
        raise ValueError("Inference requires evaluation mode")
    prepared, _ = model._prepare_generation_config(deepcopy(config))
    if prepared.to_dict() != config.to_dict():
        raise ValueError("Effective settings changed after preflight")
    inputs = {
        key: torch.tensor([value], dtype=torch.long, device=device)
        for key, value in feature.items()
    }
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=deepcopy(config))
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    if not text.strip():
        raise ValueError("Model produced an empty response")
    return text, len(output[0]) - 1


def model_file_hashes(model_dir=MODEL_DIR):
    """Stream checkpoint hashes only during an approved, preflighted inference run."""
    hashes = {}
    for name in MODEL_FILES:
        hasher = hashlib.sha256()
        with (model_dir / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        hashes[name] = hasher.hexdigest()
    return hashes


class WarningRecorder(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []
        self.phase = "initialization"

    def emit(self, record):
        self.records.append(
            {
                "phase": self.phase,
                "logger": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
            }
        )


def save_run(run_id, responses, provenance, warning_records, output_root=OUTPUT_ROOT):
    """Serialize before exclusive directory creation; never overwrite evidence."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid run ID")
    payloads = {
        "responses.jsonl": "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in responses
        ),
        "warnings.json": json.dumps(
            warning_records, indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n",
    }
    metadata = deepcopy(provenance)
    metadata["artifact_sha256"] = {
        name: digest(text.encode("utf-8")) for name, text in payloads.items()
    }
    payloads["provenance.json"] = (
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    run_dir = output_root / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing inference run: {run_dir}"
        ) from exc
    for name, text in payloads.items():
        with (run_dir / name).open("x", encoding="utf-8", newline="") as stream:
            stream.write(text)
    return run_dir


def run(args):
    instruction, context, context_source = read_inputs(args)
    validate_model_files()
    if version("transformers") != TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Effective generation resolution requires Transformers 5.18.0"
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex
    sources = {"fine_tuned": str(MODEL_DIR)}
    if args.compare_original:
        sources["original"] = BASE_MODEL
    recorder = WarningRecorder()
    loggers = [logging.getLogger(name) for name in ("transformers", "huggingface_hub")]
    for logger in loggers:
        logger.addHandler(recorder)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tokenizers = {}
            for name, source in sources.items():
                recorder.phase = f"tokenizer/{name}"
                tokenizers[name] = AutoTokenizer.from_pretrained(
                    source, local_files_only=name == "fine_tuned"
                )
            text, features, token_evidence = preflight_input(
                instruction, context, tokenizers
            )
            # No checkpoint bytes or model weights are read before input preflight.
            file_hashes = model_file_hashes()
            set_seed(SEED)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            models = {}
            LOGGER.warning(TYING_LIMITATION)
            for name, source in sources.items():
                recorder.phase = f"model/{name}"
                models[name] = (
                    AutoModelForSeq2SeqLM.from_pretrained(
                        source, local_files_only=name == "fine_tuned"
                    )
                    .to(device)
                    .eval()
                )
            recorder.phase = "resolve_generation_settings"
            config, settings = resolve_settings(models, tokenizers)
            responses = []
            for name, model in models.items():
                recorder.phase = f"generation/{name}"
                response, generated_tokens = generate_response(
                    model, tokenizers[name], features[name], device, config
                )
                responses.append(
                    {
                        "run_id": run_id,
                        "model": name,
                        "model_source": sources[name],
                        "instruction": instruction,
                        "context": context,
                        "input": text,
                        "input_sha256": digest(text.encode("utf-8")),
                        "source_tokens": token_evidence["source_token_counts"][name],
                        "response": response,
                        "response_sha256": digest(response.encode("utf-8")),
                        "generated_token_count": generated_tokens,
                        "effective_generation_settings": config.to_dict(),
                        "review_status": "unreviewed",
                    }
                )
            if model_file_hashes() != file_hashes:
                raise ValueError(
                    "Saved model files changed during inference; no results saved"
                )
        warning_records = recorder.records + [
            {
                "logger": "python.warnings",
                "level": "WARNING",
                "message": str(entry.message),
            }
            for entry in caught
        ]
    finally:
        for logger in loggers:
            logger.removeHandler(recorder)
    provenance = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "instruction": instruction,
        "context": context,
        "input": text,
        "instruction_sha256": digest(instruction.encode("utf-8")),
        "context_sha256": digest(context.encode("utf-8")),
        "input_sha256": digest(text.encode("utf-8")),
        "context_source": context_source,
        "tokenization": token_evidence,
        "models": {
            name: {
                "source": sources[name],
                "resolved_revision": getattr(model.config, "_commit_hash", None),
                "configuration": model.config.to_dict(),
                "configuration_sha256": digest(
                    json.dumps(model.config.to_dict(), sort_keys=True).encode("utf-8")
                ),
                "tokenizer_revision": tokenizers[name].init_kwargs.get("_commit_hash"),
            }
            for name, model in models.items()
        },
        "fine_tuned_file_sha256": file_hashes,
        "generation_settings": settings,
        "seed": SEED,
        "device": str(device),
        "python_version": platform.python_version(),
        "library_versions": {name: version(name) for name in ("torch", "transformers")},
        "evaluation_mode": {name: not model.training for name, model in models.items()},
        "inference_mode": True,
        "review_status": "unreviewed",
        "warnings": warning_records,
        "embedding_weight_tying_limitation": TYING_LIMITATION,
        "reproducibility_limitation": "Deterministic decoding is not a guarantee of bitwise identity across devices or versions.",
        "input_policy": "Only user instruction/context are included; no reference targets or project dataset records are loaded.",
    }
    return save_run(run_id, responses, provenance, warning_records), responses


def main(argv=None):
    parser, args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run_dir, responses = run(args)
    except (OSError, ValueError, RuntimeError, UnicodeError) as exc:
        parser.exit(2, f"Custom inference error: {exc}\n")
    for row in responses:
        print(f"{row['model']}: {row['response']}")
    print(f"Saved unreviewed custom inference: {run_dir}")


if __name__ == "__main__":
    main()
