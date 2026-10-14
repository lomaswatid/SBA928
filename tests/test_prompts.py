"""Check prompt definitions without model loading or output generation."""

import json
from pathlib import Path

from sba_928.step_02_prompt_engineering import VARIANTS, build_prompt

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_has_four_distinct_variants_per_entry():
    catalog = json.loads((ROOT / "prompts/prompt_catalog.json").read_text(encoding="utf-8"))
    assert catalog
    assert len(VARIANTS) == 4
    assert len({item["id"] for item in catalog}) == len(catalog)
    for item in catalog:
        variants = [item[field] for field in VARIANTS.values()]
        assert all(isinstance(text, str) and text.strip() for text in variants)
        normalized = {" ".join(text.split()).casefold() for text in variants}
        assert len(normalized) == 4


def test_prompt_builder_preserves_shared_facts():
    context = "Synthetic sessions: 100; purchases: 25."
    variants = ["Summarize", "Compare", "Act as an analyst", "Use only supplied facts"]
    prompts = [build_prompt(variant, context) for variant in variants]
    assert len(set(prompts)) == 4
    for variant, prompt in zip(variants, prompts):
        assert prompt == f"{variant}\n\nSource facts:\n{context}"
