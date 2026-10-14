"""Metadata checks and synthetic generation tests; never read held-out JSONL."""

import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

from sba_928 import step_01_generate_dataset as dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def manifest():
    return json.loads((ROOT / "data/processed/manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def synthetic_source():
    rows = [{"Revenue": bool(i % 2), "value": i} for i in range(100)]
    return pd.DataFrame(rows + [rows[0].copy(), rows[1].copy()])


def test_source_record_counts(manifest):
    assert manifest["source_row_count"] == 12_330
    assert manifest["source_row_counts"] == {
        "train": 9_864, "validation": 1_233, "test": 1_233
    }
    assert sum(manifest["source_row_counts"].values()) == 12_330


def test_supervised_record_counts(manifest):
    assert manifest["generated_record_counts"] == {
        "train": 18, "validation": 3, "test": 3
    }
    assert sum(manifest["generated_record_counts"].values()) == 24
    for split, count in manifest["generated_record_counts"].items():
        assert len(manifest["record_ids"][split]) == count
        assert len(dataset.TASK_ALLOCATION[split]) == count
    allocated = [task for tasks in dataset.TASK_ALLOCATION.values() for task in tasks]
    assert sorted(allocated) == list(range(1, 25))


@pytest.mark.parametrize("field", ["record_ids", "source_group_ids"])
def test_manifest_ids_are_unique_and_disjoint(manifest, field):
    identifiers = manifest[field]
    assert set(identifiers) == set(dataset.SPLITS)
    for values in identifiers.values():
        assert len(values) == len(set(values))
    for left, right in combinations(dataset.SPLITS, 2):
        assert set(identifiers[left]).isdisjoint(identifiers[right])


def test_synthetic_splits_are_exhaustive_and_reproducible(synthetic_source):
    groups = dataset.group_source_rows(synthetic_source)
    assert sorted(map(len, groups.values())).count(2) == 2
    first = dataset.split_source_groups(synthetic_source, groups)
    assert first == dataset.split_source_groups(synthetic_source, groups)
    shuffled = synthetic_source.sample(frac=1, random_state=7).reset_index(drop=True)
    shuffled_groups = dataset.group_source_rows(shuffled)
    assert first == dataset.split_source_groups(shuffled, shuffled_groups)
    assigned = [group for values in first.values() for group in values]
    assert len(assigned) == len(set(assigned)) == len(groups)
    assert set(assigned) == set(groups)
    for split, keys in first.items():
        positions = [position for key in keys for position in groups[key]]
        assert set(synthetic_source.iloc[positions]["Revenue"]) == {False, True}
        assert abs(len(positions) - len(synthetic_source) * dataset.RATIOS[split]) <= 2


def test_synthetic_record_generation_is_stable_and_excludes_provenance():
    task = {"instruction": "Summarize", "target": "Synthetic answer", "category": "test"}
    first = dataset.build_supervised_record(task, 1, "train", ["b", "a"], "hash", "Facts")
    second = dataset.build_supervised_record(task, 1, "train", ["a", "b"], "hash", "Facts")
    assert first == second
    assert dataset.format_training_input(first) == "Summarize\n\nContext:\nFacts"
    changed = dataset.build_supervised_record(task, 1, "train", ["a", "c"], "hash", "Facts")
    assert changed["record_id"] != first["record_id"]
