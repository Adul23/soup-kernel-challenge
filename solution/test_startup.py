"""The two startup patches return the same results and leave every random state alone."""

import importlib.metadata
import random

import numpy as np
import torch

from solution import startup


def test_packages_distributions_same_mapping_and_copies():
    startup.cache_packages_distributions()
    first = importlib.metadata.packages_distributions()
    startup.ENABLED = False
    try:
        fresh = importlib.metadata.packages_distributions()  # the original walk
    finally:
        startup.ENABLED = True
    assert first == fresh

    first.setdefault("torch", []).append("not-a-distribution")
    first["__mutated__"] = ["x"]
    again = importlib.metadata.packages_distributions()
    assert again == fresh, "a caller's change reached another caller"


def test_in_memory_map_identical_and_random_state_untouched(monkeypatch):
    from datasets import Dataset, fingerprint

    startup.skip_in_memory_map_hashing()
    rows = [{"text": f"row {i}", "n": i} for i in range(50)]

    def fn(batch):
        return {"length": [len(t) for t in batch["text"]], "n2": [n * 2 for n in batch["n"]]}

    calls = []
    original_update = fingerprint.update_fingerprint
    monkeypatch.setattr("datasets.arrow_dataset.update_fingerprint",
                        lambda *a, **k: calls.append(1) or original_update(*a, **k))

    def run(enabled):
        startup.ENABLED = enabled
        try:
            random.seed(3)
            np.random.seed(3)
            torch.manual_seed(3)
            ds = Dataset.from_list(rows)
            if enabled:
                assert startup.in_memory_map_shortcut(ds, (), {})
            out = ds.map(fn, batched=True, batch_size=7)
            return out, (random.getstate(), np.random.get_state()[1].copy(),
                         torch.get_rng_state().clone())
        finally:
            startup.ENABLED = True

    ref, ref_states = run(False)
    calls.clear()
    new, new_states = run(True)
    assert calls == [], "the transform was still hashed"
    # every random state ends exactly where the original map leaves it
    assert new_states[0] == ref_states[0]
    assert np.array_equal(new_states[1], ref_states[1])
    assert torch.equal(new_states[2], ref_states[2])
    assert new.to_list() == ref.to_list()
    assert new.column_names == ref.column_names


def test_shortcut_not_taken_when_a_cache_file_could_matter():
    from datasets import Dataset

    ds = Dataset.from_list([{"a": 1}])
    assert not startup.in_memory_map_shortcut(ds, (), {"cache_file_name": "x.arrow"})
    assert not startup.in_memory_map_shortcut(ds, (), {"num_proc": 2})
    assert not startup.in_memory_map_shortcut(ds, (), {"new_fingerprint": "abc"})
    assert not startup.in_memory_map_shortcut(ds, (True,), {})
