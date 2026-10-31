"""Two pieces of CPU overhead of the `soup train` command itself, not of the training step.

Allowed by the organizers (Oct 31) under the conditions noted at each patch. Both run inside
the measured process, keep nothing between runs and behave the same under measurement and tests.

1. importlib.metadata.packages_distributions() walks every installed package (~2.3 s on the
   Kaggle image). transformers computes it at import (before apply() runs) and keeps it as
   transformers.utils.import_utils.PACKAGE_DISTRIBUTION_MAPPING; trl computes it again when it is
   imported. We give later callers that same mapping, computed in this process by the same
   function, as a fresh copy on every call, so no caller can change what another one sees.
   (Installing packages during a run is not allowed, so it cannot go stale.)

2. datasets.Dataset.map hashes the transform with dill only to name a cache file. For an
   in-memory dataset (no cache files, no cache_file_name, one process: Soup builds its datasets
   with Dataset.from_list) no cache file is ever written or read, and the transform closes over
   the tokenizer, which is slow to hash. There we pass a random new_fingerprint, as datasets
   itself does when hashing fails. It comes from datasets' own random.Random instance, not from
   the global random, NumPy or torch generators, so the training's random state is untouched.
   The mapped rows, their order and everything downstream are identical. Soup's rewind
   fingerprint hashes the rows themselves, not this one.
"""

from __future__ import annotations

import functools
import importlib.metadata

ENABLED = True


def _copy(mapping):
    return {name: list(dists) for name, dists in mapping.items()}


def cache_packages_distributions():
    md = importlib.metadata
    if getattr(md.packages_distributions, "_soup_cached", False):
        return
    original = md.packages_distributions
    state = {"mapping": None}
    try:  # what transformers already computed at import, in this process
        from transformers.utils import import_utils

        mapping = getattr(import_utils, "PACKAGE_DISTRIBUTION_MAPPING", None)
        if isinstance(mapping, dict):
            state["mapping"] = _copy(mapping)
    except Exception:  # noqa: BLE001 - then the first caller computes it, once
        pass

    @functools.wraps(original)
    def packages_distributions():
        if not ENABLED:
            return original()
        if state["mapping"] is None:
            state["mapping"] = _copy(original())
        return _copy(state["mapping"])  # a fresh copy for every caller

    packages_distributions._soup_cached = True
    md.packages_distributions = packages_distributions


def in_memory_map_shortcut(dataset, args, kwargs) -> bool:
    """True when Dataset.map's fingerprint would only name a cache file nobody writes."""
    return (
        ENABLED
        and not args
        and kwargs.get("new_fingerprint") is None
        and not dataset.cache_files
        and kwargs.get("cache_file_name") is None
        and (kwargs.get("num_proc") or 1) <= 1
    )


def skip_in_memory_map_hashing():
    from datasets import arrow_dataset, fingerprint

    cls = arrow_dataset.Dataset
    if getattr(cls.map, "_soup_fingerprint", False):
        return
    original = cls.map

    @functools.wraps(original)
    def map(self, function=None, *args, **kwargs):
        if in_memory_map_shortcut(self, args, kwargs):
            kwargs["new_fingerprint"] = fingerprint.generate_random_fingerprint()
        return original(self, function, *args, **kwargs)

    map._soup_fingerprint = True
    cls.map = map


def patch_startup():
    cache_packages_distributions()  # before anything imports trl
    try:
        skip_in_memory_map_hashing()
    except ImportError:  # no datasets: nothing to do
        pass
