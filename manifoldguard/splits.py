from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np


Array = np.ndarray

_FAMILY_PATTERNS = (
    ("llama", ("llama", "tulu", "hermes")),
    ("mistral", ("mistral", "mixtral")),
    ("qwen", ("qwen", "qwq")),
    ("gemma", ("gemma",)),
    ("phi", ("phi",)),
    ("yi", ("yi-",)),
    ("falcon", ("falcon",)),
    ("deepseek", ("deepseek",)),
    ("olmo", ("olmo",)),
    ("gpt-j", ("gpt-j",)),
    ("pythia", ("pythia",)),
    ("stablelm", ("stablelm",)),
    ("dbrx", ("dbrx",)),
    ("zephyr", ("zephyr",)),
)


def infer_model_family(model_name: str) -> str:
    """Infer a coarse release family from a model identifier."""
    lowered = model_name.strip().lower()
    if not lowered:
        raise ValueError("model_name must be non-empty.")

    stem = lowered.split("/", 1)[-1]
    for family, patterns in _FAMILY_PATTERNS:
        if any(pattern in stem for pattern in patterns):
            return family

    org = lowered.split("/", 1)[0]
    return org.replace("_", "-")


def group_indices_by_family(model_names: Sequence[str]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, model_name in enumerate(model_names):
        grouped[infer_model_family(model_name)].append(idx)
    return dict(sorted(grouped.items()))


def select_family_holdout_indices(
    model_names: Sequence[str],
    test_fraction: float = 0.25,
    min_families: int = 2,
    seed: int = 0,
) -> tuple[Array, Array, list[str]]:
    """Select a stronger OOD split by holding out entire model families."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1).")
    if min_families <= 0:
        raise ValueError("min_families must be positive.")

    family_to_indices = group_indices_by_family(model_names)
    family_names = list(family_to_indices)
    if len(family_names) < min_families:
        raise ValueError("Not enough distinct model families for the requested holdout.")

    rng = np.random.default_rng(seed)
    shuffled = [family_names[i] for i in rng.permutation(len(family_names))]
    family_order = {family: order for order, family in enumerate(shuffled)}

    target_test_models = max(1, int(round(test_fraction * len(model_names))))
    held_out_families: list[str] = []
    test_indices: list[int] = []

    remaining_families = list(shuffled)
    for family in list(remaining_families):
        projected_test_size = len(test_indices) + len(family_to_indices[family])
        if len(model_names) - projected_test_size < 2:
            continue
        held_out_families.append(family)
        test_indices.extend(family_to_indices[family])
        remaining_families.remove(family)
        break

    while remaining_families:
        need_more_models = len(test_indices) < target_test_models
        need_more_families = len(held_out_families) < min_families
        if not need_more_models and not need_more_families:
            break

        candidates: list[tuple[tuple[int, int, int], str]] = []
        for family in remaining_families:
            family_indices = family_to_indices[family]
            projected_test_size = len(test_indices) + len(family_indices)
            if len(model_names) - projected_test_size < 2:
                continue

            target_gap = abs(projected_test_size - target_test_models)
            overshoot = max(projected_test_size - target_test_models, 0)
            if need_more_families:
                priority = (0, target_gap, family_order[family])
            else:
                priority = (overshoot, target_gap, family_order[family])
            candidates.append((priority, family))

        if not candidates:
            break

        _, chosen_family = min(candidates, key=lambda item: item[0])
        held_out_families.append(chosen_family)
        test_indices.extend(family_to_indices[chosen_family])
        remaining_families.remove(chosen_family)

    if len(held_out_families) < min_families:
        raise ValueError("Could not construct a valid family holdout split.")

    test_idx = np.asarray(sorted(test_indices), dtype=int)
    train_idx = np.asarray([idx for idx in range(len(model_names)) if idx not in set(test_idx.tolist())], dtype=int)
    return train_idx, test_idx, held_out_families
