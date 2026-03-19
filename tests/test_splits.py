import numpy as np

from manifoldguard.splits import infer_model_family, select_family_holdout_indices


def test_infer_model_family_matches_common_release_families() -> None:
    assert infer_model_family("meta-llama/Llama-3.1-70B") == "llama"
    assert infer_model_family("mistralai/Mixtral-8x7B-v0.1") == "mistral"
    assert infer_model_family("Qwen/QwQ-32B") == "qwen"
    assert infer_model_family("databricks/dbrx-base") == "dbrx"


def test_select_family_holdout_indices_keeps_families_disjoint() -> None:
    model_names = [
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-70B",
        "mistralai/Mixtral-8x7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-72B",
        "google/gemma-2-9b",
        "google/gemma-2-27b",
    ]

    train_idx, test_idx, held_out_families = select_family_holdout_indices(
        model_names,
        test_fraction=0.25,
        min_families=2,
        seed=4,
    )

    train_families = {infer_model_family(model_names[idx]) for idx in train_idx.tolist()}
    test_families = {infer_model_family(model_names[idx]) for idx in test_idx.tolist()}

    assert held_out_families
    assert len(held_out_families) >= 2
    assert train_idx.size + test_idx.size == len(model_names)
    assert np.intersect1d(train_idx, test_idx).size == 0
    assert train_families.isdisjoint(test_families)
