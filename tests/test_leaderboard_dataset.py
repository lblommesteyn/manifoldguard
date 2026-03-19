from manifoldguard.leaderboard_dataset import (
    CORE_BENCHMARKS,
    discover_expanded_benchmarks,
    expanded_column_names,
    extract_benchmark_scores,
    select_metric_key,
)


def test_select_metric_key_prefers_known_metric_order() -> None:
    metrics = {
        "acc,none": 0.42,
        "exact_match,none": 0.11,
        "acc_norm,none": 0.55,
    }

    assert select_metric_key(metrics) == "acc_norm,none"


def test_discover_expanded_benchmarks_uses_union_across_payloads() -> None:
    payload_a = {
        "results": {
            "leaderboard_arc_challenge": {"acc_norm,none": 0.5},
            "leaderboard_task_alpha": {"acc,none": 0.1},
        }
    }
    payload_b = {
        "results": {
            "leaderboard_arc_challenge": {"acc_norm,none": 0.6},
            "leaderboard_task_beta": {"exact_match,none": 0.2},
        }
    }

    benchmarks = discover_expanded_benchmarks([payload_a, payload_b], core_benchmarks=CORE_BENCHMARKS, extra_limit=10)

    assert "leaderboard_arc_challenge" in benchmarks
    assert "leaderboard_task_alpha" in benchmarks
    assert "leaderboard_task_beta" in benchmarks
    assert benchmarks["leaderboard_task_alpha"] == "acc,none"
    assert benchmarks["leaderboard_task_beta"] == "exact_match,none"


def test_expanded_column_names_strip_prefix() -> None:
    assert expanded_column_names(["leaderboard_arc_challenge", "leaderboard_task_alpha"]) == [
        "arc_challenge",
        "task_alpha",
    ]


def test_extract_benchmark_scores_handles_missing_values() -> None:
    payload = {
        "results": {
            "leaderboard_arc_challenge": {"acc_norm,none": 0.51234},
            "leaderboard_task_alpha": {},
        }
    }
    benchmarks = {
        "leaderboard_arc_challenge": "acc_norm,none",
        "leaderboard_task_alpha": "acc,none",
    }

    scores = extract_benchmark_scores(payload, benchmarks)

    assert scores["leaderboard_arc_challenge"] == 0.51234
    assert scores["leaderboard_task_alpha"] is None
