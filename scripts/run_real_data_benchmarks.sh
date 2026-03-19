#!/usr/bin/env bash
# Run real-data benchmarks across multiple seeds.
# Results are saved to results/real_data_benchmark/
set -e
cd "$(dirname "$0")/.."
python3 scripts/run_real_data_benchmarks.py "$@"
