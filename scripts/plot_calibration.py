"""Utility for plotting monotonic failure risk calibration curves."""

from __future__ import annotations

import numpy as np


def generate_calibration_plot(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 5,
    filename: str = "calibration_plot.png",
) -> None:
    """Bin predicted probabilities and evaluate against actual failure rates.
    
    Args:
        probs: Array of predicted failure probabilities out-of-fold.
        labels: Array of true failure labels (0 or 1).
        num_bins: Target number of equal-frequency buckets.
        filename: Destination path to save the matplotlib figure. 
    """
    if len(probs) == 0:
        print("No probabilities provided for calibration.")
        return

    # Sort probabilities to create equal-frequency buckets
    sort_idx = np.argsort(probs)
    sorted_probs = probs[sort_idx]
    sorted_labels = labels[sort_idx]

    bin_indices = np.array_split(np.arange(len(probs)), num_bins)

    mean_probs = []
    actual_rates = []
    std_errors = []

    print("\nFailure Risk Calibration:")
    print(f"Overall Risk Range: {np.min(probs):.4f} to {np.max(probs):.4f}")
    print(f"{'Bin':<5} | {'Mean Pred Risk':<15} | {'Actual Failure Rate':<20} | {'Count':<6} | {'Std Error':<9}")
    print("-" * 65)

    for i, b_idx in enumerate(bin_indices):
        if len(b_idx) == 0:
            continue

        b_probs = sorted_probs[b_idx]
        b_labels = sorted_labels[b_idx]

        mean_p = float(np.mean(b_probs))
        actual_r = float(np.mean(b_labels))
        # Standard error of the mean for binomial distribution
        std_err = float(np.sqrt(actual_r * (1.0 - actual_r) / len(b_idx)))

        mean_probs.append(mean_p)
        actual_rates.append(actual_r)
        std_errors.append(std_err)

        print(f"{i + 1:<5} | {mean_p:<15.4f} | {actual_r:<20.4f} | {len(b_idx):<6} | {std_err:<9.4f}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.errorbar(mean_probs, actual_rates, yerr=std_errors, marker="o", linestyle="-", label="Calibration Curve", capsize=4)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
        plt.xlabel("Mean Predicted Risk")
        plt.ylabel("Observed Failure Rate")
        plt.title("Failure Risk Calibration")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        print(f"\nSaved calibration plot to {filename}\n")
    except ImportError:
        print("\n[!] matplotlib is not installed. Plot was not saved, but text table was generated.")
        print("    Run `pip install matplotlib` to generate the visual plot next time.\n")
