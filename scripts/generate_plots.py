import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manifoldguard.data import generate_synthetic_scores
from manifoldguard.evaluation import evaluate_experiment, _evaluate_episode
from manifoldguard.ensemble import train_ensemble

def plot_observed_fraction(csv_path, output_path):
    df = pd.read_csv(csv_path)
    frac_df = df[df["ablation"] == "observed_fraction"].copy()
    frac_df["setting"] = frac_df["setting"].astype(float)
    frac_df = frac_df.sort_values("setting")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    plt.style.use('ggplot')

    color = 'tab:blue'
    ax1.set_xlabel('Observed Fraction')
    ax1.set_ylabel('Completion MAE', color=color)
    ax1.plot(frac_df['setting'], frac_df['completion_mae'], marker='o', color=color, linewidth=2, label='MAE')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Failure AUC', color=color)
    ax2.plot(frac_df['setting'], frac_df['failure_auc'], marker='s', color=color, linewidth=2, label='AUC')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Performance vs. Observed Benchmark Fraction')
    fig.tight_layout()
    plt.savefig(output_path)
    print(f"Saved observed fraction plot to {output_path}")

def plot_risk_stratification(output_path, seed=42):
    # Generate data
    matrix_obj = generate_synthetic_scores(num_models=100, num_benchmarks=30, seed=seed)
    matrix = matrix_obj.values
    
    # Split
    train_matrix = matrix[:70]
    test_matrix = matrix[70:]
    
    ensemble = train_ensemble(train_matrix, ensemble_size=5, rank=4, epochs=300)
    
    # Simulated episodes on test set
    from manifoldguard.episodes import simulate_new_model_episodes
    episodes = simulate_new_model_episodes(test_matrix, episodes_per_model=10, observed_fraction=0.35, seed=seed+1)
    results = [_evaluate_episode(ep, ensemble, 1e-2) for ep in episodes]
    
    maes = np.array([r.hidden_mae for r in results])
    features = np.vstack([r.features for r in results])
    
    # Simple cross-val to get probabilities
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    probs = np.zeros(len(results))
    
    # Binary label for "high error" (top 20%)
    threshold = np.percentile(maes, 80)
    labels = (maes >= threshold).astype(int)
    
    for train_idx, val_idx in kf.split(features):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(features[train_idx], labels[train_idx])
        probs[val_idx] = clf.predict_proba(features[val_idx])[:, 1]
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(probs, maes, alpha=0.5, color='purple', edgecolors='white')
    
    # Add trend line (binned average)
    bins = np.linspace(0, 1, 6)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = [maes[(probs >= bins[i]) & (probs < bins[i+1])].mean() for i in range(len(bins)-1)]
    plt.plot(bin_centers, bin_means, color='red', marker='D', linewidth=3, label='Average MAE per Risk Bin')

    plt.xlabel('Predicted Failure Probability (Risk Score)')
    plt.ylabel('Actual Completion MAE')
    plt.title('Risk Stratification: Predicted Risk vs. Actual Error')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path)
    print(f"Saved risk stratification plot to {output_path}")

def plot_qualitative_example(output_path, seed=42):
    matrix_obj = generate_synthetic_scores(num_models=30, num_benchmarks=20, seed=seed)
    matrix = matrix_obj.values
    test_model_idx = 0
    row = matrix[test_model_idx]
    
    # Training (everything except this model)
    train_matrix = np.delete(matrix, test_model_idx, axis=0)
    ensemble = train_ensemble(train_matrix, ensemble_size=5, rank=4, epochs=300)
    
    # Inference
    obs_idx = np.array([2, 5, 8, 12, 15, 18]) # Arbitrary subset
    obs_val = row[obs_idx]
    hidden_idx = np.setdiff1d(np.arange(len(row)), obs_idx)
    
    from manifoldguard.ensemble import predict_new_model
    prediction = predict_new_model(ensemble, obs_idx, obs_val)
    
    # Conformal q (approx for this demo)
    q = 0.4 
    
    # Plot
    plt.figure(figsize=(12, 5))
    x = np.arange(len(hidden_idx))
    true_vals = row[hidden_idx]
    pred_vals = prediction.mean_prediction[hidden_idx]
    
    plt.errorbar(x, pred_vals, yerr=q, fmt='o', color='tab:blue', capsize=5, label='Predicted (+/- 90% Conformal Range)')
    plt.scatter(x, true_vals, color='tab:orange', marker='x', s=100, label='True Score', zorder=3)
    
    plt.xticks(x, [matrix_obj.benchmark_names[i] for i in hidden_idx], rotation=45)
    plt.ylabel('Benchmark Score')
    plt.title(f'Qualitative Example: Scores for {matrix_obj.model_names[test_model_idx]}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved qualitative example plot to {output_path}")

if __name__ == "__main__":
    out_dir = REPO_ROOT / "results" / "plots"
    out_dir.mkdir(exist_ok=True)
    
    ablation_csv = REPO_ROOT / "results" / "ablations" / "ablation_table.csv"
    
    plot_observed_fraction(ablation_csv, out_dir / "observed_fraction.png")
    plot_risk_stratification(out_dir / "risk_stratification.png")
    plot_qualitative_example(out_dir / "qualitative_example.png")
