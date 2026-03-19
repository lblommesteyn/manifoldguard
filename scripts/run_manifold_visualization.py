"""Visualize the learned model manifold using PCA.

Usage:
    python scripts/run_manifold_visualization.py [--matplotlib]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manifoldguard.splits import infer_model_family

LATENT_CSV = REPO_ROOT / "results" / "demo" / "latent_vectors.csv"
OUT_DIR = REPO_ROOT / "results" / "demo"

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the model manifold.")
    parser.add_argument("--matplotlib", action="store_true", help="Use matplotlib for 2D visualization.")
    parser.add_argument("--output-3d", type=Path, default=OUT_DIR / "manifold_3d.html", help="Path to save 3D Plotly HTML.")
    parser.add_argument("--output-2d", type=Path, default=OUT_DIR / "manifold_2d.png", help="Path to save 2D Matplotlib PNG.")
    args = parser.parse_args()

    if not LATENT_CSV.exists():
        print(f"Error: {LATENT_CSV} not found. Run scripts/run_demo.py first.")
        sys.exit(1)

    df = pd.read_csv(LATENT_CSV)
    model_names = df["model_name"].tolist()
    
    # Extract latent columns
    latent_cols = [c for c in df.columns if c.startswith("latent_")]
    X = df[latent_cols].values

    # Infer model families for coloring
    families = [infer_model_family(name) for name in model_names]
    df["family"] = families

    if args.matplotlib:
        _plot_matplotlib(X, df, args.output_2d)
    else:
        _plot_plotly(X, df, args.output_3d)

def _plot_plotly(X: np.ndarray, df: pd.DataFrame, output_path: Path) -> None:
    import plotly.express as px
    
    # Project to 3D if possible, else 2D
    n_features = X.shape[1]
    n_components = min(3, n_features)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]
    if n_components == 3:
        df["PC3"] = X_pca[:, 2]
    else:
        df["PC3"] = 0 # Fallback
    
    # Custom marker sizes and symbols
    df["size"] = 10
    df.loc[df["type"] == "target", "size"] = 25
    
    fig = px.scatter_3d(
        df, x="PC1", y="PC2", z="PC3",
        color="family",
        symbol="type",
        hover_name="model_name",
        title=f"Model Manifold Visualization (PCA {n_components}D)",
        labels={"PC1": "Component 1", "PC2": "Component 2", "PC3": "Component 3"},
        size="size",
        size_max=20
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Latent Dim 1',
            yaxis_title='Latent Dim 2',
            zaxis_title='Latent Dim 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"3D visualization saved to {output_path}")

def _plot_matplotlib(X: np.ndarray, df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    
    # Project to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]
    
    plt.figure(figsize=(10, 7))
    
    families = df["family"].unique()
    # Use a standard qualitative colormap
    from matplotlib import cm
    num_families = len(families)
    cmap = cm.get_cmap("tab20", num_families)
    family_to_color = {f: cmap(i) for i, f in enumerate(families)}
    
    for family in families:
        mask = (df["family"] == family) & (df["type"] == "training")
        if mask.any():
            plt.scatter(
                df.loc[mask, "PC1"], df.loc[mask, "PC2"],
                label=f"{family}",
                alpha=0.7, s=60, edgecolors='none',
                color=family_to_color[family]
            )
        
    # Plot target model
    target_mask = df["type"] == "target"
    if target_mask.any():
        plt.scatter(
            df.loc[target_mask, "PC1"], df.loc[target_mask, "PC2"],
            color='red', marker='*', s=400, label=f"Target Model",
            edgecolors='black', linewidth=1.5, zorder=10
        )
    
    plt.xlabel("Latent Space Projection (PC1)")
    plt.ylabel("Latent Space Projection (PC2)")
    plt.title(f"ManifoldGuard: Learned Model Manifold (2D Projection) for {df.loc[target_mask, 'model_name'].values[0]}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Families")
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"2D visualization saved to {output_path}")
    # Don't call plt.show() here to avoid blocking execution in headless env
    plt.close()

if __name__ == "__main__":
    main()
