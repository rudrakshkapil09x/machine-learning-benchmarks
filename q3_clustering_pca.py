"""Question 3: K-Means and PCA from scratch.

Dataset: Credit Card Customers (archive/CC GENERAL.csv)

This script deliberately implements both algorithms with NumPy rather than
using scikit-learn estimators:
  * K-Means: assignment/update iterations and WCSS calculation
  * PCA: covariance matrix followed by eigendecomposition
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_STATE = 42
DATA_PATH = Path("archive/CC GENERAL.csv")
OUTPUT_DIR = Path("outputs/q3")
K_VALUES = [2, 3, 4, 5, 6]


def standardize_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Median-impute numeric columns, then return zero-mean/unit-variance data."""
    features = frame.drop(columns="CUST_ID", errors="ignore").select_dtypes(include="number")
    features = features.fillna(features.median())

    values = features.to_numpy(dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    return (values - means) / stds, list(features.columns)


def kmeans_from_scratch(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int = 300,
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Run one K-Means fit and return labels, centroids, WCSS, and iterations."""
    centroids = x[rng.choice(x.shape[0], size=k, replace=False)].copy()

    for iteration in range(1, max_iter + 1):
        squared_distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = squared_distances.argmin(axis=1)

        updated_centroids = np.empty_like(centroids)
        for cluster in range(k):
            members = x[labels == cluster]
            # Re-seed an empty cluster to a real observation.
            updated_centroids[cluster] = (
                members.mean(axis=0) if len(members) else x[rng.integers(x.shape[0])]
            )

        shift = np.linalg.norm(updated_centroids - centroids)
        centroids = updated_centroids
        if shift <= tolerance:
            break

    squared_distances = ((x - centroids[labels]) ** 2).sum(axis=1)
    wcss = float(squared_distances.sum())
    return labels, centroids, wcss, iteration


def best_kmeans_fit(x: np.ndarray, k: int, n_init: int = 10):
    """Use several random initializations and retain the lowest-WCSS solution."""
    rng = np.random.default_rng(RANDOM_STATE + k)
    best = None
    for _ in range(n_init):
        result = kmeans_from_scratch(x, k, rng)
        if best is None or result[2] < best[2]:
            best = result
    return best


def pca_from_scratch(x: np.ndarray):
    """Compute PCA from a covariance matrix and symmetric eigendecomposition."""
    centered = x - x.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    scores = centered @ eigenvectors
    return scores, eigenvalues, eigenvectors, explained_variance_ratio


def save_elbow_plot(wcss_by_k: dict[int, float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(wcss_by_k), list(wcss_by_k.values()), marker="o", linewidth=2)
    ax.set(title="K-Means Elbow Curve", xlabel="Number of clusters (K)", ylabel="WCSS")
    ax.set_xticks(K_VALUES)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "kmeans_elbow_curve.png", dpi=150)
    plt.close(fig)


def save_variance_plot(explained_variance_ratio: np.ndarray) -> None:
    cumulative = np.cumsum(explained_variance_ratio)
    components = np.arange(1, len(cumulative) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(components, cumulative, marker="o", color="tab:purple")
    ax.axhline(0.80, color="tab:gray", linestyle="--", label="80% variance")
    ax.axhline(0.90, color="tab:green", linestyle="--", label="90% variance")
    ax.set(title="PCA Cumulative Explained Variance", xlabel="Number of principal components",
           ylabel="Cumulative explained variance")
    ax.set_xticks(components)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_cumulative_explained_variance.png", dpi=150)
    plt.close(fig)


def save_pca_scatter(scores: np.ndarray, labels: np.ndarray, selected_k: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=labels, cmap="tab10", s=12, alpha=0.65)
    ax.set(title=f"Credit Card Customers in PCA Space (K={selected_k})",
           xlabel="Principal Component 1", ylabel="Principal Component 2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_first_two_components.png", dpi=150)
    plt.close(fig)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    x, feature_names = standardize_features(data)
    print(f"Loaded {len(data)} customers with {len(feature_names)} numeric features.")
    print("Missing numeric values were replaced with each feature's median before scaling.")

    wcss_by_k: dict[int, float] = {}
    kmeans_results = {}
    for k in K_VALUES:
        result = best_kmeans_fit(x, k)
        kmeans_results[k] = result
        wcss_by_k[k] = result[2]
        print(f"K={k}: WCSS={result[2]:.2f}; iterations={result[3]}")
    save_elbow_plot(wcss_by_k)

    scores, eigenvalues, eigenvectors, explained = pca_from_scratch(x)
    cumulative = np.cumsum(explained)
    pca_summary = pd.DataFrame({
        "Principal Component": [f"PC{i}" for i in range(1, len(explained) + 1)],
        "Eigenvalue": eigenvalues,
        "Explained Variance Ratio": explained,
        "Cumulative Explained Variance": cumulative,
    })
    pca_summary.to_csv(OUTPUT_DIR / "pca_variance_summary.csv", index=False)
    pd.DataFrame(eigenvectors, index=feature_names,
                 columns=[f"PC{i}" for i in range(1, len(feature_names) + 1)]).to_csv(
        OUTPUT_DIR / "pca_eigenvectors.csv"
    )
    pd.DataFrame(scores[:, :3], columns=["PC1", "PC2", "PC3"]).to_csv(
        OUTPUT_DIR / "pca_2_and_3_component_reductions.csv", index=False
    )
    save_variance_plot(explained)

    # The elbow plot is the decision evidence; the observed elbow is K=4.
    selected_k = 4
    save_pca_scatter(scores, kmeans_results[selected_k][0], selected_k)
    print(f"PC1 + PC2 explain {cumulative[1]:.2%} of variance.")
    print(f"PC1 + PC2 + PC3 explain {cumulative[2]:.2%} of variance.")
    print(f"Saved Q3 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
