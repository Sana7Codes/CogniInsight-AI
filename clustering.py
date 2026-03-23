"""
clustering.py — KMeans clustering and PCA dimensionality reduction.

Pipeline:
  1. Select features: reaction_time_ms, accuracy_pct, error_rate
  2. StandardScaler normalisation (z-score)
  3. KMeans(n_clusters=3) with multiple initialisations for stability
  4. Assign human-readable labels by inspecting cluster-centre positions
  5. PCA(n_components=2) for 2D scatter visualisation
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Features used for clustering (must exist in the input DataFrame)
CLUSTER_FEATURES = ["reaction_time_ms", "accuracy_pct", "error_rate"]

# Cluster label mapping — assigned dynamically; these are the canonical names
LABEL_FOCUSED = "Focused"
LABEL_FATIGUED = "Fatigué"
LABEL_IMPULSIVE = "Impulsif"


def _assign_labels(centers_scaled: np.ndarray, scaler: StandardScaler) -> dict[int, str]:
    """
    Map KMeans integer labels (0, 1, 2) to semantic profile names.

    Strategy:
      - Invert scaling to get interpretable centres.
      - Focused  = highest accuracy / lowest error / moderate–fast RT
      - Fatigued = highest reaction time
      - Impulsive = fastest RT but low accuracy / high error

    Parameters
    ----------
    centers_scaled : array of shape (3, n_features) — cluster centres in scaled space
    scaler         : fitted StandardScaler used for inverse_transform

    Returns
    -------
    dict mapping integer cluster id → label string
    """
    # Inverse-transform to original units: [rt, acc, err]
    centers = scaler.inverse_transform(centers_scaled)  # shape (3, 3)

    rt_vals = centers[:, 0]
    acc_vals = centers[:, 1]
    err_vals = centers[:, 2]

    # Compute a simple "performance score": high acc, low err, moderate rt
    # score = accuracy - error - 0.05 * reaction_time (weights chosen heuristically)
    score = acc_vals - err_vals - 0.05 * rt_vals

    sorted_by_score = np.argsort(score)  # ascending — worst first
    sorted_by_rt = np.argsort(rt_vals)   # ascending — fastest first

    # Fatigued = highest RT
    fatigued_id = int(np.argmax(rt_vals))

    # Among the remaining two: Focused = best score, Impulsive = the other
    remaining = [i for i in range(3) if i != fatigued_id]
    scores_remaining = [(score[i], i) for i in remaining]
    scores_remaining.sort(reverse=True)

    focused_id = scores_remaining[0][1]
    impulsive_id = scores_remaining[1][1]

    return {
        focused_id: LABEL_FOCUSED,
        fatigued_id: LABEL_FATIGUED,
        impulsive_id: LABEL_IMPULSIVE,
    }


def run_clustering(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> dict:
    """
    Fit KMeans and PCA on the user-aggregate DataFrame.

    Parameters
    ----------
    df           : DataFrame with at least the CLUSTER_FEATURES columns
    n_clusters   : number of clusters (default 3)
    random_state : reproducibility seed

    Returns
    -------
    dict with keys:
      - "labels"          : list[str] — profile name per row
      - "label_ids"       : np.ndarray — raw integer cluster ids
      - "pca_components"  : np.ndarray of shape (n, 2)
      - "pca_variance"    : tuple (var_pc1, var_pc2) in percent
      - "scaler"          : fitted StandardScaler
      - "kmeans"          : fitted KMeans model
      - "pca"             : fitted PCA model
      - "label_map"       : dict {int_id → label_string}
      - "df_clustered"    : original df with "cluster_label" column appended
    """
    X = df[CLUSTER_FEATURES].values

    # 1. Normalise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. KMeans with multiple random initialisations for stability
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=20,
        max_iter=300,
        random_state=random_state,
    )
    kmeans.fit(X_scaled)
    label_ids = kmeans.labels_

    # 3. Semantic label mapping
    label_map = _assign_labels(kmeans.cluster_centers_, scaler)
    labels = [label_map[lid] for lid in label_ids]

    # 4. PCA for 2D visualisation
    pca = PCA(n_components=2, random_state=random_state)
    pca_components = pca.fit_transform(X_scaled)
    pca_variance = (
        round(pca.explained_variance_ratio_[0] * 100, 1),
        round(pca.explained_variance_ratio_[1] * 100, 1),
    )

    # 5. Attach labels to df copy
    df_clustered = df.copy()
    df_clustered["cluster_label"] = labels

    return {
        "labels": labels,
        "label_ids": label_ids,
        "pca_components": pca_components,
        "pca_variance": pca_variance,
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "label_map": label_map,
        "df_clustered": df_clustered,
    }


def get_cluster_stats(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-cluster summary statistics for display in the dashboard.
    """
    stats = (
        df_clustered.groupby("cluster_label")[CLUSTER_FEATURES]
        .agg(["mean", "std"])
        .round(1)
    )
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    stats["n_users"] = df_clustered.groupby("cluster_label").size()
    return stats.reset_index()
