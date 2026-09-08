from __future__ import annotations
import numpy as np
import pandas as pd

CATEGORY_COLORS = {
    "Laris": "#C6EFCE",
    "Sedang": "#FFEB9C",
    "Kurang Laris": "#FFC7CE",
}


def rank_clusters(df, label_col, features):
    means = df.groupby(label_col)[features].mean().copy()
    means["score"] = means[features].mean(axis=1)
    ordered = list(means["score"].sort_values(ascending=False).index)
    names = ["Laris", "Sedang", "Kurang Laris"]
    mapping = {int(c): names[min(i, 2)] for i, c in enumerate(ordered)}
    return mapping, means


# ---------------------------------------------------------------------------
# Canonical cluster numbering
#
# K-Means/FCM assign cluster ids (0, 1, 2, ...) arbitrarily based on the
# initial centroid draw, so "Cluster 1" from one run can be the busiest
# cluster while in another run it can be the slowest-selling one. To keep
# the app (and every report generated from it) consistent, every clustering
# result is renumbered right after it is computed so that:
#   Cluster 1 (id 0) = highest average characteristic  -> "Laris"
#   Cluster 2 (id 1) = middle average characteristic    -> "Sedang"
#   Cluster 3 (id 2) = lowest average characteristic    -> "Kurang Laris"
# rank_clusters() above keeps working unchanged: once labels are canonical,
# it will simply confirm {0: "Laris", 1: "Sedang", 2: "Kurang Laris"}.
# ---------------------------------------------------------------------------

def build_canonical_permutation(df, label_col, features):
    """Return perm where perm[old_label] = canonical_label (0=highest
    average score / Laris ... n-1=lowest / Kurang Laris)."""
    means = df.groupby(label_col)[features].mean()
    score = means[features].mean(axis=1)
    ordered_old_ids = list(score.sort_values(ascending=False).index)
    n = len(ordered_old_ids)
    perm = np.zeros(n, dtype=int)
    for new_id, old_id in enumerate(ordered_old_ids):
        perm[int(old_id)] = new_id
    return perm


def _reorder_rows(arr, perm):
    arr = np.asarray(arr)
    out = np.zeros_like(arr)
    for old_id, new_id in enumerate(perm):
        out[new_id] = arr[old_id]
    return out


def _reorder_cols(arr, perm):
    arr = np.asarray(arr)
    out = np.zeros_like(arr)
    for old_id, new_id in enumerate(perm):
        out[:, new_id] = arr[:, old_id]
    return out


def apply_cluster_permutation(perm, labels, centers=None, history=None, membership=None):
    """Renumber a clustering result (labels/centers/history/membership)
    according to perm (old_id -> canonical_id). Used for both K-Means and
    Fuzzy C-Means result dictionaries produced by app.clustering."""
    labels = np.asarray(labels)
    new_labels = perm[labels]

    new_centers = _reorder_rows(centers, perm) if centers is not None else None
    new_membership = _reorder_cols(membership, perm) if membership is not None else None

    new_history = None
    if history is not None:
        new_history = []
        for h in history:
            nh = dict(h)
            for key in ("centers_before", "centers_after", "centers"):
                if nh.get(key) is not None:
                    nh[key] = _reorder_rows(nh[key], perm)
            for key in ("distances",):
                if nh.get(key) is not None:
                    nh[key] = _reorder_cols(nh[key], perm)
            for key in ("membership_before", "membership_after"):
                if nh.get(key) is not None:
                    nh[key] = _reorder_cols(nh[key], perm)
            if nh.get("labels") is not None:
                nh["labels"] = perm[np.asarray(nh["labels"])]
            new_history.append(nh)

    return new_labels, new_centers, new_history, new_membership


def canonicalize_result(source_df, features, result, label_key="labels"):
    """Renumber `result` (a run_kmeans/run_fcm dict) IN PLACE so cluster 0
    is always the highest-average-characteristic cluster (Laris), 1 is
    Sedang, and 2 is Kurang Laris. `source_df` must contain `features`
    aligned row-for-row with result[label_key] (e.g. the preprocessed/
    normalized dataframe). Returns the same `result` dict for convenience.
    """
    source = source_df.copy()
    source["_cluster"] = np.asarray(result[label_key])
    perm = build_canonical_permutation(source, "_cluster", features)

    new_labels, new_centers, new_history, new_membership = apply_cluster_permutation(
        perm,
        result[label_key],
        centers=result.get("centers"),
        history=result.get("history"),
        membership=result.get("membership"),
    )
    result[label_key] = new_labels
    if new_centers is not None:
        result["centers"] = new_centers
    if new_history is not None:
        result["history"] = new_history
    if new_membership is not None:
        result["membership"] = new_membership
    return result


def strategy_for(category):
    return {
        "Laris": "Tambah stok",
        "Sedang": "Promosi",
        "Kurang Laris": "Evaluasi",
    }.get(category, "Evaluasi")
