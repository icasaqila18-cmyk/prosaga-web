
from __future__ import annotations
import numpy as np
from sklearn.metrics import pairwise_distances

def run_kmeans(X, n_clusters=3, random_state=42, max_iter=300, tol=1e-4):
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_state)
    centers = X[rng.choice(len(X), size=n_clusters, replace=False)].copy()
    history, logs = [], []
    labels = None

    for it in range(1, max_iter + 1):
        distances = pairwise_distances(X, centers, metric="euclidean")
        new_labels = distances.argmin(axis=1)
        new_centers = centers.copy()

        for j in range(n_clusters):
            pts = X[new_labels == j]
            new_centers[j] = pts.mean(axis=0) if len(pts) else X[rng.integers(0, len(X))]

        shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
        inertia = float(np.sum((X - new_centers[new_labels]) ** 2))
        history.append({
            "iteration": it,
            "centers_before": centers.copy(),
            "distances": distances.copy(),
            "labels": new_labels.copy(),
            "centers_after": new_centers.copy(),
            "shift": shift,
            "inertia": inertia
        })
        logs.append(f"Iterasi {it}: max centroid shift={shift:.10f}; SSE={inertia:.10f}.")
        labels = new_labels
        centers = new_centers
        if shift <= tol:
            logs.append(f"Konvergen pada iterasi {it}; shift <= toleransi {tol}.")
            break

    return {
        "labels": labels.astype(int), "centers": centers,
        "history": history, "logs": logs,
        "iterations": len(history),
        "inertia": float(np.sum((X - centers[labels]) ** 2))
    }

def run_fcm(X, n_clusters=3, m=2.0, max_iter=300, tol=1e-5, random_state=42):
    if m <= 1:
        raise ValueError("Fuzziness m harus > 1.")
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_state)
    U = rng.random((len(X), n_clusters))
    U /= U.sum(axis=1, keepdims=True)
    history, logs = [], []

    for it in range(1, max_iter + 1):
        Um = U ** m
        centers = (Um.T @ X) / (Um.sum(axis=0)[:, None] + 1e-15)
        dist = np.maximum(pairwise_distances(X, centers), 1e-12)
        power = 2.0 / (m - 1.0)
        ratio = (dist[:, :, None] / dist[:, None, :]) ** power
        U_new = 1.0 / ratio.sum(axis=2)
        objective = float(np.sum((U_new ** m) * (dist ** 2)))
        change = float(np.max(np.abs(U_new - U)))

        history.append({
            "iteration": it, "membership_before": U.copy(),
            "centers": centers.copy(), "distances": dist.copy(),
            "membership_after": U_new.copy(),
            "objective": objective, "max_change": change
        })
        logs.append(f"Iterasi {it}: objective={objective:.10f}; max membership change={change:.10f}.")
        U = U_new
        if change <= tol:
            logs.append(f"Konvergen pada iterasi {it}; perubahan membership <= toleransi {tol}.")
            break

    return {
        "labels": U.argmax(axis=1).astype(int), "membership": U,
        "centers": centers, "history": history, "logs": logs,
        "iterations": len(history), "objective": float(history[-1]["objective"]), "m": m
    }
