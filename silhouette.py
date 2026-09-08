
from __future__ import annotations
import numpy as np

def calculate_silhouette(X, labels):
    """Hitung Silhouette Score secara manual sesuai rumus
    s(i) = (b(i) - a(i)) / max(a(i), b(i)).

    Dihitung manual (tidak memakai sklearn.metrics.silhouette_score/
    silhouette_samples) karena fungsi sklearn tersebut menolak data saat
    jumlah cluster == jumlah data (mis. hanya 3 baris data dengan 3
    cluster, sehingga tiap cluster berisi 1 titik). Rumus manual di
    bawah ini tidak punya batasan tersebut, sehingga analisis tetap bisa
    dijalankan walau datanya sangat sedikit (minimal 3 baris, sesuai
    batas minimum preprocessing).
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        raise ValueError("Silhouette membutuhkan minimal 2 cluster.")

    unique = np.unique(labels)
    rows = []

    for i in range(len(X)):
        c = labels[i]
        same = np.where(labels == c)[0]
        same = same[same != i]
        # a(i): jika cluster hanya berisi 1 titik (cluster singleton,
        # bisa terjadi saat data sangat sedikit), a(i) didefinisikan 0.
        a = float(np.mean(np.linalg.norm(X[same] - X[i], axis=1))) if len(same) else 0.0
        b_values = []
        for other in unique:
            if other == c:
                continue
            idx = np.where(labels == other)[0]
            b_values.append(float(np.mean(np.linalg.norm(X[idx] - X[i], axis=1))))
        b = min(b_values)
        calc_s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        rows.append({"index": i, "cluster": int(c), "a": a, "b": b, "silhouette": float(calc_s)})

    samples = np.array([r["silhouette"] for r in rows], dtype=float)
    score = float(np.mean(samples)) if len(samples) else 0.0
    return {"score": score, "samples": samples, "rows": rows}
