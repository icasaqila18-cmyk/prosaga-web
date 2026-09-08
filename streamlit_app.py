"""
PROSAGA Sales Clustering — Web Version (Streamlit)
Logika inti (preprocessing, K-Means manual, Fuzzy C-Means manual, Silhouette
manual, strategi bisnis, database) memakai ulang modul app/* yang sama
dengan versi desktop, supaya hasil perhitungan tetap konsisten.
"""
from __future__ import annotations

import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from app import db
from app.preprocessing import preprocess_dataframe
from app.clustering import run_kmeans, run_fcm
from app.silhouette import calculate_silhouette
from app.strategy import canonicalize_result, rank_clusters, strategy_for, CATEGORY_COLORS

st.set_page_config(page_title="PROSAGA Sales Clustering", page_icon="📊", layout="wide")

FEATURES = ["Quantity_sold", "transaction_frequency", "Total_sales"]


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def login_screen():
    st.title("📊 PROSAGA — Sales Clustering")
    st.caption("K-Means & Fuzzy C-Means untuk Clustering Data Penjualan")

    first_run = not db.has_any_user()

    if first_run:
        st.info("Belum ada akun. Buat akun admin pertama untuk mulai.")
        with st.form("register_form"):
            u = st.text_input("Username admin")
            p = st.text_input("Password (min. 6 karakter)", type="password")
            p2 = st.text_input("Ulangi password", type="password")
            submitted = st.form_submit_button("Buat Akun Admin", type="primary")
        if submitted:
            if p != p2:
                st.error("Password tidak sama.")
            else:
                ok, msg = db.create_user(u, p, role="admin")
                if ok:
                    st.success("Akun admin dibuat. Silakan login.")
                    st.rerun()
                else:
                    st.error(msg)
        return

    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Masuk", type="primary")
    if submitted:
        ok, msg, user = db.verify_user(u, p)
        if ok:
            st.session_state["user"] = user
            st.rerun()
        else:
            st.error(msg)

    with st.expander("Buat akun staff baru"):
        with st.form("staff_form"):
            u2 = st.text_input("Username baru", key="u2")
            p3 = st.text_input("Password baru", type="password", key="p3")
            submitted2 = st.form_submit_button("Daftar")
        if submitted2:
            ok, msg = db.create_user(u2, p3, role="staff")
            if ok:
                st.success("Akun dibuat, silakan login di atas.")
            else:
                st.error(msg)


# --------------------------------------------------------------------------
# Core pipeline (reused business logic)
# --------------------------------------------------------------------------
def run_full_analysis(raw_df: pd.DataFrame, run_name: str, source_file: str):
    display_df, X, meta = preprocess_dataframe(raw_df)
    raw_clean = meta["raw_clean"]

    km = run_kmeans(X, n_clusters=3)
    canonicalize_result(raw_clean.assign(**{f: X[:, i] for i, f in enumerate(FEATURES)}), FEATURES, km, "labels")
    sil_km = calculate_silhouette(X, km["labels"])

    fcm = run_fcm(X, n_clusters=3)
    canonicalize_result(raw_clean.assign(**{f: X[:, i] for i, f in enumerate(FEATURES)}), FEATURES, fcm, "labels")
    sil_fcm = calculate_silhouette(X, fcm["labels"])

    best_algo = "K-Means" if sil_km["score"] >= sil_fcm["score"] else "Fuzzy C-Means"
    best_labels = km["labels"] if best_algo == "K-Means" else fcm["labels"]

    score_df = raw_clean.copy()
    for i, f in enumerate(FEATURES):
        score_df[f] = X[:, i]
    score_df["_cluster"] = best_labels
    category_map, cluster_means = rank_clusters(score_df, "_cluster", FEATURES)
    strategies = {cat: strategy_for(cat) for cat in category_map.values()}

    results = {
        "kmeans": km, "fcm": fcm,
        "sil_km": sil_km, "sil_fcm": sil_fcm,
        "kmeans_silhouette": sil_km["score"], "fcm_silhouette": sil_fcm["score"],
        "best_algorithm": best_algo, "best_labels": best_labels,
        "best_category_map": category_map, "strategies": strategies,
        "preprocess_logs": meta["logs"], "_raw_clean_df": raw_clean,
    }
    params = {"k": 3, "fcm_m": 2.0}
    run_id = db.save_snapshot(None, run_name, source_file, display_df, X, results, params)
    return run_id, display_df, X, results, category_map, cluster_means


def build_result_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keep = ["product", "quantity", "frequency", "total_sales", "category", "strategy"]
    df = df[keep].rename(columns={
        "product": "Produk", "quantity": "Quantity Sold", "frequency": "Transaction Frequency",
        "total_sales": "Total Sales", "category": "Kategori", "strategy": "Strategi",
    })
    return df


def style_category(df: pd.DataFrame):
    def color_row(row):
        c = CATEGORY_COLORS.get(row["Kategori"], "")
        return [f"background-color: {c}" if c else "" for _ in row]
    return df.style.apply(color_row, axis=1)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_new_analysis():
    st.header("Analisis Baru")
    st.write("Upload file Excel (.xlsx) data penjualan, lalu jalankan clustering K-Means & Fuzzy C-Means.")

    run_name = st.text_input("Nama analisis", value=f"Analisis {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    uploaded = st.file_uploader("File Excel (.xlsx)", type=["xlsx"])

    if uploaded and st.button("Jalankan Analisis", type="primary"):
        try:
            raw_df = pd.read_excel(uploaded)
            with st.spinner("Memproses data & menjalankan K-Means + Fuzzy C-Means..."):
                run_id, display_df, X, results, category_map, cluster_means = run_full_analysis(
                    raw_df, run_name, uploaded.name
                )
            st.session_state["last_run_id"] = run_id
            st.success(f"Analisis selesai & tersimpan (ID #{run_id}).")

            c1, c2, c3 = st.columns(3)
            c1.metric("Silhouette K-Means", f"{results['sil_km']['score']:.4f}")
            c2.metric("Silhouette Fuzzy C-Means", f"{results['sil_fcm']['score']:.4f}")
            c3.metric("Algoritma Terbaik", results["best_algorithm"])

            st.subheader("Kategori Produk")
            run, rows, logs, prep = db.load_run(run_id)
            table = build_result_table(rows)
            st.dataframe(style_category(table), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Gagal memproses: {e}")


def page_dashboard():
    st.header("Dashboard — Riwayat Analisis")
    runs = db.list_runs()
    if not runs:
        st.info("Belum ada analisis. Buka menu 'Analisis Baru' untuk memulai.")
        return

    df = pd.DataFrame(runs)[["id", "run_name", "source_file", "created_at", "n_rows",
                              "kmeans_silhouette", "fcm_silhouette", "best_algorithm"]]
    df.columns = ["ID", "Nama Analisis", "File Sumber", "Dibuat", "Jumlah Baris",
                  "Silhouette K-Means", "Silhouette FCM", "Algoritma Terbaik"]
    st.dataframe(df, use_container_width=True, hide_index=True)

    ids = [r["id"] for r in runs]
    labels = {r["id"]: f"#{r['id']} — {r['run_name']} ({r['created_at']})" for r in runs}
    selected = st.selectbox("Lihat detail analisis:", ids, format_func=lambda i: labels[i])
    if st.button("Buka Detail"):
        st.session_state["view_run_id"] = selected
        st.rerun()

    if runs:
        with st.expander("Hapus analisis"):
            del_id = st.selectbox("Pilih analisis untuk dihapus:", ids, format_func=lambda i: labels[i], key="del_sel")
            if st.button("Hapus", type="secondary"):
                db.delete_run(del_id)
                st.success("Analisis dihapus.")
                st.rerun()


def page_run_detail():
    run_id = st.session_state.get("view_run_id") or st.session_state.get("last_run_id")
    if not run_id:
        st.info("Pilih analisis dari menu Dashboard terlebih dahulu.")
        return
    run, rows, logs, prep = db.load_run(run_id)
    if not run:
        st.warning("Analisis tidak ditemukan.")
        return

    st.header(f"Detail Analisis — {run['run_name']} (#{run['id']})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette K-Means", f"{run['kmeans_silhouette']:.4f}" if run['kmeans_silhouette'] is not None else "-")
    c2.metric("Silhouette Fuzzy C-Means", f"{run['fcm_silhouette']:.4f}" if run['fcm_silhouette'] is not None else "-")
    c3.metric("Algoritma Terbaik", run["best_algorithm"] or "-")

    tab1, tab2, tab3, tab4 = st.tabs(["Kategori & Strategi", "Preprocessing", "Log Iterasi", "Visualisasi"])

    with tab1:
        table = build_result_table(rows)
        st.dataframe(style_category(table), use_container_width=True, hide_index=True)

        st.subheader("5 Produk Terlaris & 5 Produk Kurang Laris")
        sorted_rows = sorted(rows, key=lambda r: r["total_sales"], reverse=True)
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**5 Produk Terlaris**")
            st.dataframe(build_result_table(sorted_rows[:5]), use_container_width=True, hide_index=True)
        with colB:
            st.markdown("**5 Produk Kurang Laris**")
            st.dataframe(build_result_table(sorted_rows[-5:][::-1]), use_container_width=True, hide_index=True)

    with tab2:
        for p in prep:
            st.write(f"{p['step_order']}. {p['message']}")

    with tab3:
        for algo in ["K-Means", "Fuzzy C-Means"]:
            algo_logs = [l for l in logs if l["algorithm"] == algo]
            if not algo_logs:
                continue
            st.subheader(algo)
            for l in algo_logs:
                detail = json.loads(l["details_json"])
                extra = f"SSE={detail.get('inertia'):.6f}" if "inertia" in detail else f"Objective={detail.get('objective'):.6f}"
                st.write(f"Iterasi {l['iteration']}: {extra}")

    with tab4:
        df_plot = pd.DataFrame(rows)
        if not df_plot.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            colors = {"Laris": "#2ca02c", "Sedang": "#ff7f0e", "Kurang Laris": "#d62728"}
            for cat, sub in df_plot.groupby("category"):
                ax.scatter(sub["quantity"], sub["total_sales"], label=cat,
                           c=colors.get(cat, "#888"), alpha=0.7, s=60)
            ax.set_xlabel("Quantity Sold")
            ax.set_ylabel("Total Sales")
            ax.set_title("Sebaran Cluster Produk")
            ax.legend()
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            counts = df_plot["category"].value_counts()
            ax2.bar(counts.index, counts.values, color=[colors.get(c, "#888") for c in counts.index])
            ax2.set_ylabel("Jumlah Produk")
            ax2.set_title("Jumlah Produk per Kategori")
            st.pyplot(fig2)


def page_comparison():
    st.header("Perbandingan Analisis")
    runs = db.list_runs()
    if len(runs) < 2:
        st.info("Minimal perlu 2 analisis tersimpan untuk membandingkan.")
        return

    ids = [r["id"] for r in runs]
    labels = {r["id"]: f"#{r['id']} — {r['run_name']} ({r['created_at']})" for r in runs}
    c1, c2 = st.columns(2)
    with c1:
        id1 = st.selectbox("Analisis 1", ids, format_func=lambda i: labels[i], key="cmp1")
    with c2:
        id2 = st.selectbox("Analisis 2", ids, index=min(1, len(ids) - 1), format_func=lambda i: labels[i], key="cmp2")

    if st.button("Bandingkan", type="primary"):
        run1, rows1, _, _ = db.load_run(id1)
        run2, rows2, _, _ = db.load_run(id2)

        c1, c2 = st.columns(2)
        for col, run, rows in [(c1, run1, rows1), (c2, run2, rows2)]:
            with col:
                st.subheader(run["run_name"])
                st.write(f"Algoritma terbaik: **{run['best_algorithm']}**")
                st.write(f"Silhouette K-Means: {run['kmeans_silhouette']:.4f}")
                st.write(f"Silhouette FCM: {run['fcm_silhouette']:.4f}")
                sorted_rows = sorted(rows, key=lambda r: r["total_sales"], reverse=True)
                st.markdown("**5 Produk Terlaris**")
                st.dataframe(build_result_table(sorted_rows[:5]), use_container_width=True, hide_index=True)
                st.markdown("**5 Produk Kurang Laris**")
                st.dataframe(build_result_table(sorted_rows[-5:][::-1]), use_container_width=True, hide_index=True)

        st.subheader("Kesimpulan")
        better = run1 if (run1["kmeans_silhouette"] or 0) + (run1["fcm_silhouette"] or 0) >= \
                          (run2["kmeans_silhouette"] or 0) + (run2["fcm_silhouette"] or 0) else run2
        st.info(f"Analisis **{better['run_name']}** memiliki kualitas cluster (silhouette) yang lebih baik "
                f"dan disarankan sebagai acuan saat ini. Disarankan menjalankan analisis ulang secara berkala "
                f"seiring bertambahnya data penjualan.")


def page_backup():
    st.header("Backup & Pemulihan Data")
    st.warning("Catatan: jika aplikasi ini dihosting di layanan gratis (mis. Streamlit Community Cloud), "
               "database bisa ter-reset saat aplikasi redeploy/sleep. Lakukan backup rutin dan simpan filenya.")

    if st.button("Download Backup Database Sekarang"):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            path = db.backup_database(tmp)
            data = pathlib.Path(path).read_bytes()
            st.download_button("Klik untuk download file .db", data, file_name=path.name)

    st.divider()
    restore_file = st.file_uploader("Pulihkan dari file backup (.db)", type=["db"])
    if restore_file and st.button("Pulihkan Database"):
        import tempfile, pathlib
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.write(restore_file.read())
            tmp_path = tmp.name
        db.restore_database(tmp_path)
        st.success("Database berhasil dipulihkan.")
        st.rerun()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    if "user" not in st.session_state:
        login_screen()
        return

    user = st.session_state["user"]
    st.sidebar.title("📊 PROSAGA")
    st.sidebar.write(f"Masuk sebagai **{user['username']}** ({user['role']})")
    page = st.sidebar.radio("Menu", [
        "Analisis Baru", "Dashboard", "Detail Analisis", "Perbandingan Analisis", "Backup & Restore"
    ])
    if st.sidebar.button("Keluar (Logout)"):
        del st.session_state["user"]
        st.rerun()

    if page == "Analisis Baru":
        page_new_analysis()
    elif page == "Dashboard":
        page_dashboard()
    elif page == "Detail Analisis":
        page_run_detail()
    elif page == "Perbandingan Analisis":
        page_comparison()
    elif page == "Backup & Restore":
        page_backup()


if __name__ == "__main__":
    main()
