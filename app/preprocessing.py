from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ALIASES = {
    "product": ["product", "product name", "product_name", "nama produk", "produk", "item", "item name", "barang", "nama barang"],
    "quantity": ["quantity sold", "quantity_sold", "quantity", "qty", "qty sold", "jumlah terjual", "jumlah barang", "kuantitas"],
    "frequency": ["transaction frequency", "transaction_frequency", "frequency", "frekuensi transaksi", "frekuensi"],
    "total_sales": ["total sales", "total_sales", "sales", "sales amount", "total penjualan", "jumlah penjualan", "revenue", "omzet"],
    "transaction_id": ["transaction id", "transaction_id", "id transaksi", "no transaksi", "nomor transaksi", "invoice", "invoice id", "order id", "order_id"],
    "unit_price": ["unit price", "unit_price", "harga satuan", "price", "harga"],
}


def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def find_col(df, aliases):
    normalized = {norm(c): c for c in df.columns}
    for a in aliases:
        if norm(a) in normalized:
            return normalized[norm(a)]
    for c in df.columns:
        nc = norm(c)
        for a in aliases:
            na = norm(a)
            if na and (na in nc or nc in na):
                return c
    return None


def numeric_clean(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip()
    s = s.str.replace(r"(?i)rp", "", regex=True)
    s = s.str.replace(r"[^\d,\.\-]", "", regex=True)

    def parse(x):
        if x in ("", "nan", "none", "nat"):
            return np.nan
        if "," in x and "." in x:
            if x.rfind(",") > x.rfind("."):
                x = x.replace(".", "").replace(",", ".")
            else:
                x = x.replace(",", "")
        elif "," in x:
            parts = x.split(",")
            x = "".join(parts[:-1]) + "." + parts[-1] if len(parts[-1]) <= 2 else "".join(parts)
        try:
            return float(x)
        except Exception:
            return np.nan

    return s.map(parse)


def preprocess_dataframe(raw_df):
    """Return display dataframe, normalized matrix, and detail metadata.

    The display dataframe contains ONLY the four requested base variables.
    The three numeric columns contain their Min-Max normalized values, while
    the cleaned raw values are kept internally in meta["raw_clean"] for
    calculation/database purposes.
    """
    logs = []
    df = raw_df.copy()
    logs.append(f"Data awal: {len(df)} baris, {len(df.columns)} kolom.")
    df.columns = [str(c).strip() for c in df.columns]

    product_col = find_col(df, ALIASES["product"])
    quantity_col = find_col(df, ALIASES["quantity"])
    frequency_col = find_col(df, ALIASES["frequency"])
    sales_col = find_col(df, ALIASES["total_sales"])
    trx_col = find_col(df, ALIASES["transaction_id"])
    price_col = find_col(df, ALIASES["unit_price"])

    if quantity_col is None:
        raise ValueError("Kolom Quantity Sold tidak ditemukan.")
    if sales_col is None and price_col is None:
        raise ValueError("Kolom Total Sales tidak ditemukan dan Unit Price/Harga juga tidak tersedia.")

    if product_col is None:
        df["product_name"] = [f"Product-{i+1}" for i in range(len(df))]
        product_col = "product_name"
        logs.append("Kolom produk tidak ditemukan; nama produk generik dibuat.")

    out = pd.DataFrame(index=df.index)
    out["product_name"] = df[product_col].astype(str).str.strip()
    out["Quantity_sold"] = numeric_clean(df[quantity_col])

    if frequency_col is not None:
        out["transaction_frequency"] = numeric_clean(df[frequency_col])
        logs.append(f"Transaction Frequency menggunakan kolom '{frequency_col}'.")
    elif trx_col is not None:
        out["transaction_frequency"] = (
            df.groupby(product_col)[trx_col]
            .transform(lambda x: x.astype(str).nunique())
            .astype(float)
        )
        logs.append(f"Transaction Frequency dihitung dari jumlah transaksi unik berdasarkan '{trx_col}'.")
    else:
        out["transaction_frequency"] = (
            df.groupby(product_col)[product_col].transform("size").astype(float)
        )
        logs.append("Transaction Frequency dihitung dari jumlah baris per produk karena ID transaksi tidak tersedia.")

    if sales_col is not None:
        out["Total_sales"] = numeric_clean(df[sales_col])
        logs.append(f"Total Sales menggunakan kolom '{sales_col}'.")
    else:
        out["Total_sales"] = numeric_clean(df[quantity_col]) * numeric_clean(df[price_col])
        logs.append("Total Sales dihitung dari Quantity Sold x Unit Price.")

    out = out.replace([np.inf, -np.inf], np.nan)

    before = len(out)
    out = out.dropna(subset=["product_name", "Quantity_sold", "transaction_frequency", "Total_sales"])
    logs.append(f"Cleaning missing/non-finite: {before - len(out)} baris dihapus.")

    # Data penjualan yang memiliki produk dan nilai penjualan sama tetap
    # dipertahankan karena dapat berasal dari transaksi yang berbeda.
    # Penghapusan duplikasi tidak dilakukan pada tahap preprocessing.
    logs.append("Cleaning duplikasi: 0 baris dihapus; seluruh transaksi dipertahankan.")

    before = len(out)
    out = out[
        (out["Quantity_sold"] >= 0)
        & (out["transaction_frequency"] >= 0)
        & (out["Total_sales"] >= 0)
    ].copy()
    logs.append(f"Cleaning nilai negatif: {before - len(out)} baris dihapus.")

    if len(out) < 3:
        raise ValueError("Data setelah preprocessing kurang dari 3 baris.")

    features = ["Quantity_sold", "transaction_frequency", "Total_sales"]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(out[features].astype(float))

    min_values = out[features].min().to_dict()
    max_values = out[features].max().to_dict()
    logs.append("Normalisasi Min-Max Scaling diterapkan pada tiga variabel clustering.")
    logs.append("Rumus: x' = (x - min(x)) / (max(x) - min(x)).")
    logs.append(f"Data akhir: {len(out)} baris.")

    # Simpan data asli hasil cleaning secara internal untuk database/detail.
    raw_clean = out.reset_index(drop=True).copy()

    # TABEL UTAMA SETELAH PREPROCESSING:
    # nilai normalisasi menggantikan nilai mentah pada tiga kolom yang sama.
    # Jadi nama kolom tetap:
    # product_name | Quantity_sold | transaction_frequency | Total_sales
    # tetapi nilai Quantity_sold, transaction_frequency, dan Total_sales
    # sudah berupa hasil Min-Max Normalization.
    display = raw_clean.copy()
    display[features] = X

    meta = {
        "feature_cols": features,
        "logs": logs,
        "mapped_columns": {
            "product_name": product_col,
            "Quantity_sold": quantity_col,
            "transaction_frequency": frequency_col or "derived",
            "Total_sales": sales_col or "derived",
            "transaction_id": trx_col,
            "unit_price": price_col,
        },
        "min_values": min_values,
        "max_values": max_values,
        "normalized": X.copy(),
        "raw_clean": raw_clean.copy(),
    }
    return display, X, meta
