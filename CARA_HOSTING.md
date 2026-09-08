# Cara Hosting PROSAGA Sales Clustering (Versi Web)

Aplikasi ini sudah diubah dari versi desktop (PySide6) menjadi versi web
(Streamlit), memakai ulang logika K-Means, Fuzzy C-Means, Silhouette,
preprocessing, dan strategi bisnis yang sama persis dengan versi desktop —
jadi angka & hasil analisisnya tetap konsisten dengan yang sudah kamu tulis
di skripsi.

Cara paling cepat & gratis untuk hosting: **Streamlit Community Cloud**.
Estimasi waktu: 10-15 menit.

## Langkah 1 — Upload ke GitHub
1. Buat akun GitHub kalau belum punya (https://github.com).
2. Buat repository baru (bisa **Private**), misal nama `prosaga-web`.
3. Upload SEMUA isi folder ini (streamlit_app.py, folder app/,
   requirements.txt, .gitignore) ke repository tersebut.
   - Paling gampang: di halaman repo GitHub, klik "Add file" →
     "Upload files", lalu drag semua file/folder ke situ, klik "Commit".

## Langkah 2 — Deploy di Streamlit Community Cloud
1. Buka https://share.streamlit.io lalu login pakai akun GitHub kamu.
2. Klik "Create app" / "New app".
3. Pilih repository `prosaga-web` yang tadi dibuat.
4. Main file path: `streamlit_app.py`
5. Klik "Deploy". Tunggu 1-3 menit sampai muncul link (contoh:
   `https://prosaga-web.streamlit.app`).
6. Link itu yang kamu kasih ke dosen/penguji — bisa dibuka dari HP,
   laptop, komputer kampus, dari mana saja, tanpa install apa pun.

## Langkah 3 — Pertama kali dibuka
- Karena belum ada akun, aplikasi akan minta kamu membuat **akun admin
  pertama** (username + password). Buat sekarang juga, sebelum sidang,
  supaya saat demo tinggal login.
- Setelah itu upload file Excel data penjualan (format sama seperti
  `test_sales.xlsx` yang disertakan) di menu "Analisis Baru".

## Hal penting yang perlu kamu tahu
- **Database (riwayat analisis) bisa ter-reset** kalau aplikasi
  "tidur" lama (tidak dibuka >±7 hari) lalu di-restart oleh Streamlit
  Cloud, atau saat kamu push perubahan kode baru. Ini keterbatasan
  paket gratis (penyimpanan tidak permanen).
  - **Sebelum sidang**: login, buat akun, upload data, jalankan
    analisis dulu sehari sebelumnya supaya aplikasi "bangun" dan siap.
  - Ada menu **Backup & Restore** di sidebar untuk download file
    database (.db) sebagai cadangan kapan saja.
- Kalau kamu butuh penyimpanan yang benar-benar permanen (tidak akan
  saya bahas di sini karena butuh setup tambahan), opsinya adalah
  pindah dari SQLite lokal ke database cloud (mis. Supabase/Postgres) —
  bisa kita kerjakan kalau waktumu masih cukup.

## Menjalankan di komputer sendiri dulu (opsional, untuk cek sebelum deploy)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Lalu buka http://localhost:8501 di browser.
