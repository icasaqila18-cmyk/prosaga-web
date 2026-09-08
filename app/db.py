from __future__ import annotations
import sqlite3, json, os, hashlib, binascii
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "prosaga_clustering.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_name TEXT NOT NULL,
        source_file TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        n_rows INTEGER,
        kmeans_silhouette REAL,
        fcm_silhouette REAL,
        best_algorithm TEXT,
        params_json TEXT
    );
    CREATE TABLE IF NOT EXISTS run_rows (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        row_index INTEGER,
        product TEXT,
        quantity REAL,
        frequency REAL,
        total_sales REAL,
        quantity_norm REAL,
        frequency_norm REAL,
        total_sales_norm REAL,
        kmeans_cluster INTEGER,
        fcm_cluster INTEGER,
        fcm_membership TEXT,
        silhouette_kmeans REAL,
        silhouette_fcm REAL,
        best_cluster INTEGER,
        category TEXT,
        strategy TEXT,
        FOREIGN KEY(run_id) REFERENCES runs(id)
    );
    CREATE TABLE IF NOT EXISTS run_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        algorithm TEXT,
        iteration INTEGER,
        details_json TEXT,
        FOREIGN KEY(run_id) REFERENCES runs(id)
    );
    CREATE TABLE IF NOT EXISTS preprocessing_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        step_order INTEGER,
        message TEXT,
        FOREIGN KEY(run_id) REFERENCES runs(id)
    );
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_salt TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'staff',
        is_active INTEGER NOT NULL DEFAULT 1,
        failed_attempts INTEGER NOT NULL DEFAULT 0,
        locked_until TEXT,
        must_change_password INTEGER NOT NULL DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        last_login TEXT
    );
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        action TEXT NOT NULL,
        description TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(run_rows)").fetchall()}
    for column, dtype in {
        "silhouette_kmeans": "REAL", "silhouette_fcm": "REAL",
        "best_cluster": "INTEGER", "category": "TEXT", "strategy": "TEXT",
    }.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE run_rows ADD COLUMN {column} {dtype}")

    # Migration for authentication/security features added after the first
    # release. Existing accounts remain usable; the oldest account becomes
    # Admin automatically.
    user_cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    for column, dtype in {
        "role": "TEXT NOT NULL DEFAULT 'staff'",
        "is_active": "INTEGER NOT NULL DEFAULT 1",
        "failed_attempts": "INTEGER NOT NULL DEFAULT 0",
        "locked_until": "TEXT",
        "must_change_password": "INTEGER NOT NULL DEFAULT 0",
        "last_login": "TEXT",
    }.items():
        if column not in user_cols:
            conn.execute(f"ALTER TABLE users ADD COLUMN {column} {dtype}")
    conn.execute("UPDATE users SET role='admin' WHERE id=(SELECT MIN(id) FROM users)")
    conn.commit(); conn.close()


def save_snapshot(run_id, run_name, source_file, df, normalized_X, results, params):
    """Create/update ONE run. Safe to call after every completed menu action."""
    init_db()
    conn = get_conn(); cur = conn.cursor()
    km_score = results.get("kmeans_silhouette")
    fcm_score = results.get("fcm_silhouette")
    best = results.get("best_algorithm")

    if run_id is None:
        cur.execute(
            "INSERT INTO runs(run_name,source_file,n_rows,kmeans_silhouette,fcm_silhouette,best_algorithm,params_json) VALUES(?,?,?,?,?,?,?)",
            (run_name, source_file, len(df), km_score, fcm_score, best, json.dumps(params)),
        )
        run_id = cur.lastrowid
    else:
        cur.execute(
            "UPDATE runs SET run_name=?, source_file=?, n_rows=?, kmeans_silhouette=?, fcm_silhouette=?, best_algorithm=?, params_json=? WHERE id=?",
            (run_name, source_file, len(df), km_score, fcm_score, best, json.dumps(params), run_id),
        )
        if cur.rowcount == 0:
            cur.execute(
                "INSERT INTO runs(run_name,source_file,n_rows,kmeans_silhouette,fcm_silhouette,best_algorithm,params_json) VALUES(?,?,?,?,?,?,?)",
                (run_name, source_file, len(df), km_score, fcm_score, best, json.dumps(params)),
            )
            run_id = cur.lastrowid

    # Snapshot the current state. This makes repeated saves idempotent.
    cur.execute("DELETE FROM run_rows WHERE run_id=?", (run_id,))
    cur.execute("DELETE FROM run_logs WHERE run_id=?", (run_id,))
    cur.execute("DELETE FROM preprocessing_logs WHERE run_id=?", (run_id,))

    raw_df = results.get("_raw_clean_df", df)
    km = results.get("kmeans")
    fcm = results.get("fcm")
    km_labels = km.get("labels") if km else None
    fcm_labels = fcm.get("labels") if fcm else None
    membership = fcm.get("membership") if fcm else None
    km_sil = results.get("sil_km", {}).get("samples") if results.get("sil_km") else None
    fcm_sil = results.get("sil_fcm", {}).get("samples") if results.get("sil_fcm") else None
    best_labels = results.get("best_labels")
    category_map = results.get("best_category_map", {})
    strategies = results.get("strategies", {})

    # DataFrame at each stage is guaranteed to have the three normalized features
    # after preprocessing. raw_df keeps the cleaned/original scale for the DB.
    for i in range(len(df)):
        raw_row = raw_df.iloc[i] if i < len(raw_df) else df.iloc[i]
        product = str(raw_row.get("product_name", df.iloc[i].get("product_name", "")))
        quantity = float(raw_row.get("Quantity_sold", 0) or 0)
        frequency = float(raw_row.get("transaction_frequency", 0) or 0)
        total_sales = float(raw_row.get("Total_sales", 0) or 0)
        norm = normalized_X[i] if normalized_X is not None and i < len(normalized_X) else [None, None, None]
        klabel = int(km_labels[i]) if km_labels is not None else None
        flabel = int(fcm_labels[i]) if fcm_labels is not None else None
        bcluster = int(best_labels[i]) if best_labels is not None else None
        category = category_map.get(bcluster) if bcluster is not None else None
        strategy = strategies.get(category) if category else None
        cur.execute("""
            INSERT INTO run_rows(
              run_id,row_index,product,quantity,frequency,total_sales,
              quantity_norm,frequency_norm,total_sales_norm,
              kmeans_cluster,fcm_cluster,fcm_membership,
              silhouette_kmeans,silhouette_fcm,best_cluster,category,strategy
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            run_id, i, product, quantity, frequency, total_sales,
            float(norm[0]) if norm[0] is not None else None,
            float(norm[1]) if norm[1] is not None else None,
            float(norm[2]) if norm[2] is not None else None,
            klabel, flabel,
            json.dumps(membership[i].tolist()) if membership is not None else None,
            float(km_sil[i]) if km_sil is not None else None,
            float(fcm_sil[i]) if fcm_sil is not None else None,
            bcluster, category, strategy,
        ))

    for i, msg in enumerate(results.get("preprocess_logs", []), 1):
        cur.execute("INSERT INTO preprocessing_logs(run_id,step_order,message) VALUES(?,?,?)", (run_id, i, msg))

    for algorithm, key in [("K-Means", "kmeans"), ("Fuzzy C-Means", "fcm")]:
        if key not in results:
            continue
        for item in results[key].get("history", []):
            safe = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in item.items()}
            cur.execute("INSERT INTO run_logs(run_id,algorithm,iteration,details_json) VALUES(?,?,?,?)",
                        (run_id, algorithm, int(item["iteration"]), json.dumps(safe)))

    conn.commit(); conn.close(); return run_id


def backup_database(dest_dir) -> Path:
    """Copy the whole database (semua riwayat analisis) ke sebuah file
    baru dengan nama bertimestamp di folder tujuan `dest_dir`.

    Memakai SQLite Online Backup API (bukan sekadar copy file) supaya
    proses ini tetap aman walau file .db sedang dipakai aplikasi.
    Mengembalikan Path file backup yang dihasilkan.
    """
    init_db()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = dest_dir / f"prosaga_backup_{timestamp}.db"

    src_conn = sqlite3.connect(DB_PATH)
    dest_conn = sqlite3.connect(backup_path)
    try:
        src_conn.backup(dest_conn)
    finally:
        dest_conn.close()
        src_conn.close()
    return backup_path


def restore_database(backup_file) -> None:
    """Pulihkan database aplikasi dari sebuah file backup .db.

    Menimpa seluruh isi database yang sedang dipakai aplikasi dengan
    isi dari `backup_file`. Dipakai saat pindah/ganti laptop: salin
    file backup ke laptop baru lalu panggil fungsi ini.
    """
    backup_file = Path(backup_file)
    if not backup_file.exists():
        raise FileNotFoundError(f"File backup tidak ditemukan: {backup_file}")

    src_conn = sqlite3.connect(backup_file)
    dest_conn = sqlite3.connect(DB_PATH)
    try:
        src_conn.backup(dest_conn)
    finally:
        dest_conn.close()
        src_conn.close()


def list_runs():
    init_db(); conn = get_conn()
    rows = conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
    conn.close(); return [dict(r) for r in rows]


def load_run(run_id):
    init_db(); conn = get_conn()
    run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    rows = conn.execute("SELECT * FROM run_rows WHERE run_id=? ORDER BY row_index", (run_id,)).fetchall()
    logs = conn.execute("SELECT * FROM run_logs WHERE run_id=? ORDER BY algorithm,iteration", (run_id,)).fetchall()
    prep = conn.execute("SELECT * FROM preprocessing_logs WHERE run_id=? ORDER BY step_order", (run_id,)).fetchall()
    conn.close()
    return (dict(run) if run else None, [dict(r) for r in rows], [dict(r) for r in logs], [dict(r) for r in prep])


def delete_run(run_id):
    """Delete one saved analysis and all of its dependent detail records."""
    init_db(); conn = get_conn(); cur = conn.cursor()
    cur.execute("DELETE FROM run_rows WHERE run_id=?", (run_id,))
    cur.execute("DELETE FROM run_logs WHERE run_id=?", (run_id,))
    cur.execute("DELETE FROM preprocessing_logs WHERE run_id=?", (run_id,))
    cur.execute("DELETE FROM runs WHERE id=?", (run_id,))
    deleted = cur.rowcount > 0
    conn.commit(); conn.close()
    return deleted


# ---------------------------------------------------------------------------
# Authentication (local, offline). Passwords are never stored in plain text:
# each one is salted with a random value and hashed with PBKDF2-HMAC-SHA256
# (200,000 iterations), which is the standard, dependency-free way to do
# this in Python. Only the salt + resulting hash are stored in SQLite.
# ---------------------------------------------------------------------------

_PBKDF2_ITERATIONS = 200_000
LOCKOUT_THRESHOLD = 5
LOCKOUT_MINUTES = 15


def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = os.urandom(16) if salt_hex is None else binascii.unhexlify(salt_hex)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return binascii.hexlify(salt).decode(), binascii.hexlify(derived).decode()


def has_any_user() -> bool:
    init_db()
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    conn.close()
    return count > 0


def _now():
    return datetime.now()


def _iso(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None


def _parse_dt(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def audit_log(username, action, description=""):
    """Record important security and business actions for the Admin audit page."""
    init_db()
    conn = get_conn()
    conn.execute(
        "INSERT INTO audit_logs(username, action, description) VALUES(?,?,?)",
        (username, action, description),
    )
    conn.commit()
    conn.close()


def get_user(username: str):
    init_db()
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", ((username or "").strip(),)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_users():
    init_db()
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, username, role, is_active, failed_attempts, locked_until, "
        "must_change_password, created_at, last_login FROM users ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_admins():
    init_db()
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) AS c FROM users WHERE role='admin' AND is_active=1").fetchone()["c"]
    conn.close()
    return int(n)


def create_user(username: str, password: str, role: str = "admin", must_change_password: bool = False,
                actor: str | None = None) -> tuple[bool, str | None]:
    username = (username or "").strip()
    role = (role or "staff").lower()
    if not username:
        return False, "Username tidak boleh kosong."
    if len(username) < 3:
        return False, "Username minimal 3 karakter."
    if len(password or "") < 6:
        return False, "Password minimal 6 karakter."
    if role not in ("admin", "staff"):
        return False, "Role akun tidak valid."

    init_db()
    conn = get_conn()
    salt_hex, hash_hex = _hash_password(password)
    try:
        conn.execute(
            """INSERT INTO users
               (username,password_salt,password_hash,role,is_active,failed_attempts,
                must_change_password)
               VALUES(?,?,?,?,1,0,?)""",
            (username, salt_hex, hash_hex, role, int(bool(must_change_password))),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username sudah digunakan."
    conn.close()
    audit_log(actor or username, "BUAT_AKUN", f"Membuat akun '{username}' dengan role {role}.")
    return True, None


def verify_user(username: str, password: str):
    """Return (ok, message, user_dict). Applies active-state and lockout rules."""
    username = (username or "").strip()
    init_db()
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if not row:
        conn.close()
        audit_log(username or "UNKNOWN", "LOGIN_GAGAL", "Username tidak ditemukan.")
        return False, "Username atau password salah.", None

    user = dict(row)
    if not user["is_active"]:
        conn.close()
        audit_log(username, "LOGIN_DITOLAK", "Akun berstatus nonaktif.")
        return False, "Akun sedang dinonaktifkan oleh Admin.", None

    now = _now()
    locked_until = _parse_dt(user.get("locked_until"))
    if locked_until and locked_until > now:
        conn.close()
        audit_log(username, "LOGIN_DITOLAK", f"Akun terkunci sampai {locked_until:%H:%M:%S}.")
        return False, f"Akun terkunci sementara sampai {locked_until:%H:%M}.", None

    if locked_until and locked_until <= now:
        conn.execute("UPDATE users SET failed_attempts=0, locked_until=NULL WHERE id=?", (user["id"],))
        user["failed_attempts"] = 0
        user["locked_until"] = None

    _, hash_hex = _hash_password(password or "", user["password_salt"])
    if hash_hex != user["password_hash"]:
        failed = int(user.get("failed_attempts") or 0) + 1
        if failed >= LOCKOUT_THRESHOLD:
            until = now.replace(microsecond=0)
            from datetime import timedelta
            until += timedelta(minutes=LOCKOUT_MINUTES)
            conn.execute(
                "UPDATE users SET failed_attempts=?, locked_until=? WHERE id=?",
                (failed, _iso(until), user["id"]),
            )
            conn.commit(); conn.close()
            audit_log(username, "AKUN_TERKUNCI", f"{LOCKOUT_THRESHOLD} kali gagal login. Terkunci {LOCKOUT_MINUTES} menit.")
            return False, f"Login gagal {LOCKOUT_THRESHOLD} kali. Akun terkunci selama {LOCKOUT_MINUTES} menit.", None
        conn.execute("UPDATE users SET failed_attempts=? WHERE id=?", (failed, user["id"]))
        conn.commit(); conn.close()
        audit_log(username, "LOGIN_GAGAL", f"Percobaan ke-{failed}.")
        return False, "Username atau password salah.", None

    conn.execute(
        "UPDATE users SET failed_attempts=0, locked_until=NULL, last_login=? WHERE id=?",
        (_iso(now), user["id"]),
    )
    conn.commit()
    user["failed_attempts"] = 0
    user["locked_until"] = None
    user["last_login"] = _iso(now)
    conn.close()
    audit_log(username, "LOGIN_BERHASIL", f"Login sebagai {user['role']}.")
    return True, "Login berhasil.", user


def set_user_active(username: str, active: bool, actor: str) -> tuple[bool, str | None]:
    username = (username or "").strip()
    if username == (actor or "").strip() and not active:
        return False, "Admin yang sedang login tidak dapat menonaktifkan akunnya sendiri."
    user = get_user(username)
    if not user:
        return False, "Akun tidak ditemukan."
    if user["role"] == "admin" and not active and count_admins() <= 1:
        return False, "Tidak dapat menonaktifkan Admin terakhir yang masih aktif."

    init_db()
    conn = get_conn()
    conn.execute("UPDATE users SET is_active=?, failed_attempts=0, locked_until=NULL WHERE username=?",
                 (1 if active else 0, username))
    conn.commit(); conn.close()
    audit_log(actor, "UBAH_STATUS_AKUN", f"Akun '{username}' menjadi {'Aktif' if active else 'Nonaktif'}.")
    return True, None


def admin_change_password(target_username: str, new_password: str, actor: str) -> tuple[bool, str | None]:
    target_username = (target_username or "").strip()
    if len(new_password or "") < 6:
        return False, "Password minimal 6 karakter."
    if not get_user(target_username):
        return False, "Akun tidak ditemukan."
    salt_hex, hash_hex = _hash_password(new_password)
    init_db()
    conn = get_conn()
    conn.execute(
        "UPDATE users SET password_salt=?, password_hash=?, must_change_password=0, "
        "failed_attempts=0, locked_until=NULL WHERE username=?",
        (salt_hex, hash_hex, target_username),
    )
    conn.commit(); conn.close()
    audit_log(actor, "UBAH_PASSWORD", f"Password akun '{target_username}' diubah oleh Admin.")
    return True, None


def get_audit_logs(limit=500):
    init_db()
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, username, action, description, created_at FROM audit_logs "
        "ORDER BY id DESC LIMIT ?", (int(limit),)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def change_password(username: str, old_password: str, new_password: str):
    """Legacy compatibility: retained for older imports.
    The current UI intentionally exposes password management to Admin only."""
    ok, msg, _ = verify_user(username, old_password)
    if not ok:
        return False, msg
    return admin_change_password(username, new_password, username)
