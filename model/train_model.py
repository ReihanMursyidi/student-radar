import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# SETUP PATH DATABASE & MODEL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(ROOT_DIR, 'students.db')
MODEL_PATH = os.path.join(BASE_DIR, 'rf_model.pkl')

print(f"🔍 Mengakses database di: {DB_PATH}")

# KONEKSI KE SQLITE
try:
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        kehadiran, nilai, pelanggaran,
        uang_saku, jml_saudara
    FROM students
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
except Exception as e:
    print(f"❌ Error membaca database: {e}")
    exit()

if df.empty:
    print("⚠️ Database masih kosong! Menggunakan data dummy untuk inisialisasi awal...")
    df = pd.DataFrame({
        'kehadiran':   [95, 80, 60, 90, 50, 98, 75, 88, 40, 85],
        'nilai':       [85, 70, 55, 88, 40, 90, 65, 78, 35, 72],
        'pelanggaran': [0, 2, 5, 0, 8, 0, 3, 1, 10, 2],
        'uang_saku':   [50000, 20000, 10000, 45000, 5000, 60000, 15000, 30000, 5000, 25000],
        'jml_saudara': [1, 3, 5, 2, 6, 1, 4, 2, 7, 3]
    })

print(f"✅ Dataset siap dengan {len(df)} data siswa.")

# REKAYASA LABEL TARGET (y)
kondisi_berisiko = (
    (df['nilai'] < 65) | 
    (df['kehadiran'] < 75) | 
    (df['pelanggaran'] >= 10) |
    ((df['uang_saku'] < 10000) & (df['jml_saudara'] >= 3))
)

df['target'] = np.where(kondisi_berisiko, 1, 0)

# PERSIAPAN TRAINING
X = df[['kehadiran', 'nilai', 'pelanggaran', 'uang_saku', 'jml_saudara']]
y = df['target']

# TRAINING MODEL RANDOM FOREST
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

if len(df) < 5:
    print("ℹ️ Data sedikit, melatih dengan seluruh dataset...")
    rf_model.fit(X, y)
else:
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model terlatih dengan akurasi: {acc*100:.2f}%")

# SIMPAN MODEL
try:
    joblib.dump(rf_model, MODEL_PATH)
    print(f"💾 Model berhasil disimpan ulang ke: {MODEL_PATH}")
    print("🚀 Sistem siap digunakan!")
except Exception as e:
    print(f"❌ Gagal menyimpan model: {e}")