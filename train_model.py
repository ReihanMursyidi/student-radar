import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# SETUP PATH DATABASE & MODEL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == 'app' else BASE_DIR
DB_PATH = os.path.join(ROOT_DIR, 'students.db')
EXCEL_PATH = os.path.join(ROOT_DIR, 'training_data.xlsx') # File Excel Anda
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

df_list = []

# AMBIL DATA DARI DATABASE
try:
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT kehadiran, nilai, pelanggaran, uang_saku, jml_saudara FROM students"
        df_db = pd.read_sql_query(query, conn)
        conn.close()

        if not df_db.empty:
            print(f"✅ Ditemukan {len(df_db)} data riil dari Database.")
            df_list.append(df_db)
        else:
            print("ℹ️ Database belum dibuat, melewati load database.")
except Exception as e:
    print(f"❌ Gagal mengakses database: {e}")

# AMBIL DATA DARI EXCEL
try:
    if os.path.exists(EXCEL_PATH):
        df_excel = pd.read_excel(EXCEL_PATH)
        df_excel.columns =[c.strip().lower() for c in df_excel.columns]

        cols_needed = ['kehadiran', 'nilai', 'pelanggaran', 'uang_saku', 'jml_saudara']
        if all(col in df_excel.columns for col in cols_needed):
            df_excel = df_excel[cols_needed]
            print(f"✅ Ditemukan {len(df_excel)} data simulasi dari Excel.")
            df_list.append(df_excel)
        else:
            print(f"❌ Kolom Excel tidak lengkap! Wajib ada: {cols_needed}")
    else:
        print(f"ℹ️ File Excel tidak ditemukan di: {EXCEL_PATH}")
except Exception as e:
    print(f"❌ Gagal membaca file Excel: {e}")

# GABUNGKAN DATA
if df_list:
    df = pd.concat(df_list, ignore_index=True)
else:
    print("❌ TIDAK ADA DATA SAMA SEKALI (Database kosong & Excel tidak ada).")
    print("👉 Buat file 'training_data.xlsx' dulu atau input data via web.")
    exit()

print(f"📊 Total Data Training: {len(df)} baris.")

# REKAYASA LABEL TARGET (y)
kondisi_berisiko = (
    (df['nilai'] < 65) | 
    (df['kehadiran'] < 75) | 
    (df['pelanggaran'] >= 10) |
    ((df['uang_saku'] < 10000) & (df['jml_saudara'] >= 3))
)

df['target'] = np.where(kondisi_berisiko, 1, 0)

# Cek keseimbangan data
counts = df['target'].value_counts()
print(f"   - Siswa Aman (0): {counts.get(0, 0)}")
print(f"   - Siswa Berisiko (1): {counts.get(1, 0)}")

# PERSIAPAN TRAINING
X = df[['kehadiran', 'nilai', 'pelanggaran', 'uang_saku', 'jml_saudara']]
y = df['target']

# TRAINING MODEL RANDOM FOREST
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Tampilkan akurasi (jika data > 1 baris)
if len(df) > 1:
    acc = accuracy_score(y, rf_model.predict(X))
    print(f"🎯 Akurasi Training: {acc * 100:.1f}%")

# SIMPAN MODEL
try:
    joblib.dump(rf_model, MODEL_PATH)
    print(f"💾 SUKSES! Model baru disimpan di: {MODEL_PATH}")
    print("🚀 Restart server 'main.py' untuk menggunakan model baru ini.")
except Exception as e:
    print(f"❌ Gagal menyimpan model: {e}")