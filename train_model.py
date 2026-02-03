import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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
            df_list.append(df_db)
            print(f"✅ Database: {len(df_db)} data.")
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
            df_list.append(df_excel)
            print(f"✅ Excel: {len(df_excel)} data.")
        else:
            print(f"❌ Excel kurang kolom. Wajib ada: {cols_needed}")
except Exception as e:
    print(f"❌ Excel Error: {e}")

# GABUNGKAN DATA
if df_list:
    df = pd.concat(df_list, ignore_index=True)
else:
    print("❌ TIDAK ADA DATA SAMA SEKALI (Database kosong & Excel tidak ada).")
    print("👉 Buat file 'training_data.xlsx' dulu atau input data via web.")
    exit()

if 'status_manual' in df.columns:
    kondisi_rumus = (
       (df['kehadiran'] < 80) |
       (df['nilai'] < 70) |
       (df['pelanggaran'] > 40) |
       ((df['uang_saku'] < 10000) & (df['jml_saudara'] >= 3))
    )
    df['target'] = np.where(
        df['status_manual'].notna(),  # Kondisi: Apakah ada isinya?
        df['status_manual'],          # Jika YA: Pakai isi manual
        np.where(kondisi_rumus, 1, 0) # Jika TIDAK: Pakai rumus
    )
else:
    print(f"⚠️ Tidak ada label manual. Menggunakan LOGIKA RUMUS (Rule-Based)")

    # REKAYASA LABEL TARGET (y)
    kondisi_berisiko = (
        (df['kehadiran'] < 80) |
        (df['nilai'] < 70) | 
        (df['pelanggaran'] > 40) |
        ((df['uang_saku'] < 10000) & (df['jml_saudara'] >= 3))
    )

    df['target'] = np.where(kondisi_berisiko, 1, 0)

df['target'] = df['target'].astype(int)

# SPLIT DATA
X = df[['kehadiran', 'nilai', 'pelanggaran', 'uang_saku', 'jml_saudara']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MULAI TRAINING
print("⚙️ Melatih Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# EVALUASI
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n📊 HASIL EVALUASI MODEL:")
print(f"🎯 Akurasi pada Data Ujian (Test Set): {acc * 100:.1f}%")
print("📝 Detail Laporan:")
print(classification_report(y_test, y_pred, zero_division=0))

# SIMPAN
joblib.dump(rf_model, MODEL_PATH)
print(f"💾 Model disimpan di: {MODEL_PATH}")