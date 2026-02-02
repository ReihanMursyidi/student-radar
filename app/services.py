import joblib
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, 'models', 'rf_model.pkl')

print(f"📂 Mencari Model di: {os.path.dirname(MODEL_DIR)}")

# Load Model RF
rf_model = None
if os.path.exists(MODEL_DIR):
    try:
        rf_model = joblib.load(MODEL_DIR)
        print(f"✅ Model berhasil dimuat dari: {MODEL_DIR}")
    except Exception as e:
        print(f"❌ File ada tapi rusak: {e}")
else:
    print(f"⚠️ Model tidak ditemukan. Pastikan sudah jalankan 'train_model.py' terlebih dahulu!")

# Setup Gemini
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
except Exception as e:
    llm = None
    print(f"⚠️ Google GenAI belum dikonfigurasi: {e}")

def predict_risk_rf(kehadiran, nilai, pelanggaran, uang_saku, saudara):
    if rf_model is None:
        return 0.5

    try:
        # Create DataFrame for prediction
        input_data = pd.DataFrame([[kehadiran, nilai, pelanggaran, uang_saku, saudara]], 
                                  columns=['kehadiran', 'nilai', 'pelanggaran', 'uang_saku', 'jml_saudara'])
        
        risk_prob = rf_model.predict_proba(input_data)[0][1]
        return risk_prob
    except Exception as e:
        print(f"⚠️ Error dalam prediksi: {e}")
        return 0.5

def analyze_with_gemini(data_siswa):
    """
    data_siswa: Dictionary lengkap berisi semua atribut
    """

    template = """
    Anda adalah asisten cerdas untuk Guru BK. Tugas Anda adalah membaca data siswa dan memberikan intisari singkat.
    DATA SISWA:
    - Nama: {nama}
    - Profil: Anak ke-{anak_ke} dari {jml_saudara} bersaudara, Ortu: {pekerjaan_ortu}, Saku: {uang_saku}
    - Minat: {organisasi}, {hobi}
    - Akademis: Absen {kehadiran}%, Nilai {nilai}, Poin {pelanggaran}
    - Skor Risiko AI: {risk_score}%
    - Catatan: "{catatan}"
    
    INSTRUKSI:
    Berikan output hanya dalam 3 poin bullet. Fokus pada korelasi antara Latar Belakang Ekonomi/Keluarga dengan Prestasi.

    OUTPUT:
    * **Akar Masalah:** [Jelaskan penyebab utama dalam 1 kalimat pendek]
    * **Potensi:** [Sebutkan hal positif dari hobi/organisasi yang bisa jadi penyelamat]
    * **Solusi:** [Saran intervensi spesifik untuk guru]
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "nama", "anak_ke", "jml_saudara", "pekerjaan_ortu", 
            "uang_saku", "organisasi", "hobi", "kehadiran", 
            "nilai", "pelanggaran", "risk_score", "catatan"
        ]
    )

    chain = prompt | llm
    # Passing semua data ke prompt
    return chain.invoke(data_siswa).content