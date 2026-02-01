import joblib
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()

# Load Model RF
try:
    rf_model = joblib.load('model/rf_model.pkl')
except:
    rf_model = None
    print("⚠️ Model belum ditemukan, jalankan train_model.py dulu!")

# Setup Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

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
    - Profil: Anak ke-{anak_ke}/{jml_saudara}, Ortu: {pekerjaan_ortu}, Saku: {uang_saku}
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