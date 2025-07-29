# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==========================
# CUSTOM THEME (warna lembut menenangkan)
# ==========================
page_bg = """
<style>
body {
    background-color: #f0f7f7; /* Latar biru kehijauan lembut */
    color: #2c3e50;
    font-family: "Segoe UI", sans-serif;
}
h1 {
    color: #2e8b57; /* Hijau pastel */
    text-align: center;
}
textarea {
    background-color: #ffffff !important;
    color: #2c3e50 !important;
    border: 1px solid #aad8d3 !important;
}
.stButton>button {
    background-color: #66b2b2 !important;
    color: white !important;
    border-radius: 8px !important;
    height: 3em;
    width: 12em;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #558b8b !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ==========================
# Load model dan vectorizer
# ==========================
model = joblib.load("svm_model_sas.pkl")            # Model SVM hasil training
vectorizer = joblib.load("tfidf_vectorizer_sas.pkl")  # Vectorizer TF-IDF

# Kamus slang untuk normalisasi kata
slang_dict = {
    "gue": "aku", "gw": "aku", "gua": "aku",
    "ga": "tidak", "gak": "tidak", "tak": "tidak",
    "yg": "yang", "bgt": "banget", "aja": "saja",
    "dah": "sudah", "tp": "tapi", "sampe": "sampai",
    "kalo": "kalau"
}

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ==========================
# Preprocessing teks
# ==========================
def preprocess(text):
    # Huruf kecil
    text = text.lower()

    # Menghapus URL, mention, dan karakter selain huruf
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Normalisasi slang
    tokens = text.split()
    tokens = [slang_dict.get(token, token) for token in tokens]

    # Stemming
    tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)

# Mapping hasil prediksi
label_map = {0: "Normal", 1: "Kecemasan"}

# ==========================
# Tampilan aplikasi
# ==========================
st.title("Prediksi Kesehatan Mental dari Teks")
st.write("Masukkan teks di bawah ini, lalu tekan tombol prediksi. "
         "Sistem akan menganalisis apakah teks termasuk **Normal** atau mengandung indikasi **Kecemasan**.")

# Input teks
user_input = st.text_area("Teks masukan", height=150)

# Tombol prediksi
if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        # Preprocessing
        clean_input = preprocess(user_input)

        # TF-IDF transform
        vector = vectorizer.transform([clean_input])

        # Prediksi
        prediction = model.predict(vector)[0]

        # Tampilkan hasil
        st.success(f"Hasil prediksi: **{label_map[prediction]}**")

# Disclaimer
st.markdown(
    "<p style='font-size: 12px; color: gray; text-align:center;'>"
    "*Aplikasi ini hanya bersifat pendukung dan tidak menggantikan peran psikolog</p>",
    unsafe_allow_html=True
)
