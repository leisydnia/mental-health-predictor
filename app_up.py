# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==========================================================
# CUSTOM TEMA LEMBUT / PASTEL
# ==========================================================
page_bg = """
<style>
/* Latar belakang pastel */
.stApp {
    background-color: #f0f7f7;
}

/* Judul */
h1 {
    color: #2e8b57;
    text-align: center;
}

/* Area teks */
textarea {
    background-color: #ffffff !important;
    color: #2c3e50 !important;
    border: 1px solid #aad8d3 !important;
}

/* Tombol */
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

# ==========================================================
# LOAD MODEL DAN VECTORIZER
# ==========================================================
model = joblib.load("svm_model_sas.pkl")
vectorizer = joblib.load("tfidf_vectorizer_sas.pkl")

# Kamus slang
slang_dict = {
    "gue": "aku", "gw": "aku", "gua": "aku",
    "ga": "tidak", "gak": "tidak", "tak": "tidak",
    "yg": "yang", "bgt": "banget", "aja": "saja",
    "dah": "sudah", "tp": "tapi", "sampe": "sampai",
    "kalo": "kalau"
}

# Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ==========================================================
# FUNGSI PREPROCESSING
# ==========================================================
def preprocess(text):
    # lowercase
    text = text.lower()

    # hapus url, mention, non-huruf
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # normalisasi slang
    tokens = text.split()
    tokens = [slang_dict.get(token, token) for token in tokens]

    # stemming
    tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)

# Mapping label
label_map = {0: "Normal", 1: "Kecemasan"}

# ==========================================================
# UI STREAMLIT
# ==========================================================
st.title("Prediksi Kesehatan Mental dari Teks")
st.write("Masukkan teks pada kolom di bawah ini. Sistem akan menganalisis apakah teks tergolong **Normal** atau mengandung indikasi **Kecemasan**.")

# Input pengguna
user_input = st.text_area("Teks masukan", height=150)

# Tombol prediksi
if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        clean_input = preprocess(user_input)
        vector = vectorizer.transform([clean_input])
        prediction = model.predict(vector)[0]

        st.success(f"Hasil prediksi: **{label_map[prediction]}**")

# Disclaimer di bagian bawah
st.markdown(
    "<p style='font-size: 12px; color: gray; text-align:center;'>"
    "*Aplikasi ini hanya bersifat pendukung dan tidak menggantikan peran psikolog</p>",
    unsafe_allow_html=True
)
