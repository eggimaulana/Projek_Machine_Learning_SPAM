import streamlit as st
import pickle
import pandas as pd
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =======================
# Load Model & Vectorizer
# =======================
@st.cache_resource

def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ===============
# Page Config
# ===============
st.set_page_config(page_title="Spam Detector", page_icon="📨", layout="centered")

# =================
# Sidebar & Header
# =================
with st.sidebar:
    st.title("📄 Tentang Aplikasi")
    st.write("""
    Aplikasi ini menggunakan model *Naive Bayes* untuk mendeteksi apakah sebuah pesan teks termasuk **spam** atau **bukan spam** (*ham*).

    Dikembangkan menggunakan:
    - Streamlit
    - scikit-learn
    - Dataset SMSSpamCollection
    """)
    st.markdown("---")
    st.caption("👨‍💻 Dibuat oleh ADVENT - Machine Learning, Nusa Putra 2024")

st.title("📧 Deteksi Pesan Spam")
st.markdown("Masukkan pesan teks di bawah untuk mendeteksi apakah itu **Spam** atau **Bukan Spam (Ham)**.")

# ====================
# Input Area
# ====================
st.text("Coba fitur tambahan seperti auto-detect atau statistik kata!")
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_area("Masukkan pesan teks:", height=150, placeholder="Contoh: Anda memenangkan hadiah uang tunai Rp10.000.000!")

with col2:
    detect_button = st.button("🔍 Deteksi")

# ====================
# Hasil Deteksi
# ====================
if detect_button:
    if user_input.strip() == "":
        st.warning("⚠️ Silakan isi pesan terlebih dahulu.")
    else:
        input_vect = vectorizer.transform([user_input])
        prediction = model.predict(input_vect)[0]

        st.subheader("📋 Hasil Deteksi:")
        if prediction == "spam":
            st.error("💥 Ini adalah SPAM!", icon="🚫")
            st.markdown("> ⚠️ Hindari membuka tautan mencurigakan.")
        else:
            st.success("✅ Ini adalah PESAN AMAN (HAM)", icon="✅")
            st.markdown("> 📨 Kemungkinan pesan ini valid atau dari pengirim yang dikenal.")

# ====================
# Statistik Dataset & Visualisasi
# ====================
with st.expander("📊 Statistik Dataset & Visualisasi"):
    try:
        df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])
        spam_count = df[df["label"] == "spam"].shape[0]
        ham_count = df[df["label"] == "ham"].shape[0]
        total = df.shape[0]

        st.metric("Total Pesan", total)
        st.metric("Pesan SPAM", spam_count)
        st.metric("Pesan HAM", ham_count)

        word_type = st.selectbox("Tampilkan Word Cloud untuk:", ["spam", "ham"])
        filtered_text = " ".join(df[df["label"] == word_type]["message"].values)
        wordcloud = WordCloud(width=700, height=300, background_color='white').generate(filtered_text)

        st.image(wordcloud.to_array(), use_column_width=True)

    except Exception as e:
        st.info("Dataset tidak ditemukan untuk statistik.")
        st.exception(e)

# ====================
# Footer
# ====================
st.markdown("---")
st.markdown("📌 *Model menggunakan Naive Bayes + CountVectorizer, dilatih pada dataset SMSSpamCollection dari UCI ML Repo.*")
