import streamlit as st
import pickle
import pandas as pd
from PIL import Image

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
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_area("Masukkan pesan teks:", height=150, placeholder="Contoh: Anda memenangkan hadiah uang tunai Rp10.000.000!")

with col2:
    if st.button("🔍 Deteksi"):
        if user_input.strip() == "":
            st.warning("⚠️ Silakan isi pesan terlebih dahulu.")
        else:
            # Proses prediksi
            input_vect = vectorizer.transform([user_input])
            prediction = model.predict(input_vect)[0]

            if prediction == "spam":
                st.error("💥 Ini adalah SPAM!", icon="🚫")
                st.markdown("> ⚠️ Hindari membuka tautan mencurigakan.")
            else:
                st.success("✅ Ini adalah PESAN AMAN (HAM)", icon="✅")
                st.markdown("> 📨 Kemungkinan pesan ini valid atau dari pengirim yang dikenal.")

# ====================
# Tambahan: Statistik Dataset
# ====================
with st.expander("📊 Statistik Dataset"):
    try:
        df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])
        spam_count = df[df["label"] == "spam"].shape[0]
        ham_count = df[df["label"] == "ham"].shape[0]
        total = df.shape[0]
        st.metric("Total Pesan", total)
        st.metric("Pesan SPAM", spam_count)
        st.metric("Pesan HAM", ham_count)
    except:
        st.info("Dataset tidak ditemukan untuk statistik.")

# ====================
# Footer
# ====================
st.markdown("---")
st.markdown("📌 *Model menggunakan Naive Bayes + CountVectorizer, dilatih pada dataset SMSSpamCollection dari UCI ML Repo.*")
