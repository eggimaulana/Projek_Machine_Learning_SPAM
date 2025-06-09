
# 📧 Aplikasi Deteksi Pesan Spam dengan Streamlit

Aplikasi ini merupakan implementasi sederhana untuk mendeteksi pesan spam menggunakan algoritma **Naive Bayes**. Dibangun menggunakan **Streamlit**, aplikasi ini memungkinkan pengguna mengetikkan pesan teks dan mendapatkan hasil klasifikasi secara langsung (spam atau ham).

---

## 🚀 Fitur

- Input pesan teks langsung dari browser.
- Deteksi otomatis apakah pesan termasuk **spam** atau **bukan spam (ham)**.
- Model dan vectorizer dilatih menggunakan dataset **SMSSpamCollection**.
- Tampilan antarmuka web interaktif menggunakan **Streamlit**.

---

## 🛠 Teknologi yang Digunakan

- Python
- scikit-learn
- pandas
- Streamlit

---

## 📁 Struktur Folder

```

.
├── train.py              # Melatih model dan menyimpan file model.pkl dan vectorizer.pkl
├── spam.py               # Aplikasi Streamlit untuk deployment
├── SMSSpamCollection     # Dataset asli (format .tsv tanpa ekstensi)
├── model.pkl             # Model yang sudah dilatih (hasil dari train.py)
├── vectorizer.pkl        # Vectorizer yang sudah dilatih
└── requirements.txt      # Daftar pustaka yang dibutuhkan

````

---

## 📦 Cara Menjalankan Lokal

1. **Clone repository ini:**

```bash
git clone https://github.com/Buluzaku12/spam-detector-streamlit.git
cd spam-detector-streamlit
````

2. **Install dependensi:**

```bash
pip install -r requirements.txt
```

3. **Latih model (jika belum ada file .pkl):**

```bash
python train.py
```

4. **Jalankan aplikasi Streamlit:**

```bash
streamlit run spam.py
```

---

## 🌐 Deployment Online

Aplikasi ini dapat diakses secara publik melalui Streamlit Cloud.
🔗 [Klik di sini untuk melihat aplikasi yang sudah dideploy](https://spam-detector-app-x6cnc4w42x6pyauer9tdxv.streamlit.app/)



---

## 📚 Dataset

* Dataset: [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
* Sumber: UCI Machine Learning Repository
* Format: tab-separated (.tsv)

---

## 🙋‍♂️ Kontributor

* Nama: **ADVENT CLEMENT ZENDRATO**
* Institusi: Universitas Nusa Putra
* Mata Kuliah: Machine Learning
* Dosen: GINA PURNAMA INSANY, S.ST., M.Kom

---
