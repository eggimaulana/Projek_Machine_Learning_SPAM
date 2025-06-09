import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])

# Preprocessing
X = df["message"]
y = df["label"]

# Vectorize pesan
vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X)

# Train model Naive Bayes
model = MultinomialNB()
model.fit(X_vect, y)

# Simpan model & vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model dan vectorizer berhasil disimpan.")
