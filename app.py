import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Fungsi untuk memuat data & model
@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")
    df['genre'] = df['genre'].fillna("")
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genre'])
    
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(tfidf_matrix)
    
    return df, tfidf, model, tfidf_matrix

# Fungsi rekomendasi
def get_recommendations(title, df, tfidf, model, tfidf_matrix, k=5):
    index = df[df['title'].str.lower() == title.lower()].index
    if len(index) == 0:
        return ["❌ Anime tidak ditemukan."]
    
    idx = index[0]
    genre = df.loc[idx, 'genre']
    vector = tfidf.transform([genre])
    
    distances, indices = model.kneighbors(vector, n_neighbors=k+1)
    
    results = []
    for i in range(1, len(indices[0])):
        results.append(df.iloc[indices[0][i]]['title'])
    
    return results

# UI Streamlit
st.title("🎌 Rekomendasi Anime - Jotjib (KNN)")
anime_input = st.text_input("Masukkan judul anime favoritmu:")

# Load data dan model
df_anime, tfidf_vectorizer, knn_model, tfidf_matrix = load_data()

# Tombol Rekomendasi
if st.button("Dapatkan Rekomendasi"):
    if anime_input.strip():
        result = get_recommendations(
            anime_input.strip(),
            df_anime,
            tfidf_vectorizer,
            knn_model,
            tfidf_matrix
        )
        st.write("### 🎯 Rekomendasi untukmu:")
        for r in result:
            st.write("🔹", r)
    else:
        st.warning("⚠️ Masukkan judul anime terlebih dahulu.")
