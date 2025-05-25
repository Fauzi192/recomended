import streamlit as st
import pandas as pd
import jotlib

# Fungsi untuk memuat data dan model dari file .jot
@st.cache_resource
def load_resources():
    df = pd.read_csv("anime.csv")
    model = jotlib.load("knn_model.jot")
    return df, model

# Fungsi untuk mendapatkan rekomendasi
def get_recommendations(title, df, model, k=5):
    # Mencari index anime berdasarkan title
    try:
        index = df[df['title'].str.lower() == title.lower()].index[0]
    except IndexError:
        return []

    vector = model['vectors'][index]
    distances, indices = model['knn'].kneighbors([vector], n_neighbors=k+1)
    
    recommendations = []
    for i in range(1, len(indices[0])):
        anime_idx = indices[0][i]
        anime_title = df.iloc[anime_idx]['title']
        recommendations.append(anime_title)
    return recommendations

# Load model dan dataset
df_anime, knn_model = load_resources()

# UI
st.title("🎌 Rekomendasi Anime Berdasarkan Genre (Jotlib + KNN)")
st.write("Masukkan anime favoritmu untuk mendapatkan rekomendasi anime serupa!")

anime_input = st.text_input("Anime favorit kamu:")

if st.button("🎯 Dapatkan Rekomendasi"):
    if anime_input.strip() == "":
        st.warning("Masukkan judul anime terlebih dahulu.")
    else:
        result = get_recommendations(anime_input, df_anime, knn_model)
        if result:
            st.success("Rekomendasi Anime:")
            for i, r in enumerate(result, 1):
                st.write(f"{i}. {r}")
        else:
            st.error("Anime tidak ditemukan. Pastikan penulisan judul benar.")
