import streamlit as st
import joblib
import pickle
import numpy as np
from PIL import Image

# Konfigurasi halaman
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Header
st.markdown("<h1 style='text-align: center; color: #F63366;'>🎌 Anime Recommender System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Masukkan judul anime favoritmu dan dapatkan rekomendasi yang mirip!</p>", unsafe_allow_html=True)

# Load model KNN
@st.cache_data
def load_resources():
    df = pd.read_csv("anime.csv")
    model = joblib.load("knn_recommender_model.pkl")  # ganti dari pickle ke joblib
    return df, model

# Dummy data anime — ganti dengan data asli Anda
anime_titles = [
    "Naruto", "Bleach", "One Piece", "Death Note", "Attack on Titan",
    "Fullmetal Alchemist", "Demon Slayer", "Jujutsu Kaisen", "My Hero Academia", "Tokyo Ghoul"
]
anime_features = np.random.rand(len(anime_titles), 10)

# Fungsi rekomendasi
def get_recommendations(title, k=5):
    if title not in anime_titles:
        return ["❌ Judul tidak ditemukan dalam database."]
    
    idx = anime_titles.index(title)
    vector = anime_features[idx].reshape(1, -1)
    
    distances, indices = model.kneighbors(vector, n_neighbors=k+1)
    
    recommendations = []
    for i in range(1, len(indices[0])):
        rec_idx = indices[0][i]
        recommendations.append(anime_titles[rec_idx])
    
    return recommendations

# Input user
with st.form("recommendation_form"):
    anime_input = st.text_input("🎬 Masukkan Judul Anime:", placeholder="Contoh: Naruto")
    submitted = st.form_submit_button("🔍 Dapatkan Rekomendasi")

# Output rekomendasi
if submitted:
    if anime_input:
        with st.spinner("Mencari rekomendasi..."):
            result = get_recommendations(anime_input.strip())
        st.success("🎯 Rekomendasi ditemukan:")
        for i, rec in enumerate(result, 1):
            st.markdown(f"- **{i}. {rec}**")
    else:
        st.warning("⚠️ Silakan masukkan judul anime terlebih dahulu.")

# Footer
st.markdown(
    "<hr><center><small>Made with ❤️ using Streamlit</small></center>",
    unsafe_allow_html=True
)
