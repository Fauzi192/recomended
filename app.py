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
def load_data():
    df = pd.read_csv("anime.csv")
    df['genre'] = df['genre'].fillna("")
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genre'])
    
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(tfidf_matrix)
    
    return df, tfidf, model, tfidf_matrix

# Dummy data anime — ganti dengan data asli Anda
anime_titles = [
    "Naruto", "Bleach", "One Piece", "Death Note", "Attack on Titan",
    "Fullmetal Alchemist", "Demon Slayer", "Jujutsu Kaisen", "My Hero Academia", "Tokyo Ghoul"
]
anime_features = np.random.rand(len(anime_titles), 10)

# Fungsi rekomendasi
def get_recommendations(title, df, model, vectorizer, k=5):
    index = df[df['title'] == title].index
    if len(index) == 0:
        return ["Anime tidak ditemukan."]
    
    index = index[0]
    genre = df.loc[index, 'genre']
    vector = vectorizer.transform([genre])
    
    distances, indices = model.kneighbors(vector, n_neighbors=k+1)

    recommended_titles = []
    for i in range(1, len(indices[0])):
        recommended_titles.append(df.iloc[indices[0][i]]['title'])

    return recommended_titles

# Input user
with st.form("recommendation_form"):
    anime_input = st.text_input("🎬 Masukkan Judul Anime:", placeholder="Contoh: Naruto")
    submitted = st.form_submit_button("🔍 Dapatkan Rekomendasi")

# Output rekomendasi
anime_input = st.text_input("Masukkan judul anime favoritmu:")

# Load data dan model
df_anime, tfidf_vectorizer, knn_model, tfidf_matrix = load_data()
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
# Footer
st.markdown(
    "<hr><center><small>Made with ❤️ using Streamlit</small></center>",
    unsafe_allow_html=True
)
