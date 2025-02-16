import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset movie
movie = pd.read_csv('D:/skripsi brok/SKRIPSI ILHAM/DEPLOY/movie recommender/movies_5000_v2.csv')
movie['overview'] = movie['overview'].fillna('')

# Inisialisasi TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Transformasi teks overview menjadi vektor numerik
tfidf_matrix = tfidf.fit_transform(movie['overview'])

# Hitung cosine similarity antar film berdasarkan keyword sinopsis
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_movies(keyword, top_n=10):
    """Merekomendasikan film berdasarkan keyword."""
    keyword_tfidf = tfidf.transform([keyword])
    similarity_scores = cosine_similarity(keyword_tfidf, tfidf_matrix).flatten()
    related_movies_idx = similarity_scores.argsort()[-top_n:][::-1]
    recommendations = movie.iloc[related_movies_idx].copy()
    recommendations['similarity_score'] = similarity_scores[related_movies_idx]
    return recommendations[['title', 'overview', 'similarity_score', 'poster_path', 'release_date']]

# Cek apakah ada parameter 'movie' di URL
query_params = st.experimental_get_query_params()
selected_movie_title = query_params.get("movie", [None])[0]

if selected_movie_title:
    # Jika ada movie yang dipilih, tampilkan detailnya
    movie_data = movie[movie['title'] == selected_movie_title].iloc[0]

    st.title(movie_data['title'])
    st.image(f"https://image.tmdb.org/t/p/w500{movie_data['poster_path']}", caption=movie_data['title'])
    st.write(f"**Tahun Rilis:** {movie_data['release_date']}")
    st.write(f"**Sinopsis:** {movie_data['overview']}")

    # Tombol kembali ke halaman utama
    st.markdown("[🔙 Kembali ke Beranda](app.py)", unsafe_allow_html=True)
else:
    # Halaman utama rekomendasi
    st.title("🎬 Rekomendasi Film Berdasarkan Sinopsis")
    st.markdown("Masukkan kata kunci untuk menemukan rekomendasi film yang relevan.")

    keyword = st.text_input("Masukkan kata kunci", "")
    if st.button("Cari"):
        if keyword.strip():
            recommendations = recommend_movies(keyword)
            if not recommendations.empty:
                for _, row in recommendations.iterrows():
                    st.subheader(row['title'])
                    st.write(f"**Similarity Score:** {row['similarity_score']:.2f}")
                    st.write(row['overview'])
                    st.markdown(f"[Lihat Detail](?movie={row['title']})", unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.warning("Tidak ada film yang cocok dengan kata kunci tersebut.")
        else:
            st.error("Kata kunci tidak boleh kosong!")
