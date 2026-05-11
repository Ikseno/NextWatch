import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="NextWatch",
    page_icon="🎬",
    layout="wide"
)

# Title
st.title("🎬 NextWatch")
st.subheader("Movie Recommendation System")

st.write(
    "Get movie recommendations using TF-IDF and Cosine Similarity"
)

# Load dataset
movies = pd.read_csv("movies.csv")

# Fill missing values
movies["genres"] = movies["genres"].fillna("")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Cosine Similarity Matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create title index mapping
indices = pd.Series(
    movies.index,
    index=movies['title']
).drop_duplicates()

# Recommendation function
def recommend(title, cosine_sim=cosine_sim):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return movies[['title', 'genres']].iloc[movie_indices]

# User input
movie_name = st.selectbox(
    "Choose a movie",
    movies["title"].sort_values().values
)

# Button
if st.button("Get Recommendations"):

    recommendations = recommend(movie_name)

    st.subheader("Recommended Movies")

    for i, row in recommendations.iterrows():
        st.write(f"🎥 {row['title']}")
        st.write(f"Genres: {row['genres']}")
        st.write("---")
