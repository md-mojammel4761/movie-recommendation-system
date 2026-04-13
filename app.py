import streamlit as st
import joblib
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- Load Data --------------------
movie_dict = joblib.load("movie_dict.jbl")
movies = pd.DataFrame(movie_dict)

# -------------------- Cache Vectors --------------------
@st.cache_data
def create_vectors(data):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    return cv.fit_transform(data["tags"]).toarray()

vectors = create_vectors(movies)

# -------------------- Fetch Poster --------------------
def fetch_poster(movie_id):
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=aa58feccdea654d6a58d5d6102717e1f&language=en-US"
        )
        data = response.json()
        return "https://image.tmdb.org/t/p/w500/" + data.get('poster_path', '')
    except:
        return ""

# -------------------- Recommend Function --------------------
def recommend(movie):
    if movie not in movies["title"].values:
        return [], []

    movie_index = movies[movies["title"] == movie].index[0]

    distances = cosine_similarity(
        vectors[movie_index].reshape(1, -1), vectors
    )[0]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_movies_poster = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_poster.append(fetch_poster(movie_id))

    return recommended_movies, recommended_movies_poster

# -------------------- UI --------------------
st.title("🎬 Movie Recommendation System")

selected_movie_name = st.selectbox(
    "Select a Movie",
    movies['title'].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie_name)

    if len(names) == 0:
        st.error("Movie not found!")
    else:
        col1, col2, col3, col4, col5 = st.columns(5)

        for idx, col in enumerate([col1, col2, col3, col4, col5]):
            with col:
                st.text(names[idx])
                st.image(posters[idx])