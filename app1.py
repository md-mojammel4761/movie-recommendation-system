from flask import Flask, render_template, request
import pickle
import pandas as pd
import requests

app = Flask(__name__)

# Load data
with open("movie_dict.pkl", "rb") as f:
    movie_dict = pickle.load(f)

with open("similarity.pkl", "rb") as f:
    similarity = pickle.load(f)

movies = pd.DataFrame(movie_dict)

TMDB_API_KEY = "aa58feccdea654d6a58d5d6102717e1f"


def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=10)
        data = response.json()

        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        return None
    except Exception:
        return None


def recommend(movie_name):
    movie_name = movie_name.strip()

    if not movie_name:
        return [], "Please enter a movie name."

    matched = movies[movies["title"].str.lower() == movie_name.lower()]

    if matched.empty:
        return [], "Movie not found. Please select a movie from the list."

    movie_index = matched.index[0]
    distances = similarity[movie_index]

    movie_scores = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for item in movie_scores:
        idx = item[0]
        title = movies.iloc[idx]["title"]
        movie_id = movies.iloc[idx]["movie_id"]
        poster = fetch_poster(movie_id)

        recommendations.append({
            "title": title,
            "poster": poster
        })

    return recommendations, ""


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    error = ""
    selected_movie = ""

    if request.method == "POST":
        selected_movie = request.form.get("movie", "")
        recommendations, error = recommend(selected_movie)

    movie_list = sorted(movies["title"].dropna().tolist())

    return render_template(
        "index.html",
        movie_list=movie_list,
        recommendations=recommendations,
        error=error,
        selected_movie=selected_movie
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)