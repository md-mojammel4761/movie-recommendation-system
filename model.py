import pandas as pd
import numpy as np
import ast
import joblib

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------- Load Data --------------------
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")

# Keep only required columns
movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

# Drop missing values
movies.dropna(inplace=True)


# -------------------- Preprocessing --------------------
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]


def convert_cast(obj):
    names = []
    for i, item in enumerate(ast.literal_eval(obj)):
        if i < 3:
            names.append(item['name'])
        else:
            break
    return names


def fetch_director(obj):
    return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director']


movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(convert_cast)
movies["crew"] = movies["crew"].apply(fetch_director)

# Split overview into words
movies["overview"] = movies["overview"].apply(lambda x: x.split())

# Remove spaces between words (e.g., "Sam Worthington" → "SamWorthington")
for col in ["genres", "keywords", "cast", "crew"]:
    movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

# Create tags
movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]

# New dataframe
df = movies[["movie_id", "title", "tags"]]

# Convert list to string
df["tags"] = df["tags"].apply(lambda x: " ".join(x))


# -------------------- Text Processing --------------------
ps = PorterStemmer()


def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])


df["tags"] = df["tags"].apply(stem)
df["tags"] = df["tags"].apply(lambda x: x.lower())


# -------------------- Vectorization --------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df["tags"]).toarray()

similarity = cosine_similarity(vectors)


# -------------------- Recommendation Function --------------------
def recommend(movie):
    try:
        movie_index = df[df["title"] == movie].index[0]
    except IndexError:
        print("Movie not found!")
        return

    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print(f"\nTop recommendations for '{movie}':\n")
    for i in movies_list:
        print(df.iloc[i[0]].title)


# -------------------- Save Files --------------------
joblib.dump(df.to_dict(), "movie_dict.jbl")
joblib.dump(similarity, "similarity.jbl")


# -------------------- Example Run --------------------
if __name__ == "__main__":
    recommend("Batman Begins")