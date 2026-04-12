import streamlit as st
import pickle
import pandas as pd
import requests


movie_dict = pickle.load(open("movie_dict.pkl",'rb'))
movies = pd.DataFrame(movie_dict)

similarity = pickle.load(open("similarity.pkl",'rb'))

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=aa58feccdea654d6a58d5d6102717e1f&language=en-US'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

def recommend(movie):
    movie_index =movies[movies["title"]== movie].index[0]
    distanc = similarity[movie_index]
    movies_list = sorted(list(enumerate(distanc)),reverse=True,key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_poster = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
       
        recommended_movies.append(movies.iloc[i[0]].title)
        #fetch_poster from API
        recommended_movies_poster.append(fetch_poster(movie_id))
    return recommended_movies,recommended_movies_poster

st.title("MOVIE RECOMMENDATION SYSTEM")
selected_movie_name = st.selectbox("Select a Movie",movies['title'].values)

if st.button("Recommend"):
    names,poster = recommend(selected_movie_name)
   
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(names[0])
        st.image(poster[0])

    with col2:
        st.text(names[1])
        st.image(poster[1])

    with col3:
        st.text(names[2])
        st.image(poster[2])

    with col4:
        st.text(names[3])
        st.image(poster[3])

    with col5:
        st.text(names[4])
        st.image(poster[4])