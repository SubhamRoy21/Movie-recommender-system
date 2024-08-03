import streamlit as st
import pickle
import pandas as pd
import requests

# Load the movie data and similarity matrix
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Function to fetch movie posters from the TMDB API
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=177588908e4b6d57a5856c161a7adef9&language=en-US"
    response = requests.get(url)
    data = response.json()
    return f"https://image.tmdb.org/t/p/original{data['poster_path']}"

# Function to recommend movies based on similarity
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))
    
    return recommended_movies, recommended_movies_posters

# Streamlit UI
st.title('Movie Recommender System')

# Movie selection dropdown
option = st.selectbox(
    'Select a movie:',
    movies['title'].values
)

# Display recommendations and posters when the button is clicked
if st.button('Recommend'):
    recommendations, posters = recommend(option)
    
    cols = st.columns(5)
    for col, rec_movie, poster in zip(cols, recommendations, posters):
        with col:
            st.markdown(f"**{rec_movie}**")
            st.image(poster, use_column_width=True)