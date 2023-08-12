import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNBasic
from collections import defaultdict
import requests
from PIL import Image
import base64

# Load datasets
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
tags_df = pd.read_csv('tags.csv')

# TMDb API setup
API_KEY = '21b1b571573d1e9d71be858159b82cb4'
BASE_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

def fetch_movie_details(movie_title):
    MOVIE_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
    response = requests.get(MOVIE_SEARCH_URL, params={
        'api_key': API_KEY,
        'query': movie_title
    })
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_poster_url(movie_data):
    if movie_data and 'results' in movie_data and movie_data['results']:
        poster_path = movie_data['results'][0]['poster_path']
        return BASE_IMAGE_URL + poster_path
    return None

# Initialize session state for ratings
from streamlit import session_state
if "ratings" not in session_state:
    session_state.ratings = {}

# Apply custom CSS styles
st.markdown("""
<style>
body {
    color: #ffffff;
    background-color: #111111;
}
.stTextInput input[type="text"], .stButton>button {
    color: #ffffff;
    background-color: #E50914;
}
.st-h2 {
    color: #E50914;
}
.stSidebar .sidebar-content {
    background-color: #111111;
}
</style>
    """, unsafe_allow_html=True)

# Display the logo with center alignment
st.markdown(
    "<div style='text-align: center;'><img src='https://raw.githubusercontent.com/LiesmannManda/WBSFLIX/30572b06dbb25dd7b94c9fa1ec3e270e3064c2d2/wbsflix%20logo.png' style='width:30%'></div>",
    unsafe_allow_html=True,
)

# Display the banner
st.image("https://raw.githubusercontent.com/LiesmannManda/WBSFLIX/30572b06dbb25dd7b94c9fa1ec3e270e3064c2d2/wbs%20flix%20banner.png", use_column_width=True)

# Sidebar with overall controls
st.sidebar.markdown("<h1 style='color: #E50914;'>Controls</h1>", unsafe_allow_html=True)
st.sidebar.write("Use these controls to adjust your recommendations.")

# Searching for movies
movie_search_query = st.sidebar.text_input("Search for a movie by title:", "")
if movie_search_query:
    matching_movies = movies_df[movies_df['title'].str.contains(movie_search_query, case=False)]
    if not matching_movies.empty:
        selected_movie = st.sidebar.selectbox("Select a movie:", matching_movies['title'].tolist())

        # Allow users to rate the selected movie
        rating = st.sidebar.slider("Rate this movie (1 to 5):", 1, 5, 3)
        if st.sidebar.button("Submit Rating"):
            session_state.ratings[selected_movie] = rating
            st.sidebar.success(f"Thanks for rating {selected_movie}!")
            
        st.write(movies_df[movies_df['title'] == selected_movie])
        movie_data = fetch_movie_details(selected_movie)
        poster_url = get_poster_url(movie_data)
        if poster_url:
            st.image(poster_url)

# ... [Rest of the code remains unchanged]

# Add a footer
st.markdown("""
<footer style="position: absolute; bottom: 0; width: 100%; height: 60px; background-color: #E50914; text-align: center; padding: 20px;">
    <p style="color: white; margin-bottom: 20px;">App by Mutale</p>
</footer>
""", unsafe_allow_html=True)
