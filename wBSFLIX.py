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
.st-h2, .st-h3 {
    color: #E50914;
}
.stSidebar .sidebar-content {
    background-color: #111111;
}
</style>
    """, unsafe_allow_html=True)

# Display the logo with center alignment and reduced size
st.markdown(
    "<div style='text-align: center;'><img src='data:image/png;base64,{}' style='width:30%'></div>".format(
        base64.b64encode(open("wbsflix_logo.png", "rb").read()).decode()
    ),
    unsafe_allow_html=True,
)

# Display the banner
banner = Image.open("wbs_flix_banner.png")
st.image(banner, use_column_width=True)

# Add a centered welcome message
st.markdown(
    "<div style='text-align: center;'><h2>Welcome to WBSFLIX! Your personal movie recommendation platform.</h2></div>",
    unsafe_allow_html=True,
)

# Sidebar with overall controls
st.sidebar.markdown("<h1 style='color: #E50914;'>Controls</h1>", unsafe_allow_html=True)
st.sidebar.write("Use these controls to adjust your recommendations.")

# Searching for movies
movie_search_query = st.sidebar.text_input("Search for a movie by title:", "")
if movie_search_query:
    matching_movies = movies_df[movies_df['title'].str.contains(movie_search_query, case=False)]
    if not matching_movies.empty:
        selected_movie = st.sidebar.selectbox("Select a movie:", matching_movies['title'].tolist())
        st.write(movies_df[movies_df['title'] == selected_movie])
        movie_data = fetch_movie_details(selected_movie)
        poster_url = get_poster_url(movie_data)
        if poster_url:
            st.image(poster_url)
    else:
        st.write("No movies found!")

# Rest of the recommendation code...

# Footer
st.markdown(
    """
    <footer style='text-align: center; color: #ffffff; background-color: #E50914; padding: 10px;'>
        <p>WBSFLIX © 2023 | Made with ❤️ by WBS Students</p>
    </footer>
    """,
    unsafe_allow_html=True,
)
