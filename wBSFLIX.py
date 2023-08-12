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
.stTextInput input[type="text"] {
    color: #ffffff;
    background-color: #333333;
}
.stButton>button {
    color: #ffffff;
    background-color: #E50914;
}
.sidebar .sidebar-content {
    background-color: #111111;
}
</style>
    """, unsafe_allow_html=True)

# Display the logo and banner with center alignment and reduced size
st.markdown(
    "<div style='text-align: center;'><img src='data:image/png;base64,{}' style='width:50%'></div>".format(
        base64.b64encode(open("/mnt/data/wbsflix logo.png", "rb").read()).decode()
    ),
    unsafe_allow_html=True,
)

# Display the banner
banner = Image.open("/mnt/data/wbs flix banner.png")
st.image(banner, use_column_width=True)

# Introduction message
st.markdown(
    """
    <div style='text-align: center; color: #E50914;'>
        <h2>Welcome to WBSFLIX!</h2>
        <p>Discover movies tailored just for you. Dive in and explore!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar with overall controls
st.sidebar.header("Controls")
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

# Collaborative Filtering Recommendations
st.markdown("<h3 style='color: #E50914;'>Collaborative Filtering Recommendations</h3>", unsafe_allow_html=True)
selected_movie_title = st.selectbox("Select a movie you like:", movies_df['title'].tolist())
if selected_movie_title:
    selected_movie_id = movies_df[movies_df['title'] == selected_movie_title]['movieId'].iloc[0]
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    similar_movie_ids = item_similarity_df[selected_movie_id].sort_values(ascending=False).index[1:11]
    recommended_movie_titles = movies_df[movies_df['movieId'].isin(similar_movie_ids)]['title'].tolist()
    st.write("Movies you might also like:")
    for movie in recommended_movie_titles:
        st.write(movie)

# Additional recommendation systems can be added below...

# Footer
st.markdown(
    """
    <footer style='text-align: center; color: #ffffff; background-color: #E50914; padding: 10px;'>
        <p>WBSFLIX © 2023 | Made with ❤️ by WBS Students</p>
    </footer>
    """,
    unsafe_allow_html=True,
)
