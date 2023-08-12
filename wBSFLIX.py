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
.st-h2 {
    color: #E50914;
}
.stSidebar .sidebar-content {
    background-color: #111111;
}
</style>
    """, unsafe_allow_html=True)

# Display the logo with center alignment and reduced size
raw_github_link_for_logo = "https://raw.githubusercontent.com/LiesmannManda/WBSFLIX/30572b06dbb25dd7b94c9fa1ec3e270e3064c2d2/wbsflix%20logo.png"
st.image(raw_github_link_for_logo, width=300, use_column_width=True)

# Display the banner
raw_github_link_for_banner = "https://raw.githubusercontent.com/LiesmannManda/WBSFLIX/30572b06dbb25dd7b94c9fa1ec3e270e3064c2d2/wbs%20flix%20banner.png"
st.image(raw_github_link_for_banner, use_column_width=True)

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

# Content-based recommendation preparation
top_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movies_with_tags = movies_df.merge(top_tags, on='movieId', how='left')
movies_with_tags['tag'].fillna("", inplace=True)
movies_with_tags['content'] = movies_with_tags['genres'] + ' ' + movies_with_tags['tag']
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_with_tags['content'])

# Collaborative Filtering Recommendations
st.subheader("Collaborative Filtering Recommendations")
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

# User-Based Collaborative Filtering using Surprise library
reader = Reader(rating_scale=(ratings_df['rating'].min(), ratings_df['rating'].max()))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
sim_options = {
    'name': 'cosine',
    'user_based': True
}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

def get_top_n_recommendations(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

st.subheader("User-Based Collaborative Filtering Recommendations")
selected_user = st.selectbox("Select a user ID:", ratings_df['userId'].unique())
if st.button("Get Recommendations for User"):
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    top_n = get_top_n_recommendations(predictions, n=10)
    user_recs = top_n[selected_user]
    st.write(f"Top recommendations for user {selected_user}:")
    for movie_id, predicted_rating in user_recs:
        movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
        st.write(f"{movie_title} (Predicted Rating: {predicted_rating:.2f})")

# Footer
st.markdown(
    "<div style='background-color: #E50914; color
