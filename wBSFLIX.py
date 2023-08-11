import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNBasic
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
tags_df = pd.read_csv('tags.csv')

# Content-based recommendation preparation
top_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movies_with_tags = movies_df.merge(top_tags, on='movieId', how='left')
movies_with_tags['tag'].fillna("", inplace=True)
movies_with_tags['content'] = movies_with_tags['genres'] + ' ' + movies_with_tags['tag']
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_with_tags['content'])

# Custom CSS for styling to mimic Netflix
st.markdown("""
<style>
body {
    background-color: #000000;
    color: #ffffff;
}
.stButton>button {
    color: #ffffff;
    background-color: #e50914;
    border: none;
}
</style>
    """, unsafe_allow_html=True)

st.title("WBSFLIX Movie Recommender")

# Sidebar with overall controls
st.sidebar.header("Controls")
st.sidebar.write("Use these controls to adjust your recommendations.")

# Page layout using columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Search Movies")
    movie_search_query = st.text_input("Search by title:", "")
    if movie_search_query:
        matching_movies = movies_df[movies_df['title'].str.contains(movie_search_query, case=False)]
        if not matching_movies.empty:
            selected_movie = st.selectbox("Select a movie:", matching_movies['title'].tolist())
            st.write(movies_df[movies_df['title'] == selected_movie])
        else:
            st.write("No movies found!")

with col2:
    st.subheader("🔍 Search Users")
    user_search_query = st.text_input("Search by user ID:", "")
    if user_search_query:
        try:
            user_id = int(user_search_query)
            if user_id in ratings_df['userId'].unique():
                st.write(f"User {user_id} found!")
            else:
                st.write("User not found!")
        except ValueError:
            st.write("Please enter a valid user ID!")

# Collaborative filtering preparation
user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Display top popular movies based on number of ratings
st.subheader("Top Popular Movies")
movie_ratings_count = ratings_df.groupby('movieId').size().reset_index(name='num_ratings')
movie_ratings_count = movie_ratings_count.merge(movies_df[['movieId', 'title']], on='movieId')
top_movies = movie_ratings_count.sort_values(by="num_ratings", ascending=False).head(10)
for index, row in top_movies.iterrows():
    st.write(row['title'])

# Collaborative Filtering Recommendations
st.subheader("Collaborative Filtering Recommendations")
selected_movie_title = st.selectbox("Select a movie you like:", movies_df['title'].tolist())
if selected_movie_title:
    selected_movie_id = movies_df[movies_df['title'] == selected_movie_title]['movieId'].iloc[0]
    similar_movie_ids = item_similarity_df[selected_movie_id].sort_values(ascending=False).index[1:11]
    recommended_movie_titles = movies_df[movies_df['movieId'].isin(similar_movie_ids)]['title'].tolist()
    st.write("Movies you might also like:")
    for movie in recommended_movie_titles:
        st.write(movie)

# Prepare data for Surprise
reader = Reader(rating_scale=(ratings_df['rating'].min(), ratings_df['rating'].max()))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# User-based collaborative filtering with Surprise
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
selected_user = st.selectbox("Select a user
