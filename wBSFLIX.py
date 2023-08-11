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

# Display the logo
st.image("/mnt/data/wbsflix logo.png", use_column_width=True)

# Custom CSS for styling resembling Netflix
st.markdown("""
<style>
    body {
        color: #ffffff;
        background-color: #141414;
    }
    div[role="listbox"] {
        background-color: #5a5a5a;
        color: #ffffff;
    }
    .st-d9 {
        background-color: #5a5a5a;
    }
    div[data-baseweb="select"] > div {
        background-color: #5a5a5a;
    }
    .st-gb {
        color: #ffffff;
    }
</style>
    """, unsafe_allow_html=True)

# Content-based recommendation preparation
top_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movies_with_tags = movies_df.merge(top_tags, on='movieId', how='left')
movies_with_tags['tag'].fillna("", inplace=True)
movies_with_tags['content'] = movies_with_tags['genres'] + ' ' + movies_with_tags['tag']
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_with_tags['content'])

# Searching for movies
movie_search_query = st.text_input("Search for a movie by title:", "")
if movie_search_query:
    matching_movies = movies_df[movies_df['title'].str.contains(movie_search_query, case=False)]
    if not matching_movies.empty:
        selected_movie = st.selectbox("Select a movie:", matching_movies['title'].tolist())
        st.write(movies_df[movies_df['title'] == selected_movie])
    else:
        st.write("No movies found!")

# Searching for users
user_search_query = st.text_input("Search for a user by ID:", "")
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
trainset = data.build_full_trainset()  # Use full data for training

# Set up algorithm for user-based collaborative filtering
sim_options = {
    'name': 'cosine',
    'user_based': True
}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

# Function to get top-N recommendations
def get_top_n_recommendations(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

# Streamlit UI for user-based recommendations
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

# User-Based Collaborative Filtering from scratch functions
user_item_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def predict_rating(user_id, movie_id, user_similarity_df, user_item_matrix, k=20):
    """Predict the rating of a user for a movie."""
    # Get the top K similar users to the target user
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:k+1]
    
    # Compute the predicted rating
    weighted_sum, sum_of_weights = 0, 0
    for user in similar_users:
        weighted_sum += user_similarity_df.loc[user_id, user] * user_item_matrix.loc[user, movie_id]
        sum_of_weights += abs(user_similarity_df.loc[user_id, user])
    
    if sum_of_weights == 0:
        return 0
    else:
        return weighted_sum / sum_of_weights

def get_top_n_recommendations_for_user(user_id, user_similarity_df, user_item_matrix, n=10):
    """Get the top N movie recommendations for a user."""
    # Get the list of movies the user hasn't rated yet
    unrated_movies = user_item_matrix.columns[user_item_matrix.loc[user_id].isnull()].tolist()
    
    # Predict ratings for all unrated movies
    predicted_ratings = {}
    for movie_id in unrated_movies:
        predicted_ratings[movie_id] = predict_rating(user_id, movie_id, user_similarity_df, user_item_matrix)
    
    # Get the top N movie recommendations
    recommended_movies = sorted(predicted_ratings, key=predicted_ratings.get, reverse=True)[:n]
    
    return recommended_movies

# Streamlit UI for User-Based Collaborative Filtering (from scratch)
st.subheader("User-Based Collaborative Filtering Recommendations (From Scratch)")
selected_user_scratch = st.selectbox("Select a user ID:", ratings_df['userId'].unique(), key='user_select_scratch')
if st.button("Get Recommendations for User (From Scratch)"):
    recommended_movie_ids = get_top_n_recommendations_for_user(selected_user_scratch, user_similarity_df, user_item_matrix)
    recommended_movie_titles = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]['title'].tolist()
    st.write(f"Top recommendations for user {selected_user_scratch} (from scratch):")
    for movie_title in recommended_movie_titles:
        st.write(movie_title)
