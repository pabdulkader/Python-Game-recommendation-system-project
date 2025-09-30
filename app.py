import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Set Streamlit page config
st.set_page_config(page_title="Game Recommender", layout="centered")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("imdb-videogames.csv")
    return df

df = load_data()

# TF-IDF Matrix
df['name'] = df['name'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['name'])

# Function to compute similarities on-the-fly
def get_similar_games(input_index, tfidf_matrix, top_n=10):
    input_vector = tfidf_matrix[input_index]
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    similar_indices = similarities.argsort()[::-1][1:top_n+1]
    return similar_indices

# UI - Title
st.markdown("<h1 style='text-align: center; font-size: 36px;'>🎮 Game Recommendation System</h1>", unsafe_allow_html=True)

# UI - Input
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    user_input = st.text_input("", placeholder="Enter a Game Title You Like", label_visibility="collapsed")
    submit = st.button("Submit", use_container_width=True)

# Recommendation Logic
if submit and user_input:
    game_titles = df['name'].tolist()
    close_matches = difflib.get_close_matches(user_input, game_titles)

    if close_matches:
        best_match = close_matches[0]
        st.markdown(f"<h3 style='text-align: center;'>Recommendations for <strong>{best_match}</strong></h3>", unsafe_allow_html=True)

        idx = df[df.name == best_match].index[0]
        similar_indices = get_similar_games(idx, tfidf_matrix)

        st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
        for i in similar_indices:
            st.markdown(f"<p style='font-size: 18px; margin: 4px 0;'>{df.iloc[i]['name']}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"No close match found for '{user_input}'. Try another title.")
elif submit and not user_input:
    st.warning("Please enter a game title.")

# Style tweaks
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stTextInput input {
            font-size: 18px;
            padding: 10px;
            border-radius: 8px;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-size: 16px;
            padding: 10px 16px;
            border-radius: 8px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)
