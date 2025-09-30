
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# ✅ This must be the first Streamlit command
st.set_page_config(page_title="Game Recommender", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('imdb-videogames.csv')
    return df


df = load_data()

# Make sure to use the correct column name here
df['name'] = df['name'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['name'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# UI
st.title("🎮 Game Recommendation System")

user_input = st.text_input("Enter a Game Title You Like")

if user_input:
    game_titles = df['name'].tolist()
    close_matches = difflib.get_close_matches(user_input, game_titles)

    if close_matches:
        best_match = close_matches[0]
        st.write(f"Showing recommendations for **{best_match}**:")

        idx = df[df.name == best_match].index[0]
        scores = list(enumerate(cosine_sim[idx]))
        sorted_games = sorted(scores, key=lambda x: x[1], reverse=True)

        recommended_titles = [df.iloc[i[0]].name for i in sorted_games[1:11]]
        st.write("### 🕹️ Recommended Games:")
        for game in recommended_titles:
            st.write(f"- {game}")
    else:
        st.warning("No close match found. Try another title.")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
