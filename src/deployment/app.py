import os
import requests
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv("PREDICT_API_URL")

GENRE_COLUMNS = [
    "Action", "Adult", "Adventure", "Animation", "Biography", "Comedy",
    "Crime", "Documentary", "Drama", "Family", "Fantasy", "Film-Noir",
    "Game-Show", "History", "Horror", "Music", "Musical", "Mystery",
    "News", "Reality-TV", "Romance", "Sci-Fi", "Sport", "Talk-Show",
    "Thriller", "Unknown", "War", "Western"
]

st.set_page_config(page_title="Movie Ratings Prediction", layout="wide")

st.title("Predicting Movie Average Rating")

with st.form("Prediction"):
    runtime_minutes = st.number_input("Runtime Minutes", min_value=10, step=10)
    num_votes = st.number_input("Number of Votes", min_value=100, step=100)
    genres = st.multiselect("Movie Genres", GENRE_COLUMNS)
    submit = st.form_submit_button("Predict")

if submit:
    if not genres:
        st.warning("Please select a genre! If you are not sure then select `Unknown` genre")
    else:
        payload = {
            "runtimeMinutes": runtime_minutes,
            "numVotes": num_votes,
            "genres": genres
        }

        try:
            with st.spinner("Calling prediction API..."):
                r = requests.post(API_URL, json=payload, timeout=5)

            if r.status_code == 200:
                result = r.json()
                st.success("Prediction successful")
                st.metric("Predicted Rating", result["predicted_label"])
            else:
                st.error(r.text)

        except requests.exceptions.RequestException as e:
            st.error(f"API connection failed: {e}")
