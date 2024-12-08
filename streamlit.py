import streamlit as st
import requests

#API endpoint
API_URL = "http://127.0.0.1:8000/add_review_and_analyze"

#Spotify colors
SPOTIFY_GREEN = "#1DB954"
BACKGROUND_COLOR = "#191414"
TEXT_COLOR = "#FFFFFF"

#Streamlit page settings
st.set_page_config(
    page_title="Spotify Review Analysis",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

#Header and background
st.markdown(
    f"""
    <style>
        body {{
            background-color: {BACKGROUND_COLOR};
            color: {TEXT_COLOR};
        }}
        .stButton>button {{
            background-color: {SPOTIFY_GREEN};
            color: {TEXT_COLOR};
            font-size: 16px;
            border-radius: 10px;
            padding: 10px 20px;
        }}
        .stButton>button:hover {{
            background-color: #1ed760;
            color: {TEXT_COLOR};
        }}
        textarea {{
            font-size: 16px;
            border: 2px solid {SPOTIFY_GREEN};
            border-radius: 10px;
            padding: 10px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

#Header
st.markdown(
    f"<h1 style='text-align: center; color: {SPOTIFY_GREEN};'>Spotify Review Analysis</h1>",
    unsafe_allow_html=True,
)

#Input from user
user_input = st.text_area(
    "🎧🎵 🎧🎵 🎧🎵 🎧🎵 ",
    height=150,
    placeholder="Write your review about Spotify here ",
)



#Analysis button and result
if st.button("🎤 Analyze"):

    if user_input:
        #Send review data to API
        response = requests.post(API_URL, json={"review": user_input})

        #Control API's response
        if response.status_code == 200:

            data = response.json()
            conclusion = data.get("Conclusion", "No conclusion generated")
            label = data.get("label", "No label")
            st.subheader("Analysis Result")
            st.write(f"Sentiment Label: {label}")
            st.write(f"Summary of Reviews: {conclusion['content']}")

        else:
            st.error("Error with the API request.")
            
    else:
        st.warning("Please enter a review before analyzing.")