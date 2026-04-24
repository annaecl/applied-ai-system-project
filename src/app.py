"""
app.py — Streamlit web UI for VibeMatcher.

Run from the project root:
    streamlit run src/app.py

The app implements a RAG pipeline:
  1. User types a natural-language request
  2. Gemini extracts structured preferences (LLM call 1)
  3. recommend_songs() retrieves top-k matches from the catalog (the Retrieval step)
  4. Gemini generates a personalized response using the retrieved songs (LLM call 2)
"""

import logging
import os
import sys

# Make project root importable when running as a Streamlit script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from src.ai_interface import (
    VALID_GENRES,
    VALID_MOODS,
    extract_preferences,
    generate_recommendation,
    load_gemini_client,
)
from src.recommender import load_songs, recommend_songs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("recommender.log"),
    ],
)
logging.getLogger().handlers[1].setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VibeMatcher",
    page_icon="🎵",
    layout="centered",
)

st.title("🎵 VibeMatcher")
st.caption("AI-powered music recommendations — describe what you're in the mood for, or build your profile directly")

# ---------------------------------------------------------------------------
# Cached loaders (run once per session)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_client():
    return load_gemini_client()


@st.cache_data
def get_songs():
    return load_songs("data/songs.csv")


client = get_client()
songs = get_songs()

if client is None:
    st.error(
        "**GOOGLE_API_KEY not set.** "
        "Create a `.env` file in the project root with:\n\n"
        "```\nGOOGLE_API_KEY=your-key-here\n```\n\n"
        "Then restart the Streamlit server."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — catalog info
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("About")
    st.write(
        "VibeMatcher uses Google Gemini to understand your request, "
        "then searches a catalog of songs using a scoring algorithm, "
        "and finally asks Gemini to explain the picks in plain language."
    )
    st.divider()
    st.metric("Songs in catalog", len(songs))
    st.caption(f"Genres: {', '.join(sorted(VALID_GENRES))}")
    st.caption(f"Moods: {', '.join(sorted(VALID_MOODS))}")

# ---------------------------------------------------------------------------
# Main input — two tabs
# ---------------------------------------------------------------------------

tab_describe, tab_profile = st.tabs(["Describe it", "Build your profile"])

with tab_describe:
    query = st.text_input(
        "What are you in the mood for?",
        placeholder="e.g. something chill and acoustic for studying",
        max_chars=500,
        key="nl_query",
    )
    top_k_nl = st.slider(
        "Songs to retrieve before generating response",
        min_value=3, max_value=10, value=5,
        key="top_k_nl",
    )
    run_nl = st.button(
        "Get Recommendations",
        disabled=not query.strip(),
        type="primary",
        key="run_nl",
    )

with tab_profile:
    st.write("Set your music preferences directly — no need to describe them in words.")

    col1, col2 = st.columns(2)
    with col1:
        prof_genre = st.selectbox("Favorite genre", sorted(VALID_GENRES), key="prof_genre")
        prof_mood = st.selectbox("Favorite mood", sorted(VALID_MOODS), key="prof_mood")
        prof_acoustic = st.checkbox("I prefer acoustic songs", key="prof_acoustic")
    with col2:
        prof_energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.05,
                                help="0 = very calm, 1 = very intense", key="prof_energy")
        prof_valence = st.slider("Positivity (valence)", 0.0, 1.0, 0.5, 0.05,
                                 help="0 = dark/sad, 1 = bright/happy", key="prof_valence")
        prof_dance = st.slider("Danceability", 0.0, 1.0, 0.5, 0.05, key="prof_dance")
        prof_acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, 0.05,
                                      help="0 = fully electronic, 1 = fully acoustic",
                                      key="prof_acousticness")
        prof_tempo = st.slider("Tempo (BPM)", 60, 180, 100, key="prof_tempo")

    prof_context = st.text_input(
        "Optional: add any extra context for the AI response",
        placeholder="e.g. for a late-night drive",
        max_chars=200,
        key="prof_context",
    )
    top_k_prof = st.slider(
        "Songs to retrieve before generating response",
        min_value=3, max_value=10, value=5,
        key="top_k_prof",
    )
    run_prof = st.button("Get Recommendations", type="primary", key="run_prof")

# ---------------------------------------------------------------------------
# RAG pipeline — shared output section
# ---------------------------------------------------------------------------

prefs = None
query_for_gen = None
top_k = None

if run_nl and query.strip():
    top_k = top_k_nl
    query_for_gen = query
    with st.spinner("Analyzing your request with Gemini..."):
        try:
            prefs = extract_preferences(query, client)
        except ValueError as e:
            st.error(f"Could not understand your request: {e}")
            st.stop()

elif run_prof:
    top_k = top_k_prof
    query_for_gen = prof_context.strip() or (
        f"{prof_mood} {prof_genre} music, energy {prof_energy:.2f}"
    )
    prefs = {
        "favorite_genre": prof_genre,
        "favorite_mood": prof_mood,
        "target_energy": prof_energy,
        "target_tempo_bpm": float(prof_tempo),
        "target_valence": prof_valence,
        "target_danceability": prof_dance,
        "target_acousticness": prof_acousticness,
        "likes_acoustic": prof_acoustic,
    }

if prefs is not None:
    with st.spinner("Retrieving best matches from catalog..."):
        retrieved = recommend_songs(prefs, songs, k=top_k)

    with st.spinner("Generating your personalized recommendation..."):
        try:
            response_text = generate_recommendation(query_for_gen, retrieved, client)
        except Exception as e:
            st.error(f"Gemini could not generate a response: {e}")
            response_text = None

    st.divider()

    # --- Recommendation ---
    st.subheader("Your Recommendation")
    if response_text:
        st.write(response_text)
    else:
        st.warning("AI response unavailable — showing scored results below.")

    # --- Extracted preferences (expandable) ---
    with st.expander("Profile used for this recommendation"):
        display_prefs = {
            "Genre": prefs["favorite_genre"],
            "Mood": prefs["favorite_mood"],
            "Energy": f"{prefs['target_energy']:.2f}  (0 = calm, 1 = intense)",
            "Tempo": f"{prefs['target_tempo_bpm']:.0f} BPM",
            "Positivity": f"{prefs['target_valence']:.2f}  (0 = dark, 1 = bright)",
            "Danceability": f"{prefs['target_danceability']:.2f}",
            "Acousticness": f"{prefs['target_acousticness']:.2f}  (0 = electronic, 1 = acoustic)",
        }
        for label, value in display_prefs.items():
            st.text(f"  {label:<14} {value}")

    # --- Retrieved songs (expandable) ---
    with st.expander(f"Top {len(retrieved)} songs from the catalog (ranked by score)"):
        for rank, (song, score, explanation) in enumerate(retrieved, start=1):
            st.markdown(f"**#{rank}  {song['title']}** — {song['artist']}")
            cols = st.columns(3)
            cols[0].caption(f"Genre: {song['genre']}")
            cols[1].caption(f"Mood: {song['mood']}")
            cols[2].caption(f"Score: {score:.2f}")
            st.caption(f"Why: {explanation}")
            if rank < len(retrieved):
                st.divider()
