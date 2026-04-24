"""
ai_interface.py — Gemini API integration for the music recommender.

Implements a two-step RAG pipeline:
  1. extract_preferences(user_text, client)        → dict of structured user prefs
  2. generate_recommendation(user_text, songs, client) → natural-language response

The scoring algorithm in recommender.py is the Retrieval step of RAG.
The retrieved songs are passed as context to Gemini for generation.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.0-flash"

VALID_GENRES = {
    "pop", "lofi", "rock", "jazz", "ambient", "synthwave",
    "indie pop", "hip-hop", "country", "r&b", "metal",
    "folk", "electronic", "blues", "classical",
}
VALID_MOODS = {
    "happy", "chill", "intense", "focused", "moody", "energetic",
    "melancholic", "relaxed", "nostalgic", "romantic", "angry", "sad",
}

_EXTRACT_PROMPT_TEMPLATE = """\
You are a music preference extractor. Given a user's request, return a JSON object \
with exactly these keys and types:

  "favorite_genre"      — string, MUST be exactly one of: pop, lofi, rock, jazz, \
ambient, synthwave, indie pop, hip-hop, country, r&b, metal, folk, electronic, \
blues, classical
  "favorite_mood"       — string, MUST be exactly one of: happy, chill, intense, \
focused, moody, energetic, melancholic, relaxed, nostalgic, romantic, angry, sad
  "target_energy"       — float 0.0–1.0 (0 = very calm, 1 = very intense)
  "target_tempo_bpm"    — float 60–180 (beats per minute)
  "target_valence"      — float 0.0–1.0 (0 = sad/dark, 1 = happy/bright)
  "target_danceability" — float 0.0–1.0
  "target_acousticness" — float 0.0–1.0 (0 = electronic, 1 = fully acoustic)
  "likes_acoustic"      — boolean

Rules:
- Return ONLY the JSON object. No markdown fences. No explanation text.
- All float values must be within their stated ranges.
- Use moderate defaults (0.5 for floats, 100 for tempo) when the request is vague.
- genre and mood must match one of the listed values exactly.

User request: "{user_text}"
"""

_GENERATE_PROMPT_TEMPLATE = """\
You are a friendly music recommendation assistant.

User request: "{user_text}"

Songs retrieved by the scoring system (ranked best to worst match):
{context_block}

Write a warm, conversational response (3–5 sentences) that:
- Acknowledges what the user is looking for
- Highlights 2–3 of the top songs by name and artist
- Briefly explains why each fits, in plain English
Do not mention "scores", "algorithms", or technical terms. Do not list all songs — curate.
"""


def load_gemini_client() -> Optional[genai.Client]:
    """
    Initialize the Gemini client from GOOGLE_API_KEY in the environment.

    Returns None (with a logged warning) if the key is missing, allowing the
    app to fall back to profile-only mode gracefully.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not set — AI features disabled.")
        return None
    client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialized. Model: %s", MODEL_NAME)
    return client


def extract_preferences(user_text: str, client: genai.Client) -> Dict:
    """
    Use Gemini to translate a free-text music request into structured preferences.

    Args:
        user_text: Raw user input, e.g. "something chill to study to"
        client:    Initialized Gemini Client

    Returns:
        A dict matching the PROFILES schema in main.py (8 keys).

    Raises:
        ValueError: if input is empty, or Gemini's response cannot be parsed
                    or is missing required keys.
    """
    user_text = user_text.strip()
    if not user_text:
        raise ValueError("User input cannot be empty.")
    if len(user_text) > 500:
        logger.warning("User input truncated from %d to 500 characters.", len(user_text))
        user_text = user_text[:500]

    prompt = _EXTRACT_PROMPT_TEMPLATE.format(user_text=user_text)
    logger.info("Calling Gemini to extract preferences.")

    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    raw = response.text.strip()

    # Strip markdown fences if Gemini adds them despite instructions
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    logger.debug("Gemini extraction raw response: %s", raw)

    try:
        prefs = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Gemini returned non-JSON: %s", raw)
        raise ValueError(f"Gemini returned non-JSON response: {e}") from e

    prefs = _validate_and_clamp_preferences(prefs)
    logger.info(
        "Extracted preferences: genre=%s, mood=%s, energy=%.2f",
        prefs["favorite_genre"], prefs["favorite_mood"], prefs["target_energy"],
    )
    return prefs


def _validate_and_clamp_preferences(prefs: Dict) -> Dict:
    """
    Validate and clamp Gemini's extracted preference dict.

    Checks all required keys exist, clamps floats to valid ranges, and
    falls back unknown genres/moods to safe defaults. Logs a warning for
    every correction made so the log file shows exactly what was sanitized.
    """
    required = [
        "favorite_genre", "favorite_mood", "target_energy", "target_tempo_bpm",
        "target_valence", "target_danceability", "target_acousticness", "likes_acoustic",
    ]
    for key in required:
        if key not in prefs:
            raise ValueError(f"Gemini response missing required key: '{key}'")

    for key in ("target_energy", "target_valence", "target_danceability", "target_acousticness"):
        original = prefs[key]
        prefs[key] = max(0.0, min(1.0, float(prefs[key])))
        if prefs[key] != original:
            logger.warning("Clamped %s from %s to %.2f", key, original, prefs[key])

    original_tempo = prefs["target_tempo_bpm"]
    prefs["target_tempo_bpm"] = max(60.0, min(180.0, float(prefs["target_tempo_bpm"])))
    if prefs["target_tempo_bpm"] != original_tempo:
        logger.warning(
            "Clamped target_tempo_bpm from %s to %.1f", original_tempo, prefs["target_tempo_bpm"]
        )

    if prefs["favorite_genre"] not in VALID_GENRES:
        logger.warning(
            "Unknown genre '%s' — falling back to 'pop'", prefs["favorite_genre"]
        )
        prefs["favorite_genre"] = "pop"

    if prefs["favorite_mood"] not in VALID_MOODS:
        logger.warning(
            "Unknown mood '%s' — falling back to 'chill'", prefs["favorite_mood"]
        )
        prefs["favorite_mood"] = "chill"

    prefs["likes_acoustic"] = bool(prefs["likes_acoustic"])
    return prefs


def generate_recommendation(
    user_text: str,
    retrieved_songs: List[Tuple[Dict, float, str]],
    client: genai.Client,
) -> str:
    """
    Use Gemini to write a natural-language recommendation from retrieved songs.

    This is the Augment + Generate step of the RAG pipeline. The retrieved
    songs (from recommend_songs()) are injected as context so Gemini can
    write a response grounded in actual catalog data.

    Args:
        user_text:       The original user request (for tone and framing).
        retrieved_songs: List of (song_dict, score, explanation) tuples from
                         recommend_songs() — already ranked best to worst.
        client:          Initialized Gemini Client.

    Returns:
        A natural-language string to display to the user.
    """
    context_lines = []
    for rank, (song, score, explanation) in enumerate(retrieved_songs, start=1):
        context_lines.append(
            f'#{rank}: "{song["title"]}" by {song["artist"]}'
            f' (genre: {song["genre"]}, mood: {song["mood"]},'
            f' energy: {song["energy"]:.2f}, score: {score:.2f})'
            f'\n    Scoring reasons: {explanation}'
        )
    context_block = "\n".join(context_lines)

    prompt = _GENERATE_PROMPT_TEMPLATE.format(
        user_text=user_text,
        context_block=context_block,
    )

    logger.info(
        "Calling Gemini to generate recommendation. %d songs in context.", len(retrieved_songs)
    )
    logger.debug("Generation context:\n%s", context_block)

    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    result = response.text.strip()
    logger.info("Gemini generated recommendation (%d characters).", len(result))
    return result
