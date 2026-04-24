"""
Command-line runner for the Music Recommender Simulation.

Loads the song catalog, defines user taste profiles (including adversarial
edge cases), runs the recommender, and prints ranked results with scores
and explanations.

Usage
-----
Run all profiles (original behavior):
    python -m src.main

Run a single profile by name:
    python -m src.main --profile "Chill Lofi"

List available profile names:
    python -m src.main --list

AI mode — single natural language query:
    python -m src.main --query "something chill to study to"

AI mode — interactive loop:
    python -m src.main --ai
"""

import argparse
import logging

from .ai_interface import extract_preferences, generate_recommendation, load_gemini_client
from .recommender import load_songs, recommend_songs

# ---------------------------------------------------------------------------
# Logging — INFO to console, DEBUG to file
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("recommender.log"),
    ],
)
logging.getLogger().handlers[1].setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Profile registry — add new profiles here; they are picked up automatically
# ---------------------------------------------------------------------------

PROFILES = {
    # --- Normal personas ---

    "High-Energy Pop": {
        # Upbeat, danceable electronic/pop — workout playlist or city drive.
        "favorite_genre":      "pop",
        "favorite_mood":       "happy",
        "target_energy":        0.72,
        "target_tempo_bpm":   118.0,
        "target_valence":       0.88,
        "target_danceability":  0.78,
        "target_acousticness":  0.30,
        "likes_acoustic":      False,
    },

    "Chill Lofi": {
        # Low-key lofi beats for studying or winding down.
        "favorite_genre":      "lofi",
        "favorite_mood":       "chill",
        "target_energy":        0.28,
        "target_tempo_bpm":    78.0,
        "target_valence":       0.48,
        "target_danceability":  0.42,
        "target_acousticness":  0.72,
        "likes_acoustic":      True,
    },

    "Deep Intense Rock": {
        # Heavy, raw rock — headbanging or volume-maxed night drives.
        "favorite_genre":      "rock",
        "favorite_mood":       "intense",
        "target_energy":        0.90,
        "target_tempo_bpm":   145.0,
        "target_valence":       0.28,
        "target_danceability":  0.48,
        "target_acousticness":  0.10,
        "likes_acoustic":      False,
    },

    # --- Adversarial / edge-case profiles ---

    "High Energy + Sad [ADVERSARIAL]": {
        # Conflict: wants brutal energy (0.9) but sad mood.
        # Sad songs (folk/blues) are typically low-energy — does the mood
        # bonus of +0.15 beat the energy-proximity penalty?
        "favorite_genre":      "folk",
        "favorite_mood":       "sad",
        "target_energy":        0.90,
        "target_tempo_bpm":   140.0,
        "target_valence":       0.10,
        "target_danceability":  0.30,
        "target_acousticness":  0.20,
        "likes_acoustic":      False,
    },

    "Impossible Tempo [ADVERSARIAL]": {
        # target_tempo_bpm: 300 is above the scorer's 200-BPM normalization
        # ceiling.  tempo proximity = 1 - |song/200 - 300/200| can go
        # negative, producing sub-zero component scores.  Does the final
        # score still make sense?
        "favorite_genre":      "electronic",
        "favorite_mood":       "energetic",
        "target_energy":        0.95,
        "target_tempo_bpm":   300.0,
        "target_valence":       0.70,
        "target_danceability":  0.80,
        "target_acousticness":  0.05,
        "likes_acoustic":      False,
    },

    "Ghost Genre & Mood [ADVERSARIAL]": {
        # favorite_genre/mood don't exist in the catalog, so neither
        # categorical bonus (+0.15 each) ever fires.  Every song is ranked
        # purely on numeric proximity — exposes how much genre/mood bonuses
        # normally shift the ranking.
        "favorite_genre":      "reggae",
        "favorite_mood":       "euphoric",
        "target_energy":        0.60,
        "target_tempo_bpm":   100.0,
        "target_valence":       0.70,
        "target_danceability":  0.65,
        "target_acousticness":  0.40,
        "likes_acoustic":      False,
    },

    "Dead Center [ADVERSARIAL]": {
        # Everything at the midpoint (0.5).  All songs should receive nearly
        # the same numeric score — a flat ranking reveals whether any feature
        # weighting is accidentally dominant or whether ties are broken
        # arbitrarily.
        "favorite_genre":      "pop",
        "favorite_mood":       "happy",
        "target_energy":        0.50,
        "target_tempo_bpm":   100.0,
        "target_valence":       0.50,
        "target_danceability":  0.50,
        "target_acousticness":  0.50,
        "likes_acoustic":      True,
    },

    "Contradictory Acoustic [ADVERSARIAL]": {
        # likes_acoustic=True signals an acoustic preference, but
        # target_acousticness=0.0 asks for fully electronic sound.
        # score_song uses target_acousticness for scoring and ignores
        # likes_acoustic — so the boolean has zero effect.  This profile
        # makes that silent bug visible.
        "favorite_genre":      "jazz",
        "favorite_mood":       "relaxed",
        "target_energy":        0.35,
        "target_tempo_bpm":    90.0,
        "target_valence":       0.55,
        "target_danceability":  0.40,
        "target_acousticness":  0.00,   # says "electronic" …
        "likes_acoustic":      True,    # … but boolean says "acoustic"
    },
}


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def print_results(profile_name: str, recommendations: list) -> None:
    print("\n" + "=" * 60)
    print(f"  {profile_name.upper()}")
    print(f"  Top {len(recommendations)} recommendations")
    print("=" * 60)
    for i, rec in enumerate(recommendations, start=1):
        song, score, explanation = rec
        print(f"\n#{i}  {song['title']}  —  {song['artist']}")
        print(f"    Score : {score:.2f}")
        print(f"    Why   :", end="")
        reasons = [r.strip() for r in explanation.split(",")]
        for j, reason in enumerate(reasons):
            prefix = " " if j == 0 else "            "
            print(f"{prefix}{reason}")
    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# AI mode helper
# ---------------------------------------------------------------------------

def run_ai_mode(query: str, songs: list, client, k: int = 5) -> None:
    """
    Run the full RAG pipeline for a single natural-language query.

    Steps:
      1. Gemini extracts structured preferences from the query
      2. recommend_songs() retrieves top-k songs (the Retrieval step in RAG)
      3. Gemini generates a natural-language response using the retrieved songs
    """
    print(f'\nProcessing: "{query}"')
    print("-" * 60)

    try:
        prefs = extract_preferences(query, client)
        print(
            f"Extracted: genre={prefs['favorite_genre']}, "
            f"mood={prefs['favorite_mood']}, "
            f"energy={prefs['target_energy']:.2f}"
        )
    except ValueError as e:
        print(f"Error extracting preferences: {e}")
        logger.error("Preference extraction failed: %s", e)
        return

    retrieved = recommend_songs(prefs, songs, k=k)
    print(f"Retrieved {len(retrieved)} songs via scoring algorithm.")

    try:
        response = generate_recommendation(query, retrieved, client)
        print("\nRecommendation:")
        print("=" * 60)
        print(response)
        print("=" * 60)
    except Exception as e:
        logger.error("Recommendation generation failed: %s", e)
        print("(AI generation failed — showing scored results instead)")
        print_results("Fallback", retrieved)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the music recommender against one or all user profiles."
    )
    parser.add_argument(
        "--profile",
        metavar="NAME",
        help="Run only the named profile (use quotes for names with spaces).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available profile names and exit.",
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        help="Interactive AI mode: enter natural language queries.",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help='Single natural language query for AI mode (e.g. "chill music to study to").',
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="K",
        help="Number of songs to retrieve in AI mode (default: 5).",
    )
    args = parser.parse_args()

    if args.list:
        print("Available profiles:")
        for name in PROFILES:
            print(f"  • {name}")
        return

    # --- AI mode ---
    if args.ai or args.query:
        client = load_gemini_client()
        if client is None:
            print("ERROR: GOOGLE_API_KEY not set.")
            print("Add it to a .env file in the project root:")
            print("  GOOGLE_API_KEY=your-key-here")
            return

        songs = load_songs("data/songs.csv")
        logger.info("Loaded %d songs from catalog.", len(songs))
        print(f"Loaded {len(songs)} songs.")

        if args.query:
            run_ai_mode(args.query, songs, client, k=args.top_k)
        else:
            print("AI Music Recommender — type your request, or 'quit' to exit.")
            while True:
                try:
                    query = input("\nWhat are you in the mood for? ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break
                if query.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                if not query:
                    print("Please enter a request.")
                    continue
                run_ai_mode(query, songs, client, k=args.top_k)
        return

    # --- Profile mode (original behavior) ---
    songs = load_songs("data/songs.csv")
    logger.info("Loaded %d songs from catalog.", len(songs))
    print(f"Loaded {len(songs)} songs.")

    if args.profile:
        if args.profile not in PROFILES:
            print(f"Unknown profile: '{args.profile}'")
            print("Run with --list to see available profiles.")
            return
        to_run = [(args.profile, PROFILES[args.profile])]
    else:
        to_run = list(PROFILES.items())

    for profile_name, user_prefs in to_run:
        recommendations = recommend_songs(user_prefs, songs, k=5)
        print_results(profile_name, recommendations)


if __name__ == "__main__":
    main()
