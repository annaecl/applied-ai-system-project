from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its audio/metadata attributes.

    Attributes:
        id: Unique identifier for the song.
        title: Title of the song.
        artist: Name of the artist.
        genre: Musical genre (e.g. "pop", "rock").
        mood: Emotional mood label (e.g. "happy", "melancholic").
        energy: Intensity level from 0.0 (calm) to 1.0 (intense).
        tempo_bpm: Tempo in beats per minute.
        valence: Positivity from 0.0 (negative) to 1.0 (positive).
        danceability: How suitable for dancing, 0.0 (least) to 1.0 (most).
        acousticness: Acoustic quality from 0.0 (electronic) to 1.0 (acoustic).
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's music taste preferences.

    Attributes:
        favorite_genre: Preferred genre for genre-match scoring (e.g. "pop").
        favorite_mood: Preferred mood for mood-match scoring (e.g. "happy").
        target_energy: Desired energy level from 0.0 (calm) to 1.0 (intense).
        likes_acoustic: Whether the user prefers acoustic over electronic sounds.
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP interface for generating song recommendations.

    Wraps a catalog of Song objects and exposes methods for ranking
    and explaining recommendations based on a UserProfile.
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """
        Return the top-k songs best matching the given user profile.

        Args:
            user: The user's taste preferences.
            k: Number of songs to return (default 5).

        Returns:
            A list of up to k Song objects ranked by match quality.
        """
        user_dict = _profile_to_dict(user)
        scored = [
            (song, score_song(user_dict, _song_to_dict(song))[0])
            for song in self.songs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """
        Generate a human-readable explanation for why a song was recommended.

        Args:
            user: The user's taste preferences used during scoring.
            song: The song being explained.

        Returns:
            A string describing the factors that contributed to the recommendation.
        """
        _, reasons = score_song(_profile_to_dict(user), _song_to_dict(song))
        return ", ".join(reasons)

def _profile_to_dict(user: "UserProfile") -> Dict:
    return {
        "favorite_genre":      user.favorite_genre,
        "favorite_mood":       user.favorite_mood,
        "target_energy":       user.target_energy,
        "target_acousticness": 0.8 if user.likes_acoustic else 0.3,
        "target_valence":      0.5,
        "target_danceability": 0.5,
        "target_tempo_bpm":    100.0,
    }


def _song_to_dict(song: "Song") -> Dict:
    return {
        "genre":        song.genre,
        "mood":         song.mood,
        "energy":       song.energy,
        "acousticness": song.acousticness,
        "valence":      song.valence,
        "danceability": song.danceability,
        "tempo_bpm":    song.tempo_bpm,
    }


_MAX_SCORE = 1.0 + 0.15 + 0.15  # numeric ceiling + genre bonus + mood bonus


def confidence_score(raw_score: float) -> float:
    """
    Normalize a raw score to [0.0, 1.0] as a match-confidence percentage.

    0.0 = worst possible match, 1.0 = perfect match on every feature plus
    both categorical bonuses.
    """
    return min(1.0, max(0.0, raw_score / _MAX_SCORE))


WEIGHTS = {
    "energy":        0.25,
    "acousticness":  0.25,
    "valence":       0.20,
    "danceability":  0.20,
    "tempo":         0.10,
}

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Score a single song against user preferences.

    Computes a weighted proximity score across numeric audio features, then
    adds categorical bonuses for genre and mood matches.

    Args:
        user_prefs: Dict of user preference keys (e.g. "target_energy",
                    "favorite_genre") matching the structure used in main.py.
        song: Dict of song attributes matching the structure returned by load_songs.

    Returns:
        A tuple of (total_score, reasons) where total_score is a float and
        reasons is a list of strings explaining each scoring contribution.
    """
    reasons = []

    # Numeric proximity scores (1.0 = perfect match, 0.0 = furthest apart)
    proximities = {
        "energy":        1 - abs(song["energy"]        - user_prefs["target_energy"]),
        "acousticness":  1 - abs(song["acousticness"]  - user_prefs["target_acousticness"]),
        "valence":       1 - abs(song["valence"]        - user_prefs["target_valence"]),
        "danceability":  1 - abs(song["danceability"]   - user_prefs["target_danceability"]),
        "tempo":         1 - abs(song["tempo_bpm"] / 200 - user_prefs["target_tempo_bpm"] / 200),
    }

    total_weight = sum(WEIGHTS.values())
    numeric_score = sum(WEIGHTS[f] * proximities[f] for f in WEIGHTS) / total_weight

    reasons.append(f"audio features score: {numeric_score:.2f}")

    # Categorical bonuses
    bonus = 0.0
    if song["genre"] == user_prefs["favorite_genre"]:
        bonus += 0.15
        reasons.append("genre match (+0.15)")
    if song["mood"] == user_prefs["favorite_mood"]:
        bonus += 0.15
        reasons.append("mood match (+0.15)")

    return numeric_score + bonus, reasons


def load_songs(csv_path: str) -> List[Dict]:
    """
    Load songs from a CSV file into a list of dicts.

    Resolves relative paths from the project root (the parent of src/).
    Each row is cast to the appropriate Python type (int/float/str).

    Args:
        csv_path: Path to the CSV file, absolute or relative to the project root.

    Returns:
        A list of song dicts with keys: id, title, artist, genre, mood,
        energy, tempo_bpm, valence, danceability, acousticness.
    """
    import csv
    import os

    # Resolve relative paths from the project root (parent of src/)
    if not os.path.isabs(csv_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, csv_path)

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":           int(row["id"]),
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "energy":       float(row["energy"]),
                "tempo_bpm":    float(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Return the top-k songs ranked by how well they match user preferences.

    Scores every song using score_song, then returns the highest-scoring k
    songs in descending order.

    Args:
        user_prefs: Dict of user preference keys (see main.py for the full schema).
        songs: List of song dicts as returned by load_songs.
        k: Maximum number of recommendations to return (default 5).

    Returns:
        A list of (song, score, explanation) tuples sorted by score descending,
        where explanation is a comma-separated string of scoring reasons.
    """
    scored = [
        (song, score, ", ".join(reasons))
        for song in songs
        for score, reasons in [score_song(user_prefs, song)]
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
