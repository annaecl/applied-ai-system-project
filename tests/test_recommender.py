from src.recommender import Song, UserProfile, Recommender, score_song, confidence_score

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# ---------------------------------------------------------------------------
# Confidence scoring tests
# ---------------------------------------------------------------------------

_PERFECT_USER = {
    "favorite_genre":      "pop",
    "favorite_mood":       "happy",
    "target_energy":        0.8,
    "target_acousticness":  0.2,
    "target_valence":       0.9,
    "target_danceability":  0.8,
    "target_tempo_bpm":   120.0,
}

_PERFECT_SONG = {
    "genre": "pop", "mood": "happy",
    "energy": 0.8, "acousticness": 0.2,
    "valence": 0.9, "danceability": 0.8, "tempo_bpm": 120.0,
}

_POOR_SONG = {
    "genre": "metal", "mood": "angry",
    "energy": 0.0, "acousticness": 1.0,
    "valence": 0.0, "danceability": 0.0, "tempo_bpm": 60.0,
}


def test_confidence_score_is_in_range():
    for raw in (0.0, 0.5, 1.0, 1.3, 1.5):
        c = confidence_score(raw)
        assert 0.0 <= c <= 1.0, f"confidence_score({raw}) = {c} out of [0, 1]"


def test_confidence_score_perfect_match_is_one():
    raw, _ = score_song(_PERFECT_USER, _PERFECT_SONG)
    assert confidence_score(raw) == 1.0


def test_confidence_score_good_match_above_threshold():
    raw, _ = score_song(_PERFECT_USER, _PERFECT_SONG)
    assert confidence_score(raw) >= 0.7, (
        f"Expected confidence ≥ 0.7 for a perfect match, got {confidence_score(raw):.2f}"
    )


def test_confidence_score_poor_match_below_threshold():
    raw, _ = score_song(_PERFECT_USER, _POOR_SONG)
    assert confidence_score(raw) < 0.7, (
        f"Expected confidence < 0.7 for a poor match, got {confidence_score(raw):.2f}"
    )


def test_explain_includes_genre_match_reason():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(user, rec.songs[0])
    assert "genre match" in explanation
