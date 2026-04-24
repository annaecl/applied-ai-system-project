"""
test_ai_consistency.py — Reliability testing for Gemini preference extraction.

This module is the "Reliability / Testing System" advanced feature.
It verifies that extract_preferences() produces schema-valid, in-range,
and consistent results. Most tests use mocked Gemini responses so they run
without an API key. Pass --live to run a real consistency check against
the live Gemini API.

Run mocked tests (no API key required):
    pytest tests/test_ai_consistency.py -v

Run live consistency test (requires GOOGLE_API_KEY in .env):
    pytest tests/test_ai_consistency.py -v --live
"""

import json
import os
from unittest.mock import MagicMock

import pytest

from src.ai_interface import (
    VALID_GENRES,
    VALID_MOODS,
    _validate_and_clamp_preferences,
    extract_preferences,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SAMPLE_PREFS = {
    "favorite_genre":      "lofi",
    "favorite_mood":       "chill",
    "target_energy":        0.30,
    "target_tempo_bpm":    78.0,
    "target_valence":       0.55,
    "target_danceability":  0.48,
    "target_acousticness":  0.80,
    "likes_acoustic":      True,
}

REQUIRED_KEYS = list(SAMPLE_PREFS.keys())


def make_mock_client(response_dict: dict):
    """Return a mock Gemini client whose models.generate_content() returns response_dict as JSON."""
    mock_response = MagicMock()
    mock_response.text = json.dumps(response_dict)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# Schema tests (mocked — always run)
# ---------------------------------------------------------------------------

def test_extract_returns_all_required_keys():
    """All 8 keys the scoring algorithm needs must be present in the output."""
    model = make_mock_client(SAMPLE_PREFS)
    prefs = extract_preferences("chill lofi for studying", model)
    for key in REQUIRED_KEYS:
        assert key in prefs, f"Missing required key: '{key}'"


def test_extract_values_in_valid_ranges():
    """Float preferences must be within their valid ranges."""
    model = make_mock_client(SAMPLE_PREFS)
    prefs = extract_preferences("chill lofi for studying", model)

    for key in ("target_energy", "target_valence", "target_danceability", "target_acousticness"):
        assert 0.0 <= prefs[key] <= 1.0, f"{key} = {prefs[key]} is out of [0, 1]"

    assert 60.0 <= prefs["target_tempo_bpm"] <= 180.0, (
        f"target_tempo_bpm = {prefs['target_tempo_bpm']} is out of [60, 180]"
    )


def test_extract_genre_is_valid():
    """The extracted genre must be one of the catalog genres."""
    model = make_mock_client(SAMPLE_PREFS)
    prefs = extract_preferences("chill lofi for studying", model)
    assert prefs["favorite_genre"] in VALID_GENRES


def test_extract_mood_is_valid():
    """The extracted mood must be one of the recognized moods."""
    model = make_mock_client(SAMPLE_PREFS)
    prefs = extract_preferences("chill lofi for studying", model)
    assert prefs["favorite_mood"] in VALID_MOODS


def test_extract_likes_acoustic_is_bool():
    """likes_acoustic must be a Python bool."""
    model = make_mock_client(SAMPLE_PREFS)
    prefs = extract_preferences("chill lofi for studying", model)
    assert isinstance(prefs["likes_acoustic"], bool)


# ---------------------------------------------------------------------------
# Guardrail tests (call _validate_and_clamp_preferences directly)
# ---------------------------------------------------------------------------

def test_guardrail_clamps_energy_above_one():
    bad = {**SAMPLE_PREFS, "target_energy": 1.8}
    result = _validate_and_clamp_preferences(bad)
    assert result["target_energy"] == 1.0


def test_guardrail_clamps_valence_below_zero():
    bad = {**SAMPLE_PREFS, "target_valence": -0.3}
    result = _validate_and_clamp_preferences(bad)
    assert result["target_valence"] == 0.0


def test_guardrail_clamps_tempo_above_ceiling():
    bad = {**SAMPLE_PREFS, "target_tempo_bpm": 300.0}
    result = _validate_and_clamp_preferences(bad)
    assert result["target_tempo_bpm"] == 180.0


def test_guardrail_clamps_tempo_below_floor():
    bad = {**SAMPLE_PREFS, "target_tempo_bpm": 30.0}
    result = _validate_and_clamp_preferences(bad)
    assert result["target_tempo_bpm"] == 60.0


def test_guardrail_unknown_genre_falls_back_to_pop():
    bad = {**SAMPLE_PREFS, "favorite_genre": "reggaeton"}
    result = _validate_and_clamp_preferences(bad)
    assert result["favorite_genre"] == "pop"


def test_guardrail_unknown_mood_falls_back_to_chill():
    bad = {**SAMPLE_PREFS, "favorite_mood": "ethereal"}
    result = _validate_and_clamp_preferences(bad)
    assert result["favorite_mood"] == "chill"


def test_guardrail_raises_on_missing_key():
    incomplete = {k: v for k, v in SAMPLE_PREFS.items() if k != "target_energy"}
    with pytest.raises(ValueError, match="missing required key"):
        _validate_and_clamp_preferences(incomplete)


def test_guardrail_casts_likes_acoustic_to_bool():
    # Gemini might return 1 or "true" instead of True
    prefs_with_int = {**SAMPLE_PREFS, "likes_acoustic": 1}
    result = _validate_and_clamp_preferences(prefs_with_int)
    assert result["likes_acoustic"] is True
    assert isinstance(result["likes_acoustic"], bool)


# ---------------------------------------------------------------------------
# Input guardrail tests
# ---------------------------------------------------------------------------

def test_extract_raises_on_empty_input():
    model = make_mock_client(SAMPLE_PREFS)
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_preferences("", model)


def test_extract_raises_on_whitespace_only_input():
    model = make_mock_client(SAMPLE_PREFS)
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_preferences("   ", model)


def test_extract_raises_on_non_json_response():
    mock_response = MagicMock()
    mock_response.text = "Sorry, I don't understand music."
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    with pytest.raises(ValueError, match="non-JSON"):
        extract_preferences("give me music", mock_client)


# ---------------------------------------------------------------------------
# Consistency test (mocked)
# ---------------------------------------------------------------------------

def test_consistency_same_mock_produces_identical_results():
    """
    Given the same mocked Gemini response, extract_preferences must return
    identical dicts across multiple calls — the pipeline must be deterministic.
    """
    model = make_mock_client(SAMPLE_PREFS)
    results = [extract_preferences("something chill to study to", model) for _ in range(3)]
    assert all(r == results[0] for r in results), (
        f"Results differ across runs: {results}"
    )


# ---------------------------------------------------------------------------
# Live API consistency test (skipped unless --live)
# ---------------------------------------------------------------------------

def test_live_consistency(live_mode):
    """
    LIVE TEST — requires --live flag and GOOGLE_API_KEY in environment.

    Calls the real Gemini API three times with the same query and checks that:
    - All extracted genres and moods are valid
    - Energy values do not swing more than 0.3 across runs (consistency check)
    - All float values remain within their valid ranges
    """
    if not live_mode:
        pytest.skip("Pass --live to run real Gemini API tests.")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set.")

    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    client = genai.Client(api_key=api_key)

    query = "I want something chill and acoustic for studying"
    results = [extract_preferences(query, client) for _ in range(3)]

    genres = [r["favorite_genre"] for r in results]
    moods = [r["favorite_mood"] for r in results]
    energies = [r["target_energy"] for r in results]

    print(f"\n--- Live Consistency Report ---")
    print(f"Query: '{query}'")
    print(f"Genres:  {genres}")
    print(f"Moods:   {moods}")
    print(f"Energies: {energies}  (range: {max(energies) - min(energies):.2f})")

    for r in results:
        assert r["favorite_genre"] in VALID_GENRES, f"Invalid genre: {r['favorite_genre']}"
        assert r["favorite_mood"] in VALID_MOODS, f"Invalid mood: {r['favorite_mood']}"
        for key in ("target_energy", "target_valence", "target_danceability", "target_acousticness"):
            assert 0.0 <= r[key] <= 1.0, f"{key} out of range: {r[key]}"

    energy_range = max(energies) - min(energies)
    assert energy_range <= 0.3, (
        f"Energy inconsistency too high across 3 runs: {energies} (range {energy_range:.2f})"
    )
