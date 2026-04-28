# VibeMatcher — AI Music Recommender

## Original Project (Modules 1–3)

This project started as **VibeMatcher 1.0**, a rules-based music recommender simulation. Its original goal was to demonstrate how recommendation systems work by scoring 18 songs against a user's hardcoded taste profile — matching on audio features like energy, tempo, and acousticness using a weighted proximity formula. It could rank songs and generate basic scoring explanations, but every user profile was a fixed Python dictionary and the "AI" was just math — no language model, no natural language input, no dynamic behavior.

---

## Title and Summary

**VibeMatcher** is an AI-powered music recommender that accepts plain-language requests — "something chill and acoustic to study to" — and returns personalized song picks with natural-language explanations. It matters because it demonstrates how real AI products are built: not just a language model alone, but a system where retrieval and generation work together. Gemini understands what you want; a scoring algorithm finds the best matches; Gemini explains the picks as if talking to a friend.

---

## Architecture Overview

> **Diagrams:** The Mermaid source for the system diagram is at [`system_diagram.mmd`](system_diagram.mmd). A rendered image of the full pipeline is at [`final_project_diagram.png`](final_project_diagram.png).

VibeMatcher uses a **Retrieval-Augmented Generation (RAG)** pipeline:

```
User types: "something chill and acoustic to study to"
         │
         ▼
 ┌─────────────────────────────────────────────┐
 │  LLM Call 1 — Preference Extraction         │
 │  Gemini reads the request and outputs a     │
 │  structured JSON dict:                      │
 │    genre=lofi, mood=chill, energy=0.28,     │
 │    acousticness=0.80, tempo=75 BPM, ...     │
 └─────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────┐
 │  Retrieval — Scoring Algorithm               │
 │  Scores all 100 catalog songs by proximity  │
 │  to the extracted preferences.              │
 │  Returns top-k ranked results.              │
 └─────────────────────────────────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────┐
 │  LLM Call 2 — Augmented Generation          │
 │  Gemini receives the original request +     │
 │  the retrieved songs as context, and writes │
 │  a warm, personalized recommendation.       │
 └─────────────────────────────────────────────┘
         │
         ▼
  User sees a natural-language recommendation
  + expandable panels showing retrieved songs
    and extracted preferences
```

The scoring formula (unchanged from the original project):

```
numeric_score = weighted average of proximity scores across:
    energy (w=0.25), acousticness (w=0.25), valence (w=0.20),
    danceability (w=0.20), tempo normalized to 200 BPM (w=0.10)

total_score = numeric_score
            + 0.15 if genre matches
            + 0.15 if mood matches
```

### How Gemini and the scoring algorithm interact

**Gemini does not override, influence, or rerank the scoring algorithm's results.**

The two Gemini calls and the scoring algorithm each own a distinct part of the pipeline:

| Step | Who does it | What it produces |
|---|---|---|
| Preference extraction | Gemini (Call 1) | Structured dict (genre, mood, energy, tempo, …) |
| Song retrieval & ranking | Scoring algorithm | Ranked list of `(song, score, explanation)` tuples |
| Natural-language response | Gemini (Call 2) | Conversational text shown to the user |

The only way Gemini touches the ranking is indirectly: if Call 1 extracts slightly different numeric values (e.g., `energy=0.30` vs `energy=0.38`), the algorithm will score songs differently. Once those preferences are fixed, the algorithm runs deterministically and Gemini has no further input.

Call 2 receives the already-ranked list and is instructed to write about the top songs — it cannot promote a low-scoring song or demote a high-scoring one. The scores and order the user sees are always the algorithm's output, not Gemini's opinion.

---

## Setup Instructions

### Prerequisites
- Python 3.9+
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier available)

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd applied-ai-system-project
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Mac / Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your-key-here
```

> This file is in `.gitignore` and is never committed.

### 5. Run the app

**Streamlit web UI (recommended):**
```bash
streamlit run src/app.py
```
Open the URL shown in the terminal (usually `http://localhost:8501`).

**CLI — single query:**
```bash
python3 -m src.main --query "something chill to study to"
```

**CLI — interactive loop:**
```bash
python3 -m src.main --ai
```

**Original profile mode (no API key needed):**
```bash
python3 -m src.main --profile "Chill Lofi"
python3 -m src.main        # runs all 7 hardcoded profiles
```

### 6. Run tests

```bash
pytest                                        # all tests, mocked (no API key needed)
pytest tests/test_ai_consistency.py -v        # reliability tests only
pytest tests/test_ai_consistency.py --live    # live Gemini API consistency check
```

---

## Sample Interactions

Examples 1 and 2 use the natural-language CLI (`--query`). Example 3 uses the hardcoded profile mode. All outputs are captured directly from the running system.

---

### Example 1 — Study Music (natural language)

**Input:** `"something chill and acoustic to study to"`

**Extracted by Gemini:** `genre=lofi  mood=chill  energy=0.30`

**AI-generated recommendation:**
> I've got some wonderfully chill tunes perfect for studying to, just what you asked for! For that peaceful, focused atmosphere, I'd highly recommend **Soft Hours by Dusk Tape**, which has a lovely gentle lofi feel that's great for background listening. You might also like **Library Rain by Paper Lanterns** for a similar calming vibe. And if you prefer something a bit more atmospheric and dreamlike to help you concentrate, **Spacewalk Thoughts by Orbit Bloom** would be a fantastic choice.

---

### Example 2 — Morning Run (natural language)

**Input:** `"upbeat pop music for a morning run"`

**Extracted by Gemini:** `genre=pop  mood=energetic  energy=0.90`

**AI-generated recommendation:**
> Ready to energize your morning run with some fantastic upbeat pop! I've picked out a few tracks that should get you moving and grooving.
>
> First up, try **Summer Static by Coastal Line** — it's super energetic and has that perfect summery vibe to power your run. You might also love **Dancing in Neon by Prism Beat** and **Sunrise City by Neon Echo**, both of which are wonderfully happy and uplifting, making them ideal for an upbeat workout. Enjoy your run!

---

### Example 3 — Chill Lofi (hardcoded profile)

**Profile:** `genre=lofi  mood=chill  energy=0.28  tempo=78 BPM  acousticness=0.72`

**Scored results:**
```
#1  Midnight Coding  —  LoRoom           Score: 1.21
    audio features score: 0.91  |  genre match (+0.15)  |  mood match (+0.15)

#2  Soft Hours       —  Dusk Tape        Score: 1.20
    audio features score: 0.90  |  genre match (+0.15)  |  mood match (+0.15)

#3  Library Rain     —  Paper Lanterns   Score: 1.19
    audio features score: 0.89  |  genre match (+0.15)  |  mood match (+0.15)

#4  Gray Afternoon   —  Dusk Tape        Score: 1.08
    audio features score: 0.93  |  genre match (+0.15)

#5  Rainy Afternoon  —  LoRoom           Score: 1.07
    audio features score: 0.92  |  genre match (+0.15)
```

**AI-generated recommendation:**
> I've got just the thing for your chill lofi needs, perfect for studying or winding down!
>
> For that gentle, low-energy vibe, I think **Midnight Coding by LoRoom** would be a fantastic start — its subtle sounds are great for focus or just relaxing. You might also love **Soft Hours by Dusk Tape**, which truly embodies a peaceful atmosphere ideal for quiet moments. And if you're looking for something serene, **Library Rain by Paper Lanterns** perfectly captures that tranquil, reflective mood you described. Enjoy your calming listening!

---

## Design Decisions

### Why RAG instead of asking Gemini to recommend directly?

The project already had a working scoring algorithm from Modules 1–3. Replacing it entirely with an LLM would throw away a transparent, explainable system that we understood and could test. The RAG pattern let us keep what worked — the retrieval and ranking — and add what was missing: natural language input and output. Gemini handles the hard parts of understanding what someone *means* and explaining picks in plain language. The scoring algorithm handles the hard parts of *finding* songs with the right audio characteristics. Neither could do the full job alone.

### Why two separate Gemini calls?

A single call asking Gemini to both understand preferences and recommend songs would skip the scoring algorithm entirely, making it impossible to test, debug, or extend. Separating extraction from generation also means we can validate the structured output (call 1) before it touches the scoring algorithm, which is where the guardrail layer lives.

### Why a 100-song catalog?

The original 18-song catalog was too small for meaningful retrieval — almost any query would return the same 5 songs because there weren't enough options. At 100 songs, there's real differentiation: a rock query gets different results than a metal query, a "chill" lofi request gets different songs than a "focused" lofi request. A real system would have millions, but 100 is enough to demonstrate the RAG pattern meaningfully.

### Trade-offs

| Decision | Trade-off |
|---|---|
| Flat genre/mood bonuses (+0.15) | Simple and transparent, but can let a matching genre outscore a better overall audio match |
| Gemini for preference extraction | Human-readable and flexible, but adds latency and can be slightly inconsistent across runs |
| Scored retrieval over embeddings | More explainable and testable, but can't capture semantic similarity (e.g., "jazz" and "blues" scored separately even when both match a vibe) |
| `google-genai` SDK + `gemini-2.5-flash` | Current, supported SDK — but free tier quota is limited for high-volume use |

---

## Testing Summary

### What was tested

**Unit tests (`tests/test_recommender.py`)** — validated the original scoring logic: sorting by score, non-empty explanations. These passed without changes after the catalog expansion.

**Reliability tests (`tests/test_ai_consistency.py`)** — 17 tests covering:
- Schema completeness: all 8 required keys must be present in Gemini's output
- Value range validation: floats in `[0, 1]`, tempo in `[60, 180]`
- Guardrail clamping: out-of-range values are corrected, not rejected
- Fallback behavior: unknown genres/moods fall back to safe defaults rather than crashing
- Input edge cases: empty input and non-JSON responses raise clean `ValueError`s
- Consistency: same mocked response always produces identical output (pipeline is deterministic)
- Live consistency test (`--live`): real Gemini API, 3 runs of same query, energy variance must stay below 0.3

```
pytest tests/ -v
# Result: 24 passed, 1 skipped (live test)
```

### What worked well

- The guardrail layer caught every edge case we tested: out-of-range values, unknown genres, non-JSON responses, empty input
- The scoring algorithm performed consistently after the catalog expanded — no changes were needed
- Mocked tests ran in under 1 second and required no API key, making them safe to run anywhere
- Separating the pipeline into two functions (`extract_preferences` + `generate_recommendation`) made both easy to test independently

### What didn't work as expected

- The `likes_acoustic` boolean field in `UserProfile` (from the original project) has no effect on scoring — it's captured in the extracted preferences dict but `score_song()` uses `target_acousticness` only. This is a known limitation inherited from the original design.
- Gemini's extracted preferences can vary slightly across runs for ambiguous queries — "something chill" might map to `energy=0.30` one run and `energy=0.38` another. This is why the live consistency test exists.
- The `google-genai` free tier has strict per-minute rate limits, which can cause 429 errors during rapid consecutive calls.

### What we learned

Writing tests before seeing real Gemini output forced us to think clearly about what the system *should* do versus what it actually does. The guardrail tests in particular exposed that we needed to handle five different failure modes (missing keys, out-of-range floats, unknown categories, non-JSON output, empty input) — none of which were obvious until we tried to write a test that broke the system.

---

## Reflection

I really enjoyed making this project — it let me integrate several things we learned such as how to use Streamlit, apply RAG, and using Mermaid.js to plan and visualize the system design. But building it also pushed me to think beyond just "does it work?" to "is it responsible?"

**Limitations and biases.** The catalog is entirely fictional and manually created, which means it reflects my own assumptions about what genres and moods sound like. The flat +0.15 genre and mood bonuses in the scoring formula can let a weaker song with a matching genre outscore a better overall audio match — a structural bias baked into the weights. Gemini's preference extraction also carries whatever biases exist in its training data: if someone describes a cultural style of music that isn't well-represented in Western music datasets, the extracted features will likely map it to the closest Western equivalent rather than representing it accurately.

**Misuse.** A music recommender seems low-stakes, but the same RAG pattern applies to higher-risk domains. In this system, a user could craft a query designed to extract the system prompt or manipulate Gemini's structured output — prompt injection is a real concern whenever user text goes directly into an LLM call. The guardrail layer partially addresses this by validating and clamping Gemini's output before it touches the scoring algorithm, so even a malicious or garbled response can't crash the system or produce nonsense results. A more hardened version would also sanitize the input before passing it to Gemini in the first place.

**Surprises during reliability testing.** The biggest surprise was how many distinct failure modes appeared once I actually tried to break the system. Before writing tests, I assumed Gemini would either return valid JSON or clearly fail — but in practice there were five separate ways it could produce bad output (missing keys, out-of-range floats, unknown genre/mood values, non-JSON text, and empty input), each of which needed its own handling. I also didn't expect the live consistency test to matter much, but it exposed that "something chill" could map to `energy=0.30` in one run and `energy=0.38` in another — a small difference that meaningfully changes which songs score highest.

**Collaboration with AI.** I used Claude throughout the project for both design and implementation. One instance where AI gave a genuinely helpful suggestion: when I was designing the two-call pipeline, Claude pointed out that separating preference extraction from response generation would make each step independently testable — which turned out to be exactly right, since I could mock Gemini's output in tests without needing a real API key. One instance where AI was wrong: early on, Claude suggested using a single Gemini call to handle both preference extraction and song recommendation together. That would have bypassed the scoring algorithm entirely and made the system untestable and opaque. I pushed back and kept the two-stage design, which was the better call.

---

## What Changed from the Original

| | Original (Modules 1–3) | Current version |
|---|---|---|
| **Input** | Hardcoded Python profile dicts only | Natural language queries + hardcoded profiles |
| **AI** | None — scoring math only | Two Gemini calls (preference extraction + response generation) |
| **Output** | Ranked list with score breakdowns | Ranked list + conversational natural-language recommendation |
| **Catalog** | 18 songs | 100 songs |
| **Interface** | CLI only | CLI + Streamlit web UI |
| **Guardrails** | None | Validator clamps out-of-range values and rejects bad AI output |
| **Tests** | Basic unit tests (scoring, sorting) | Unit tests + 17 AI reliability tests + live consistency check |
