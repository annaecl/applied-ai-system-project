# VibeMatcher — AI Music Recommender

## Original Project (Modules 1–3)

This project started as **VibeMatcher 1.0**, a rules-based music recommender simulation built during Modules 1–3. Its original goal was to demonstrate how recommendation systems work by scoring 18 songs against a user's hardcoded taste profile — matching on audio features like energy, tempo, and acousticness using a weighted proximity formula. It could rank songs and generate basic scoring explanations, but every user profile was a fixed Python dictionary and the "AI" was just math — no language model, no natural language input, no dynamic behavior.

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

The scoring algorithm output shown below is exact. The Gemini-generated recommendations are representative of what the system produces.

---

### Example 1 — Study Music

**Input:** `"something chill and acoustic to study to"`

**Extracted preferences:**
```
genre=lofi  mood=chill  energy=0.28  tempo=75 BPM
valence=0.55  danceability=0.48  acousticness=0.80
```

**Songs retrieved by scoring algorithm:**
```
#1  Soft Hours         — Dusk Tape        (score=1.25)
#2  Library Rain       — Paper Lanterns   (score=1.24)
#3  Midnight Coding    — LoRoom           (score=1.21)
#4  Rainy Afternoon    — LoRoom           (score=1.12)
#5  Warm Light         — Paper Lanterns   (score=1.10)
```

**AI-generated recommendation:**
> It sounds like you're looking for some gentle background music to help you focus without distraction. I'd start with **Soft Hours by Dusk Tape** — it has a warm, unhurried feel that's perfect for settling into a study session. **Library Rain by Paper Lanterns** is another great pick, with a soft, slightly nostalgic texture that keeps things calm without putting you to sleep. If you want something with just a bit more life to it, **Midnight Coding by LoRoom** has that cozy late-night energy that makes long study sessions feel manageable.

---

### Example 2 — Workout Music

**Input:** `"I need high energy intense music for lifting weights"`

**Extracted preferences:**
```
genre=rock  mood=intense  energy=0.90  tempo=145 BPM
valence=0.42  danceability=0.60  acousticness=0.08
```

**Songs retrieved by scoring algorithm:**
```
#1  Storm Runner          — Voltline          (score=1.26)
#2  Blue Collar Anthem    — Working Crew      (score=1.24)
#3  Concrete Wings        — Voltline          (score=1.10)
#4  Dark Signal           — Iron Circuit      (score=1.09)
#5  Aftershock            — Red Fault         (score=1.09)
```

**AI-generated recommendation:**
> Time to push your limits! **Storm Runner by Voltline** is the obvious opener — it hits hard right from the start with the kind of raw, driving energy that makes you want to move more weight. **Blue Collar Anthem by Working Crew** pairs well if you want something with a bit more grit and attitude, perfect for the middle of a tough set. And when you need a final push, **Aftershock by Red Fault** delivers exactly what the name promises — an aggressive surge that'll carry you through the last rep.

---

### Example 3 — Late Night Drive

**Input:** `"moody synthwave for a late night drive"`

**Extracted preferences:**
```
genre=synthwave  mood=moody  energy=0.74  tempo=112 BPM
valence=0.45  danceability=0.76  acousticness=0.12
```

**Songs retrieved by scoring algorithm:**
```
#1  Digital Rain     — Binary Pulse      (score=1.28)
#2  Night Drive Loop — Neon Echo         (score=1.26)
#3  Night Moves      — Dark Circuit      (score=1.11)
#4  Heartbeats       — Echo Shore        (score=1.09)
#5  Retro Signal     — Volt City         (score=1.07)
```

**AI-generated recommendation:**
> For a late-night drive with some atmospheric edge, you're in the right place. **Night Drive Loop by Neon Echo** is practically made for this — that slow-building synthwave pulse feels exactly like watching city lights blur past at 2am. **Digital Rain by Binary Pulse** adds a darker, more mysterious layer if you want to lean into the mood, with a rhythm that matches the hum of the highway. Round it out with **Retro Signal by Volt City** for a touch of nostalgia that feels like a drive through a city that never quite existed.

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
| `google-genai` SDK | Current, supported SDK — but free tier quota is limited for high-volume use |

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
# Result: 19 passed, 1 skipped (live test)
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

Building VibeMatcher taught me that AI systems are not a single model — they're pipelines where each component needs to be individually reliable. The scoring algorithm is simple enough to reason about and test directly. The Gemini calls are powerful but non-deterministic, which is why the guardrail layer exists: it translates Gemini's flexibility into something the downstream code can trust.

The biggest shift in thinking was moving from "I'll just ask Gemini to do everything" to understanding *why* RAG is more robust. When retrieval is separate from generation, you can log what was retrieved, test that step independently, and swap out the retrieval strategy without changing the generation. That architectural clarity — knowing which part of your system does what — is what makes AI products maintainable rather than magical black boxes.

From a bias and fairness perspective: the catalog reflects whatever preferences went into designing the 100 songs. Genres like K-pop, Latin, Afrobeats, and dozens of others aren't represented, which means the system can't serve users with those tastes well — even if Gemini perfectly extracts their preferences, the retrieval step would fall back to a poor approximation. Real recommenders face this same problem at scale, and it's a reminder that the data you retrieve from shapes the quality of what you generate.
