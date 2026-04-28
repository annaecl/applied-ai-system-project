# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **VibeFinder 1.0**  

**Response**

**VibeMatcher 2.0**

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

**Response**

VibeMatcher recommends songs from a 100-song catalog based on a user's music preferences. Users can either describe what they want in plain language ("something chill and acoustic to study to") or fill out a profile directly with sliders in the web UI. The system returns the top 5 most fitting songs with a natural-language explanation.

Under the hood, it uses a two-call RAG pipeline powered by Google Gemini 2.5 Flash: the first call extracts structured preferences from natural language; a scoring algorithm retrieves and ranks songs; the second call generates a warm, conversational recommendation grounded in the retrieved songs.

This is a classroom simulation demonstrating how real AI products are built. It is not intended for production use.

---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What user preferences are considered  
- How does the model turn those into a score  
- What changes did you make from the starter logic  

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

**Response**

VibeMatcher uses a three-step pipeline:

**Step 1 — Preference Extraction (Gemini 2.5 Flash).** If you type a natural language request, Gemini reads it and converts it into a structured set of numbers: a favorite genre, a mood, and numeric targets for energy, acousticness, positivity (valence), danceability, and tempo. This is the only step that uses the LLM to understand meaning.

**Step 2 — Retrieval (Scoring Algorithm).** Every song in the 100-song catalog gets a score based on how closely its audio features match your preferences. Five features are weighted: energy and acousticness count for 25% each, valence and danceability for 20% each, and tempo for 10%. If a song's genre matches your target genre, it gets a flat +0.15 bonus; a mood match adds another +0.15. Songs are ranked highest to lowest and the top 5 are returned.

**Step 3 — Generation (Gemini 2.5 Flash).** Gemini receives your original request plus the top 5 ranked songs as context, and writes a warm, conversational recommendation explaining why the top picks fit what you asked for. It cannot change the ranking — it can only describe the results the algorithm already produced.

Gemini does not influence, override, or rerank the scoring algorithm's output. The two LLM calls handle language understanding and language generation; everything in between is deterministic math.

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  

**Response**

The catalog has 100 songs (expanded from 18 in the original project). It covers 15 genres: pop, lofi, rock, ambient, jazz, synthwave, indie pop, classical, hip-hop, country, r&b, metal, folk, electronic, and blues.

Moods include happy, chill, intense, focused, melancholic, energetic, nostalgic, romantic, angry, sad, moody, and relaxed.

All songs are fictional and were created for this simulation. Because the catalog was written by hand, it reflects assumptions about what genres and moods typically sound like — which introduces bias. Many real-world genres are missing entirely (no Latin, no K-pop, no reggae, no EDM subgenres). Songs also cluster in the mid-range for energy and acousticness values, so users with extreme preferences (very high or very low energy) have fewer close matches.

---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

**Response**

The system works best for clear, genre-specific requests. Natural language queries like "something chill and acoustic to study to" correctly extract `genre=lofi, mood=chill, energy=0.30` and surface low-energy lofi songs (Soft Hours, Library Rain, Spacewalk Thoughts). Upbeat queries like "upbeat pop music for a morning run" correctly extract `genre=pop, mood=energetic, energy=0.90` and return high-energy pop tracks.

The weighted numeric scoring does a decent job of separating songs when genre and mood bonuses do not apply. The "Ghost Genre & Mood" adversarial test showed that numeric proximity alone still produces a sensible ranking even when no categorical bonuses fire.

Gemini's natural language understanding is generally reliable for common descriptions. Vague queries ("something nice") produce reasonable mid-range preferences rather than crashing.

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  

**Response**
- The flat +0.15 genre and mood bonuses are too powerful. A song with a matching genre but poor audio feature alignment can outscore a better overall match. This is a structural bias in the scoring formula — the bonuses should be scaled rather than flat.
- Gemini's extracted preferences can vary slightly across runs for ambiguous queries. "Something chill" might map to `energy=0.30` in one run and `energy=0.38` in another, which changes which songs score highest even though the user's request was identical.
- The `likes_acoustic` boolean in the user profile is captured but never used by the scoring algorithm — `score_song()` uses `target_acousticness` only. The flag is silently irrelevant.
- The catalog is entirely fictional and reflects the creator's assumptions about genre/mood audio characteristics, which may not match what a real listener expects.

---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested  
- What you looked for in the recommendations  
- What surprised you  
- Any simple tests or comparisons you ran  

No need for numeric metrics unless you created some.

**Response**

The full test suite has 25 tests (24 passing, 1 skipped).

**Unit tests (`tests/test_recommender.py`, 7 tests)** — validated the scoring logic: correct sorting by score, non-empty explanations, score ranges, and genre match reasons.

**AI reliability tests (`tests/test_ai_consistency.py`, 17 tests + 1 live)** — tested the guardrail layer using mocked Gemini responses:
- Schema completeness: all 8 required keys must be present
- Value range validation: floats in `[0, 1]`, tempo in `[60, 180]`
- Guardrail clamping: out-of-range values are corrected, not rejected
- Fallback behavior: unknown genres/moods fall back to safe defaults
- Input edge cases: empty input and non-JSON responses raise clean `ValueError`s
- Consistency: same mocked response always produces identical output
- Live consistency test (skipped by default): real Gemini API, 3 runs of the same query, energy variance must stay below 0.3

**Live output tests** — two natural language queries and one profile were run against the live system:

*Query: "something chill and acoustic to study to"*
Extracted: `genre=lofi, mood=chill, energy=0.30`
Top picks: Soft Hours (Dusk Tape), Library Rain (Paper Lanterns), Spacewalk Thoughts (Orbit Bloom)

*Query: "upbeat pop music for a morning run"*
Extracted: `genre=pop, mood=energetic, energy=0.90`
Top picks: Summer Static (Coastal Line), Dancing in Neon (Prism Beat), Sunrise City (Neon Echo)

*Profile: Chill Lofi* (`genre=lofi, mood=chill, energy=0.28, acousticness=0.72`)
`#1 Midnight Coding — LoRoom (1.21)  #2 Soft Hours — Dusk Tape (1.20)  #3 Library Rain — Paper Lanterns (1.19)`
Gemini: *"For that gentle, low-energy vibe, Midnight Coding by LoRoom would be a fantastic start — its subtle sounds are great for focus or just relaxing. You might also love Soft Hours by Dusk Tape, which truly embodies a peaceful atmosphere ideal for quiet moments."*

**What surprised me?**
The biggest surprise was how many distinct ways the AI layer could fail before adding the guardrail validator — missing keys, out-of-range floats, unknown genre/mood values, non-JSON text, and empty input each needed separate handling. Also, Gemini's preference extraction can vary slightly between runs for vague queries, which is why a live consistency test exists.


---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

**Response**

The genre and mood bonuses feel too powerful. A better approach would be to fold them into the weighted formula as scaled scores instead of flat add-ons, so they can't overwhelm a better overall audio match.

The catalog is still small at 100 songs. Adding hundreds more per genre would make numeric scoring much more meaningful — right now some genres have only a handful of songs competing.

The `likes_acoustic` boolean is never used in scoring — it should either be removed or wired into the acousticness weight.

A diversity filter would help. Right now the top 5 results can all come from the same artist or genre. Forcing representation across at least 2–3 genres would make recommendations feel less repetitive.

For natural language mode, adding a confidence indicator — something like "I'm not sure what genre fits this, defaulting to pop" — would make the system more transparent when Gemini's extraction is uncertain.

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  

**Response**

Implementing the recommender system was relatively simple, but its effectiveness is based on subjective decisions (such as what matters more, energy or genre) that may lead to counterintuitive recommendations. This explains why when apps like Spotify make suggestions, they reflect strategic choices that may or may not result in a positive user experience. If I had more time, I would like to hammer out some of the kinks in the system (ex: the over-emphasis on genre and mood) and refine how the recommendations are visually presented to the user. I would also add a larger collection of songs. 