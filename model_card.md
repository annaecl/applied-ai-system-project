# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **VibeFinder 1.0**  

**Response**

**VibeMatcher 1.0**

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

**Response**

VibeMatcher recommends songs from a small catalog based on a user's audio preferences. It takes in a user profile — including a favorite genre, mood, and numeric targets for energy, tempo, valence, danceability, and acousticness — and returns the top 5 most fitting songs.

It assumes the user knows what kind of music they want and can describe it with numbers. This is a classroom simulation, not a production app.

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

Every song in the catalog gets a score from 0 to 1 (plus potential bonuses). The score is based on how close the song's audio features are to what the user wants.

Five features are compared: energy, acousticness, valence (positivity), danceability, and tempo. Each one gets a proximity score — the closer the song is to the target, the higher the score. The five scores are then combined with different weights. Energy and acousticness each count for 25%, valence and danceability for 20% each, and tempo for 10%.

On top of that, if a song's genre matches the user's favorite genre, it gets a +0.15 bonus. If the mood matches, it gets another +0.15. Songs are then ranked from highest to lowest score and the top 5 are returned.

The starter code only had a placeholder that returned the first 5 songs in the list. The actual scoring logic — weighted proximity, genre/mood bonuses, and ranking — was built from scratch.

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  

**Response**

The catalog has 18 songs. It covers genres including pop, lofi, rock, ambient, jazz, synthwave, indie pop, classical, hip-hop, country, r&b, metal, folk, electronic, and blues.

Moods include happy, chill, intense, focused, melancholic, energetic, nostalgic, romantic, angry, sad, moody, and relaxed.

No songs were added or removed from the original starter dataset. The catalog is tiny compared to a real music service. Many genres are missing entirely — no Latin, no K-pop, no reggae, no EDM subgenres. Most songs cluster in the mid-range for acoustic and energy values, so extreme preferences (very loud or very quiet) have few close matches.

---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

**Response**

The system works best for users whose preferences match a clear genre or mood already in the catalog. The "Chill Lofi" profile consistently got low-energy, acoustic songs back, which felt correct. The "Deep Intense Rock" profile found the metal and rock songs reliably.

The weighted numeric scoring does a decent job of separating songs when genre and mood bonuses do not apply. The "Ghost Genre & Mood" adversarial test showed the numeric scores alone still produce a sensible ranking — songs with similar energy and tempo to the target floated to the top.

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  

**Response**
- Interestingly, using bonsus to give boosts to genre and mood matches can lead to poor recommendations. This is because a song is more likely to be recommended if it is of the same genre or mood, even if the tempo or desired energy level does not appropriately fit the user profile. It might be better to incorporate the bonuses into the weighted formula, or to create a pure point system. 

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

Seven user profiles were tested in total: three normal personas and four adversarial edge cases.

**Normal personas:**

- **High-Energy Pop** — a listener who wants upbeat, danceable pop music with high tempo and positive valence, like a workout or city-drive playlist. Recommendations were checked to confirm they clustered around high-energy, happy songs.
- **Chill Lofi** — a listener who prefers low-energy lofi beats for studying or winding down, with a preference for acoustic texture and slow tempo. Recommendations were checked for low-energy, mellow songs.
- **Deep Intense Rock** — a listener who wants heavy, raw rock at high tempo and high energy but low valence (dark mood). Recommendations were checked to see whether the system could distinguish intense-but-sad from intense-but-happy.

**Adversarial edge cases:**

- **High Energy + Sad** — conflicting signals: brutally high energy target paired with a sad folk mood preference. Sad songs tend to be low-energy, so the test examined whether the mood bonus (+0.15) could override a large energy-proximity penalty, potentially producing counterintuitive results.
- **Impossible Tempo** — a target tempo of 300 BPM, far above any real song in the catalog and above the scorer's 200-BPM normalization ceiling. This tested whether the scoring formula could produce negative component scores without breaking the overall ranking logic.
- **Ghost Genre & Mood** — favorite genre ("reggae") and mood ("euphoric") that do not exist in the catalog, so the categorical bonuses never fire. This isolated the numeric-proximity scoring and revealed how much the genre and mood bonuses normally shift rankings when they do match.
- **Dead Center** — every numeric preference set to exactly 0.5 (the midpoint). Because all songs are equidistant from the target, this exposed whether any feature weight accidentally dominates and whether ties resolve arbitrarily or consistently.
- **Contradictory Acoustic** — `likes_acoustic=True` but `target_acousticness=0.0`, a direct contradiction between the boolean flag and the numeric target. Since the scoring formula uses only `target_acousticness` for computation and ignores the boolean, this profile made the silent irrelevance of `likes_acoustic` visible in the output.

**What suprised me?**
The biggest surprise was that the top recommendations for each profile were almost always genre or mood matches (because of the bonuses). I would have preferred there be some variation with the dominant factor in the recommendation; for example, I think energy should probably play a larger role. 


---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

**Response**

The genre and mood bonuses feel too powerful. A better approach would be to fold them into the weighted formula as scaled scores instead of flat add-ons.

The catalog is too small. Adding hundreds of songs would make the numeric scoring much more meaningful, since right now there are very few songs competing in each genre.

The `likes_acoustic` boolean is never used in scoring — it should either be removed or wired into the acousticness weight.

A diversity filter would help. Right now the top 5 songs can all come from the same genre. Forcing at least 2-3 different genres in the results would make the recommendations feel more interesting.

Explanations could also be improved. Saying "audio features score: 0.74, genre match (+0.15)" is accurate but not very friendly. A plain-English explanation like "matches your energy level and genre preference" would feel better to a real user.

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  

**Response**

Building this taught me that even a simple scoring formula has a lot of hidden design decisions. Choosing weights, deciding how to handle categorical features, and figuring out what to do with edge cases all required judgment calls that were not obvious up front.

The most interesting discovery was how much the genre and mood bonuses dominated the output. I expected numeric proximity to carry most of the weight, but a +0.15 bonus is actually huge relative to the numeric scores — it can push an otherwise mediocre song to the top.

I now think about music apps like Spotify differently. When a recommendation feels slightly off — right genre but wrong energy — it is probably because the system is over-indexing on one feature the way this one over-indexes on genre and mood. Real systems must tune these trade-offs very carefully.
