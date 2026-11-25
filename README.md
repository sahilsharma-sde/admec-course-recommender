# Course Recommender

A visitor types a goal ("I want to become a video editor") and gets the best-matching courses, with a match strength, the skills that matched, career paths, and similar courses. One input, no forms, runs on a plain CPU.

The included `data/courses.json` is **dummy data** (24 sample courses). Replace it with your institute's real courses; nothing else needs to change.

## Run it

```bash
cd course-recommender
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 for the page, or http://127.0.0.1:8000/docs for the interactive API docs.

## How it works

1. Each course becomes one text document: title, category, skills, careers and description (title, skills and careers are repeated so they outweigh the long description).
2. Two TF-IDF models are built over those documents: one on words (1 to 2 word phrases) and one on character n-grams, which tolerates typos such as "photshop".
3. The visitor's goal is expanded with catalog vocabulary (`EXPANSIONS` in `app/recommender.py`: "youtuber" adds "video editing premiere pro"), then compared with every course by cosine similarity.
4. The final score is 65% word match plus 35% character match. Courses below `MIN_SCORE` are not shown. The match bar compares each result with the best one for that query (best = 100%); it is a ranking aid, not a probability.
5. "Similar courses" uses the same similarity between a course and all the others.

Training happens at startup in milliseconds, so there is no model file to manage.

## Use your own courses

Edit `data/courses.json`. Each course is one object:

```json
{
  "id": "c01",
  "title": "Graphic Design with Photoshop & Illustrator",
  "category": "Design",
  "level": "Beginner",
  "duration": "3 months",
  "description": "One or two sentences on what students learn.",
  "skills": ["Photoshop", "Illustrator", "Typography"],
  "careers": ["Graphic Designer", "Brand Designer"],
  "url": "https://your-institute.com/courses/graphic-design"
}
```

- `id` must be unique. `level` should be Beginner, Intermediate or Advanced (the level filter uses these exact words).
- `url` is optional. When present, the page shows a "View course" link.
- The better the `skills` and `careers` lists, the better the matches. Write them the way students would search.
- Restart the server after editing the file.

To point at a different file, set the `COURSES_PATH` environment variable.

## Improve the matching

- **Add everyday words** students use to the `EXPANSIONS` dictionary (for example "bank exam" mapped to your accounting course).
- **Tune the cut-off**: raise `MIN_SCORE` for stricter results, lower it to always show something.
- **Semantic matching** (understanding "make money online" without keyword overlap): replace `_scores` with sentence embeddings from `sentence-transformers` (`all-MiniLM-L6-v2` runs on CPU). It adds a model download of roughly 90 MB and a few hundred MB of dependencies, so only do it if keyword matching proves too limited.

## Add it to the institute website

**Link or embed.** Host the app, then embed it in the portfolio page:

```html
<iframe src="https://your-recommender-host/" title="Course finder"
        style="width:100%;height:760px;border:0;" loading="lazy"></iframe>
```

**Separate hosting for page and API.** Put `static/index.html` on the main site and open it with `?api=https://your-api-host`. On the API host, restrict who may call it:

```bash
export ALLOWED_ORIGINS="https://your-institute.com,https://www.your-institute.com"
```

**Hosting options.** Any host that runs Python works, for example Render, Railway, or a Hugging Face Space. Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/recommend` | Body: `{"query": "...", "top_k": 5, "level": "Beginner"}` (`level` and `category` optional) |
| GET | `/api/similar/{course_id}?top_k=3` | Courses with overlapping topics |
| GET | `/api/courses?category=&level=` | The catalog |
| GET | `/api/meta` | Categories, levels, example goals |
| GET | `/api/health` | Health check |

## Project layout

```
course-recommender/
  app/main.py          FastAPI routes
  app/recommender.py   TF-IDF engine and query expansion
  data/courses.json    course catalog (dummy data)
  static/index.html    the page (no build step)
  requirements.txt
```

## Privacy

The app does not store goals or any visitor data. Your host's own server logs are separate, so check them if that matters.
