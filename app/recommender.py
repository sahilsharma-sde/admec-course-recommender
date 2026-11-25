"""Course recommender: TF-IDF on words + character n-grams, with light query expansion.

CPU only, no model downloads, answers in a few milliseconds. For a catalog of
tens or hundreds of courses this is accurate enough and very easy to explain in class.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(
    os.getenv("COURSES_PATH", Path(__file__).resolve().parent.parent / "data" / "courses.json")
)

# Below this combined score a course is not shown (the query matched nothing meaningful).
MIN_SCORE = 0.06

# People describe goals in everyday words, not course vocabulary. Each key (a word the
# user types) adds the words that describe the same intent in the catalog.
EXPANSIONS = {
    "youtube": "video editing premiere pro content creation",
    "youtuber": "video editing premiere pro content creation social media",
    "reels": "video editing social media content creation",
    "film": "video editing motion graphics",
    "movies": "video editing motion graphics animation",
    "vfx": "after effects 3d blender animation",
    "cartoon": "2d animation motion graphics",
    "animator": "animation 2d 3d motion graphics",
    "drawing": "graphic design illustrator creative",
    "draw": "graphic design illustrator photoshop creative",
    "art": "graphic design illustrator photoshop creative",
    "artist": "graphic design illustrator photoshop animation creative",
    "sketch": "graphic design illustrator creative",
    "creative": "graphic design video editing animation content creation",
    "logo": "graphic design illustrator photoshop branding",
    "poster": "graphic design photoshop illustrator",
    "designer": "graphic design ui ux figma",
    "design": "graphic design ui ux figma",
    "website": "web development html css javascript wordpress",
    "websites": "web development html css javascript wordpress",
    "web": "web development html css javascript",
    "app": "android app development mobile",
    "apps": "android app development mobile",
    "coding": "programming python java javascript",
    "coder": "programming python javascript",
    "programmer": "programming python java c++",
    "programming": "programming python java c++ javascript",
    "software": "programming developer python java",
    "c++": "c c++ programming object oriented data structures",
    "cpp": "c c++ programming object oriented data structures",
    "ai": "machine learning artificial intelligence generative ai",
    "ml": "machine learning python scikit-learn",
    "chatgpt": "generative ai llm prompt engineering",
    "chatbot": "generative ai llm rag langchain",
    "chatbots": "generative ai llm rag langchain",
    "data": "data analytics excel sql power bi",
    "analyst": "data analytics excel sql power bi",
    "analytics": "data analytics excel sql power bi dashboards",
    "scientist": "data science machine learning python",
    "hacker": "ethical hacking cyber security penetration testing",
    "hacking": "ethical hacking cyber security penetration testing",
    "security": "cyber security ethical hacking network",
    "cybersecurity": "cyber security ethical hacking network",
    "server": "linux system administration cloud devops",
    "servers": "linux system administration cloud devops",
    "sysadmin": "linux system administration shell scripting",
    "cloud": "aws devops docker cloud computing",
    "devops": "devops aws docker linux ci/cd",
    "seo": "digital marketing seo google ads",
    "marketing": "digital marketing seo social media",
    "ads": "digital marketing google ads social media",
    "business": "digital marketing e-commerce shopify tally",
    "shop": "e-commerce shopify online store",
    "store": "e-commerce shopify online store",
    "sell": "e-commerce shopify online store dropshipping",
    "accounts": "tally gst accounting billing",
    "accounting": "tally gst accounting billing",
    "accountant": "tally gst accounting billing",
    "gst": "tally gst accounting billing",
    "game": "game development unity c#",
    "games": "game development unity c#",
    "gaming": "game development unity c#",
    "3d": "blender 3d modeling animation",
    "freelance": "freelancer web design graphic design content creation",
    "freelancing": "freelancer web design graphic design content creation",
    "office": "advanced excel mis reporting",
    "mis": "advanced excel mis reporting",
    "excel": "excel formulas pivot tables dashboards",
}

_TOKEN = re.compile(r"[a-z0-9+#/]+")
_SKIP = {"and", "with", "for", "the", "pro", "basics", "design"}


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def expand_query(query: str) -> str:
    """Append catalog vocabulary for everyday words found in the query."""
    extra = [EXPANSIONS[t] for t in dict.fromkeys(_tokens(query)) if t in EXPANSIONS]
    return " ".join([query] + extra)


def _course_text(c: dict) -> str:
    # Title, skills and careers are repeated so they outweigh the long description.
    parts = [c["title"]] * 3 + [c["category"]] + c["skills"] * 2 + c["careers"] * 2 + [c["description"]]
    return " ".join(parts)


class Recommender:
    def __init__(self, path: Path = DATA_PATH):
        self.courses: list[dict] = json.loads(Path(path).read_text(encoding="utf-8"))
        self.by_id = {c["id"]: i for i, c in enumerate(self.courses)}
        docs = [_course_text(c) for c in self.courses]

        self.word = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        # Character n-grams make matching tolerant to typos like "photshop" or "hakcing".
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.Xw = self.word.fit_transform(docs)
        self.Xc = self.char.fit_transform(docs)

    # ---- scoring -------------------------------------------------------------
    def _scores(self, query: str) -> np.ndarray:
        q = expand_query(query)
        sw = cosine_similarity(self.word.transform([q]), self.Xw)[0]
        sc = cosine_similarity(self.char.transform([q]), self.Xc)[0]
        return 0.65 * sw + 0.35 * sc

    def _matched_skills(self, course: dict, query_tokens: set[str]) -> list[str]:
        """Skills the goal actually points at. A multi-word skill needs most of its words
        present, so "Sound editing" does not match just because the goal says "editing"."""
        hits = []
        for skill in course["skills"]:
            words = [w for w in _tokens(skill) if len(w) > 1 and w not in _SKIP]
            if words and sum(w in query_tokens for w in words) / len(words) > 0.5:
                hits.append(skill)
        return hits

    @staticmethod
    def _join(items: list[str]) -> str:
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    def _card(self, i: int, score: float, query_tokens: set[str] | None = None) -> dict:
        c = self.courses[i]
        card = {**c, "score": round(float(score), 4)}
        if query_tokens is not None:
            matched = self._matched_skills(c, query_tokens)
            card["matched_skills"] = matched
            card["why"] = (
                f"Teaches {self._join(matched[:3])}, which fits your goal."
                if matched
                else f"Closest fit in {c['category']} for your goal."
            )
        return card

    @staticmethod
    def _add_relative_match(cards: list[dict]) -> list[dict]:
        """match = how strong a result is compared with the best one for this query (best = 100).
        It is a ranking aid, not a probability; weak queries are already cut off by MIN_SCORE."""
        if cards:
            top = max(c["score"] for c in cards) or 1.0
            for c in cards:
                c["match"] = int(round(100 * c["score"] / top))
        return cards

    # ---- public API ----------------------------------------------------------
    def recommend(self, query: str, top_k: int = 5, level: str | None = None, category: str | None = None) -> list[dict]:
        scores = self._scores(query)
        q_tokens = set(_tokens(expand_query(query)))
        results = []
        for i in np.argsort(-scores):
            c = self.courses[i]
            if level and c["level"].lower() != level.lower():
                continue
            if category and c["category"].lower() != category.lower():
                continue
            if scores[i] < MIN_SCORE:
                break
            results.append(self._card(int(i), scores[i], q_tokens))
            if len(results) >= top_k:
                break
        return self._add_relative_match(results)

    def similar(self, course_id: str, top_k: int = 3) -> list[dict]:
        if course_id not in self.by_id:
            raise KeyError(course_id)
        idx = self.by_id[course_id]
        sims = 0.65 * cosine_similarity(self.Xw[idx], self.Xw)[0] + 0.35 * cosine_similarity(self.Xc[idx], self.Xc)[0]
        sims[idx] = -1
        return [self._card(int(i), sims[i]) for i in np.argsort(-sims)[:top_k]]

    def meta(self) -> dict:
        return {
            "categories": sorted({c["category"] for c in self.courses}),
            "levels": ["Beginner", "Intermediate", "Advanced"],
            "total_courses": len(self.courses),
        }
