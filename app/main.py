"""FastAPI service for the course recommender.

Run from the project root:  uvicorn app.main:app --reload
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .recommender import Recommender

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

# Written as completions of the page's "I want to ..." headline.
EXAMPLES = [
    "become a YouTuber",
    "draw and design for a living",
    "build AI chatbots",
    "learn hacking and cyber security",
    "work in accounts and GST",
    "start a career in data",
    "make websites for clients",
]

app = FastAPI(title="Course Recommender API", version="1.0.0")

# Set ALLOWED_ORIGINS="https://your-institute-site.com" in production.
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

engine = Recommender()


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=300, description="Goal or interest in plain words")
    top_k: int = Field(5, ge=1, le=10)
    level: Optional[str] = None
    category: Optional[str] = None


@app.get("/api/health")
def health():
    return {"status": "ok", "courses": len(engine.courses)}


@app.get("/api/meta")
def meta():
    return {**engine.meta(), "examples": EXAMPLES}


@app.get("/api/courses")
def courses(category: Optional[str] = None, level: Optional[str] = None):
    items = engine.courses
    if category:
        items = [c for c in items if c["category"].lower() == category.lower()]
    if level:
        items = [c for c in items if c["level"].lower() == level.lower()]
    return {"count": len(items), "courses": items}


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    results = engine.recommend(req.query, req.top_k, req.level, req.category)
    return {"query": req.query, "count": len(results), "results": results}


@app.get("/api/similar/{course_id}")
def similar(course_id: str, top_k: int = Query(3, ge=1, le=6)):
    try:
        return {"course_id": course_id, "results": engine.similar(course_id, top_k)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Course not found")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
