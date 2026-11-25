#!/usr/bin/env python3
"""Build the recommender's course data from a course sitemap.

Reads the sitemap, downloads every course page, pulls out the useful fields and writes
data/courses.generated.json in the recommender's format, plus a review report.

Setup (once):      pip install requests beautifulsoup4
Trial run:         python tools/build_courses.py --limit 5
Inspect one page:  python tools/build_courses.py --url https://www.admecindia.co.in/course/power-bi-course/ --debug
Full run:          python tools/build_courses.py

Pages are cached in .cache/pages, so you can tweak the rules below and re-run without
downloading everything again (use --refresh to download again).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SITEMAP = "https://www.admecindia.co.in/course-sitemap.xml"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CourseDataBuilder/1.0; internal use)"}

# ---------------------------------------------------------------------------
# Rules you can edit
# ---------------------------------------------------------------------------

# Pages whose slug matches are left out (school classes, kids classes, free intros).
# Excluded pages still appear in the review report so you can decide.
EXCLUDE_SLUG = re.compile(r"6th|9th|11th|kids|free-intro|free-computer|_\d+$")  # "_0" = duplicate WordPress page

# First matching rule wins. Matched against the title plus the URL slug, lowercase.
CATEGORY_RULES = [
    ("Architecture & CAD", r"autocad|revit|sketchup|lumion|v-?ray|architect|\bcad\b|textile|draftsman"),
    # Film and video words come before the AI rule, so "AI Filmmaking" is Multimedia, not AI & Data.
    ("Multimedia", r"film|video|premiere|after effects|\bvfx\b|visual effects|animat|photograph"),
    ("AI & Data", r"machine learning|\bai\b|generative|claude|n8n|data analytic|data science|power bi|tableau|statistics|\bsql\b|\bdba\b|database|\bmis\b|excel"),
    ("Marketing", r"\bseo\b|digital marketing|google ads|google analytics|social media"),
    ("Programming", r"web develop|html|\bcss\b|javascript|jquery|react|angular|vue|node|\bmern\b|\bmean\b|\bphp\b|laravel|magento|drupal|wordpress|bootstrap|\bjava\b|python|\bc\b|c plus plus|flutter|ionic|programming|full stack|web design (and )?development|app development|mobile app"),
    ("Multimedia", r"animat|video|premiere|after effects|\bvfx\b|visual effects|motion|maya|3ds|cinema 4d|blender|photograph|filmmaking|final cut|lightroom|multimedia"),
    ("Design", r"photoshop|illustrator|graphic|\bux\b|\bui\b|figma|\bxd\b|sketch|indesign|painting|fine arts|\bart\b|advertising|visual design|creative cloud|web design|responsive|photo.?edit"),
]

# Words dropped from the URL slug to find "the same course in different versions".
# Pages with the same remaining words share a group; the recommender shows one card per group
# and lists the other versions underneath.
GROUP_STOPWORDS = set(
    "best most advanced master plus standard premium short term course courses in at delhi rohini "
    "institute classes class training online certificate certified diploma for the and of beginners "
    "beginner free expert professional professionals adobe autodesk ms microsoft".split()
)
GROUP_NORMALIZE = {"designing": "design", "developer": "development"}
# Whole group names that are really the same subject spelled differently.
GROUP_ALIAS = {"nodejs": "node-js", "c-programming": "c", "core-java": "java"}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def read_sitemap(url: str) -> list[str]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [e.text.strip() for e in root.findall(".//s:loc", ns) if e.text]


def fetch(url: str, cache_dir: Path, delay: float, refresh: bool) -> str | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".html")
    if f.exists() and not refresh:
        return f.read_text(encoding="utf-8", errors="ignore")
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                f.write_text(r.text, encoding="utf-8")
                time.sleep(delay)  # be polite to the server
                return r.text
            print(f"    HTTP {r.status_code}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"    retry {attempt + 1}: {e}", file=sys.stderr)
        time.sleep(2 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Parse one page
# ---------------------------------------------------------------------------

HEADINGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
CAREER_H = re.compile(r"career (options|opportunit|after|path|prospect)|job (role|profile|opportunit)", re.I)
ELIGIBILITY_H = re.compile(r"entry requirement|prerequisite|eligibility", re.I)
OUTCOME_H = re.compile(r"learning outcome|what (will )?you('| wi)?ll? learn", re.I)
TOOLS_H = re.compile(r"\btools\b|software|technolog|what will you learn", re.I)
# "Module 1: X", "Module I- X", "Module 2 – X", "I: X", "3. X". A bare roman numeral needs a space after the
# separator, so "V-Ray Course" is not mistaken for module "V".
MODULE_H = re.compile(
    r"^\s*(?:module\s*(?:\d{1,2}|[ivx]{1,5})\s*[:.)\-–—]\s*|(?:\d{1,2}|[ivx]{1,5})\s*[:.)\-–—]\s+)(.+)$", re.I)
# Headings that introduce a curriculum written as a list or as lines of text instead of "Module N" headings.
STRUCTURE_H = re.compile(r"course structure|structure and modules|\bmodules?\b|syllabus|curriculum|course content", re.I)
STRUCTURE_SKIP = re.compile(r"^\s*(get|download|request|apply|check|understand|know|see|view)\b|^\s*module", re.I)
# Text that is never a job role or a learning outcome.
CAREER_NOISE = re.compile(
    r"after completing|institute|\bcourses?\b|training (institute|course|program)|\bclasses\b|admec|placement|"
    r"salary|hiring|\bjob in\b|apply|enquire|free trial|brochure", re.I)
OUTCOME_NOISE = re.compile(r"hiring|salary|placement|apply|enquire|brochure|free trial|admec", re.I)
# Lead-ins such as "you can work as ..." / "roles such as ..." so only the roles after them are kept.
ROLE_LEAD = re.compile(
    r"\b(?:work(?:ing)? as|becomes? (?:an? )?|such as|including|include|roles? (?:like|as)|"
    r"opportunit(?:y|ies) (?:as|like)|careers? (?:as|in)|jobs? (?:as|like)|positions? (?:as|like)|hired as|placed as)\b[:\s]*",
    re.I)
LABELS = ["Duration", "Course Type", "Training Type", "Training Mode"]


def clean(text: str | None) -> str:
    text = re.sub(r"\[/?gdlr[^\]]*\]?", " ", text or "")  # stray page-builder shortcodes: [gdlr_core_icon icon=...
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def label_value(soup: BeautifulSoup, label: str) -> str:
    """Value for label/value pairs such as 'Duration  06 Months'."""
    inline = re.compile(rf"^\s*{re.escape(label)}\s*:\s*(\S.{{0,58}})$", re.I)  # "Duration: 6 Months"
    for node in soup.find_all(string=inline):
        return clean(inline.match(node).group(1))
    pat = re.compile(rf"^\s*{re.escape(label)}\s*:?\s*$", re.I)
    others = re.compile("|".join(LABELS), re.I)
    for node in soup.find_all(string=pat):
        el = node.parent
        for _ in range(3):  # the value usually lives in the label's parent or grandparent
            if el is None:
                break
            rest = clean(re.sub(re.escape(label), "", el.get_text(" ", strip=True), count=1, flags=re.I)).strip(" :")
            if rest and len(rest) <= 60 and not others.search(rest):
                return rest
            el = el.parent
    for node in soup.find_all(string=pat):  # fallback: the next piece of text after the label
        nxt = node.find_next(string=lambda s: s and clean(s) and clean(s) != clean(node))
        if nxt and len(clean(nxt)) <= 60:
            return clean(nxt)
    return ""


def split_sections(soup: BeautifulSoup) -> list[tuple[str, str, list]]:
    """(heading text, heading tag, tags between this heading and the next heading)."""
    out = []
    for h in soup.find_all(HEADINGS):
        content = []
        for el in h.next_elements:
            if getattr(el, "name", None) in HEADINGS:
                break
            if getattr(el, "name", None):
                content.append(el)
        out.append((clean(h.get_text(" ")), h.name, content))
    return out


def li_texts(content: list) -> list[str]:
    seen: list[str] = []
    for el in content:
        if el.name == "li" and not el.find("li"):
            t = clean(el.get_text(" "))
            if t and t not in seen:
                seen.append(t)
    return seen


def paragraphs(content: list) -> list[str]:
    return [t for t in (clean(el.get_text(" ")) for el in content if el.name == "p") if t]


CTA = re.compile(r"^(apply|enquire|enrol|read more|view|know more|request|download|get |click|call|contact|submit|learn more|explore|book)", re.I)


def leaf_texts(content: list) -> list[str]:
    """Short text blocks at the deepest level of a section (cards, divs, spans).
    Used when a section is built from cards instead of a proper bulleted list."""
    out: list[str] = []
    for el in content:
        if el.name in ("script", "style", "img", "br", "svg"):
            continue
        if any(clean(child.get_text(" ")) for child in el.find_all(True)):
            continue  # a child carries text, so a deeper element will be used instead
        t = clean(el.get_text(" "))
        if 2 <= len(t) <= 60 and not CTA.match(t) and t not in out:
            out.append(t)
    return out


BULLET = re.compile(r"^\s*[•·▪●◦✓✔➤►»]\s*")
# Last words of a job title. Only required when the roles had to be pulled out of plain prose.
ROLE_NOUN = re.compile(
    r"(designer|developer|engineer|analyst|expert|specialist|manager|editor|artist|executive|professional|programmer|"
    r"consultant|administrator|architect|animator|marketer|photographer|writer|strategist|researcher|tester|officer|"
    r"assistant|associate|lead|head|director|trainer|teacher|freelancer|creator|producer|operator|technician|"
    r"draftsman|illustrator|modell?er|scientist|coordinator|supervisor|planner|estimator|surveyor|accountant|auditor|"
    r"influencer|youtuber|blogger|entrepreneur|instructor|tutor|generalist|practitioner|handler)s?$", re.I)
# Context that follows the job title in a sentence: "... a Google Analytics Expert in the industry."
ROLE_TAIL = re.compile(
    r"\s+(?:in|at|across|within|with|for)\s+(?:the\s+)?(?:industry|industries|market|field|companies|"
    r"organi[sz]ations|sector|top\b|leading\b).*$", re.I)
TOOL_NOISE = re.compile(r"\s+(?:courses?|training|classes|class|software|tools?|institute)\b.*$|\s+in\s+(?:delhi|rohini)\b.*$", re.I)


def section_strings(content: list) -> list[str]:
    """Every piece of text in a section, in page order, one entry per line/text node. This reaches text that
    sits directly inside a <p> or <div> next to <br> tags, which the list and card readers above cannot see."""
    out: list[str] = []
    for el in content:
        for s in el.stripped_strings:
            s = clean(s)
            if s and s not in out:
                out.append(s)
    return out


def bullet_lines(content: list) -> list[str]:
    """Lines that start with a typed bullet ("• Monitor your traffic") rather than a real <li>."""
    return [BULLET.sub("", s) for s in section_strings(content) if BULLET.match(s)]


def clean_tool(name: str) -> str:
    """'Photoshop Course in Rohini' -> 'Photoshop', 'miro software' -> 'Miro' (image alt texts are written for SEO)."""
    name = TOOL_NOISE.sub("", name).strip(" -–:")
    return name[:1].upper() + name[1:] if name.islower() else name


def split_roles(text: str, strict: bool = False) -> list[str]:
    """Job titles from one line: a list ("A, B, C etc."), a sentence ("... you can work as a X in the industry")
    or a single title. strict=True is for plain prose, where a piece must also end like a job title."""
    text = BULLET.sub("", clean(text))
    lead = ROLE_LEAD.search(text)
    if lead:
        text = text[lead.end():]
    text = ROLE_TAIL.sub("", text).rstrip(" .")
    parts = [p for p in re.split(r"\s*(?:,|;|\||/)\s*", text) if p.strip()]
    roles: list[str] = []
    for i, part in enumerate(parts):
        halves = re.split(r"\s+(?:and|or|&)\s+", part)
        # "A, B and C" lists the last two; a lone "X and Y" is split only when both sides look like titles.
        if len(halves) > 1 and ((len(parts) > 1 and i == len(parts) - 1) or all(len(h.split()) >= 2 for h in halves)):
            pieces = halves
        else:
            pieces = [part]
        for piece in pieces:
            piece = re.sub(r"^(?:an?|the)\s+", "", re.sub(r"\b(?:etc|and more)\b\.?", "", piece, flags=re.I).strip(" .,;:-"), flags=re.I)
            words = piece.split()
            if not words or len(words) > 5 or CAREER_NOISE.search(piece) or re.search(r"\d", piece):
                continue
            if not lead and piece[0].islower():
                continue  # a sentence fragment, not a title
            if strict and not lead and not ROLE_NOUN.search(piece):
                continue
            piece = piece[0].upper() + piece[1:]
            if piece.lower() not in {r.lower() for r in roles}:
                roles.append(piece)
    return roles


def extract_careers(content: list) -> list[str]:
    """Job titles from a careers section. Tries a real list, then cards, then lines of text, then prose;
    the first way that yields real titles wins."""
    for items, strict in ((li_texts(content), False), (leaf_texts(content), False), (section_strings(content), True)):
        roles: list[str] = []
        for item in items:
            for role in split_roles(item, strict):
                if role.lower() not in {r.lower() for r in roles}:
                    roles.append(role)
        if roles:
            return roles
    return []


def curriculum_items(content: list) -> tuple[list[str], bool]:
    """Curriculum with no 'Module N' headings: a bullet list, typed bullets, or (last resort) prose.
    Returns (items, is_prose). Prose is kept as one long entry: it is used for matching, never shown."""
    raw_items = li_texts(content) or bullet_lines(content)
    raw_items = [re.sub(r"^(?:module|unit|day|week|session|lesson)\s*\d+\s*[:.\-–—]\s*", "", t, flags=re.I) for t in raw_items]
    items = tidy_list(raw_items, max_words=10, limit=12)
    if items:
        return items, False
    prose = " ".join(s for s in section_strings(content) if len(s.split()) >= 4 and not CAREER_NOISE.search(s))
    return ([shorten(prose, 400)] if prose else []), True


def img_alts(content: list) -> list[str]:
    out = []
    for el in content:
        if el.name == "img":
            alt = clean(el.get("alt"))
            alt = re.sub(r"\b(logo|icon|image|img|badge|png|jpe?g|svg|webp)\b", "", alt, flags=re.I)
            alt = clean_tool(clean(re.sub(r"[-_]+", " ", alt)))
            if 2 <= len(alt) <= 40 and "admec" not in alt.lower() and alt.lower() not in {a.lower() for a in out}:
                out.append(alt)
    return out


def tidy_list(items: list[str], max_words: int = 7, limit: int = 10) -> list[str]:
    out: list[str] = []
    for it in items:
        it = clean(re.sub(r"\b(etc|and more)\b\.?", "", it, flags=re.I)).strip(" .,;:-")
        if it and len(it.split()) <= max_words and it.lower() not in {o.lower() for o in out}:
            out.append(it)
    return out[:limit]


def meta(soup: BeautifulSoup, **attrs) -> str:
    tag = soup.find("meta", attrs=attrs)
    return clean(tag.get("content")) if tag and tag.get("content") else ""


def shorten(text: str, limit: int = 260) -> str:
    text = clean(text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if "." in cut[80:]:
        return cut[: cut.rfind(".") + 1]
    return cut[: cut.rfind(" ")].rstrip(" ,;:") + "…"


def parse_page(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # Read page-level fields first, before boilerplate is stripped.
    h1 = soup.find("h1")
    title = clean(h1.get_text(" ")) if h1 else meta(soup, property="og:title")
    description = meta(soup, property="og:description") or meta(soup, name="description")
    description = re.sub(r"\s*[|\-–]\s*ADMEC.*$", "", description, flags=re.I)
    labels = {name: label_value(soup, name) for name in LABELS}

    for tag in soup(["script", "style", "noscript", "nav", "footer", "form", "iframe", "svg"]):
        tag.decompose()
    sections = split_sections(soup)

    careers, eligibility, outcomes, tools, topics, structure = [], [], [], [], [], []
    for text, level, content in sections:
        if CAREER_H.search(text):
            careers += extract_careers(content)
        if ELIGIBILITY_H.search(text):
            eligibility += li_texts(content) or paragraphs(content)
        if OUTCOME_H.search(text):
            outcomes += [o for o in (li_texts(content) or bullet_lines(content)) if not OUTCOME_NOISE.search(o)]
        if TOOLS_H.search(text):
            tools += img_alts(content) + [clean_tool(t) for t in li_texts(content) if len(t.split()) <= 4]
        m = MODULE_H.match(text)
        if m and level in ("h3", "h4"):
            topics.append(clean(m.group(1)))
        elif STRUCTURE_H.search(text) and not STRUCTURE_SKIP.search(text):
            structure.append(content)

    topics = tidy_list(topics, max_words=12, limit=20)
    if not topics:  # no "Module N" headings: read the curriculum as a list, typed bullets or prose
        for content in structure:
            items, _is_prose = curriculum_items(content)
            if items:
                topics = items
                break

    if len(description) < 60:  # fall back to the first real paragraph
        for _, _, content in sections:
            long_p = [p for p in paragraphs(content) if len(p) >= 80]
            if long_p:
                description = long_p[0]
                break

    return {
        "url": url,
        "slug": url.rstrip("/").split("/")[-1],
        "source_title": title,
        "description": shorten(description),
        "duration_text": labels["Duration"],
        "course_type": labels["Course Type"],
        "training_type": labels["Training Type"],
        "training_mode": labels["Training Mode"],
        "tools": tidy_list(tools, max_words=4, limit=12),
        "outcomes": tidy_list(outcomes, max_words=8, limit=12),
        "topics": topics,
        "careers": tidy_list(careers, max_words=6, limit=8),
        "eligibility": tidy_list(eligibility, max_words=30, limit=6),
    }


# ---------------------------------------------------------------------------
# Raw page data -> recommender course
# ---------------------------------------------------------------------------

def clean_title(t: str) -> str:
    t = re.sub(r"^(best|top)\s+", "", t, flags=re.I)
    t = re.sub(r"\bbest\s+", "", t, flags=re.I)  # "...with Best Adobe Premiere Pro Course"
    for _ in range(2):  # sales phrases in front of the real name: "Join the", "Get Certified with the", "No.1" ...
        t = re.sub(r"^(#\s*1|no\.?\s*1)\s+", "", t, flags=re.I)
        t = re.sub(r"^(join the|get certified with the|get certified with|learn with|get your|institute for|the)\s+", "", t, flags=re.I)
    t = re.sub(r"\s+(in|at|from)\s+(delhi|rohini|ncr|india)\b.*$", "", t, flags=re.I)
    t = re.sub(r"\s*[|\-–]\s*ADMEC.*$", "", t, flags=re.I)
    return t.strip(" ,:-") or t


def normalize_duration(text: str) -> tuple[str, float | None]:
    text = text.replace("½", ".5").replace("¼", ".25").replace("¾", ".75")  # "1½ Months" -> "1.5 Months"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(year|month|week|day)s?", text, re.I)
    if not m:
        return clean(text), None
    n, unit = float(m.group(1)), m.group(2).lower()
    months = n * {"year": 12, "month": 1, "week": 0.25, "day": 0.033}[unit]
    return f"{n:g} {unit}{'' if n == 1 else 's'}", months


def infer_level(slug: str, months: float | None, title: str = "") -> tuple[str, str]:
    """Most subjects are sold as a ladder: Standard < Premium / Master < Master Plus.
    The tier named in the page title is the most reliable signal, so it is read first, then the slug
    (its "advanced-" prefix is also used on plain Master courses, so it is not trusted alone), then duration."""
    for text, source in ((title.lower(), "title"), (slug.lower().replace("-", " "), "slug")):
        if re.search(r"master plus", text):
            return "Advanced", source
        if re.search(r"\bstandard\b|short.term|\bcertificate\b", text):
            return "Beginner", source
        if source == "title" and re.search(r"\badvanced\b", text):
            return "Advanced", source
        if re.search(r"\bmaster\b|premium|expert", text):
            return "Intermediate", source
        if "diploma" in text:
            return ("Advanced" if months is not None and months >= 10 else "Intermediate"), source
        if re.search(r"beginner|basic|\bintro", text):
            return "Beginner", source
    if "advanced" in slug:
        return "Advanced", "slug"
    if months is not None:
        return ("Beginner" if months <= 3 else "Intermediate" if months <= 8 else "Advanced"), "duration"
    return "Intermediate", "default"


def infer_category(title: str, slug: str) -> str:
    text = f"{title} {slug.replace('-', ' ')}".lower()
    # "AI Integrated Java" is a Java course that mentions AI, not an AI course.
    text = re.sub(r"\bai[- ](integrated|packed|powered|enhanced|driven)\b|with artificial intelligence", " ", text)
    for category, pattern in CATEGORY_RULES:
        if re.search(pattern, text):
            return category
    return "Other"


def group_key(slug: str) -> str:
    words = [GROUP_NORMALIZE.get(w, w) for w in re.split(r"[^a-z0-9]+", slug.lower()) if w]
    kept = [w for w in words if w not in GROUP_STOPWORDS and not w.isdigit()]
    key = "-".join(kept) or slug
    return GROUP_ALIAS.get(key, key)


def build_skills(raw: dict) -> list[str]:
    """Tools first, then short learning outcomes. Module titles are only a fallback,
    because they already help matching through the 'topics' field."""
    skills: list[str] = []

    def add(items: list[str]) -> None:
        for item in items:
            item = clean(item).strip(" .;:-")
            if len(skills) < 8 and 2 <= len(item) <= 40 and len(item.split()) <= 5 \
                    and item.lower() not in {s.lower() for s in skills}:
                skills.append(item)

    add(raw["tools"][:6])
    add(raw["outcomes"])
    if len(skills) < 3:
        add(raw["topics"])
    return skills


def to_course(raw: dict, override: dict) -> tuple[dict, dict]:
    """Returns (course, review info)."""
    duration, months = normalize_duration(raw["duration_text"])
    level, level_source = infer_level(raw["slug"], months, raw["source_title"])
    title = clean_title(raw["source_title"]) or raw["slug"].replace("-", " ").title()
    course = {
        "id": raw["slug"],
        "title": title,
        "category": infer_category(raw["source_title"], raw["slug"]),
        "level": level,
        "duration": duration,
        "description": raw["description"],
        "skills": build_skills(raw),
        # Long learning outcomes make poor skill chips, but they are good matching text, so they join the topics.
        "topics": raw["topics"] + [o for o in raw["outcomes"] if len(o.split()) > 5 and o not in raw["topics"]],
        "careers": raw["careers"],
        "url": raw["url"],
        "group": group_key(raw["slug"]),
    }
    for key, value in override.items():
        if key in course:
            course[key] = value
    flags = [f"no_{k}" for k in ("duration", "description", "skills", "topics", "careers") if not course[k]]
    if course["category"] == "Other":
        flags.append("category_other")
    return course, {"level_source": level_source, "flags": flags}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_outputs(rows: list[dict], out: Path, raw_out: Path, report: Path) -> None:
    for p in (out, raw_out, report):
        p.parent.mkdir(parents=True, exist_ok=True)
    kept = [r["course"] for r in rows if not r["excluded"] and r["course"]]
    out.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
    raw_out.write_text(json.dumps([r["raw"] for r in rows if r["raw"]], indent=2, ensure_ascii=False), encoding="utf-8")
    with report.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["slug", "included", "title", "category", "level", "level_source", "duration",
                    "skills", "topics", "careers", "group", "flags"])
        for r in rows:
            c = r["course"] or {}
            flags = list(r["flags"]) + (["excluded"] if r["excluded"] else []) + (["download_failed"] if not r["raw"] else [])
            w.writerow([r["slug"], "no" if r["excluded"] or not r["raw"] else "yes", c.get("title", ""),
                        c.get("category", ""), c.get("level", ""), r["level_source"], c.get("duration", ""),
                        len(c.get("skills", [])), len(c.get("topics", [])), len(c.get("careers", [])),
                        c.get("group", ""), " ".join(flags)])


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # so "–" prints correctly when output is redirected to a file on Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sitemap", default=SITEMAP)
    ap.add_argument("--url", help="process only this course page")
    ap.add_argument("--limit", type=int, help="process only the first N pages (trial run)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds to wait between downloads")
    ap.add_argument("--out", default="data/courses.generated.json")
    ap.add_argument("--raw", default="data/courses_raw.json")
    ap.add_argument("--report", default="data/review_report.csv")
    ap.add_argument("--overrides", default="data/overrides.json")
    ap.add_argument("--cache", default=".cache/pages")
    ap.add_argument("--refresh", action="store_true", help="download pages again instead of using the cache")
    ap.add_argument("--keep-excluded", action="store_true", help="do not apply the EXCLUDE_SLUG rule")
    ap.add_argument("--debug", action="store_true", help="print everything found on the page (use with --url)")
    args = ap.parse_args()

    overrides = {}
    if Path(args.overrides).exists():
        overrides = json.loads(Path(args.overrides).read_text(encoding="utf-8"))

    urls = [args.url] if args.url else [u for u in read_sitemap(args.sitemap) if "/course/" in u]
    if args.limit:
        urls = urls[: args.limit]
    print(f"{len(urls)} course pages to process")

    rows = []
    for i, url in enumerate(urls, 1):
        slug = url.rstrip("/").split("/")[-1]
        print(f"[{i}/{len(urls)}] {slug}")
        html = fetch(url, Path(args.cache), args.delay, args.refresh)
        if html is None:
            rows.append({"slug": slug, "raw": None, "course": None, "excluded": False, "level_source": "", "flags": []})
            continue
        raw = parse_page(html, url)
        ov = overrides.get(slug, {})
        course, info = to_course(raw, ov)
        excluded = (bool(EXCLUDE_SLUG.search(slug)) and not args.keep_excluded) or bool(ov.get("exclude"))
        if ov.get("include"):
            excluded = False
        if args.debug:
            soup = BeautifulSoup(html, "html.parser")
            print("\n--- headings found ---")
            for text, level, content in split_sections(soup):
                print(f"{level}: {text[:90]}  [{len(li_texts(content))} list items, {len(leaf_texts(content))} text blocks, {len(img_alts(content))} images]")
                if CAREER_H.search(text) or (STRUCTURE_H.search(text) and not STRUCTURE_SKIP.search(text)):
                    print(f"      text lines in this section: {section_strings(content)[:10]}")
            print("\n--- extracted ---")
            print(json.dumps({"raw": raw, "course": course, **info}, indent=2, ensure_ascii=False))
        rows.append({"slug": slug, "raw": raw, "course": course, "excluded": excluded, **info})

    if args.debug and args.url:
        return 0

    write_outputs(rows, Path(args.out), Path(args.raw), Path(args.report))

    included = [r for r in rows if r["course"] and not r["excluded"]]
    failed = [r["slug"] for r in rows if not r["raw"]]
    groups: dict[str, list[str]] = {}
    for r in included:
        groups.setdefault(r["course"]["group"], []).append(r["slug"])
    print(f"\nWrote {len(included)} courses to {args.out}")
    print(f"Excluded by rule: {sum(1 for r in rows if r['excluded'])} | download failed: {len(failed)}")
    for flag in ("no_description", "no_skills", "no_topics", "no_careers", "no_duration", "category_other"):
        n = sum(1 for r in included if flag in r["flags"])
        if n:
            print(f"  {flag}: {n} courses (see {args.report})")
    multi = {g: s for g, s in groups.items() if len(s) > 1}
    print(f"{len(groups)} distinct course groups; {len(multi)} groups have several versions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())