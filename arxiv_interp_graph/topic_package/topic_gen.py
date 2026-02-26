"""
Step B: Topic generation — LLM (structured JSON) + fallback (keyword-based).
"""

import asyncio
import json
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

ABSTRACT_MAX_CHARS = 900


def _truncate_abstract(abstract: str, max_chars: int = ABSTRACT_MAX_CHARS) -> str:
    if not abstract:
        return ""
    if len(abstract) <= max_chars:
        return abstract
    return abstract[: max_chars - 3].rsplit(" ", 1)[0] + "..."


def _papers_to_input_text(papers: List[Dict[str, Any]]) -> str:
    """Format papers for LLM: title + truncated abstract per paper."""
    lines = []
    for i, p in enumerate(papers, 1):
        pid = p.get("paper_id", "")
        title = (p.get("title") or "").strip()
        abstract = _truncate_abstract(p.get("abstract") or "")
        year = p.get("year", "")
        lines.append(f"[{i}] paper_id: {pid}")
        lines.append(f"    title: {title}")
        if abstract:
            lines.append(f"    abstract: {abstract}")
        if year:
            lines.append(f"    year: {year}")
        lines.append("")
    return "\n".join(lines)


SCHEMA_PROMPT = """You are a research synthesizer. Given a list of papers (with paper_id, title, abstract, year), produce a single coherent topic package in JSON only. No markdown, no explanation.

Output a single JSON object with exactly these keys (use empty arrays/strings where needed):
- topic_title: object with "main" (string), "alt1" (string), "alt2" (string)
- one_sentence_thesis: string
- scope_in: string (what this topic package covers)
- scope_out: string (what it does not cover, to avoid scope creep)
- keywords: array of 10-15 strings
- subthemes: array of 3-5 objects, each { "name": string, "paper_ids": array of 2-4 paper_id strings }
- outline: array of 6-8 objects, each { "section": string, "paper_ids": array of paper_id strings }
- reading_order: array of paper_id strings (recommended reading order)
- confidence: number between 0 and 1 (your confidence in this synthesis)
"""


async def generate_topic_llm(
    papers: List[Dict[str, Any]],
    generate_fn: Callable[[str, str], Any],
) -> Optional[Dict[str, Any]]:
    """
    Call LLM with papers text; expect JSON. generate_fn(system_message, user_prompt) should return response text.
    """
    text = _papers_to_input_text(papers)
    user = f"Papers:\n{text}\n\nProduce the topic package JSON as specified."
    try:
        if asyncio.iscoroutinefunction(generate_fn):
            out = await generate_fn(SCHEMA_PROMPT, user)
        else:
            out = generate_fn(SCHEMA_PROMPT, user)
    except Exception:
        return None
    if not out:
        return None
    out = (out or "").strip()
    # Strip markdown code block if present
    if out.startswith("```"):
        out = re.sub(r"^```\w*\n?", "", out)
        out = re.sub(r"\n?```\s*$", "", out)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def _fallback_keywords(papers: List[Dict[str, Any]], top_k: int = 12) -> List[str]:
    stop = {"the", "a", "an", "and", "or", "of", "in", "for", "on", "with", "to", "from", "by", "using", "via"}
    counts: Counter = Counter()
    for p in papers:
        title = (p.get("title") or "").lower()
        abstract = (p.get("abstract") or "").lower()
        for word in re.findall(r"[a-z0-9\-]+", title + " " + abstract):
            if len(word) >= 2 and word not in stop:
                counts[word] += 1
    return [w for w, _ in counts.most_common(top_k)]


def generate_topic_fallback(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Non-LLM fallback: keyword-based topic_title, simple structure."""
    if not papers:
        return _empty_package()
    keywords = _fallback_keywords(papers, 12)
    main_title = " ".join(keywords[:6]) if keywords else "Literature Topic"
    paper_ids = [p.get("paper_id", "") for p in papers if p.get("paper_id")]
    return {
        "topic_title": {"main": main_title, "alt1": "", "alt2": ""},
        "one_sentence_thesis": f"Synthesized topic from {len(papers)} papers: {main_title}.",
        "scope_in": "Papers in the selected sample.",
        "scope_out": "Papers outside the sample.",
        "keywords": keywords,
        "subthemes": [{"name": "Core papers", "paper_ids": paper_ids[:4]}],
        "outline": (
            [{"section": "Overview", "paper_ids": paper_ids[:3]}]
            + [
                {"section": f"Section {i+2}", "paper_ids": paper_ids[3+i*2:3+(i+1)*2]}
                for i in range(min(5, max(0, (len(paper_ids) - 3) // 2)))
            ]
        ),
        "reading_order": paper_ids,
        "confidence": 0.5,
    }


def _empty_package() -> Dict[str, Any]:
    return {
        "topic_title": {"main": "", "alt1": "", "alt2": ""},
        "one_sentence_thesis": "",
        "scope_in": "",
        "scope_out": "",
        "keywords": [],
        "subthemes": [],
        "outline": [],
        "reading_order": [],
        "confidence": 0.0,
    }
