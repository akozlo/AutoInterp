"""Topic naming, representative papers, and keyword extraction (TF-IDF style)."""

import math
import re
from collections import Counter
from typing import Any, Dict, List

import networkx as nx


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9\-]+", (text or "").lower())
    stop = {
        "the", "a", "an", "and", "or", "of", "in", "for", "on",
        "with", "to", "from", "by", "using", "via", "into",
    }
    return [t for t in tokens if len(t) >= 2 and t not in stop]


def extract_keywords_tf(
    G: nx.DiGraph,
    node_ids: List[str],
    top_k: int = 10,
) -> List[str]:
    """
    Extract keywords from titles (and abstract if in node attrs) using term frequency.
    Returns top_k terms by frequency.
    """
    all_tokens: List[str] = []
    for nid in node_ids:
        if nid not in G.nodes:
            continue
        attrs = G.nodes[nid]
        title = (attrs.get("title") or "").strip()
        abstract = (attrs.get("abstract") or "").strip()
        text = f"{title} {abstract}".strip()
        all_tokens.extend(_tokenize(text))
    if not all_tokens:
        return []
    counts = Counter(all_tokens)
    return [t for t, _ in counts.most_common(top_k)]


def topic_name_from_keywords(keywords: List[str], max_words: int = 5) -> str:
    """Short topic name from top keywords."""
    return " ".join(keywords[:max_words]) if keywords else "Unnamed Topic"


def representative_papers(
    G: nx.DiGraph,
    node_ids: List[str],
    top_n: int = 10,
    use_pagerank: bool = False,
) -> List[Dict[str, Any]]:
    """
    Rank papers by in_degree (within graph) + citation_count; optionally add PageRank.
    Returns list of dicts: paper_id, title, year, citation_count, in_degree, url.
    """
    sub = G.subgraph(node_ids)
    if sub.number_of_nodes() == 0:
        return []
    in_deg = dict(sub.in_degree())
    scores: List[tuple] = []
    for nid in node_ids:
        attrs = G.nodes.get(nid, {})
        ccount = attrs.get("citation_count") or 0
        indeg = in_deg.get(nid, 0)
        # combined: in_degree (strong signal in-graph) + log(1+citation_count) for global impact
        s = indeg + 0.5 * (1 + math.log(1 + ccount))
        scores.append((s, nid))
    if use_pagerank and sub.number_of_edges() > 0:
        try:
            pr = nx.pagerank(sub)
            scores = [(s + 0.3 * pr.get(nid, 0), nid) for s, nid in scores]
        except Exception:
            pass
    scores.sort(reverse=True, key=lambda x: x[0])
    out: List[Dict[str, Any]] = []
    for _, nid in scores[:top_n]:
        attrs = G.nodes.get(nid, {})
        out.append({
            "paper_id": nid,
            "title": attrs.get("title") or "",
            "year": attrs.get("year"),
            "citation_count": attrs.get("citation_count") or 0,
            "in_degree": in_deg.get(nid, 0),
            "url": attrs.get("url"),
        })
    return out
