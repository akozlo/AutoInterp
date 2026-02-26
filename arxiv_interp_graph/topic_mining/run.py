"""
Main entry: load graph, run topic generation (graph / embed / hybrid), output topics.json + report.
Idempotent and cache-aware: skip recompute if outputs exist and --force not set.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

from .communities import (
    DETERMINISTIC_SEED,
    communities_embedding,
    communities_graph,
    communities_hybrid,
)
from .embeddings import embed_papers_cached, get_embedder
from .graph_loader import load_graph_for_topic_mining
from .naming import (
    extract_keywords_tf,
    representative_papers,
    topic_name_from_keywords,
)
from .scoring import topic_importance_score


def _text_for_node(G: nx.DiGraph, nid: str) -> str:
    """Title; if abstract in attrs, title + abstract."""
    attrs = G.nodes.get(nid, {})
    title = (attrs.get("title") or "").strip()
    abstract = (attrs.get("abstract") or "").strip()
    if abstract:
        return f"{title} {abstract}".strip()
    return title


def run_topic_mining(
    graph_path: str | Path,
    output_dir: str | Path,
    topic_mode: str = "graph",
    hybrid_mode: str = "union",
    knn_k: int = 10,
    sim_threshold: float = 0.3,
    resolution: float = 1.0,
    embedder_backend: str = "sentence-transformers",
    embedder_model: Optional[str] = None,
    embed_cache_dir: Optional[str | Path] = None,
    top_representatives: int = 10,
    top_keywords: int = 10,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Run topic mining and write topics.json, topics_report.md, optional per-topic subgraphs.
    Returns dict with keys: topics (list), output_paths (dict).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    topics_path = output_dir / "topics.json"

    if not force and topics_path.exists():
        with open(topics_path) as f:
            data = json.load(f)
        return {"topics": data, "output_paths": {"topics_json": str(topics_path)}}

    G = load_graph_for_topic_mining(graph_path)
    node_ids = list(G.nodes())
    if not node_ids:
        result = {"topics": [], "output_paths": {}}
        with open(topics_path, "w") as f:
            json.dump([], f, indent=2)
        return result

    communities: List[List[str]] = []
    method_used = topic_mode

    if topic_mode == "graph":
        communities = communities_graph(G, resolution=resolution, seed=DETERMINISTIC_SEED)
    elif topic_mode == "embed":
        cache_dir = Path(embed_cache_dir or output_dir / "embeddings")
        cache_dir.mkdir(parents=True, exist_ok=True)
        texts = [_text_for_node(G, nid) for nid in node_ids]
        embedder = get_embedder(backend=embedder_backend, model=embedder_model)
        vectors = embed_papers_cached(node_ids, texts, embedder, cache_dir)
        emb = np.array(vectors, dtype=np.float32)
        communities = communities_embedding(
            emb, node_ids, method="hdbscan", min_cluster_size=5, seed=DETERMINISTIC_SEED
        )
    elif topic_mode == "hybrid":
        cache_dir = Path(embed_cache_dir or output_dir / "embeddings")
        cache_dir.mkdir(parents=True, exist_ok=True)
        texts = [_text_for_node(G, nid) for nid in node_ids]
        embedder = get_embedder(backend=embedder_backend, model=embedder_model)
        vectors = embed_papers_cached(node_ids, texts, embedder, cache_dir)
        emb = np.array(vectors, dtype=np.float32)
        communities = communities_hybrid(
            G, emb, node_ids,
            hybrid_mode=hybrid_mode,
            knn_k=knn_k,
            sim_threshold=sim_threshold,
            resolution=resolution,
            seed=DETERMINISTIC_SEED,
        )
    else:
        raise ValueError(f"Unknown topic_mode: {topic_mode}")

    topics_payload: List[Dict[str, Any]] = []
    for i, comm in enumerate(communities):
        if not comm:
            continue
        keywords = extract_keywords_tf(G, comm, top_k=top_keywords)
        name = topic_name_from_keywords(keywords, max_words=5)
        score = topic_importance_score(G, comm)
        reps = representative_papers(G, comm, top_n=top_representatives, use_pagerank=True)
        topics_payload.append({
            "topic_id": f"topic_{i}",
            "method": method_used,
            "size": len(comm),
            "score": round(score, 4),
            "keywords": keywords,
            "representatives": reps,
            "node_ids": comm,
        })

    topics_payload.sort(key=lambda x: (-x["score"], -x["size"]))

    to_dump = [{k: v for k, v in t.items() if k != "node_ids"} for t in topics_payload]
    with open(topics_path, "w") as f:
        json.dump(to_dump, f, indent=2)

    report_path = output_dir / "topics_report.md"
    _write_report(topics_payload, report_path, output_dir)

    subgraph_dir = output_dir / "subgraphs"
    subgraph_dir.mkdir(parents=True, exist_ok=True)
    def _sanitize(H):
        H = H.copy()
        for _, attrs in H.nodes(data=True):
            for k, v in list(attrs.items()):
                if v is None:
                    attrs[k] = ""
                elif not isinstance(v, (str, int, float, bool)):
                    attrs[k] = str(v)
        return H

    for t in topics_payload:
        tid = t["topic_id"]
        nodes = t.get("node_ids") or []
        if not nodes:
            continue
        sub = G.subgraph(nodes)
        path = subgraph_dir / f"{tid}.graphml"
        try:
            nx.write_graphml(_sanitize(sub), path)
        except Exception:
            pass

    return {
        "topics": topics_payload,
        "output_paths": {
            "topics_json": str(topics_path),
            "topics_report_md": str(report_path),
            "subgraphs_dir": str(subgraph_dir),
        },
    }


def _write_report(topics: List[Dict], report_path: Path, output_dir: Path) -> None:
    lines = [
        "# Topic Mining Report",
        "",
        "| Topic ID | Name (keywords) | Size | Score |",
        "|----------|-----------------|------|-------|",
    ]
    for t in topics[:30]:
        name = topic_name_from_keywords(t.get("keywords", [])[:5], max_words=5)
        lines.append(f"| {t['topic_id']} | {name} | {t['size']} | {t['score']:.3f} |")
    lines.extend(["", "## Top representatives per topic", ""])
    for t in topics[:15]:
        lines.append(f"### {t['topic_id']} (size={t['size']}, score={t['score']:.3f})")
        for r in (t.get("representatives") or [])[:5]:
            title = (r.get("title") or "")[:70]
            lines.append(f"- {title} (in_degree={r.get('in_degree', 0)}, year={r.get('year')})")
        lines.append("")
    lines.append("## Subgraphs")
    lines.append(f"Per-topic subgraphs (GraphML): `{output_dir / 'subgraphs'}/`")
    report_path.write_text("\n".join(lines))