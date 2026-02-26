"""
Step A: Sampling — random_seed (PPR from weighted seed) and community (Louvain/Leiden + representative + random).
"""

import random
from typing import Any, Dict, List, Optional

import networkx as nx

try:
    from ..topic_mining.communities import communities_graph
except ImportError:
    from topic_mining.communities import communities_graph

DEFAULT_SEED = 42


def _paper_record(G: nx.DiGraph, nid: str) -> Dict[str, Any]:
    """Build paper dict: paper_id, title, abstract, year, url, citation_count."""
    attrs = G.nodes.get(nid, {})
    return {
        "paper_id": nid,
        "title": (attrs.get("title") or "").strip(),
        "abstract": (attrs.get("abstract") or "").strip(),
        "year": attrs.get("year"),
        "url": attrs.get("url"),
        "citation_count": attrs.get("citation_count") or 0,
    }


def sample_random_seed(
    G: nx.DiGraph,
    k: int,
    year_min: Optional[int] = 2012,
    min_in_degree: int = 0,
    use_pagerank_weight: bool = False,
    ppr_alpha: float = 0.85,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """
    Sample one seed node (weighted by in_degree+1 or PageRank), then expand to k papers via PPR.
    Returns list of paper records (paper_id, title, abstract, year, url, citation_count).
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    in_deg = dict(G.in_degree())
    # Filter
    if year_min is not None:
        nodes = [n for n in nodes if (G.nodes[n].get("year") or 0) >= year_min]
    if min_in_degree > 0:
        nodes = [n for n in nodes if in_deg.get(n, 0) >= min_in_degree]
    if not nodes:
        return []

    if use_pagerank_weight:
        try:
            pr = nx.pagerank(G)
            weights = [pr.get(n, 0) + 1e-6 for n in nodes]
        except Exception:
            weights = [in_deg.get(n, 0) + 1 for n in nodes]
    else:
        weights = [in_deg.get(n, 0) + 1 for n in nodes]
    total = sum(weights)
    if total <= 0:
        return []
    r = rng.random() * total
    for i, n in enumerate(nodes):
        r -= weights[i]
        if r <= 0:
            seed_node = n
            break
    else:
        seed_node = nodes[-1]

    # PPR from seed
    try:
        ppr = nx.pagerank(G, alpha=ppr_alpha, personalization={seed_node: 1.0})
    except Exception:
        # Fallback: 1-hop + 2-hop neighbors, sort by in_degree
        one_hop = set(G.successors(seed_node)) | set(G.predecessors(seed_node)) | {seed_node}
        two_hop = set(one_hop)
        for n in list(one_hop):
            two_hop.update(G.successors(n))
            two_hop.update(G.predecessors(n))
        candidates = [(n, in_deg.get(n, 0) + (1 if n == seed_node else 0)) for n in two_hop]
        candidates.sort(key=lambda x: -x[1])
        top_k_ids = [n for n, _ in candidates[:k]]
        return [_paper_record(G, nid) for nid in top_k_ids]

    sorted_nodes = sorted(ppr.keys(), key=lambda x: -ppr[x])
    top_k_ids = [n for n in sorted_nodes if n in G.nodes][:k]
    return [_paper_record(G, nid) for nid in top_k_ids]


def sample_community(
    G: nx.DiGraph,
    k: int,
    community_id: Optional[int] = None,
    core_ratio: float = 0.4,
    resolution: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """
    Run Louvain/Leiden, pick one community (largest by default or community_id), then sample k papers:
    top core_ratio*k by in_degree+citation_count as core, rest random.
    """
    communities = communities_graph(G, resolution=resolution, seed=seed)
    if not communities:
        return []
    # Sort by size descending; pick by community_id (0 = largest)
    communities = sorted(communities, key=len, reverse=True)
    idx = community_id if community_id is not None else 0
    idx = max(0, min(idx, len(communities) - 1))
    comm = communities[idx]
    if not comm:
        return []

    in_deg = dict(G.in_degree())
    def score(n):
        attrs = G.nodes.get(n, {})
        return in_deg.get(n, 0) + (attrs.get("citation_count") or 0) / 1000.0

    scored = [(n, score(n)) for n in comm]
    scored.sort(key=lambda x: -x[1])
    m = max(1, int(k * core_ratio))
    core_ids = [n for n, _ in scored[:m]]
    rest = [n for n, _ in scored[m:] if n not in core_ids]
    rng = random.Random(seed)
    rng.shuffle(rest)
    fill = k - len(core_ids)
    chosen_ids = core_ids + rest[:fill]
    return [_paper_record(G, nid) for nid in chosen_ids]
