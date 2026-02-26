"""
Build a 3-paper context pack: 1 seed + 1–2 forward (citing seed) + 1 backward (cited by seed).
Forward/backward from graph; if forward too few, fill via S2 "citing papers" API.
"""

import random
from typing import Any, Dict, List, Optional

import networkx as nx


def _node_to_paper(G: nx.DiGraph, nid: str, relation: str, source: str = "graph") -> Dict[str, Any]:
    attrs = G.nodes.get(nid, {})
    return {
        "paperId": nid,
        "title": (attrs.get("title") or "").strip(),
        "year": attrs.get("year"),
        "abstract": (attrs.get("abstract") or "").strip(),
        "relation": relation,
        "source": source,
    }


def _s2_citing_to_paper(item: dict, relation: str = "forward", source: str = "s2") -> Dict[str, Any]:
    return {
        "paperId": (item.get("paperId") or "").strip(),
        "title": (item.get("title") or "").strip(),
        "year": item.get("year"),
        "abstract": (item.get("abstract") or "").strip(),
        "relation": relation,
        "source": source,
    }


def build_context_pack(
    G: nx.DiGraph,
    seed_id: Optional[str] = None,
    s2_client: Optional[Any] = None,
    n_forward: int = 2,
    n_backward: int = 1,
    min_forward_from_graph: int = 0,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Assemble context pack = {seed + 2 others} (fixed 3 papers).

    - Forward = papers that cite the seed (in graph: predecessors of seed).
    - If graph has none or too few forward, call S2 "citing papers" to fill.
    - From forward pick 1–2 (simple: random). If no forward, we still try to fill with backward.
    - Backward = papers the seed cites (in graph: successors of seed). Pick 1, then fill rest from backward until we have 3 total.

    Returns list of 3 paper dicts with keys: paperId, title, year, abstract, relation (seed|forward|backward), source (graph|s2).
    """
    # When seed is None: new RNG each run (different papers every time). When seed is set: reproducible.
    if seed is None:
        rng = random.Random()
    else:
        rng = random.Random(seed)
    nodes = list(G.nodes())
    if not nodes:
        return []

    # 1) Choose seed
    if seed_id is None or seed_id not in G:
        seed_id = rng.choice(nodes)
    pack: List[Dict[str, Any]] = [_node_to_paper(G, seed_id, "seed", "graph")]

    # 2) Forward = papers that cite the seed (predecessors in G: edge X->seed means X cites seed)
    forward_in_G = list(G.predecessors(seed_id))
    forward_candidates: List[Dict[str, Any]] = [_node_to_paper(G, n, "forward", "graph") for n in forward_in_G]

    if s2_client and len(forward_candidates) < min_forward_from_graph:
        try:
            citing = s2_client.get_citations(seed_id)
            for item in citing or []:
                pid = item.get("paperId")
                if not pid or any(p["paperId"] == pid for p in forward_candidates):
                    continue
                forward_candidates.append(_s2_citing_to_paper(item, "forward", "s2"))
        except Exception:
            pass

    # 3) Pick 1–2 from forward (simple: random; if not enough, take all)
    n_from_forward = min(n_forward, len(forward_candidates))
    if n_from_forward > 0:
        chosen_forward = rng.sample(forward_candidates, n_from_forward)
        pack.extend(chosen_forward)

    # 4) Backward = papers the seed cites (successors in G)
    backward_in_G = list(G.successors(seed_id))
    backward_candidates = [_node_to_paper(G, n, "backward", "graph") for n in backward_in_G]

    # 5) Fill to 3: add 1 from backward (or more until we have 3)
    need = 3 - len(pack)
    if need > 0 and backward_candidates:
        # Avoid duplicates
        existing_ids = {p["paperId"] for p in pack}
        available = [p for p in backward_candidates if p["paperId"] not in existing_ids]
        n_take = min(need, len(available))
        if n_take > 0:
            chosen_backward = rng.sample(available, n_take)
            pack.extend(chosen_backward)

    # If still short (e.g. no backward), fill from remaining forward/backward
    need = 3 - len(pack)
    if need > 0:
        existing_ids = {p["paperId"] for p in pack}
        rest_forward = [p for p in forward_candidates if p["paperId"] not in existing_ids]
        rest_backward = [p for p in backward_candidates if p["paperId"] not in existing_ids]
        rest = rest_forward + rest_backward
        if rest:
            chosen = rng.sample(rest, min(need, len(rest)))
            pack.extend(chosen)

    return pack[:3]
