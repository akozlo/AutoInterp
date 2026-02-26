"""Topic importance score: log(size), internal density, bridge-ness proxy, recency."""

import math
from typing import List

import networkx as nx


def topic_importance_score(
    G: nx.DiGraph,
    node_ids: List[str],
    weight_size: float = 0.3,
    weight_density: float = 0.3,
    weight_bridge: float = 0.2,
    weight_recency: float = 0.2,
) -> float:
    """
    Weighted combination of:
    - log(size): log(1 + len(node_ids))
    - internal density: edges within community / possible edges
    - bridge-ness proxy: 1 - (internal_degree / total_degree) averaged (how much the topic is "bridging")
    - recency: mean year normalized to [0,1] (e.g. (year - 2017) / 10), then normalized score
    """
    if not node_ids:
        return 0.0
    sub = G.subgraph(node_ids)
    n = sub.number_of_nodes()
    m = sub.number_of_edges()
    possible = n * (n - 1) if n > 1 else 1
    density = m / possible if possible > 0 else 0.0
    size_score = math.log(1 + n) / math.log(1 + max(n, G.number_of_nodes()))
    density_score = min(1.0, density * 10)

    # Bridge: nodes that cite / are cited outside the community
    in_deg = dict(G.in_degree(node_ids))
    out_deg = dict(G.out_degree(node_ids))
    internal_in = sum(sub.in_degree(n) for n in node_ids)
    internal_out = sum(sub.out_degree(n) for n in node_ids)
    total_in = sum(in_deg.get(n, 0) for n in node_ids)
    total_out = sum(out_deg.get(n, 0) for n in node_ids)
    total_degree = total_in + total_out
    internal_degree = internal_in + internal_out
    if total_degree > 0:
        bridge_score = 1.0 - (internal_degree / total_degree)
    else:
        bridge_score = 0.0

    years = [G.nodes[n].get("year") for n in node_ids if G.nodes[n].get("year") is not None]
    if years:
        mean_year = sum(years) / len(years)
        recency_score = (mean_year - 2017) / 10.0
        recency_score = max(0, min(1, recency_score))
    else:
        recency_score = 0.5

    return (
        weight_size * size_score
        + weight_density * density_score
        + weight_bridge * bridge_score
        + weight_recency * recency_score
    )
