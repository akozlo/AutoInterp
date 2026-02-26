"""Community detection: graph-based (Louvain/Leiden), embedding-based (HDBSCAN/KMeans), hybrid fusion."""

import math
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

DETERMINISTIC_SEED = 42


def _to_undirected(G: nx.DiGraph) -> nx.Graph:
    return G.to_undirected()


def communities_graph(
    G: nx.DiGraph,
    resolution: float = 1.0,
    seed: int = DETERMINISTIC_SEED,
) -> List[List[str]]:
    """
    Run Louvain (or Leiden if available) on undirected projection.
    Returns list of communities, each a list of node IDs.
    """
    U = _to_undirected(G)
    try:
        import leidenalg
        import igraph as ig
        node_list = list(U.nodes())
        node2idx = {n: i for i, n in enumerate(node_list)}
        edges = [(node2idx[u], node2idx[v]) for u, v in U.edges()]
        g_ig = ig.Graph(n=len(node_list), edges=edges)
        part = leidenalg.find_partition(
            g_ig,
            leidenalg.ModularityVertexPartition,
            resolution_parameter=resolution,
            seed=seed,
            n_iterations=-1,
        )
        return [[node_list[i] for i in comm] for comm in part]
    except ImportError:
        pass
    # Louvain via networkx
    from networkx.algorithms import community as nx_comm
    part = nx_comm.louvain_communities(U, resolution=resolution, seed=seed)
    return [list(c) for c in part]


def communities_embedding(
    embeddings: np.ndarray,
    node_ids: List[str],
    method: str = "hdbscan",
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    n_clusters: Optional[int] = None,
    seed: int = DETERMINISTIC_SEED,
) -> List[List[str]]:
    """
    Cluster in embedding space. HDBSCAN preferred; KMeans fallback.
    Returns list of communities (list of node IDs). Noise points (label -1) form their own singleton "communities" for simplicity.
    """
    if method == "hdbscan":
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples or min_cluster_size,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(embeddings)
        except ImportError:
            method = "kmeans"
    if method == "kmeans":
        if n_clusters is None:
            n_clusters = max(2, min(50, len(node_ids) // 10))
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = km.fit_predict(embeddings)
    # labels: one per row; -1 for noise in HDBSCAN
    id_arr = np.array(node_ids)
    communities: List[List[str]] = []
    for label in np.unique(labels):
        mask = labels == label
        comm = id_arr[mask].tolist()
        if comm:
            communities.append(comm)
    return communities


def build_knn_graph(
    embeddings: np.ndarray,
    node_ids: List[str],
    k: int,
    metric: str = "cosine",
) -> nx.Graph:
    """Build k-NN graph in embedding space (edges from each node to its k nearest neighbors)."""
    from sklearn.metrics.pairwise import cosine_distances
    n = len(node_ids)
    if metric == "cosine":
        sim = 1.0 - cosine_distances(embeddings)
    else:
        from sklearn.metrics.pairwise import euclidean_distances
        sim = -euclidean_distances(embeddings)
    G = nx.Graph()
    for i, nid in enumerate(node_ids):
        G.add_node(nid)
    for i in range(n):
        row = sim[i]
        topk = np.argsort(row)[::-1][1 : k + 1]
        for j in topk:
            if row[j] > 0:
                G.add_edge(node_ids[i], node_ids[j], weight=float(row[j]))
    return G


def fuse_graphs(
    citation_G: nx.DiGraph,
    knn_G: nx.Graph,
    mode: str = "union",
    sim_threshold: float = 0.0,
) -> nx.Graph:
    """
    Fuse citation graph (as undirected) with kNN graph.
    - union: keep all edges from both; weight 1 for citation, weight from kNN for embedding edges.
    - intersection: keep edge only if it appears in citation graph OR (in kNN and weight >= sim_threshold).
    Returns undirected weighted graph.
    """
    U = _to_undirected(citation_G)
    F = nx.Graph()
    for n in U.nodes():
        F.add_node(n)
    for u, v in U.edges():
        F.add_edge(u, v, weight=1.0)
    for u, v in knn_G.edges():
        w = knn_G.edges[u, v].get("weight", 0.0)
        if mode == "union":
            if F.has_edge(u, v):
                F.edges[u, v]["weight"] = max(F.edges[u, v]["weight"], w)
            else:
                F.add_edge(u, v, weight=w)
        else:
            if not F.has_edge(u, v) and w >= sim_threshold:
                F.add_edge(u, v, weight=w)
    return F


def communities_hybrid(
    G: nx.DiGraph,
    embeddings: np.ndarray,
    node_ids: List[str],
    hybrid_mode: str = "union",
    knn_k: int = 10,
    sim_threshold: float = 0.3,
    resolution: float = 1.0,
    seed: int = DETERMINISTIC_SEED,
) -> List[List[str]]:
    """
    Build kNN graph, fuse with citation graph, run community detection on fused graph.
    """
    knn_G = build_knn_graph(embeddings, node_ids, k=knn_k, metric="cosine")
    fused = fuse_graphs(G, knn_G, mode=hybrid_mode, sim_threshold=sim_threshold)
    # Run Louvain/Leiden on fused (weighted)
    try:
        import leidenalg
        import igraph as ig
        node_list = list(fused.nodes())
        node2idx = {n: i for i, n in enumerate(node_list)}
        edges = [(node2idx[u], node2idx[v]) for u, v in fused.edges()]
        weights = [fused.edges[u, v].get("weight", 1.0) for u, v in fused.edges()]
        g_ig = ig.Graph(n=len(node_list), edges=edges)
        g_ig.es["weight"] = weights
        part = leidenalg.find_partition(
            g_ig,
            leidenalg.ModularityVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=seed,
            n_iterations=-1,
        )
        return [[node_list[i] for i in comm] for comm in part]
    except ImportError:
        pass
    from networkx.algorithms import community as nx_comm
    part = nx_comm.louvain_communities(
        fused, weight="weight", resolution=resolution, seed=seed
    )
    return [list(c) for c in part]
