"""Load citation graph from graph_state.json or GraphML fallback. Preserves node attributes."""

import json
from pathlib import Path

import networkx as nx


def load_graph_for_topic_mining(
    graph_path: str | Path,
) -> nx.DiGraph:
    """
    Load graph from graph_state.json (preferred) or citation_graph.graphml.
    Returns a directed graph; node attributes include title, year, citation_count, authors, venue, url, wave, group, is_seed.
    Abstract is not in current graph_state.json; callers should use title (and abstract if present) for embeddings.
    """
    path = Path(graph_path)
    if not path.exists():
        raise FileNotFoundError(f"Graph path does not exist: {path}")

    if path.suffix == ".json" or path.name == "graph_state.json":
        with open(path) as f:
            state = json.load(f)
        G = nx.DiGraph()
        for entry in state["nodes"]:
            node_id = entry.pop("id")
            G.add_node(node_id, **entry)
        for entry in state["edges"]:
            src = entry.pop("source")
            dst = entry.pop("target")
            G.add_edge(src, dst, **entry)
        return G

    if path.suffix.lower() == ".graphml":
        G = nx.read_graphml(path)
    elif path.suffix.lower() == ".gexf":
        G = nx.read_gexf(path)
    else:
        raise ValueError(f"Unsupported graph format: {path.suffix}. Use .json or .graphml")
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)
    return G
