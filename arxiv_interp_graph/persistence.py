"""Save and load graph state as JSON for resumability."""

import json
import os

import networkx as nx

from config import GRAPH_STATE_PATH, OUTPUT_DIR


def save_graph(G: nx.DiGraph, completed_waves: int,
               path: str = GRAPH_STATE_PATH) -> None:
    """Serialize graph + metadata to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    nodes = []
    for node_id, attrs in G.nodes(data=True):
        entry = {"id": node_id}
        entry.update(attrs)
        nodes.append(entry)

    edges = []
    for src, dst, attrs in G.edges(data=True):
        entry = {"source": src, "target": dst}
        entry.update(attrs)
        edges.append(entry)

    state = {
        "completed_waves": completed_waves,
        "nodes": nodes,
        "edges": edges,
    }

    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp_path, path)
    print(f"  Saved graph state ({len(nodes)} nodes, {len(edges)} edges) to {path}")


def load_graph(path: str = GRAPH_STATE_PATH) -> tuple[nx.DiGraph, int]:
    """Deserialize graph state. Returns (graph, completed_waves)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved state at {path}")

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

    return G, state["completed_waves"]


def state_exists(path: str = GRAPH_STATE_PATH) -> bool:
    return os.path.exists(path)
