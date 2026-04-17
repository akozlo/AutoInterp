"""
Enrich existing graph: fetch abstract for all nodes from Semantic Scholar and update graph_state.json.
Uses batch API (500 per request); responses are cached in .cache so re-runs are fast.
"""

import os
import sys

# Run from citation_graph root
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from api_client import SemanticScholarClient
from config import BATCH_CHUNK_SIZE, GRAPH_STATE_PATH
from persistence import load_graph, save_graph


def main(graph_path: str = None):
    path = graph_path or GRAPH_STATE_PATH
    if not os.path.exists(path):
        print(f"Graph not found: {path}")
        sys.exit(1)

    print(f"Loading graph from {path} ...")
    G, completed_waves = load_graph(path)
    node_ids = list(G.nodes())
    n = len(node_ids)
    print(f"Graph has {n} nodes. Fetching abstracts via Semantic Scholar batch API ...")

    client = SemanticScholarClient()
    fields = "paperId,abstract"
    updated = 0
    for i in range(0, n, BATCH_CHUNK_SIZE):
        chunk = node_ids[i : i + BATCH_CHUNK_SIZE]
        batch = client.get_papers_batch(chunk, fields=fields)
        for paper in batch:
            if not paper or not paper.get("paperId"):
                continue
            pid = paper["paperId"]
            if pid not in G:
                continue
            abstract = paper.get("abstract") or ""
            if isinstance(abstract, str):
                abstract = abstract.strip()
            else:
                abstract = ""
            G.nodes[pid]["abstract"] = abstract
            if abstract:
                updated += 1
        print(f"  Processed {min(i + BATCH_CHUNK_SIZE, n)} / {n} nodes ...")

    save_graph(G, completed_waves, path=path)
    print(f"Done. Updated graph saved to {path} (nodes with non-empty abstract: {updated})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
