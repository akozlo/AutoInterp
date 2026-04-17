"""
Enrich existing graph: fetch externalIds and openAccessPdf for all nodes from
Semantic Scholar. Stores arxiv_id on every node, and open_access_url on nodes
that lack an arxiv_id (Distill, Transformer Circuits Thread, conference-only, etc.).
Uses batch API (500 per request).
"""

import os
import sys

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
    print(f"Graph has {n} nodes. Fetching externalIds + openAccessPdf via batch API ...")

    client = SemanticScholarClient()
    fields = "paperId,externalIds,openAccessPdf"
    has_arxiv = 0
    has_oa_url = 0
    no_source = 0

    for i in range(0, n, BATCH_CHUNK_SIZE):
        chunk = node_ids[i : i + BATCH_CHUNK_SIZE]
        batch = client.get_papers_batch(chunk, fields=fields)
        for paper in batch:
            if not paper or not paper.get("paperId"):
                continue
            pid = paper["paperId"]
            if pid not in G:
                continue

            # Extract arxiv_id
            ext = paper.get("externalIds") or {}
            arxiv_id = ext.get("ArXiv") or ext.get("arXiv")
            G.nodes[pid]["arxiv_id"] = arxiv_id

            # For papers without arxiv_id, store openAccessPdf URL
            if arxiv_id:
                has_arxiv += 1
            else:
                oa = paper.get("openAccessPdf")
                oa_url = None
                if isinstance(oa, dict) and oa.get("url"):
                    oa_url = oa["url"]
                G.nodes[pid]["open_access_url"] = oa_url
                if oa_url:
                    has_oa_url += 1
                else:
                    no_source += 1

        print(f"  Processed {min(i + BATCH_CHUNK_SIZE, n)} / {n} nodes ...")

    save_graph(G, completed_waves, path=path)
    print(f"\nDone. Updated graph saved to {path}")
    print(f"  {has_arxiv} nodes have arxiv_id")
    print(f"  {has_oa_url} nodes have open_access_url (no arxiv_id)")
    print(f"  {no_source} nodes have neither")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
