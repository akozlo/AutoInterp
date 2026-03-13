"""
Enrich papers that lack both arxiv_id and open_access_url.

Strategy:
1. S2 batch API (cache=False) to re-fetch externalIds + openAccessPdf for all
   49 missing papers in a single request — avoids per-paper rate limits.
2. For papers still missing after S2, try arxiv API title search (with 4s delay
   between requests to respect rate limits).

Usage:
    cd arxiv_interp_graph && python enrich_missing_urls.py [--dry-run]
"""

import os
import re
import sys
import time
import json

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from api_client import SemanticScholarClient
from config import GRAPH_STATE_PATH, S2_API_BASE
from persistence import load_graph, save_graph

import requests


def _arxiv_search(title: str, max_results: int = 3) -> str | None:
    """Search arxiv API by title and return arxiv_id if a close match is found."""
    try:
        clean = re.sub(r"[^\w\s]", " ", title).strip()
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f'ti:"{clean[:200]}"',
            "max_results": max_results,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()

        ids = re.findall(r"<id>http://arxiv\.org/abs/([^<]+)</id>", r.text)
        if ids:
            arxiv_id = re.sub(r"v\d+$", "", ids[0])
            return arxiv_id
    except Exception as e:
        print(f"    arxiv search failed: {e}")
    return None


def main(dry_run: bool = False):
    path = GRAPH_STATE_PATH
    if not os.path.exists(path):
        print(f"Graph not found: {path}")
        sys.exit(1)

    print(f"Loading graph from {path} ...")
    G, completed_waves = load_graph(path)

    # Find papers missing both arxiv_id and open_access_url
    missing = {}
    for nid, attrs in G.nodes(data=True):
        has_arxiv = attrs.get("arxiv_id") and isinstance(attrs["arxiv_id"], str)
        has_oa = attrs.get("open_access_url") and isinstance(attrs["open_access_url"], str)
        if not has_arxiv and not has_oa:
            missing[nid] = attrs

    print(f"Found {len(missing)} papers with no stored download URL.\n")
    if not missing:
        print("Nothing to do.")
        return

    client = SemanticScholarClient()
    recovered_arxiv = 0
    recovered_oa = 0
    resolved = {}  # nid -> ("arxiv_id", value) or ("open_access_url", value)

    # ------------------------------------------------------------------
    # Phase 1: S2 batch API (cache=False) — one request for all 49 papers
    # ------------------------------------------------------------------
    print("Phase 1: S2 batch API re-fetch (cache bypass) ...")
    missing_ids = list(missing.keys())
    fields = "paperId,title,externalIds,openAccessPdf"
    batch_url = f"{S2_API_BASE}/paper/batch"

    batch_result = client._request(
        "POST",
        batch_url,
        params={"fields": fields},
        json_body={"ids": missing_ids},
        cache=False,
    )

    if batch_result and isinstance(batch_result, list):
        for paper in batch_result:
            if not paper or not paper.get("paperId"):
                continue
            pid = paper["paperId"]
            if pid not in missing:
                continue

            ext = paper.get("externalIds") or {}
            arxiv_id = ext.get("ArXiv") or ext.get("arXiv")
            if arxiv_id and isinstance(arxiv_id, str):
                resolved[pid] = ("arxiv_id", arxiv_id)
                recovered_arxiv += 1
                print(f"  [S2 batch] {missing[pid].get('title', '')[:60]} -> arxiv:{arxiv_id}")
            else:
                oa = paper.get("openAccessPdf")
                if isinstance(oa, dict) and oa.get("url"):
                    resolved[pid] = ("open_access_url", oa["url"])
                    recovered_oa += 1
                    print(f"  [S2 batch] {missing[pid].get('title', '')[:60]} -> oa_url:{oa['url'][:60]}")

    s2_resolved = len(resolved)
    print(f"\nPhase 1 complete: {s2_resolved} recovered from S2 batch.\n")

    # ------------------------------------------------------------------
    # Phase 2: arxiv API title search for remaining papers
    # ------------------------------------------------------------------
    still_missing = {nid: attrs for nid, attrs in missing.items() if nid not in resolved}

    if still_missing:
        print(f"Phase 2: arxiv API title search for {len(still_missing)} remaining papers ...")
        for i, (nid, attrs) in enumerate(still_missing.items()):
            title = attrs.get("title", "(no title)")
            print(f"  [{i+1}/{len(still_missing)}] {title[:70]}")

            arxiv_id = _arxiv_search(title)
            if arxiv_id:
                resolved[nid] = ("arxiv_id", arxiv_id)
                recovered_arxiv += 1
                print(f"    -> arxiv:{arxiv_id}")
            else:
                print(f"    -> not found")

            # arxiv rate limit: ~1 req per 3 seconds; be conservative
            time.sleep(4)

    # ------------------------------------------------------------------
    # Apply results to graph
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Recovered arxiv_id:        {recovered_arxiv}")
    print(f"  Recovered open_access_url: {recovered_oa}")
    print(f"  Still missing:             {len(missing) - len(resolved)}")
    print(f"  Total processed:           {len(missing)}")

    if resolved:
        remaining_missing = []
        for nid, attrs in missing.items():
            if nid in resolved:
                field, value = resolved[nid]
                title = attrs.get("title", "(no title)")[:60]
                print(f"  {field}: {title} -> {value}")
                if not dry_run:
                    G.nodes[nid][field] = value
            else:
                remaining_missing.append(attrs.get("title", nid))

        if remaining_missing:
            print(f"\nStill unresolved ({len(remaining_missing)} papers):")
            for t in remaining_missing:
                print(f"  - {t[:80]}")

    if not dry_run and resolved:
        save_graph(G, completed_waves, path=path)
        print(f"\nGraph saved to {path}")
    elif dry_run:
        print(f"\n[DRY RUN] No changes written.")
    else:
        print(f"\nNo new URLs found, graph unchanged.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
