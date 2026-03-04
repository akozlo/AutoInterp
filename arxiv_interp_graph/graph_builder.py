"""Wave expansion algorithm for building the citation graph."""

from collections import defaultdict

import networkx as nx

from api_client import SemanticScholarClient
from config import (
    DEFAULT_MAX_CANDIDATES_PER_WAVE,
    DEFAULT_MAX_WAVES,
    DEFAULT_MIN_CITATION_OVERLAP,
    DEFAULT_MIN_CITATIONS,
    DEFAULT_OVERLAP_GROWTH,
    DEFAULT_YEAR_MAX,
    DEFAULT_YEAR_MIN,
    PAPER_FIELDS,
)
from persistence import save_graph
from seeds import SEED_PAPERS


def _paper_to_node_attrs(paper: dict, wave: int, group: str = "") -> dict:
    """Extract node attributes from an S2 API paper dict (includes abstract when available)."""
    authors = paper.get("authors") or []
    author_names = [a.get("name", "") for a in authors[:10]]
    abstract = paper.get("abstract") or ""
    if isinstance(abstract, str):
        abstract = abstract.strip()
    else:
        abstract = ""
    external_ids = paper.get("externalIds") or {}
    arxiv_id = external_ids.get("ArXiv") or external_ids.get("arXiv")
    # For non-arxiv papers, store openAccessPdf URL (Distill, Transformer Circuits, etc.)
    open_access_url = None
    if not arxiv_id:
        oa = paper.get("openAccessPdf")
        if isinstance(oa, dict) and oa.get("url"):
            open_access_url = oa["url"]
    return {
        "title": paper.get("title", ""),
        "year": paper.get("year"),
        "citation_count": paper.get("citationCount", 0),
        "venue": paper.get("venue", ""),
        "authors": "; ".join(author_names),
        "url": paper.get("url", ""),
        "abstract": abstract,
        "arxiv_id": arxiv_id,
        "open_access_url": open_access_url,
        "wave": wave,
        "group": group,
    }


def resolve_seeds(client: SemanticScholarClient) -> nx.DiGraph:
    """Wave 0: resolve all seed papers and return the initial graph."""
    G = nx.DiGraph()
    print(f"Resolving {len(SEED_PAPERS)} seed papers...")

    for i, seed in enumerate(SEED_PAPERS):
        title = seed["title"]
        print(f"  [{i + 1}/{len(SEED_PAPERS)}] {title}")
        paper = client.resolve_paper(seed["id"], title, fields=PAPER_FIELDS)
        if not paper or not paper.get("paperId"):
            print(f"    WARNING: Could not resolve, skipping.")
            continue
        pid = paper["paperId"]
        attrs = _paper_to_node_attrs(paper, wave=0, group=seed.get("group", "seed"))
        attrs["is_seed"] = True
        G.add_node(pid, **attrs)
        print(f"    -> {pid} (citations: {attrs['citation_count']})")

    # Add edges between seeds based on their references
    seed_ids = set(G.nodes())
    print(f"\nFetching references for {len(seed_ids)} seeds to find inter-seed edges...")
    for pid in list(seed_ids):
        refs = client.get_references(pid)
        for ref in refs:
            ref_id = ref.get("paperId")
            if ref_id and ref_id in seed_ids:
                G.add_edge(pid, ref_id)

    print(f"Wave 0 complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def expand_wave(client: SemanticScholarClient, G: nx.DiGraph, wave_num: int,
                min_overlap: int = DEFAULT_MIN_CITATION_OVERLAP,
                max_candidates: int = DEFAULT_MAX_CANDIDATES_PER_WAVE,
                year_min: int = DEFAULT_YEAR_MIN,
                year_max: int = DEFAULT_YEAR_MAX,
                min_citations: int = DEFAULT_MIN_CITATIONS) -> set[str]:
    """Expand the graph by one wave. Returns the set of newly added paper IDs."""

    # Identify the frontier: papers added in the previous wave
    if wave_num == 1:
        frontier = {n for n in G.nodes() if G.nodes[n].get("is_seed")}
    else:
        frontier = {n for n in G.nodes() if G.nodes[n].get("wave") == wave_num - 1}

    if not frontier:
        print(f"Wave {wave_num}: no frontier papers, stopping.")
        return set()

    print(f"\nWave {wave_num}: expanding from {len(frontier)} frontier papers...")
    graph_ids = set(G.nodes())

    # Track how many existing graph papers each candidate cites
    # candidate_id -> set of graph paper IDs it cites
    candidate_links: dict[str, set[str]] = defaultdict(set)

    # For each frontier paper, get its citers
    for i, pid in enumerate(sorted(frontier)):
        title = G.nodes[pid].get("title", pid)
        print(f"  [{i + 1}/{len(frontier)}] Fetching citers of: {title[:60]}")
        citers = client.get_citations(pid)
        print(f"    -> {len(citers)} citers")

        for citer in citers:
            cid = citer.get("paperId")
            if not cid or cid in graph_ids:
                # Already in graph — but we should still record the edge
                if cid and cid in graph_ids:
                    G.add_edge(cid, pid)
                continue
            # Record that candidate cid cites frontier paper pid
            candidate_links[cid].add(pid)

    print(f"  Found {len(candidate_links)} unique candidate papers")

    # Filter candidates by overlap threshold
    # Optimization: for candidates with overlap = min_overlap - 1,
    # fetch their reference list to check if they cite other graph papers
    qualified = {}  # cid -> set of graph papers cited
    borderline = {}

    for cid, cited_graph_papers in candidate_links.items():
        if len(cited_graph_papers) >= min_overlap:
            qualified[cid] = cited_graph_papers
        elif len(cited_graph_papers) == min_overlap - 1:
            borderline[cid] = cited_graph_papers

    print(f"  Directly qualified: {len(qualified)}, borderline: {len(borderline)}")

    # Check borderline candidates by fetching their references (capped to avoid API overload)
    BORDERLINE_CAP = 200
    if borderline and min_overlap > 1:
        borderline_list = sorted(borderline.keys(),
                                 key=lambda c: len(candidate_links.get(c, set())),
                                 reverse=True)[:BORDERLINE_CAP]
        print(f"  Checking {len(borderline_list)} borderline candidates "
              f"(of {len(borderline)} total, capped at {BORDERLINE_CAP})...")
        for j, cid in enumerate(borderline_list):
            cited_set = borderline[cid]
            refs = client.get_references(cid)
            ref_ids = {r.get("paperId") for r in refs if r.get("paperId")}
            extra = (ref_ids & graph_ids) - cited_set
            if extra:
                cited_set.update(extra)
                if len(cited_set) >= min_overlap:
                    qualified[cid] = cited_set

    print(f"  Total qualified after borderline check: {len(qualified)}")

    if not qualified:
        print(f"Wave {wave_num}: no qualified candidates.")
        return set()

    # Fetch metadata for qualified candidates (batch)
    candidate_ids = list(qualified.keys())
    print(f"  Fetching metadata for {len(candidate_ids)} candidates...")
    papers = client.get_papers_batch(candidate_ids)

    # Build a lookup
    paper_lookup = {}
    for p in papers:
        if p and p.get("paperId"):
            paper_lookup[p["paperId"]] = p

    # Filter by year, sort by citation count, cap
    scored = []
    for cid in qualified:
        p = paper_lookup.get(cid)
        if not p:
            continue
        year = p.get("year")
        if year is not None and (year < year_min or year > year_max):
            continue
        cite_count = p.get("citationCount", 0) or 0
        if cite_count < min_citations:
            continue
        scored.append((cite_count, cid, p))

    scored.sort(reverse=True)
    if len(scored) > max_candidates:
        print(f"  Capping from {len(scored)} to {max_candidates} candidates")
        scored = scored[:max_candidates]

    # Add nodes and edges
    new_ids = set()
    for _, cid, p in scored:
        attrs = _paper_to_node_attrs(p, wave=wave_num)
        G.add_node(cid, **attrs)
        new_ids.add(cid)
        # Add edges: this paper cites these graph papers
        for cited_pid in qualified[cid]:
            G.add_edge(cid, cited_pid)

    print(f"Wave {wave_num} complete: added {len(new_ids)} papers. "
          f"Graph now has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return new_ids


def build_graph(client: SemanticScholarClient | None = None,
                max_waves: int = DEFAULT_MAX_WAVES,
                min_overlap: int = DEFAULT_MIN_CITATION_OVERLAP,
                overlap_growth: int = DEFAULT_OVERLAP_GROWTH,
                max_candidates: int = DEFAULT_MAX_CANDIDATES_PER_WAVE,
                year_min: int = DEFAULT_YEAR_MIN,
                year_max: int = DEFAULT_YEAR_MAX,
                min_citations: int = DEFAULT_MIN_CITATIONS,
                resume_graph: nx.DiGraph | None = None,
                resume_wave: int = 0) -> nx.DiGraph:
    """Build the full citation graph from seeds through N waves."""
    if client is None:
        client = SemanticScholarClient()

    if resume_graph is not None:
        G = resume_graph
        start_wave = resume_wave + 1
        print(f"Resuming from wave {resume_wave} ({G.number_of_nodes()} nodes)")
    else:
        G = resolve_seeds(client)
        save_graph(G, completed_waves=0)
        start_wave = 1

    for wave in range(start_wave, max_waves + 1):
        effective_overlap = min_overlap + overlap_growth * (wave - 1)
        print(f"[Overlap threshold for wave {wave}: {effective_overlap}]")
        expand_wave(client, G, wave, min_overlap=effective_overlap,
                    max_candidates=max_candidates, year_min=year_min, year_max=year_max,
                    min_citations=min_citations)
        save_graph(G, completed_waves=wave)

    return G
