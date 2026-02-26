"""
One topic package per run: load graph -> sample (random_seed | community) -> generate topic -> write JSON + MD + optional subgraph.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

try:
    from ..topic_mining.graph_loader import load_graph_for_topic_mining
except ImportError:
    from topic_mining.graph_loader import load_graph_for_topic_mining

from .llm_client import get_llm_generate_fn
from .sampling import sample_community, sample_random_seed
from .topic_gen import generate_topic_fallback, generate_topic_llm


def _render_md(package: Dict[str, Any], papers: List[Dict[str, Any]]) -> str:
    """Turn topic_package JSON into human-readable topic_package.md."""
    lines = ["# Topic Package", ""]
    tt = package.get("topic_title") or {}
    lines.append(f"## Title\n- **Main:** {tt.get('main', '')}")
    if tt.get("alt1"):
        lines.append(f"- Alt 1: {tt['alt1']}")
    if tt.get("alt2"):
        lines.append(f"- Alt 2: {tt['alt2']}")
    lines.extend(["", f"## One-sentence thesis\n{package.get('one_sentence_thesis', '')}", ""])
    lines.extend(["## Scope", f"- **In:** {package.get('scope_in', '')}", f"- **Out:** {package.get('scope_out', '')}", ""])
    kw = package.get("keywords") or []
    lines.append("## Keywords\n" + ", ".join(kw) + "\n")
    subthemes = package.get("subthemes") or []
    if subthemes:
        lines.append("## Subthemes")
        for s in subthemes:
            lines.append(f"- **{s.get('name', '')}**: {', '.join(s.get('paper_ids', []))}")
        lines.append("")
    outline = package.get("outline") or []
    if outline:
        lines.append("## Outline")
        for o in outline:
            lines.append(f"- **{o.get('section', '')}**: {', '.join(o.get('paper_ids', []))}")
        lines.append("")
    ro = package.get("reading_order") or []
    if ro:
        lines.append("## Recommended reading order\n" + " → ".join(ro) + "\n")
    lines.append(f"**Confidence:** {package.get('confidence', 0):.2f}")
    lines.append("")
    lines.append("---")
    lines.append("## Papers in this package")
    for p in papers:
        lines.append(f"- **[{p.get('paper_id', '')}]** {p.get('title', '')} ({p.get('year', '')})")
        abstract = (p.get("abstract") or "").strip()
        if abstract:
            # Truncate for readability (e.g. 500 chars)
            ab = abstract if len(abstract) <= 500 else abstract[:497].rsplit(" ", 1)[0] + "..."
            lines.append(f"  {ab}")
    return "\n".join(lines)


def run_topic_package(
    graph_path: str | Path,
    output_dir: str | Path,
    sampling: str = "random_seed",
    k: int = 20,
    community_id: Optional[int] = None,
    year_min: Optional[int] = 2012,
    min_in_degree: int = 0,
    core_ratio: float = 0.4,
    resolution: float = 1.0,
    export_subgraph: bool = True,
    seed: int = 42,
    llm_generate_fn: Optional[Any] = None,
    llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load graph, sample k papers (random_seed or community), generate topic package (LLM or fallback), write:
    - topic_package.json
    - topic_package.md
    - topic_subgraph.graphml (if export_subgraph)
    Returns dict with keys: papers, package, output_paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    G = load_graph_for_topic_mining(graph_path)
    papers: List[Dict[str, Any]] = []
    if sampling == "random_seed":
        papers = sample_random_seed(G, k=k, year_min=year_min, min_in_degree=min_in_degree, seed=seed)
    elif sampling == "community":
        papers = sample_community(G, k=k, community_id=community_id, core_ratio=core_ratio, resolution=resolution, seed=seed)
    else:
        raise ValueError(f"Unknown sampling: {sampling}. Use random_seed or community.")

    if not papers:
        return {"papers": [], "package": {}, "output_paths": {}}

    # Step B: topic generation (LLM if available, else fallback; reuse provider/model from llm_config when given)
    import asyncio
    package: Dict[str, Any] = {}
    if not llm_generate_fn and llm_config:
        llm_generate_fn = get_llm_generate_fn(
            provider=llm_config.get("provider"),
            model=llm_config.get("model"),
        )
    gen_fn = llm_generate_fn or get_llm_generate_fn()
    if gen_fn:
        try:
            pkg = asyncio.run(generate_topic_llm(papers, gen_fn))
            if pkg and pkg.get("topic_title"):
                package = pkg
        except Exception as e:
            pass  # fallback below
    if not package or not package.get("topic_title"):
        package = generate_topic_fallback(papers)

    # Add sampling metadata
    package["_meta"] = {"sampling": sampling, "k": len(papers), "seed": seed}

    json_path = output_dir / "topic_package.json"
    with open(json_path, "w") as f:
        json.dump({"package": package, "papers": papers}, f, indent=2)

    md_path = output_dir / "topic_package.md"
    md_path.write_text(_render_md(package, papers))

    subgraph_path = None
    if export_subgraph:
        node_ids = [p["paper_id"] for p in papers]
        sub = G.subgraph(node_ids)
        H = sub.copy()
        for _, attrs in H.nodes(data=True):
            for key, val in list(attrs.items()):
                if val is None:
                    attrs[key] = ""
        subgraph_path = output_dir / "topic_subgraph.graphml"
        nx.write_graphml(H, subgraph_path)

    return {
        "papers": papers,
        "package": package,
        "output_paths": {
            "topic_package_json": str(json_path),
            "topic_package_md": str(md_path),
            "topic_subgraph_graphml": str(subgraph_path) if subgraph_path else None,
        },
    }
