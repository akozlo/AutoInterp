#!/usr/bin/env python3
"""CLI entry point for the citation graph builder."""

import argparse
import os
import re
import sys

from api_client import SemanticScholarClient
from config import (
    DEFAULT_MAX_CANDIDATES_PER_WAVE,
    DEFAULT_MAX_WAVES,
    DEFAULT_MIN_CITATION_OVERLAP,
    DEFAULT_MIN_CITATIONS,
    DEFAULT_OVERLAP_GROWTH,
    DEFAULT_YEAR_MAX,
    DEFAULT_YEAR_MIN,
)
from graph_builder import build_graph
from persistence import load_graph, state_exists
from visualization import export_gexf, export_graphml, export_interactive, print_stats


def cmd_build(args):
    client = SemanticScholarClient()

    resume_graph = None
    resume_wave = 0
    if args.resume:
        if not state_exists():
            print("No saved state to resume from. Starting fresh.")
        else:
            resume_graph, resume_wave = load_graph()
            print(f"Loaded saved state: wave {resume_wave}, "
                  f"{resume_graph.number_of_nodes()} nodes")

    G = build_graph(
        client=client,
        max_waves=args.max_waves,
        min_overlap=args.min_overlap,
        overlap_growth=args.overlap_growth,
        max_candidates=args.max_candidates,
        year_min=args.year_min,
        year_max=args.year_max,
        min_citations=args.min_citations,
        resume_graph=resume_graph,
        resume_wave=resume_wave,
    )

    print_stats(G)
    print("Build complete.")


def cmd_stats(args):
    if not state_exists():
        print("No saved graph state found. Run 'build' first.")
        sys.exit(1)
    G, wave = load_graph()
    print(f"Loaded graph (completed through wave {wave})")
    print_stats(G)


def cmd_export(args):
    if not state_exists():
        print("No saved graph state found. Run 'build' first.")
        sys.exit(1)
    G, wave = load_graph()
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    fmt = args.format
    if fmt in ("graphml", "all"):
        export_graphml(G)
    if fmt in ("gexf", "all"):
        export_gexf(G)
    if fmt in ("html", "all"):
        export_interactive(G)


def cmd_topic_mining(args):
    """Run topic mining: graph / embed / hybrid modes."""
    from topic_mining.run import run_topic_mining

    graph_path = args.graph
    if not os.path.isabs(graph_path):
        base = os.path.dirname(os.path.abspath(__file__))
        graph_path = os.path.join(base, graph_path)
    if not os.path.exists(graph_path):
        print(f"Graph path not found: {graph_path}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(os.path.dirname(graph_path), "topic_mining")
    result = run_topic_mining(
        graph_path=graph_path,
        output_dir=output_dir,
        topic_mode=args.topic_mode,
        hybrid_mode=args.hybrid_mode,
        knn_k=args.knn_k,
        sim_threshold=args.sim_threshold,
        resolution=args.resolution,
        embedder_backend=args.embedder_backend or "sentence-transformers",
        embedder_model=args.embedder_model,
        embed_cache_dir=args.embed_cache_dir,
        top_representatives=args.top_representatives,
        top_keywords=args.top_keywords,
        force=args.force,
    )
    print(f"Topics: {len(result['topics'])}")
    print(f"Output: {result['output_paths']}")


def cmd_topic_package(args):
    """Generate one topic package: sample -> topic_package.json + topic_package.md + optional topic_subgraph.graphml.
    Uses same LLM as main flow: config from AutoInterp config.yaml (../config.yaml) when present.
    """
    from topic_package.run import run_topic_package

    graph_path = args.graph
    if not os.path.isabs(graph_path):
        base = os.path.dirname(os.path.abspath(__file__))
        graph_path = os.path.join(base, graph_path)
    if not os.path.exists(graph_path):
        print(f"Graph path not found: {graph_path}")
        sys.exit(1)

    llm_config = None
    if not getattr(args, "no_llm", False):
        # Prefer last model selected in main.py (.last_llm.json), then AutoInterp config.yaml
        base = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.join(base, "..")
        try:
            import json
            last_llm_path = os.path.join(repo_root, ".last_llm.json")
            if os.path.exists(last_llm_path):
                with open(last_llm_path) as f:
                    llm_config = json.load(f)
                if llm_config and (llm_config.get("provider") or llm_config.get("model")):
                    print("Using LLM (last selected in main): {} / {} — calling model to generate topic.".format(
                        llm_config.get("provider", ""), llm_config.get("model", "")))
        except Exception:
            pass
        if not llm_config or (not llm_config.get("provider") and not llm_config.get("model")):
            try:
                import yaml
                config_path = os.path.join(repo_root, "config.yaml")
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        root_config = yaml.safe_load(f) or {}
                    llm_config = root_config.get("llm") or {}
                    if llm_config and (llm_config.get("provider") or llm_config.get("model")):
                        print("Using LLM from config.yaml: {} / {} — calling model to generate topic.".format(
                            llm_config.get("provider", ""), llm_config.get("model", "")))
            except Exception:
                pass
        if not llm_config or (not llm_config.get("provider") and not llm_config.get("model")):
            print("No LLM config (no .last_llm.json or config.yaml). Using keyword fallback for topic (no model call).")
            llm_config = None

    output_dir = args.output_dir or os.path.join(os.path.dirname(graph_path), "topic_package")
    result = run_topic_package(
        graph_path=graph_path,
        output_dir=output_dir,
        sampling=args.sampling,
        k=args.k,
        community_id=args.community_id,
        year_min=args.year_min,
        min_in_degree=args.min_in_degree,
        core_ratio=args.core_ratio,
        resolution=args.resolution,
        export_subgraph=not args.no_subgraph,
        seed=args.seed,
        llm_generate_fn=None,
        llm_config=llm_config,
    )
    print(f"Sampled {len(result['papers'])} papers (sampling={args.sampling})")
    pkg = result.get("package", {})
    tt = pkg.get("topic_title") or {}
    print(f"Topic: {tt.get('main', '(fallback)')}")
    for k, v in result.get("output_paths", {}).items():
        if v:
            print(f"  {k}: {v}")


def cmd_context_pack(args):
    """Build 3-paper context pack (seed + forward + backward), download PDFs, manifest, optional LLM question."""
    from context_pack.run import run_context_pack

    base = os.path.dirname(os.path.abspath(__file__))
    graph_path = args.graph
    if not os.path.isabs(graph_path):
        graph_path = os.path.join(base, graph_path)
    if not os.path.exists(graph_path):
        print(f"Graph not found: {graph_path}")
        sys.exit(1)

    if args.output_dir is None:
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
        repo_root = os.path.join(base, "..")
        output_dir = os.path.join(repo_root, "projects", f"context_pack_{ts}", "questions")
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")
    client = SemanticScholarClient()

    llm_generate_fn = None
    if not getattr(args, "no_llm", False):
        repo_root = os.path.join(base, "..")
        try:
            import json
            last_llm = os.path.join(repo_root, ".last_llm.json")
            if os.path.exists(last_llm):
                with open(last_llm) as f:
                    llm_config = json.load(f)
            else:
                import yaml
                cfg_path = os.path.join(repo_root, "config.yaml")
                llm_config = {}
                if os.path.exists(cfg_path):
                    with open(cfg_path) as f:
                        root = yaml.safe_load(f) or {}
                    llm_config = root.get("llm") or {}
            if llm_config and (llm_config.get("provider") or llm_config.get("model")):
                from topic_package.llm_client import get_llm_generate_fn
                llm_generate_fn = get_llm_generate_fn(
                    provider=llm_config.get("provider"),
                    model=llm_config.get("model"),
                )
                print("Using LLM for research question: {} / {}".format(
                    llm_config.get("provider"), llm_config.get("model")))
        except Exception as e:
            print("LLM config load failed:", e)

    result = run_context_pack(
        graph_path=graph_path,
        output_dir=output_dir,
        seed_id=args.seed_id,
        s2_client=client,
        seed=args.seed,
        download_pdfs=not getattr(args, "no_download", False),
        llm_generate_fn=llm_generate_fn,
    )
    papers = result.get("papers", [])
    print(f"Context pack: {len(papers)} papers")
    for p in papers:
        print(f"  [{p.get('relation')}] {p.get('paperId')} {p.get('title', '')[:50]}...")
    if result.get("manifest_path"):
        print(f"  manifest: {result['manifest_path']}")
    if result.get("question_path"):
        print(f"  question: {result['question_path']}")
    # Auto-rename project folder from context_pack_<ts> to <slug>_<ts> using QUESTION line
    if args.output_dir is None and result.get("question_text"):
        qtext = result["question_text"]
        q_match = re.search(r"QUESTION:\s*(.+?)(?:\n|$)", qtext, re.IGNORECASE | re.DOTALL)
        if q_match:
            title = q_match.group(1).strip()
            slug = re.sub(r"[^\w\s\-]", "", title).strip()
            slug = re.sub(r"\s+", "_", slug).lower()[:50].strip("_")
            if slug:
                old_parent = os.path.dirname(output_dir)
                new_name = f"{slug}_{ts}"
                projects_base = os.path.join(repo_root, "projects")
                new_parent = os.path.join(projects_base, new_name)
                if old_parent != new_parent and os.path.exists(old_parent) and not os.path.exists(new_parent):
                    try:
                        os.rename(old_parent, new_parent)
                        print(f"  Renamed project to: {os.path.join(new_parent, 'questions')}")
                    except Exception as e:
                        print(f"  Could not rename to {new_name}: {e}")


def cmd_enrich_abstracts(args):
    """Fetch abstract for all nodes from Semantic Scholar and save updated graph."""
    from enrich_abstracts import main as enrich_main
    graph_path = args.graph
    if not os.path.isabs(graph_path):
        base = os.path.dirname(os.path.abspath(__file__))
        graph_path = os.path.join(base, graph_path)
    enrich_main(graph_path)


def main():
    parser = argparse.ArgumentParser(
        description="AI Interpretability Citation Graph Builder"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build
    build_parser = subparsers.add_parser("build", help="Build the citation graph")
    build_parser.add_argument("--max-waves", type=int, default=DEFAULT_MAX_WAVES,
                              help=f"Max expansion waves (default: {DEFAULT_MAX_WAVES})")
    build_parser.add_argument("--min-overlap", type=int, default=DEFAULT_MIN_CITATION_OVERLAP,
                              help=f"Min citation overlap to qualify at wave 1 (default: {DEFAULT_MIN_CITATION_OVERLAP})")
    build_parser.add_argument("--overlap-growth", type=int, default=DEFAULT_OVERLAP_GROWTH,
                              help=f"Overlap threshold increase per wave (default: {DEFAULT_OVERLAP_GROWTH})")
    build_parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES_PER_WAVE,
                              help=f"Max papers added per wave (default: {DEFAULT_MAX_CANDIDATES_PER_WAVE})")
    build_parser.add_argument("--year-min", type=int, default=DEFAULT_YEAR_MIN,
                              help=f"Earliest year filter (default: {DEFAULT_YEAR_MIN})")
    build_parser.add_argument("--year-max", type=int, default=DEFAULT_YEAR_MAX,
                              help=f"Latest year filter (default: {DEFAULT_YEAR_MAX})")
    build_parser.add_argument("--min-citations", type=int, default=DEFAULT_MIN_CITATIONS,
                              help=f"Min citation count for candidates (default: {DEFAULT_MIN_CITATIONS})")
    build_parser.add_argument("--resume", action="store_true",
                              help="Resume from saved state")
    build_parser.set_defaults(func=cmd_build)

    # stats
    stats_parser = subparsers.add_parser("stats", help="Print graph statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # export
    export_parser = subparsers.add_parser("export", help="Export graph files")
    export_parser.add_argument("--format", choices=["graphml", "gexf", "html", "all"],
                               default="all", help="Export format (default: all)")
    export_parser.set_defaults(func=cmd_export)

    # topic-mining
    topic_parser = subparsers.add_parser("topic-mining", help="Generate topics from citation graph (graph/embed/hybrid)")
    topic_parser.add_argument("--graph", default="output/graph_state.json", help="Path to graph_state.json or .graphml")
    topic_parser.add_argument("--output-dir", help="Output directory (default: <graph_dir>/topic_mining)")
    topic_parser.add_argument("--topic-mode", choices=["graph", "embed", "hybrid"], default="graph",
                              help="Topic generation mode (default: graph)")
    topic_parser.add_argument("--hybrid-mode", choices=["union", "intersection"], default="union",
                              help="For hybrid: fuse citation + kNN edges (default: union)")
    topic_parser.add_argument("--knn-k", type=int, default=10, help="k for k-NN in embedding space (default: 10)")
    topic_parser.add_argument("--sim-threshold", type=float, default=0.3,
                              help="Similarity threshold for intersection hybrid (default: 0.3)")
    topic_parser.add_argument("--resolution", type=float, default=1.0,
                              help="Community resolution (default: 1.0)")
    topic_parser.add_argument("--embedder-backend", choices=["sentence-transformers", "openai"], default="sentence-transformers")
    topic_parser.add_argument("--embedder-model", help="Model name for embedder (optional)")
    topic_parser.add_argument("--embed-cache-dir", help="Cache dir for embeddings (default: output/embeddings)")
    topic_parser.add_argument("--top-representatives", type=int, default=10)
    topic_parser.add_argument("--top-keywords", type=int, default=10)
    topic_parser.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    topic_parser.set_defaults(func=cmd_topic_mining)

    # topic-package: one topic package per run (sample + topic_package.json + .md + optional subgraph)
    pkg_parser = subparsers.add_parser("topic-package", help="Generate one topic package (sample k papers -> JSON + MD + optional subgraph)")
    pkg_parser.add_argument("--graph", default="output/graph_state.json", help="Path to graph_state.json or .graphml")
    pkg_parser.add_argument("--output-dir", help="Output directory (default: <graph_dir>/topic_package)")
    pkg_parser.add_argument("--sampling", choices=["random_seed", "community"], default="random_seed",
                            help="Sampling method (default: random_seed)")
    pkg_parser.add_argument("--k", type=int, default=20, help="Number of papers to sample (default: 20)")
    pkg_parser.add_argument("--community-id", type=int, default=None,
                            help="For community: which community (0=largest, 1=2nd, ...); default 0")
    pkg_parser.add_argument("--year-min", type=int, default=2012, help="Exclude papers before this year (random_seed)")
    pkg_parser.add_argument("--min-in-degree", type=int, default=0, help="Exclude nodes with in_degree below this (random_seed)")
    pkg_parser.add_argument("--core-ratio", type=float, default=0.4,
                            help="For community: ratio of k as core (by in_degree); rest random (default: 0.4)")
    pkg_parser.add_argument("--resolution", type=float, default=1.0, help="Community detection resolution (default: 1.0)")
    pkg_parser.add_argument("--no-subgraph", action="store_true", help="Do not export topic_subgraph.graphml")
    pkg_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    pkg_parser.add_argument("--no-llm", action="store_true", help="Do not use LLM; use keyword fallback only")
    pkg_parser.set_defaults(func=cmd_topic_package)

    # context-pack: seed + forward/backward -> 3 papers -> PDFs + manifest -> optional LLM question
    ctx_parser = subparsers.add_parser("context-pack", help="Build 3-paper context pack (seed + citing + cited), download PDFs, manifest, optional research question)")
    ctx_parser.add_argument("--graph", default="output/graph_state.json", help="Path to graph_state.json")
    ctx_parser.add_argument("--output-dir", default=None, help="Output directory; default: auto from generated question")
    ctx_parser.add_argument("--seed-id", default=None, help="Seed paper ID (default: random from graph)")
    ctx_parser.add_argument("--seed", type=int, default=None, help="Random seed (default: different each run; set e.g. 42 for reproducibility)")
    ctx_parser.add_argument("--no-download", action="store_true", help="Do not download PDFs")
    ctx_parser.add_argument("--no-llm", action="store_true", help="Do not generate research question with LLM")
    ctx_parser.set_defaults(func=cmd_context_pack)

    # enrich-abstracts: fetch abstract for all nodes and update graph_state.json
    enrich_parser = subparsers.add_parser("enrich-abstracts", help="Fetch abstracts for all papers from Semantic Scholar and update graph_state.json")
    enrich_parser.add_argument("--graph", default="output/graph_state.json", help="Path to graph_state.json")
    enrich_parser.set_defaults(func=cmd_enrich_abstracts)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
