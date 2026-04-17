"""Export and visualization utilities for the citation graph."""

import os
from typing import Optional

import networkx as nx

from config import OUTPUT_DIR

VIZ_DIR = os.path.join(os.path.dirname(__file__), "viz")


def print_stats(G: nx.DiGraph) -> None:
    """Print summary statistics about the graph."""
    print(f"\n{'=' * 60}")
    print(f"Citation Graph Statistics")
    print(f"{'=' * 60}")
    print(f"Nodes (papers):  {G.number_of_nodes()}")
    print(f"Edges (cites):   {G.number_of_edges()}")

    if G.number_of_nodes() == 0:
        return

    # Wave breakdown
    wave_counts: dict[int, int] = {}
    for _, attrs in G.nodes(data=True):
        w = attrs.get("wave", -1)
        wave_counts[w] = wave_counts.get(w, 0) + 1
    for w in sorted(wave_counts):
        label = f"Wave {w}" if w >= 0 else "Unknown"
        if w == 0:
            label = "Seeds (wave 0)"
        print(f"  {label}: {wave_counts[w]} papers")

    # Density
    n = G.number_of_nodes()
    possible = n * (n - 1)
    if possible > 0:
        print(f"Density:         {G.number_of_edges() / possible:.6f}")

    # Degree stats
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    if in_degrees:
        print(f"In-degree:  avg={sum(in_degrees)/len(in_degrees):.1f}, "
              f"max={max(in_degrees)}")
        print(f"Out-degree: avg={sum(out_degrees)/len(out_degrees):.1f}, "
              f"max={max(out_degrees)}")

    # Most-cited papers in the graph
    print(f"\nTop 10 most-cited papers (by in-degree within graph):")
    top = sorted(G.nodes(data=True), key=lambda x: G.in_degree(x[0]), reverse=True)[:10]
    for pid, attrs in top:
        title = attrs.get("title", pid)[:70]
        indeg = G.in_degree(pid)
        year = attrs.get("year", "?")
        print(f"  [{indeg:3d} in-graph cites] ({year}) {title}")

    # Connected components (undirected view)
    U = G.to_undirected()
    components = list(nx.connected_components(U))
    print(f"\nConnected components: {len(components)}")
    if len(components) > 1:
        sizes = sorted([len(c) for c in components], reverse=True)
        print(f"  Sizes: {sizes[:10]}")

    print(f"{'=' * 60}\n")


def _sanitize_for_export(G: nx.DiGraph) -> nx.DiGraph:
    """Return a copy with None attribute values replaced for GraphML/GEXF compat."""
    H = G.copy()
    for _, attrs in H.nodes(data=True):
        for k, v in attrs.items():
            if v is None:
                attrs[k] = ""
    for _, _, attrs in H.edges(data=True):
        for k, v in attrs.items():
            if v is None:
                attrs[k] = ""
    return H


def export_graphml(G: nx.DiGraph, path: Optional[str] = None) -> str:
    """Export graph as GraphML."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(OUTPUT_DIR, "citation_graph.graphml")
    nx.write_graphml(_sanitize_for_export(G), path)
    print(f"Exported GraphML to {path}")
    return path


def export_gexf(G: nx.DiGraph, path: Optional[str] = None) -> str:
    """Export graph as GEXF (for Gephi)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(OUTPUT_DIR, "citation_graph.gexf")
    nx.write_gexf(_sanitize_for_export(G), path)
    print(f"Exported GEXF to {path}")
    return path


def export_plot(G: nx.DiGraph, path: Optional[str] = None) -> str:
    """Render a matplotlib visualization of the graph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(OUTPUT_DIR, "citation_graph.png")

    fig, ax = plt.subplots(1, 1, figsize=(20, 16))

    # Color by wave
    wave_colors = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71", 3: "#f39c12"}
    colors = []
    sizes = []
    for _, attrs in G.nodes(data=True):
        w = attrs.get("wave", 0)
        colors.append(wave_colors.get(w, "#95a5a6"))
        cc = attrs.get("citation_count", 1) or 1
        sizes.append(min(30 + cc * 0.3, 500))

    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, arrows=True,
                           arrowsize=5, edge_color="#cccccc")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                           alpha=0.8, linewidths=0.5, edgecolors="white")

    # Label seed papers
    seed_labels = {}
    for nid, attrs in G.nodes(data=True):
        if attrs.get("is_seed"):
            title = attrs.get("title", "")
            # Truncate long titles
            label = title[:35] + "..." if len(title) > 35 else title
            seed_labels[nid] = label
    if seed_labels:
        nx.draw_networkx_labels(G, pos, labels=seed_labels, ax=ax,
                                font_size=6, font_weight="bold")

    ax.set_title(f"AI Interpretability Citation Graph ({G.number_of_nodes()} papers, "
                 f"{G.number_of_edges()} citations)", fontsize=14)
    ax.axis("off")

    # Legend
    for w, color in sorted(wave_colors.items()):
        if any(attrs.get("wave") == w for _, attrs in G.nodes(data=True)):
            label = f"Wave {w}" + (" (seeds)" if w == 0 else "")
            ax.scatter([], [], c=color, s=60, label=label)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Exported plot to {path}")
    return path


def export_interactive(G: nx.DiGraph, path: Optional[str] = None) -> str:
    """Render an interactive Plotly visualization with hover tooltips."""
    import plotly.graph_objects as go

    os.makedirs(VIZ_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(VIZ_DIR, "citation_graph.html")

    # Community-seeded spring layout: detect communities, place them in
    # separate clusters as initial positions, then refine with spring forces.
    # This avoids the ring/shell artifacts of Kamada-Kawai and plain spring.
    import math
    U = G.to_undirected()
    communities = list(nx.community.louvain_communities(U, seed=42))

    # Assign each community a center point arranged in a circle
    init_pos = {}
    for i, comm in enumerate(communities):
        angle = 2 * math.pi * i / len(communities)
        cx, cy = math.cos(angle), math.sin(angle)
        for j, node in enumerate(comm):
            # Scatter nodes around community center
            jitter_angle = 2 * math.pi * j / len(comm)
            r = 0.15 * (len(comm) ** 0.3)
            init_pos[node] = (
                cx + r * math.cos(jitter_angle),
                cy + r * math.sin(jitter_angle),
            )

    pos = nx.spring_layout(G, pos=init_pos, k=1.5 / (G.number_of_nodes() ** 0.3),
                           iterations=150, seed=42)

    # Build edge traces
    edge_x, edge_y = [], []
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.3, color="#cccccc"),
        hoverinfo="none",
        mode="lines",
    )

    # Build node traces, one per wave for legend grouping
    wave_colors = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71", 3: "#f39c12", 4: "#9b59b6"}
    waves_present = sorted(set(
        attrs.get("wave", 0) for _, attrs in G.nodes(data=True)
    ))

    node_traces = []
    for w in waves_present:
        wave_nodes = [(nid, attrs) for nid, attrs in G.nodes(data=True)
                      if attrs.get("wave", 0) == w]
        if not wave_nodes:
            continue

        x_vals, y_vals, hover_texts, sizes = [], [], [], []
        for nid, attrs in wave_nodes:
            x, y = pos[nid]
            x_vals.append(x)
            y_vals.append(y)

            title = attrs.get("title", nid)
            cc = attrs.get("citation_count", 0) or 0
            year = attrs.get("year", "?")
            in_deg = G.in_degree(nid)
            out_deg = G.out_degree(nid)
            authors = attrs.get("authors", "")
            # Truncate long author lists
            if len(authors) > 80:
                authors = authors[:80] + "..."

            hover = (
                f"<b>{title}</b><br>"
                f"Year: {year}<br>"
                f"S2 Citations: {cc}<br>"
                f"In-graph citations: {in_deg} in / {out_deg} out<br>"
                f"Authors: {authors}"
            )
            hover_texts.append(hover)
            sizes.append(max(4, min(3 + cc ** 0.4, 30)))

        color = wave_colors.get(w, "#95a5a6")
        label = f"Wave {w}" + (" (seeds)" if w == 0 else "")
        node_traces.append(go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers",
            marker=dict(size=sizes, color=color, line=dict(width=0.5, color="white")),
            text=hover_texts,
            hoverinfo="text",
            name=label,
        ))

    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title=dict(
                text=f"AI Interpretability Citation Graph ({G.number_of_nodes()} papers, "
                     f"{G.number_of_edges()} citations)",
                font=dict(size=16),
            ),
            showlegend=True,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            width=1400,
            height=1000,
        ),
    )

    fig.write_html(path)
    print(f"Exported interactive plot to {path}")
    return path
