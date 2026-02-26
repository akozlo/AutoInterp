"""Generate histogram visualizations of the citation graph."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# Allow importing from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from persistence import load_graph

VIZ_DIR = os.path.join(os.path.dirname(__file__), "..", "viz")


def plot_citation_counts(G: nx.DiGraph, out_dir: str) -> None:
    """Histogram of Semantic Scholar citation counts (global, not in-graph)."""
    counts = [attrs.get("citation_count", 0) or 0
              for _, attrs in G.nodes(data=True)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    ax = axes[0]
    ax.hist(counts, bins=50, color="#3498db", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Citation Count")
    ax.set_ylabel("Number of Papers")
    ax.set_title("Citation Count Distribution")
    ax.axvline(np.median(counts), color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Median = {np.median(counts):.0f}")
    ax.axvline(np.mean(counts), color="#f39c12", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(counts):.0f}")
    ax.legend()

    # Log scale
    ax = axes[1]
    log_bins = np.logspace(np.log10(max(1, min(counts))),
                           np.log10(max(counts)), 40)
    ax.hist(counts, bins=log_bins, color="#3498db", edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Citation Count (log scale)")
    ax.set_ylabel("Number of Papers")
    ax.set_title("Citation Count Distribution (Log Scale)")

    fig.suptitle(f"Semantic Scholar Citation Counts ({G.number_of_nodes()} papers)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "citation_counts.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_degree(G: nx.DiGraph, out_dir: str) -> None:
    """Histograms of in-degree, out-degree, and total degree (within the graph)."""
    in_deg = [d for _, d in G.in_degree()]
    out_deg = [d for _, d in G.out_degree()]
    total_deg = [G.in_degree(n) + G.out_degree(n) for n in G.nodes()]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, data, label, color in [
        (axes[0], in_deg, "In-Degree", "#e74c3c"),
        (axes[1], out_deg, "Out-Degree", "#2ecc71"),
        (axes[2], total_deg, "Total Degree", "#9b59b6"),
    ]:
        ax.hist(data, bins=50, color=color, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel("Number of Papers")
        ax.set_title(f"{label} Distribution")
        ax.axvline(np.median(data), color="black", linestyle="--", linewidth=1.2,
                   label=f"Median = {np.median(data):.0f}")
        ax.axvline(np.mean(data), color="gray", linestyle="--", linewidth=1.2,
                   label=f"Mean = {np.mean(data):.1f}")
        ax.legend(fontsize=9)

    fig.suptitle(f"In-Graph Degree Distributions ({G.number_of_nodes()} papers, "
                 f"{G.number_of_edges()} edges)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "degree_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_eigenvector_centrality(G: nx.DiGraph, out_dir: str) -> None:
    """Histogram of eigenvector centrality scores."""
    # Use the undirected graph for eigenvector centrality (standard approach).
    # For directed graphs, nx.eigenvector_centrality can fail to converge,
    # so we convert and use the undirected version.
    U = G.to_undirected()
    try:
        ec = nx.eigenvector_centrality(U, max_iter=1000, tol=1e-06)
    except nx.PowerIterationFailedConvergence:
        print("Warning: eigenvector centrality did not converge, "
              "trying with numpy fallback")
        ec = nx.eigenvector_centrality_numpy(U)

    scores = list(ec.values())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full distribution
    ax = axes[0]
    ax.hist(scores, bins=50, color="#e67e22", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Eigenvector Centrality")
    ax.set_ylabel("Number of Papers")
    ax.set_title("Eigenvector Centrality Distribution")
    ax.axvline(np.median(scores), color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Median = {np.median(scores):.4f}")
    ax.axvline(np.mean(scores), color="#2c3e50", linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(scores):.4f}")
    ax.legend()

    # Top papers bar chart
    ax = axes[1]
    top_n = 15
    sorted_ec = sorted(ec.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = []
    vals = []
    for pid, score in sorted_ec:
        title = G.nodes[pid].get("title", pid)
        short = title[:40] + "..." if len(title) > 40 else title
        names.append(short)
        vals.append(score)
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color="#e67e22", edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Eigenvector Centrality")
    ax.set_title(f"Top {top_n} Papers by Eigenvector Centrality")

    fig.suptitle(f"Eigenvector Centrality ({G.number_of_nodes()} papers)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "eigenvector_centrality.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_publication_year(G: nx.DiGraph, out_dir: str) -> None:
    """Histogram of publication years."""
    years = [attrs.get("year") for _, attrs in G.nodes(data=True)
             if attrs.get("year") is not None]

    fig, ax = plt.subplots(figsize=(10, 5))

    min_year, max_year = min(years), max(years)
    bins = np.arange(min_year, max_year + 2) - 0.5  # center bars on year values
    ax.hist(years, bins=bins, color="#1abc9c", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("Number of Papers")
    ax.set_xticks(range(min_year, max_year + 1))
    ax.set_title(f"Publication Year Distribution ({len(years)} papers)")

    median_year = int(np.median(years))
    ax.axvline(median_year, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Median = {median_year}")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "publication_years.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    os.makedirs(VIZ_DIR, exist_ok=True)

    print("Loading graph...")
    G, waves = load_graph()
    print(f"Loaded graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges, {waves} waves completed")

    plot_citation_counts(G, VIZ_DIR)
    plot_degree(G, VIZ_DIR)
    plot_eigenvector_centrality(G, VIZ_DIR)
    plot_publication_year(G, VIZ_DIR)

    print(f"\nAll visualizations saved to {VIZ_DIR}/")


if __name__ == "__main__":
    main()
