# AI Interpretability Citation Graph

Builds a citation graph of AI interpretability research starting from 26 seed papers (Distill Circuits thread, Anthropic blog posts, key academic papers) and expanding outward via the [Semantic Scholar API](https://www.semanticscholar.org/product/api) to discover densely connected related work.

A completed 4-wave uncapped build produces **563 papers and 4,654 citation edges** in a single connected component.

## How it works

The builder uses a **wave expansion** algorithm:

1. **Wave 0** — Resolve seed papers through the S2 API (by ID or title-match fallback), then fetch inter-seed references to build initial edges.
2. **Wave N** — For each paper added in wave N-1 (the "frontier"), fetch all citing papers. A candidate qualifies if it cites >= the overlap threshold papers already in the graph.
3. **Progressive overlap** — The overlap requirement increases each wave (`min_overlap + overlap_growth * (wave - 1)`). With defaults: wave 1 requires >= 2, wave 2 >= 3, wave 3 >= 4, etc. This keeps later-wave additions tightly connected.
4. Candidates are filtered by year range and minimum citation count, sorted by citation count (most-cited first), and optionally capped per wave.

All API responses are cached to disk (keyed by SHA-256 of the request), so rebuilds and interrupted runs avoid re-fetching.

## Setup

**Dependencies:**

```bash
pip install networkx requests plotly matplotlib numpy
```

**API key** (recommended): Get a free key from [Semantic Scholar](https://www.semanticscholar.org/product/api#api-key) and create a `.env` file:

```
S2_API_KEY=your_key_here
```

Without a key, requests are rate-limited more aggressively and more prone to 429 errors.

## Usage

### Build the graph

```bash
python cli.py build                                              # Default: 2 waves, overlap 2+, min 10 cites, 200 cap
python cli.py build --max-waves 4 --max-candidates 999999        # 4 waves, uncapped
python cli.py build --overlap-growth 0                           # Flat overlap (no increase per wave)
python cli.py build --min-citations 50                           # Only papers with >= 50 S2 citations
python cli.py build --resume                                     # Resume interrupted build
```

### View statistics

```bash
python cli.py stats
```

### Export

```bash
python cli.py export --format all       # GraphML + GEXF + interactive HTML
python cli.py export --format html      # Interactive HTML only (hover for paper details)
python cli.py export --format graphml   # GraphML only (for network tools)
python cli.py export --format gexf      # GEXF only (for Gephi)
```

### Analysis visualizations

```bash
python analysis/histograms.py
```

Generates histograms in `viz/`:
- **`citation_counts.png`** — Distribution of Semantic Scholar citation counts (linear + log scale)
- **`degree_distributions.png`** — In-degree, out-degree, and total degree within the graph
- **`eigenvector_centrality.png`** — Eigenvector centrality distribution + top-15 papers bar chart
- **`publication_years.png`** — Publication year distribution

## Seed papers

The graph starts from 20 papers across three groups:

| Group | Papers | Examples |
|-------|--------|----------|
| Distill Circuits | 4 | Zoom In, Curve Detectors, Naturally Occurring Equivariance |
| Anthropic | 3 | Induction Heads, Toy Models of Superposition, Scaling Monosemanticity |
| Academic | 13 | ROME, IOI Circuit, Grokking, Othello-GPT, ACDC, Contrastive Activation Addition, Geometry of Truth, Emergent Misalignment, Persona Vectors |

## File structure

```
cli.py              CLI entry point (build / stats / export subcommands)
config.py           Constants, API URLs, rate limits, expansion parameters
seeds.py            Curated list of 26 seed papers
api_client.py       S2 API client with disk caching, rate limiting, retries
graph_builder.py    Wave expansion algorithm (NetworkX DiGraph)
persistence.py      JSON save/load for graph state (atomic writes)
visualization.py    GraphML/GEXF export, interactive Plotly visualization, stats printing
analysis/
  histograms.py     Generates histogram visualizations into viz/
output/
  graph_state.json  Serialized graph (for resumability)
  citation_graph.*  Exported graph files (graphml, gexf)
viz/
  citation_graph.html       Interactive network visualization (Plotly)
  citation_counts.png       Citation count histograms
  degree_distributions.png  Degree distribution histograms
  eigenvector_centrality.png  Centrality analysis
  publication_years.png     Year distribution histogram
.cache/             Disk cache of S2 API responses (two-level directory)
.env                S2_API_KEY (not committed)
```

## Build parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--max-waves` | 2 | Number of expansion waves |
| `--min-overlap` | 2 | Minimum citation overlap at wave 1 |
| `--overlap-growth` | 1 | Overlap threshold increase per wave |
| `--max-candidates` | 200 | Max papers added per wave (use 999999 to uncap) |
| `--year-min` | 2017 | Earliest publication year |
| `--year-max` | 2026 | Latest publication year |
| `--min-citations` | 10 | Minimum S2 citation count for candidates |
| `--resume` | — | Resume from saved state |
