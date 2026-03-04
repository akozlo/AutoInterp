"""Constants and configuration for the citation graph builder."""

import os

# Load .env file if present (no dependency needed)
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

# Semantic Scholar API
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = os.environ.get("S2_API_KEY", "")

# Rate limiting (seconds between requests)
RATE_LIMIT_AUTH = 1.0       # With API key: ~1 req/s
RATE_LIMIT_UNAUTH = 1.1     # Without key: ~1 req/s

# Retry / backoff
MAX_RETRIES = 6
BACKOFF_BASE = 2.0          # Exponential base (seconds)
BACKOFF_MAX = 120.0          # Cap on backoff delay
JITTER_MAX = 1.0             # Random jitter added to backoff

# Pagination
PAGE_SIZE = 100              # Results per page (S2 max is 1000 for some endpoints)

# Batch endpoint
BATCH_CHUNK_SIZE = 500       # Max papers per batch request

# Disk cache
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")

# Expansion algorithm
DEFAULT_MAX_WAVES = 2
DEFAULT_MIN_CITATION_OVERLAP = 2
DEFAULT_OVERLAP_GROWTH = 1          # Overlap threshold increases by this much per wave
DEFAULT_MAX_CANDIDATES_PER_WAVE = 200
DEFAULT_YEAR_MIN = 2017
DEFAULT_YEAR_MAX = 2026
DEFAULT_MIN_CITATIONS = 10

# Paper fields to fetch (abstract for topic package / LLM summarization)
PAPER_FIELDS = "paperId,externalIds,openAccessPdf,title,year,citationCount,authors,venue,url,abstract"
CITATION_FIELDS = "paperId,externalIds,title,year,citationCount"
REFERENCE_FIELDS = "paperId"

# Output
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
GRAPH_STATE_PATH = os.path.join(OUTPUT_DIR, "graph_state.json")
