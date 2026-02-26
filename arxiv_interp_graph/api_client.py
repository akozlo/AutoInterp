"""Semantic Scholar API client with disk caching, rate limiting, and retries."""

import hashlib
import json
import os
import random
import time
from urllib.parse import urlencode

import requests

from config import (
    BACKOFF_BASE,
    BACKOFF_MAX,
    BATCH_CHUNK_SIZE,
    CACHE_DIR,
    CITATION_FIELDS,
    JITTER_MAX,
    MAX_RETRIES,
    PAGE_SIZE,
    PAPER_FIELDS,
    RATE_LIMIT_AUTH,
    RATE_LIMIT_UNAUTH,
    REFERENCE_FIELDS,
    S2_API_BASE,
    S2_API_KEY,
)


class SemanticScholarClient:
    def __init__(self, api_key: str = S2_API_KEY, cache_dir: str = CACHE_DIR):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.rate_limit = RATE_LIMIT_AUTH if api_key else RATE_LIMIT_UNAUTH
        self._last_request_time = 0.0
        self.session = requests.Session()
        if api_key:
            self.session.headers["x-api-key"] = api_key
        os.makedirs(cache_dir, exist_ok=True)

    # --- Cache helpers ---

    def _cache_key(self, method: str, url: str, params: dict | None, body: str | None) -> str:
        raw = f"{method}|{url}|{json.dumps(params, sort_keys=True)}|{body or ''}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> str:
        subdir = os.path.join(self.cache_dir, key[:2])
        os.makedirs(subdir, exist_ok=True)
        return os.path.join(subdir, f"{key}.json")

    def _cache_get(self, key: str):
        path = self._cache_path(key)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def _cache_set(self, key: str, data):
        path = self._cache_path(key)
        with open(path, "w") as f:
            json.dump(data, f)

    # --- Rate limiting & retries ---

    def _wait_for_rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

    def _request(self, method: str, url: str, params: dict | None = None,
                 json_body=None, cache: bool = True) -> dict | list | None:
        body_str = json.dumps(json_body, sort_keys=True) if json_body else None
        if cache:
            key = self._cache_key(method, url, params, body_str)
            cached = self._cache_get(key)
            if cached is not None:
                return cached

        for attempt in range(MAX_RETRIES + 1):
            self._wait_for_rate_limit()
            self._last_request_time = time.time()
            try:
                resp = self.session.request(method, url, params=params, json=json_body, timeout=30)
            except requests.RequestException as e:
                print(f"  [request error] {e}")
                if attempt == MAX_RETRIES:
                    return None
                self._backoff(attempt)
                continue

            if resp.status_code == 200:
                data = resp.json()
                if cache:
                    self._cache_set(key, data)
                return data
            elif resp.status_code == 404:
                return None
            elif resp.status_code in (429, 500, 502, 503, 504):
                print(f"  [HTTP {resp.status_code}] retrying ({attempt + 1}/{MAX_RETRIES})...")
                if attempt == MAX_RETRIES:
                    return None
                self._backoff(attempt)
            else:
                print(f"  [HTTP {resp.status_code}] {resp.text[:200]}")
                return None
        return None

    def _backoff(self, attempt: int):
        delay = min(BACKOFF_BASE ** (attempt + 1), BACKOFF_MAX)
        delay += random.uniform(0, JITTER_MAX)
        time.sleep(delay)

    # --- Paper lookup ---

    def get_paper(self, paper_id: str, fields: str = PAPER_FIELDS) -> dict | None:
        """Fetch a single paper by S2 ID, ARXIV:id, DOI:id, or URL:url."""
        url = f"{S2_API_BASE}/paper/{paper_id}"
        return self._request("GET", url, params={"fields": fields})

    def search_paper_by_title(self, title: str, fields: str = PAPER_FIELDS) -> dict | None:
        """Fallback: use /paper/search/match for title-based lookup."""
        url = f"{S2_API_BASE}/paper/search/match"
        params = {"query": title, "fields": fields}
        result = self._request("GET", url, params=params)
        if result and isinstance(result, dict) and "data" in result:
            matches = result["data"]
            if matches:
                return matches[0]
        return None

    def resolve_paper(self, paper_id: str | None, title: str, fields: str = PAPER_FIELDS) -> dict | None:
        """Try ID-based lookup first, then fall back to title match."""
        if paper_id:
            result = self.get_paper(paper_id, fields=fields)
            if result:
                return result
            print(f"  ID lookup failed for {paper_id}, trying title match...")
        return self.search_paper_by_title(title, fields=fields)

    # --- Citations & references ---

    def get_citations(self, paper_id: str, fields: str = CITATION_FIELDS) -> list[dict]:
        """Fetch all papers that cite the given paper (paginated)."""
        return self._paginate(f"{S2_API_BASE}/paper/{paper_id}/citations",
                              fields=fields, result_key="citingPaper")

    def get_references(self, paper_id: str, fields: str = REFERENCE_FIELDS) -> list[dict]:
        """Fetch all papers referenced by the given paper (paginated)."""
        return self._paginate(f"{S2_API_BASE}/paper/{paper_id}/references",
                              fields=fields, result_key="citedPaper")

    def _paginate(self, url: str, fields: str, result_key: str) -> list[dict]:
        all_results = []
        offset = 0
        while True:
            params = {"fields": fields, "offset": offset, "limit": PAGE_SIZE}
            data = self._request("GET", url, params=params)
            if not data or "data" not in data:
                break
            items = data["data"]
            if not items:
                break
            for item in items:
                if item is None:
                    continue
                paper = item.get(result_key)
                if paper and paper.get("paperId"):
                    all_results.append(paper)
            if not items or len(items) < PAGE_SIZE or "next" not in data:
                break
            offset += PAGE_SIZE
        return all_results

    # --- Batch endpoint ---

    def get_papers_batch(self, paper_ids: list[str], fields: str = PAPER_FIELDS) -> list[dict]:
        """Fetch metadata for many papers efficiently via POST /paper/batch."""
        results = []
        for i in range(0, len(paper_ids), BATCH_CHUNK_SIZE):
            chunk = paper_ids[i:i + BATCH_CHUNK_SIZE]
            url = f"{S2_API_BASE}/paper/batch"
            params = {"fields": fields}
            data = self._request("POST", url, params=params, json_body={"ids": chunk}, cache=True)
            if data and isinstance(data, list):
                results.extend([p for p in data if p is not None])
        return results
