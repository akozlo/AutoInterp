"""
Download articles for literature search papers and write manifest.json.

Supports both PDF (arxiv, conference open access) and HTML (Distill, Transformer
Circuits Thread) articles. Prefers arxiv PDF URLs constructed from stored arxiv_id.
Falls back to stored open_access_url, then live S2 API lookup.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


def _safe_filename(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]+", "_", (text or "").strip())
    cleaned = cleaned.strip("_") or fallback
    return cleaned[:100]


def _arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def _get_article_url(paper: Dict[str, Any], s2_client: Optional[Any] = None) -> Optional[str]:
    """
    Resolve a download URL for the paper. Priority order:

    1. Stored arxiv_id (from graph) -> construct arxiv PDF URL (no API call)
    2. Stored open_access_url (from graph enrichment, e.g. Distill DOI)
    3. openAccessPdf already in paper dict
    4. Live S2 API call -> try externalIds.ArXiv first, then openAccessPdf
    """
    title = (paper.get("title") or paper.get("paperId") or "unknown")[:80]

    # 1) Stored arxiv_id from graph — best case, no API call needed
    arxiv_id = paper.get("arxiv_id")
    if arxiv_id and isinstance(arxiv_id, str):
        url = _arxiv_pdf_url(arxiv_id)
        logger.debug("Using stored arxiv_id for '%s': %s", title, url)
        return url

    # 2) Stored open_access_url (for Distill, Transformer Circuits, etc.)
    oa_url = paper.get("open_access_url")
    if oa_url and isinstance(oa_url, str):
        logger.debug("Using stored open_access_url for '%s': %s", title, oa_url)
        return oa_url

    # 3) openAccessPdf already in paper dict (e.g., from a prior API enrichment)
    oa = paper.get("openAccessPdf") if isinstance(paper.get("openAccessPdf"), dict) else None
    if oa and oa.get("url"):
        logger.debug("Using openAccessPdf for '%s': %s", title, oa["url"])
        return oa["url"]

    # 4) Live S2 API call as last resort
    if s2_client and paper.get("paperId"):
        logger.debug("No stored URL for '%s'; fetching from S2 API...", title)
        try:
            fields = "paperId,title,year,openAccessPdf,externalIds"
            p = s2_client.get_paper(paper["paperId"], fields=fields)
            if p:
                # Prefer arxiv from externalIds
                ext = p.get("externalIds") or {}
                arxiv = ext.get("ArXiv") or ext.get("arXiv")
                if arxiv and isinstance(arxiv, str):
                    url = _arxiv_pdf_url(arxiv)
                    logger.debug("S2 API returned arxiv_id for '%s': %s", title, url)
                    return url
                # Fall back to openAccessPdf
                oa = p.get("openAccessPdf") if isinstance(p.get("openAccessPdf"), dict) else None
                if oa and oa.get("url"):
                    logger.debug("S2 API returned openAccessPdf for '%s': %s", title, oa["url"])
                    return oa["url"]
                logger.warning("S2 API returned no source for '%s'", title)
        except Exception as e:
            logger.warning("S2 API call failed for '%s': %s", title, e)

    logger.warning("No download URL found for '%s'", title)
    return None


def _download_article(url: str, base_path: Path, timeout: int = 30) -> Tuple[Optional[Path], str]:
    """
    Download an article from url. Returns (saved_path, format) where format
    is 'pdf' or 'html'. Follows redirects (e.g. DOI -> distill.pub).

    If the response is HTML, saves with .html extension instead of .pdf.
    Returns (None, '') on failure.
    """
    try:
        r = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "")

        if "text/html" in content_type:
            # HTML article (Distill, Transformer Circuits, etc.)
            html_path = base_path.with_suffix(".html")
            html_path.parent.mkdir(parents=True, exist_ok=True)
            with html_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            return html_path, "html"

        # Assume PDF for application/pdf or other binary types
        pdf_path = base_path.with_suffix(".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with pdf_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        return pdf_path, "pdf"

    except requests.exceptions.HTTPError as e:
        logger.warning("HTTP error downloading %s: %s", url, e)
        return None, ""
    except requests.exceptions.Timeout:
        logger.warning("Timeout downloading %s (limit=%ds)", url, timeout)
        return None, ""
    except Exception as e:
        logger.warning("Failed to download %s: %s", url, e)
        return None, ""


def download_literature_search_pdfs(
    papers: List[Dict[str, Any]],
    output_dir: Path,
    s2_client: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    For each paper, resolve article URL and download to output_dir/pdfs/.
    Handles both PDF and HTML articles. Returns same list with download metadata set.
    """
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(papers):
        title = (p.get("title") or f"paper_{i+1}")[:80]
        url = _get_article_url(p, s2_client)
        p["article_url"] = url

        if not url:
            p["download_path"] = None
            p["pdf_path"] = None
            p["article_format"] = None
            continue

        base_name = _safe_filename(p.get("title"), f"paper_{i+1}")
        base_path = pdf_dir / base_name  # extension added by _download_article

        # Check if already downloaded (either format)
        existing_pdf = base_path.with_suffix(".pdf")
        existing_html = base_path.with_suffix(".html")
        if existing_pdf.exists():
            logger.debug("Already downloaded '%s': %s", title, existing_pdf)
            p["download_path"] = str(existing_pdf)
            p["pdf_path"] = str(existing_pdf)
            p["article_format"] = "pdf"
            continue
        if existing_html.exists():
            logger.debug("Already downloaded '%s': %s", title, existing_html)
            p["download_path"] = str(existing_html)
            p["pdf_path"] = str(existing_html)
            p["article_format"] = "html"
            continue

        logger.debug("Downloading '%s' from %s", title, url)
        saved_path, fmt = _download_article(url, base_path)

        if saved_path and saved_path.exists():
            logger.info("Downloaded: %s (%s)", title, fmt.upper())
            p["download_path"] = str(saved_path)
            p["pdf_path"] = str(saved_path)
            p["article_format"] = fmt
        else:
            logger.warning("Download failed for '%s'.", title)
            p["download_path"] = None
            p["pdf_path"] = None
            p["article_format"] = None

    n_pdf = sum(1 for p in papers if p.get("article_format") == "pdf")
    n_html = sum(1 for p in papers if p.get("article_format") == "html")
    n_fail = sum(1 for p in papers if p.get("download_path") is None)
    logger.info("Downloaded %d PDF(s), %d HTML article(s), %d failed.", n_pdf, n_html, n_fail)
    return papers


def write_manifest(papers: List[Dict[str, Any]], output_dir: Path) -> Path:
    """
    Write manifest.json with paper metadata and download results.
    """
    entries = []
    for p in papers:
        entries.append({
            "paperId": p.get("paperId"),
            "title": p.get("title"),
            "year": p.get("year"),
            "arxiv_id": p.get("arxiv_id"),
            "relation": p.get("relation"),
            "source": p.get("source"),
            "article_url": p.get("article_url"),
            "article_format": p.get("article_format"),
            "download_path": p.get("download_path"),
        })
    path = output_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"papers": entries}, f, ensure_ascii=False, indent=2)
    return path
