"""
Download PDFs for context pack papers and write manifest.json.
Uses S2 openAccessPdf or arxiv PDF URL from externalIds when available.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def _safe_filename(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]+", "_", (text or "").strip())
    cleaned = cleaned.strip("_") or fallback
    return cleaned[:100]


def _get_pdf_url(paper: Dict[str, Any], s2_client: Optional[Any] = None) -> Optional[str]:
    # 1) openAccessPdf.url (if present in paper or fetch from S2)
    oa = paper.get("openAccessPdf") if isinstance(paper.get("openAccessPdf"), dict) else None
    if oa and oa.get("url"):
        return oa.get("url")
    # 2) Fetch from S2 if we have client and paperId
    if s2_client and paper.get("paperId"):
        try:
            fields = "paperId,title,year,openAccessPdf,externalIds"
            p = s2_client.get_paper(paper["paperId"], fields=fields)
            if p:
                oa = p.get("openAccessPdf") if isinstance(p.get("openAccessPdf"), dict) else None
                if oa and oa.get("url"):
                    return oa.get("url")
                ext = p.get("externalIds") or {}
                arxiv = ext.get("ArXiv") or ext.get("arXiv")
                if arxiv:
                    aid = arxiv if isinstance(arxiv, str) else None
                    if aid:
                        return f"https://arxiv.org/pdf/{aid}.pdf"
        except Exception:
            pass
    # 3) externalIds.ArXiv in paper
    ext = paper.get("externalIds") or {}
    arxiv = ext.get("ArXiv") or ext.get("arXiv")
    if arxiv:
        aid = arxiv if isinstance(arxiv, str) else None
        if aid:
            return f"https://arxiv.org/pdf/{aid}.pdf"
    return None


def _download_pdf(url: str, path: Path, timeout: int = 30) -> bool:
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False


def download_context_pack_pdfs(
    papers: List[Dict[str, Any]],
    output_dir: Path,
    s2_client: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    For each paper, resolve PDF URL (openAccessPdf or arxiv), download to output_dir/pdfs, set pdf_path.
    Returns same list with pdf_path and download_path set where successful.
    """
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(papers):
        url = _get_pdf_url(p, s2_client)
        p["pdf_url"] = url
        if url:
            title = (p.get("title") or "").strip()
            filename = _safe_filename(title, f"paper_{i+1}") + ".pdf"
            path = pdf_dir / filename
            if not path.exists():
                _download_pdf(url, path)
            if path.exists():
                p["download_path"] = str(path)
                p["pdf_path"] = str(path)
            else:
                p["download_path"] = None
                p["pdf_path"] = None
        else:
            p["download_path"] = None
            p["pdf_path"] = None
    return papers


def write_manifest(papers: List[Dict[str, Any]], output_dir: Path) -> Path:
    """
    Write manifest.json with paperId, title, year, relation, source, download_path (and pdf_path).
    """
    entries = []
    for p in papers:
        entries.append({
            "paperId": p.get("paperId"),
            "title": p.get("title"),
            "year": p.get("year"),
            "relation": p.get("relation"),
            "source": p.get("source"),
            "download_path": p.get("download_path"),
            "pdf_path": p.get("pdf_path"),
        })
    path = output_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"papers": entries}, f, ensure_ascii=False, indent=2)
    return path
