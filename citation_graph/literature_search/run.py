"""
Run full literature search pipeline: load graph -> seed + forward/backward -> 3 papers -> download PDFs + manifest -> optional LLM (one research question).
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import networkx as nx

from .download import download_literature_search_pdfs, write_manifest
from .sampling import _has_download_url, _node_to_paper, build_literature_search

logger = logging.getLogger(__name__)


def _load_graph(graph_path: Union[str, Path]) -> nx.DiGraph:
    """Load citation graph from graph_state.json or GraphML/GEXF fallback."""
    path = Path(graph_path)
    if not path.exists():
        raise FileNotFoundError(f"Graph path does not exist: {path}")

    if path.suffix == ".json" or path.name == "graph_state.json":
        with open(path) as f:
            state = json.load(f)
        G = nx.DiGraph()
        for entry in state["nodes"]:
            node_id = entry.pop("id")
            G.add_node(node_id, **entry)
        for entry in state["edges"]:
            src = entry.pop("source")
            dst = entry.pop("target")
            G.add_edge(src, dst, **entry)
        return G

    if path.suffix.lower() == ".graphml":
        G = nx.read_graphml(path)
    elif path.suffix.lower() == ".gexf":
        G = nx.read_gexf(path)
    else:
        raise ValueError(f"Unsupported graph format: {path.suffix}. Use .json or .graphml")
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)
    return G

# When sending whole PDFs via OpenRouter, use the instruction-only prompt (no paper_content).
LITERATURE_SEARCH_QUESTION_PROMPT_WITH_PDFS = """You are an expert in LLM interpretability research. I want you to generate a new research question in the field of AI interpretability. It should be a question that can be empirically investigated by analyzing the activations and outputs of a small open source LLM. I have attached three papers as PDFs. Use them for inspiration. Your question should be theoretically important and advance the literature in a meaningful way.

Requirements:
1. Ground the question in these papers' themes (use them for inspiration).
2. Answerable via a small open-source language model through activations, outputs, or weights—no training/fine-tuning, no external datasets.
3. Specific and testable in a few days of experimentation.

Format your response as exactly two lines:
QUESTION: <one sentence question>
RATIONALE: <one short paragraph>
"""

# Fallback when not sending PDFs: use extracted text or abstracts.
LITERATURE_SEARCH_QUESTION_PROMPT_WITH_TEXT = """You are an expert in LLM interpretability research. I want you to generate a new research question in the field of AI interpretability. It should be a question that can be empirically investigated by analyzing the activations and outputs of a small open source LLM. Use the three papers below for inspiration. Your question should be theoretically important and advance the literature in a meaningful way.

Requirements:
1. Ground the question in these papers' themes (use them for inspiration).
2. Answerable via a small open-source language model through activations, outputs, or weights—no training/fine-tuning, no external datasets.
3. Specific and testable in a few days of experimentation.

Format your response as exactly two lines:
QUESTION: <one sentence question>
RATIONALE: <one short paragraph>

Content from the three papers (excerpts from the PDFs):
---
{paper_content}
---
"""


def _extract_pdf_text(path: Path, max_pages: int = 6, max_chars: int = 6000) -> str:
    """Extract text from PDF for LLM; uses pypdf if available."""
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages[:max_pages]:
            page_text = (page.extract_text() or "").strip()
            if page_text:
                texts.append(page_text)
        merged = "\n".join(texts).strip()
        return merged[:max_chars] if max_chars and len(merged) > max_chars else merged
    except Exception:
        return ""


def _extract_html_text(path: Path, max_chars: int = 6000) -> str:
    """Extract readable text from an HTML article (Distill, Transformer Circuits, etc.)."""
    import re as _re
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        # Remove script and style blocks
        raw = _re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=_re.DOTALL | _re.IGNORECASE)
        # Strip HTML tags
        text = _re.sub(r"<[^>]+>", " ", raw)
        # Collapse whitespace
        text = _re.sub(r"\s+", " ", text).strip()
        return text[:max_chars] if max_chars and len(text) > max_chars else text
    except Exception:
        return ""


def _extract_article_text(path: Path, max_chars: int = 6000, max_pdf_pages: int = 6) -> str:
    """Extract text from a downloaded article (PDF or HTML) based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".html" or suffix == ".htm":
        return _extract_html_text(path, max_chars=max_chars)
    return _extract_pdf_text(path, max_pages=max_pdf_pages, max_chars=max_chars)


def _paper_content_for_llm(
    p: Dict[str, Any],
    max_chars: int = 6000,
    max_pdf_pages: int = 6,
    max_abstract: int = 600,
) -> str:
    """Build content for LLM: prefer article excerpt (PDF or HTML) when available, else abstract."""
    title = (p.get("title") or "").strip()
    year = p.get("year") or "n.d."
    rel = p.get("relation", "")
    header = f"[{rel.upper()}] {title} ({year})\n\n"
    dl_path = p.get("download_path") or p.get("pdf_path")
    if dl_path and Path(dl_path).exists():
        text = _extract_article_text(Path(dl_path), max_chars=max_chars, max_pdf_pages=max_pdf_pages)
        if text:
            return header + text + "\n\n"
    abstract = (p.get("abstract") or "").strip()
    if len(abstract) > max_abstract:
        abstract = abstract[:max_abstract] + "..."
    return header + abstract + "\n\n"


def _generate_question_llm(
    papers: List[Dict[str, Any]],
    generate_fn: Callable[..., str],
) -> str:
    system = "You output only the requested format. No markdown, no extra text."
    pdf_paths = []
    for p in papers:
        path = p.get("download_path") or p.get("pdf_path")
        if path and Path(path).exists() and Path(path).suffix.lower() == ".pdf":
            pdf_paths.append(path)
    # Prefer sending whole PDFs to OpenRouter (Sonnet 4.6) when we have all 3
    if len(pdf_paths) == 3:
        prompt = LITERATURE_SEARCH_QUESTION_PROMPT_WITH_PDFS
        try:
            out = generate_fn(system, prompt, pdf_paths=pdf_paths)
            return (out or "").strip()
        except TypeError:
            # generate_fn doesn't accept pdf_paths (e.g. non-OpenRouter)
            pass
    # Fallback: send extracted text or abstracts
    content = "\n".join(_paper_content_for_llm(p) for p in papers)
    prompt = LITERATURE_SEARCH_QUESTION_PROMPT_WITH_TEXT.format(paper_content=content)
    try:
        out = generate_fn(system, prompt)
        return (out or "").strip()
    except Exception as e:
        import sys
        print(f"[literature_search] LLM call failed: {e}", file=sys.stderr)
        return ""


def _retry_failed_downloads(
    papers: List[Dict[str, Any]],
    G: nx.DiGraph,
    output_dir: Path,
    s2_client: Optional[Any],
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Replace papers whose download failed with other downloadable papers
    from the graph. Tries up to *max_retries* replacements per slot.
    """
    rng = random.Random()
    tried_ids = {p["paperId"] for p in papers}

    for idx, p in enumerate(papers):
        if p.get("download_path") is not None:
            continue  # downloaded OK

        title = (p.get("title") or p.get("paperId", "?"))[:60]
        logger.info("Download failed for '%s'; attempting replacement...", title)

        # Gather candidates: downloadable nodes not already in the pack
        candidates = [
            nid for nid in G.nodes()
            if _has_download_url(G, nid) and nid not in tried_ids
        ]
        rng.shuffle(candidates)

        replaced = False
        for attempt, nid in enumerate(candidates[:max_retries]):
            tried_ids.add(nid)
            replacement = _node_to_paper(G, nid, p.get("relation", "replacement"), "graph")
            downloaded = download_literature_search_pdfs([replacement], output_dir, s2_client)
            replacement = downloaded[0]
            if replacement.get("download_path") is not None:
                logger.info(
                    "Replaced '%s' with '%s' (attempt %d)",
                    title, (replacement.get("title") or "")[:60], attempt + 1,
                )
                papers[idx] = replacement
                replaced = True
                break
            logger.debug("Replacement attempt %d also failed, trying next...", attempt + 1)

        if not replaced:
            logger.warning("Could not find a downloadable replacement for '%s'", title)

    return papers


def run_literature_search(
    graph_path: Union[str, Path],
    output_dir: Union[str, Path],
    seed_id: Optional[str] = None,
    s2_client: Optional[Any] = None,
    n_forward: int = 2,
    n_backward: int = 1,
    min_forward_from_graph: int = 0,
    seed: Optional[int] = None,
    download_pdfs: bool = True,
    llm_generate_fn: Optional[Callable[[str, str], str]] = None,
    n_papers: int = 3,
) -> Dict[str, Any]:
    """
    Load graph, build literature search (seed + forward + backward), download PDFs, write manifest.
    If llm_generate_fn is provided, generate 1 research question and save to output_dir.

    Returns dict with keys: papers, manifest_path, question_path (if LLM ran), question_text.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    G = _load_graph(graph_path)
    papers = build_literature_search(
        G,
        seed_id=seed_id,
        s2_client=s2_client,
        n_forward=n_forward,
        n_backward=n_backward,
        min_forward_from_graph=min_forward_from_graph,
        seed=seed,
        n_papers=n_papers,
    )
    if len(papers) < n_papers:
        return {"papers": papers, "manifest_path": None, "question_path": None, "question_text": None}

    if download_pdfs:
        papers = download_literature_search_pdfs(papers, output_dir, s2_client)
        # Retry-replace any papers whose download failed
        n_failed = sum(1 for p in papers if p.get("download_path") is None)
        if n_failed:
            logger.info("%d of %d downloads failed; retrying with replacements...", n_failed, len(papers))
            papers = _retry_failed_downloads(papers, G, output_dir, s2_client)
    manifest_path = write_manifest(papers, output_dir)

    question_path = None
    question_text = None
    if llm_generate_fn:
        question_text = _generate_question_llm(papers, llm_generate_fn)
        if question_text:
            question_path = output_dir / "literature_search_question.txt"
            question_path.write_text(question_text, encoding="utf-8")

    return {
        "papers": papers,
        "manifest_path": str(manifest_path),
        "question_path": str(question_path) if question_path else None,
        "question_text": question_text,
    }
