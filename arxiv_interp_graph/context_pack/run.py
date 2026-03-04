"""
Run full context pack pipeline: load graph -> seed + forward/backward -> 3 papers -> download PDFs + manifest -> optional LLM (one research question).
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from ..topic_mining.graph_loader import load_graph_for_topic_mining
except ImportError:
    from topic_mining.graph_loader import load_graph_for_topic_mining

from .download import download_context_pack_pdfs, write_manifest
from .sampling import build_context_pack

# When sending whole PDFs via OpenRouter, use the instruction-only prompt (no paper_content).
CONTEXT_PACK_QUESTION_PROMPT_WITH_PDFS = """You are an expert in LLM interpretability research. I want you to generate a new research question in the field of AI interpretability. It should be a question that can be empirically investigated by analyzing the activations and outputs of a small open source LLM. I have attached three papers as PDFs. Use them for inspiration. Your question should be theoretically important and advance the literature in a meaningful way.

Requirements:
1. Ground the question in these papers' themes (use them for inspiration).
2. Answerable via a small open-source language model through activations, outputs, or weights—no training/fine-tuning, no external datasets.
3. Specific and testable in a few days of experimentation.

Format your response as exactly two lines:
QUESTION: <one sentence question>
RATIONALE: <one short paragraph>
"""

# Fallback when not sending PDFs: use extracted text or abstracts.
CONTEXT_PACK_QUESTION_PROMPT_WITH_TEXT = """You are an expert in LLM interpretability research. I want you to generate a new research question in the field of AI interpretability. It should be a question that can be empirically investigated by analyzing the activations and outputs of a small open source LLM. Use the three papers below for inspiration. Your question should be theoretically important and advance the literature in a meaningful way.

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
        prompt = CONTEXT_PACK_QUESTION_PROMPT_WITH_PDFS
        try:
            out = generate_fn(system, prompt, pdf_paths=pdf_paths)
            return (out or "").strip()
        except TypeError:
            # generate_fn doesn't accept pdf_paths (e.g. non-OpenRouter)
            pass
    # Fallback: send extracted text or abstracts
    content = "\n".join(_paper_content_for_llm(p) for p in papers)
    prompt = CONTEXT_PACK_QUESTION_PROMPT_WITH_TEXT.format(paper_content=content)
    try:
        out = generate_fn(system, prompt)
        return (out or "").strip()
    except Exception as e:
        import sys
        print(f"[context_pack] LLM call failed: {e}", file=sys.stderr)
        return ""


def run_context_pack(
    graph_path: str | Path,
    output_dir: str | Path,
    seed_id: Optional[str] = None,
    s2_client: Optional[Any] = None,
    n_forward: int = 2,
    n_backward: int = 1,
    min_forward_from_graph: int = 0,
    seed: Optional[int] = None,
    download_pdfs: bool = True,
    llm_generate_fn: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, Any]:
    """
    Load graph, build 3-paper context pack (seed + forward + backward), download PDFs, write manifest.
    If llm_generate_fn is provided, generate 1 research question and save to output_dir.

    Returns dict with keys: papers, manifest_path, question_path (if LLM ran), question_text.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    G = load_graph_for_topic_mining(graph_path)
    papers = build_context_pack(
        G,
        seed_id=seed_id,
        s2_client=s2_client,
        n_forward=n_forward,
        n_backward=n_backward,
        min_forward_from_graph=min_forward_from_graph,
        seed=seed,
    )
    if len(papers) < 3:
        return {"papers": papers, "manifest_path": None, "question_path": None, "question_text": None}

    if download_pdfs:
        papers = download_context_pack_pdfs(papers, output_dir, s2_client)
    manifest_path = write_manifest(papers, output_dir)

    question_path = None
    question_text = None
    if llm_generate_fn:
        question_text = _generate_question_llm(papers, llm_generate_fn)
        if question_text:
            question_path = output_dir / "context_pack_question.txt"
            question_path.write_text(question_text, encoding="utf-8")

    return {
        "papers": papers,
        "manifest_path": str(manifest_path),
        "question_path": str(question_path) if question_path else None,
        "question_text": question_text,
    }
