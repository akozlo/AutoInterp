"""
Upload a completed AutoInterp project to Future Science.

Endpoint: POST https://future-science.org/api/v1/contributions/api-bots
Auth:     x-api-key header
Format:   multipart/form-data

Fields sent:
  data                   - JSON string with contribution metadata
  file                   - Main Markdown report
  cover                  - First figure image (optional)
  additionalMaterialsZip - ZIP of repo/ scripts, results, notebooks, README (optional)

API-only metadata fields:
  researchOrchestrator   - Name of human researcher who orchestrated the AI-assisted research
  agentDescription       - Short description of the agent (displayed as italic text)
  agentName              - Name of the AI agent (used as author if author array is omitted)
"""

import io
import json
import logging
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import re

import requests

logger = logging.getLogger(__name__)

FUTURE_SCIENCE_ENDPOINT = "https://future-science.org/api/v1/contributions/api-bots"

# Model ID → human-readable display name (for agent_description templating)
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-opus-4-6": "Claude Opus 4.6",
    "gpt-5.4": "GPT-5.4",
    "gpt-5-2025-08-07": "GPT-5",
    "gpt-5-mini-2025-08-07": "GPT-5-mini",
}


# ---------------------------------------------------------------------------
# Markdown title/abstract stripping
# ---------------------------------------------------------------------------

def strip_title_and_abstract(content: str) -> str:
    """
    Remove the title (# heading) and abstract (## Abstract section) from
    the beginning of a Markdown document.  Returns the remainder starting
    from the first ## heading that is not "Abstract".

    The original file on disk is never modified — this operates on a string.
    """
    lines = content.splitlines(keepends=True)
    # Find the first ## heading that is NOT "## Abstract"
    body_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^##\s+", stripped) and not re.match(r"^##\s+Abstract", stripped, re.IGNORECASE):
            body_start = i
            break

    if body_start is None:
        return content

    # Strip any leading blank lines or --- separators before the body heading
    while body_start > 0 and lines[body_start - 1].strip() in ("", "---"):
        body_start -= 1

    result = "".join(lines[body_start:])
    # Clean up any remaining leading blank lines / separators
    result = re.sub(r"^(\s*\n|\s*---\s*\n)*", "", result)
    return result


def safe_strip_title_and_abstract(content: str, filename: str = "") -> str:
    """
    Strip title/abstract with safety checks.  Raises RuntimeError if the
    result looks wrong (nothing removed, or too much removed).
    """
    stripped = strip_title_and_abstract(content)
    orig_len = len(content)
    stripped_len = len(stripped)

    if stripped_len == orig_len:
        raise RuntimeError(
            f"Title/abstract stripping removed nothing from {filename}. "
            "The markdown may use a non-standard heading structure. "
            "Review the file and publish with --dry-run first."
        )

    removed_pct = (orig_len - stripped_len) / orig_len * 100
    if removed_pct > 20:
        raise RuntimeError(
            f"Title/abstract stripping removed {removed_pct:.0f}% of {filename} "
            f"({orig_len - stripped_len} of {orig_len} chars). "
            "This likely indicates a non-standard document structure. "
            "Review the file and publish with --dry-run first."
        )

    return stripped


# ---------------------------------------------------------------------------
# Publish stamping — mark run_metadata.json after successful upload
# ---------------------------------------------------------------------------

def _stamp_published(project_dir: Path, document_id: str = "") -> None:
    """Write published=True and published_date into run_metadata.json."""
    from datetime import datetime, timezone

    meta_path = project_dir / "run_metadata.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {meta_path}: {e}")

    meta["published"] = True
    meta["published_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if document_id:
        meta["published_document_id"] = document_id

    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Stamped {meta_path} as published (documentId: {document_id})")
    except Exception as e:
        raise RuntimeError(
            f"Failed to stamp {meta_path} as published after successful API submission. "
            f"The paper was uploaded to Future Science but metadata write failed: {e}"
        )


def _unstamp_published(project_dir: Path) -> None:
    """Remove the published stamp from run_metadata.json (used on definitive 4xx rejection)."""
    meta_path = project_dir / "run_metadata.json"
    if not meta_path.exists():
        return
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return
    meta.pop("published", None)
    meta.pop("published_date", None)
    meta.pop("published_document_id", None)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Un-stamped {meta_path} (definitive API rejection)")


def is_published(project_dir: Path) -> bool:
    """Check whether a project has already been published to Future Science."""
    meta_path = project_dir / "run_metadata.json"
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            return json.load(f).get("published", False)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Project file discovery
# ---------------------------------------------------------------------------

def find_report(project_dir: Path) -> Optional[Path]:
    """
    Return the main Markdown report file for a completed project.

    Search order:
    1. repo/paper/*.md  (excluding README.md, research_question.md, *_log.md)
    2. reports/*.md     (excluding README.md, research_question.md, *_log.md)

    Prefers non-revision files (picks foo.md over foo_revision_1.md).
    """
    def is_valid_report(p: Path) -> bool:
        """Check if a markdown file should be considered as a report."""
        name_lower = p.name.lower()
        # Exclude metadata, log, and question files
        if name_lower in ("readme.md", "research_question.md"):
            return False
        if name_lower.endswith("_log.md") or "reporter_log" in name_lower:
            return False
        return True

    def sort_key(p: Path) -> tuple:
        """Sort by: (has_revision, -mtime). Prefers non-revision files."""
        has_revision = "_revision" in p.name.lower()
        mtime = p.stat().st_mtime
        return (has_revision, -mtime)

    paper_dir = project_dir / "repo" / "paper"
    if paper_dir.is_dir():
        candidates = sorted(
            [p for p in paper_dir.glob("*.md") if is_valid_report(p)],
            key=sort_key,
        )
        if candidates:
            return candidates[0]

    reports_dir = project_dir / "reports"
    if reports_dir.is_dir():
        candidates = sorted(
            [p for p in reports_dir.glob("*.md") if is_valid_report(p)],
            key=sort_key,
        )
        if candidates:
            return candidates[0]

    return None


def find_cover_image(project_dir: Path) -> Optional[Path]:
    """Return the first figure PNG to use as the cover image."""
    for search_dir in [project_dir / "repo" / "paper", project_dir / "visualizations"]:
        if not search_dir.is_dir():
            continue
        pngs = sorted(search_dir.glob("*.png"))
        if pngs:
            return pngs[0]
    return None


def build_paper_zip(project_dir: Path) -> Optional[bytes]:
    """
    Build a ZIP archive of the entire repo/paper/ directory (markdown + figures).

    Markdown files have their title and abstract stripped (those are sent via
    API metadata fields instead) so they are not rendered twice on Future Science.

    Excludes metadata/log files (research_question.md, *_log.md, Reporter_log.md).

    Returns None if the paper directory doesn't exist or is empty.
    """
    paper_dir = project_dir / "repo" / "paper"
    if not paper_dir.is_dir():
        return None

    def should_include_md(filename: str) -> bool:
        """Check if a markdown file should be included (not a metadata/log file)."""
        name_lower = filename.lower()
        if name_lower == "research_question.md":
            return False
        if name_lower.endswith("_log.md") or "reporter_log" in name_lower:
            return False
        return True

    buf = io.BytesIO()
    any_added = False

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(paper_dir.rglob("*")):
            if f.is_file():
                arcname = str(f.relative_to(paper_dir))
                if f.suffix.lower() == ".md":
                    # Skip metadata/log files
                    if not should_include_md(f.name):
                        continue
                    content = f.read_text(encoding="utf-8", errors="replace")
                    zf.writestr(arcname, safe_strip_title_and_abstract(content, arcname))
                else:
                    zf.write(f, arcname)
                any_added = True

    if not any_added:
        return None
    return buf.getvalue()


def build_additional_zip(project_dir: Path) -> Optional[bytes]:
    """
    Build a ZIP archive containing repo/ supplementary materials:
    scripts/, results/, notebooks/, and README.md.

    Returns None if none of those exist.
    """
    repo_dir = project_dir / "repo"
    if not repo_dir.is_dir():
        return None

    include_dirs = ["scripts", "results", "notebooks"]
    buf = io.BytesIO()
    any_added = False

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        readme = repo_dir / "README.md"
        if readme.exists():
            zf.write(readme, "README.md")
            any_added = True

        for subdir_name in include_dirs:
            subdir = repo_dir / subdir_name
            if not subdir.is_dir():
                continue
            for f in sorted(subdir.rglob("*")):
                if f.is_file():
                    arcname = f.relative_to(repo_dir)
                    zf.write(f, str(arcname))
                    any_added = True

    if not any_added:
        return None
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Metadata extraction via Claude CLI
# ---------------------------------------------------------------------------

def extract_metadata_with_claude(report_path: Path) -> Dict[str, Any]:
    """
    Use the Claude Code CLI to extract title, abstract, and keywords from the
    report Markdown file.

    Returns a dict with keys:
        "title"    – str
        "abstract" – str
        "keywords" – list of {"text": str} dicts (3–8 items)

    Raises RuntimeError if the claude CLI is not found or exits non-zero.
    """
    if not shutil.which("claude"):
        raise RuntimeError(
            "claude CLI not found. Install Claude Code to enable metadata extraction."
        )

    prompt = (
        f"Read the Markdown file at {report_path} and extract the following metadata. "
        "Return ONLY a JSON object with these exact fields:\n"
        '  "title":    the main title of the paper (string)\n'
        '  "abstract": the full abstract text (string)\n'
        '  "keywords": an array of 3–8 keywords, each as {"text": "<keyword>"}\n\n'
        "If a field is not explicitly present, infer the best value from the content. "
        "Output ONLY the raw JSON object — no markdown fences, no explanation."
    )

    result = subprocess.run(
        ["claude", "-p", "--dangerously-skip-permissions", prompt],
        capture_output=True,
        text=True,
        cwd=str(report_path.parent),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Claude CLI exited with code {result.returncode}: {result.stderr[:500]}"
        )

    output = result.stdout.strip()
    # Strip markdown fences if the model adds them anyway
    if output.startswith("```"):
        lines = output.splitlines()
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        output = "\n".join(lines[1:end])

    return json.loads(output)


# ---------------------------------------------------------------------------
# Main publish function
# ---------------------------------------------------------------------------

def publish_project(
    project_dir: Path,
    api_key: str,
    initiative_id: str = "",
    contributor_email: str = "",
    author_first_name: str = "",
    author_last_name: str = "",
    author_institution: str = "",
    author_email: str = "",
    agent_name: str = "AutoInterp",
    research_orchestrator: str = "",
    agent_description: str = "",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Upload a completed AutoInterp project to Future Science.

    Args:
        project_dir:            Path to the project directory.
        api_key:                Future Science API key.
        initiative_id:          documentId of the target initiative (may be empty).
        contributor_email:      Email of the submitting contributor.
        author_*:               Author metadata fields (optional; omitted if agent_name is set).
        agent_name:             Name of the AI agent (used as author when author fields are empty).
        research_orchestrator:  Name of the human researcher who orchestrated the research.
        agent_description:      Short description of the agent (displayed as italic text).
        dry_run:                If True, print what would be sent but don't POST.

    Returns:
        API response dict on success, or raises on error.
    """
    project_dir = Path(project_dir).resolve()
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    # --- Locate report ---
    report_path = find_report(project_dir)
    if report_path is None:
        raise FileNotFoundError(
            f"No Markdown report found in {project_dir}. "
            "Run the full pipeline first (repo assembly or report generation)."
        )
    logger.info(f"Report: {report_path}")

    # --- Read run metadata (provider/model/status) ---
    run_meta_path = project_dir / "run_metadata.json"
    run_meta: Dict[str, Any] = {}
    if run_meta_path.exists():
        try:
            with open(run_meta_path) as f:
                run_meta = json.load(f)
        except Exception:
            pass

    # Block publishing incomplete projects
    run_status = run_meta.get("status", "")
    if run_status and run_status != "completed":
        raise RuntimeError(
            f"Project run status is '{run_status}', not 'completed'. "
            "Only completed projects can be published."
        )

    # Template {model} in agent_description from run metadata
    model_id = run_meta.get("model", "")
    run_model_display = MODEL_DISPLAY_NAMES.get(model_id, model_id)
    if agent_description and run_model_display:
        agent_description = agent_description.replace("{model}", run_model_display)

    # --- Extract metadata via Claude ---
    metadata = extract_metadata_with_claude(report_path)
    title = metadata.get("title") or report_path.stem.replace("_", " ")
    abstract = metadata.get("abstract") or title
    keywords = metadata.get("keywords") or [{"text": "mechanistic interpretability"}]

    payload: Dict[str, Any] = {
        "data": {
            "title": title,
            "type": "Article",
            "abstract": abstract,
            "language": "English",
            "langId": "",
            "keywords": keywords,
            "coauthorsNotified": True,
            "isFormatCompliant": True,
            "sourceLinkCorrect": True,
            "isOriginal": True,
            "isMarkdown": True,
            "comments": "",
            "linkOriginalContribution": "",
        }
    }

    # Use explicit author fields if provided, otherwise fall back to agentName
    has_author = any([author_first_name, author_last_name])
    if has_author:
        payload["data"]["author"] = [{
            "firstName": author_first_name,
            "lastName": author_last_name,
            "institution": author_institution,
            "email": author_email or contributor_email,
            "links": [],
        }]
    elif agent_name:
        payload["data"]["agentName"] = agent_name

    if initiative_id:
        payload["data"]["initiative"] = initiative_id
    if contributor_email:
        payload["data"]["contributorEmail"] = contributor_email
    if research_orchestrator:
        payload["data"]["researchOrchestrator"] = research_orchestrator
    if agent_description:
        payload["data"]["agentDescription"] = agent_description

    # --- Build multipart files dict ---
    paper_zip = build_paper_zip(project_dir)
    if paper_zip:
        logger.info("Paper ZIP (markdown + figures): included")
        files: Dict[str, Any] = {
            "data": (None, json.dumps(payload), "application/json"),
            "file": ("paper.zip", paper_zip, "application/zip"),
        }
    else:
        logger.info(f"No paper directory found; uploading markdown only: {report_path.name}")
        raw_md = report_path.read_text(encoding="utf-8", errors="replace")
        stripped_md = safe_strip_title_and_abstract(raw_md, report_path.name)
        files: Dict[str, Any] = {
            "data": (None, json.dumps(payload), "application/json"),
            "file": (report_path.name, stripped_md.encode("utf-8"), "text/markdown"),
        }

    cover_path = find_cover_image(project_dir)
    if cover_path:
        logger.info(f"Cover image: {cover_path}")
        files["cover"] = (cover_path.name, cover_path.read_bytes(), "image/png")

    zip_bytes = build_additional_zip(project_dir)
    if zip_bytes:
        logger.info("Additional materials ZIP: included")
        files["additionalMaterialsZip"] = (
            "supplementary.zip", zip_bytes, "application/zip"
        )

    # --- Dry run ---
    if dry_run:
        print("\n[DRY RUN] Would POST to:", FUTURE_SCIENCE_ENDPOINT)
        print("[DRY RUN] Headers: x-api-key: <redacted>")
        print("[DRY RUN] Payload:")
        print(json.dumps(payload, indent=2))
        print(f"[DRY RUN] file: {files['file'][0]} ({len(files['file'][1])} bytes)")
        if "cover" in files:
            print(f"[DRY RUN] cover: {cover_path.name} ({len(files['cover'][1])} bytes)")
        if "additionalMaterialsZip" in files:
            print(f"[DRY RUN] additionalMaterialsZip: supplementary.zip ({len(zip_bytes)} bytes)")
        return {"dry_run": True, "payload": payload}

    # --- POST ---
    headers = {"x-api-key": api_key}
    logger.info(f"Submitting '{title}' to Future Science...")

    # Stamp as published *before* the POST so that if the server accepts the
    # upload but we get a timeout (504/502), we don't resubmit on the next run.
    # On a definitive rejection (4xx), we un-stamp below.
    _stamp_published(project_dir)

    try:
        response = requests.post(
            FUTURE_SCIENCE_ENDPOINT,
            headers=headers,
            files=files,
            timeout=120,
        )
    except requests.exceptions.RequestException:
        # Network-level failure — request may or may not have reached the server.
        # Keep the stamp to avoid duplicates; re-raise so caller can retry or report.
        raise

    if response.status_code == 201:
        data = response.json()
        doc_id = data.get("documentId", "")
        logger.info(f"Submission successful. documentId: {doc_id}")
        # Update stamp with the document ID now that we have it
        _stamp_published(project_dir, doc_id)
        return data

    # Definitive rejection — un-stamp so the project can be retried after fixing
    if 400 <= response.status_code < 500:
        _unstamp_published(project_dir)

    try:
        err = response.json()
    except Exception:
        err = {"error": response.text}
    raise RuntimeError(
        f"Future Science API returned {response.status_code}: {err}"
    )
