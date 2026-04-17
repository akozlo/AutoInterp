"""
Run an external AI agent (claude CLI or codex CLI) to conduct a literature
review autonomously within one subprocess invocation.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from AutoInterp.src.core.agent_subprocess import (
    MilestonePattern,
    MilestoneSpec,
    run_agent_with_polling,
)
from AutoInterp.src.core.utils import PACKAGE_ROOT, prepend_persona

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent command construction
# ---------------------------------------------------------------------------

def _get_literature_review_agent_command(
    provider: str,
    prompt_text: str,
    literature_dir: Path,
    model: str = "",
    sandbox_bypass: bool = False,
) -> Optional[Tuple[List[str], Dict[str, Any]]]:
    """
    Return ``(cmd_list, subprocess_kwargs)`` for the selected provider's CLI
    agent, or ``None`` if the provider/CLI is not available.
    """
    provider_lower = (provider or "").lower()

    if provider_lower == "anthropic":
        cli = "claude"
        if not shutil.which(cli):
            return None
        cmd = [cli, "-p", "--dangerously-skip-permissions"]
        if model:
            cmd += ["--model", model]
        cmd.append(prompt_text)
        return cmd, {"cwd": str(literature_dir)}

    if provider_lower == "openai":
        cli = "codex"
        if not shutil.which(cli):
            return None
        if sandbox_bypass:
            cmd = [cli, "exec", "--dangerously-bypass-approvals-and-sandbox"]
        else:
            cmd = [cli, "exec", "-s", "workspace-write"]
        if model:
            cmd += ["-m", model]
        cmd.append(prompt_text)
        return cmd, {"cwd": str(literature_dir)}

    return None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_literature_review_prompt(
    prompt_template: str,
    prioritized_research_question: str,
    lit_count: int = 8,
) -> str:
    """
    Substitute ``{prioritized_research_question}`` and ``{lit_count}`` in
    the prompt template.
    """
    filled = prompt_template.replace(
        "{prioritized_research_question}", prioritized_research_question
    ).replace("{lit_count}", str(lit_count))
    return prepend_persona(filled, "agent_literature_review.yaml")


# ---------------------------------------------------------------------------
# Agent subprocess execution
# ---------------------------------------------------------------------------

def run_literature_review_agent(
    provider: str,
    literature_dir: Path,
    prompt_text: str,
    timeout: int = 1800,
    on_progress: Optional[Callable[[str], None]] = None,
    model: str = "",
    sandbox_bypass: bool = False,
) -> Dict[str, Any]:
    """
    Launch the CLI agent subprocess for literature review and return the result.

    Returns ``{"success": bool, "stdout": str, "stderr": str, "returncode": int}``.
    """
    result = _get_literature_review_agent_command(
        provider, prompt_text, literature_dir, model=model,
        sandbox_bypass=sandbox_bypass
    )
    if result is None:
        cli_name = "claude" if (provider or "").lower() == "anthropic" else "codex"
        logger.warning(
            "Agent CLI '%s' not found or provider '%s' unsupported.",
            cli_name,
            provider,
        )
        return {
            "success": False,
            "stdout": "",
            "stderr": f"CLI '{cli_name}' not found",
            "returncode": -1,
        }

    cmd, kwargs = result
    cwd = Path(kwargs["cwd"])
    cwd.mkdir(parents=True, exist_ok=True)

    # Ensure pdfs/ subdirectory exists for the agent to download into
    (cwd / "pdfs").mkdir(parents=True, exist_ok=True)

    logger.debug("Running literature review agent: %s (timeout=%ds)", cmd[0], timeout)
    print(f"[AUTOINTERP] Running {cmd[0]} literature review agent (timeout={timeout}s)...")

    milestone = MilestoneSpec(
        watch_dir=cwd,
        patterns=[
            MilestonePattern(
                glob="*.md",
                message_fn=lambda fname: f"Wrote summary: {fname}",
            ),
            MilestonePattern(
                glob="Final_Research_Question.txt",
                message_fn=lambda _: "Wrote Final_Research_Question.txt",
            ),
            MilestonePattern(
                glob="Literature_Review.md",
                message_fn=lambda _: "Wrote Literature_Review.md",
            ),
            MilestonePattern(
                glob="pdfs/*.pdf",
                message_fn=lambda fname: f"Downloaded paper: {fname}",
            ),
        ],
    )

    proc_result = run_agent_with_polling(
        cmd=cmd,
        cwd=cwd,
        timeout=timeout,
        milestone=milestone,
        on_progress=on_progress,
    )

    success = proc_result["success"]
    if not success:
        logger.warning(
            "Literature review agent exited with code %d. stderr: %s",
            proc_result["returncode"],
            proc_result["stderr"][:500],
        )
        print(
            f"[AUTOINTERP] Literature review agent exited with code {proc_result['returncode']}"
        )

    return proc_result


# ---------------------------------------------------------------------------
# Reading agent outputs
# ---------------------------------------------------------------------------

def read_literature_review_outputs(literature_dir: Path) -> Dict[str, Any]:
    """
    Read the files produced by the literature review agent.

    Returns a dict with:
      - ``has_final_question``: bool
      - ``final_question_text``: str (contents of Final_Research_Question.txt)
      - ``has_literature_review``: bool
      - ``literature_review_text``: str (contents of Literature_Review.md)
      - ``paper_summaries``: list of (filename, text) tuples for .md summary files
      - ``downloaded_pdfs``: list of PDF filenames in pdfs/
    """
    outputs: Dict[str, Any] = {
        "has_final_question": False,
        "final_question_text": "",
        "has_literature_review": False,
        "literature_review_text": "",
        "paper_summaries": [],
        "downloaded_pdfs": [],
    }

    if not literature_dir.exists():
        return outputs

    # Final_Research_Question.txt
    fq_path = literature_dir / "Final_Research_Question.txt"
    if fq_path.exists():
        outputs["has_final_question"] = True
        try:
            outputs["final_question_text"] = fq_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
        except OSError:
            pass

    # Literature_Review.md
    lr_path = literature_dir / "Literature_Review.md"
    if lr_path.exists():
        outputs["has_literature_review"] = True
        try:
            outputs["literature_review_text"] = lr_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
        except OSError:
            pass

    # Paper summary .md files (excluding Literature_Review.md itself)
    excluded = {"Literature_Review.md", "Research_Questions.txt"}
    for md_file in sorted(literature_dir.glob("*.md")):
        if md_file.name in excluded:
            continue
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace").strip()
            outputs["paper_summaries"].append((md_file.name, text))
        except OSError:
            pass

    # Downloaded PDFs
    pdfs_dir = literature_dir / "pdfs"
    if pdfs_dir.exists():
        outputs["downloaded_pdfs"] = sorted(
            f.name for f in pdfs_dir.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"
        )

    return outputs


# ---------------------------------------------------------------------------
# Prompt template loading helper
# ---------------------------------------------------------------------------

def load_literature_review_prompt_template() -> str:
    """Load the agent literature review prompt template from prompts/agent_literature_review.yaml."""
    prompt_path = PACKAGE_ROOT / "prompts" / "agent_literature_review.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Literature review prompt template not found: {prompt_path}"
        )
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("prompt_template", "")
