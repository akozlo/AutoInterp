"""
Run an external AI agent (claude CLI or codex CLI) to create a self-contained,
executable Jupyter notebook from the finalized repo/ directory.
"""

import logging
import os
import shutil
import subprocess
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

def _get_notebook_agent_command(
    provider: str,
    prompt_text: str,
    repo_dir: Path,
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
        return cmd, {"cwd": str(repo_dir)}

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
        return cmd, {"cwd": str(repo_dir)}

    return None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_notebook_prompt(prompt_template: str, project_dir: Path) -> str:
    """
    Substitute placeholders in the prompt template.

    Placeholders:
    - ``{repo_listing}`` — directory tree of ``repo/``
    - ``{report_excerpt}`` — first ~200 lines of the most current report
    """
    repo_dir = project_dir / "repo"

    # Build repo listing via tree-like output
    repo_listing = _build_repo_listing(repo_dir)

    # Extract report excerpt
    report_excerpt = _get_report_excerpt(project_dir)

    return prepend_persona(
        prompt_template.replace("{repo_listing}", repo_listing).replace(
            "{report_excerpt}", report_excerpt
        ),
        "agent_notebook.yaml",
    )


def _build_repo_listing(repo_dir: Path) -> str:
    """Build a simple directory tree of repo/."""
    if not repo_dir.exists():
        return "(repo/ directory not found)"
    lines = []
    for fpath in sorted(repo_dir.rglob("*")):
        if fpath.is_file():
            rel = fpath.relative_to(repo_dir)
            lines.append(str(rel))
    if not lines:
        return "(repo/ is empty)"
    return "\n".join(lines)


def _get_report_excerpt(project_dir: Path) -> str:
    """Read the first ~200 lines of the most current report from repo/paper/."""
    paper_dir = project_dir / "repo" / "paper"
    if not paper_dir.exists():
        return "(no report found)"
    md_files = sorted(paper_dir.glob("*.md"))
    if not md_files:
        return "(no report found in repo/paper/)"
    # Pick the last one (highest revision number if multiple)
    report_path = md_files[-1]
    try:
        lines = report_path.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[:200])
    except OSError:
        return f"(could not read {report_path.name})"


# ---------------------------------------------------------------------------
# Agent subprocess execution
# ---------------------------------------------------------------------------

def run_notebook_agent(
    provider: str,
    project_dir: Path,
    prompt_text: str,
    timeout: int = 4200,
    on_progress: Optional[Callable[[str], None]] = None,
    model: str = "",
    sandbox_bypass: bool = False,
) -> Dict[str, Any]:
    """
    Launch the CLI agent subprocess for notebook creation and return the result.

    Returns ``{"success": bool, "stdout": str, "stderr": str, "returncode": int}``.
    """
    repo_dir = project_dir / "repo"
    result = _get_notebook_agent_command(provider, prompt_text, repo_dir, model=model, sandbox_bypass=sandbox_bypass)
    if result is None:
        cli_name = "claude" if (provider or "").lower() == "anthropic" else "codex"
        logger.warning(
            "Agent CLI '%s' not found or provider '%s' unsupported.",
            cli_name,
            provider,
        )
        return {"success": False, "stdout": "", "stderr": f"CLI '{cli_name}' not found", "returncode": -1}

    cmd, kwargs = result
    cwd = Path(kwargs["cwd"])

    # Ensure notebooks/ exists
    notebooks_dir = cwd / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    # Notebook JSON can exceed the default 32K output-token cap — raise it to 64K
    # so that large notebooks aren't silently truncated and the agent doesn't fail.
    os.environ.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")

    logger.debug("Running notebook agent: %s (timeout=%ds)", cmd[0], timeout)
    print(f"[AUTOINTERP] Running {cmd[0]} notebook agent (timeout={timeout}s)...")

    milestone = MilestoneSpec(
        watch_dir=cwd,
        patterns=[
            MilestonePattern(
                glob="notebooks/*.ipynb",
                message_fn=lambda p: f"Wrote notebooks/{Path(p).name}",
            ),
            MilestonePattern(
                glob="notebooks/*.py",
                message_fn=lambda p: f"Wrote notebooks/{Path(p).name}",
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
            "Notebook agent exited with code %d. stderr: %s",
            proc_result["returncode"],
            proc_result["stderr"][:500],
        )
        print(f"[AUTOINTERP] Notebook agent exited with code {proc_result['returncode']}")

    return proc_result


# ---------------------------------------------------------------------------
# Reading agent outputs
# ---------------------------------------------------------------------------

def read_notebook_outputs(project_dir: Path) -> Dict[str, Any]:
    """
    Read the files produced by the notebook agent.

    Returns a dict with keys: ``notebook_path``, ``all_files``.
    """
    notebooks_dir = project_dir / "repo" / "notebooks"
    outputs: Dict[str, Any] = {
        "notebook_path": None,
        "all_files": [],
    }

    if not notebooks_dir.exists():
        return outputs

    for fpath in sorted(notebooks_dir.glob("*.ipynb")):
        outputs["all_files"].append(str(fpath))
        if outputs["notebook_path"] is None:
            outputs["notebook_path"] = str(fpath)

    return outputs


# ---------------------------------------------------------------------------
# Prompt template loading helper
# ---------------------------------------------------------------------------

def load_notebook_prompt_template() -> str:
    """Load the agent notebook prompt template from prompts/agent_notebook.yaml."""
    prompt_path = PACKAGE_ROOT / "prompts" / "agent_notebook.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Notebook prompt template not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("prompt_template", "")
