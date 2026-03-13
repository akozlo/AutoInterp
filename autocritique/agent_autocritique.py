"""
Run an external AI agent (claude CLI or codex CLI) to perform an automated
peer review (AutoCritique) of the generated research report.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from AutoInterp.core.agent_subprocess import (
    MilestonePattern,
    MilestoneSpec,
    run_agent_with_polling,
)
from AutoInterp.core.utils import PACKAGE_ROOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent command construction (mirrors agent_report.py)
# ---------------------------------------------------------------------------

def _get_autocritique_agent_command(
    provider: str,
    prompt_text: str,
    project_dir: Path,
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
        cmd = [cli, "-p", "--dangerously-skip-permissions", prompt_text]
        return cmd, {"cwd": str(project_dir)}

    if provider_lower == "openai":
        cli = "codex"
        if not shutil.which(cli):
            return None
        cmd = [cli, "exec", "-s", "workspace-write", prompt_text]
        return cmd, {"cwd": str(project_dir)}

    return None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_autocritique_prompt(prompt_template: str) -> str:
    """
    Substitute placeholders in the prompt template.

    Currently the autocritique prompt has no dynamic placeholders — the agent
    discovers everything it needs by reading the filesystem.  This function
    exists for forward-compatibility with future placeholders.
    """
    return prompt_template


# ---------------------------------------------------------------------------
# Agent subprocess execution
# ---------------------------------------------------------------------------

def run_autocritique_agent(
    provider: str,
    project_dir: Path,
    prompt_text: str,
    timeout: int = 600,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Launch the CLI agent subprocess for autocritique and return the result.

    Returns ``{"success": bool, "stdout": str, "stderr": str, "returncode": int}``.
    """
    result = _get_autocritique_agent_command(provider, prompt_text, project_dir)
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
    critique_dir = cwd / "autocritique"
    critique_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Running autocritique agent: %s (timeout=%ds)", cmd[0], timeout)
    print(f"[AUTOINTERP] Running {cmd[0]} autocritique agent (timeout={timeout}s)...")

    milestone = MilestoneSpec(
        watch_dir=critique_dir,
        patterns=[
            MilestonePattern(
                glob="AutoCritique_log.md",
                message_fn=lambda _: "Wrote AutoCritique_log.md",
            ),
            MilestonePattern(
                glob="AutoCritique_review.md",
                message_fn=lambda _: "Wrote AutoCritique_review.md",
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
            "AutoCritique agent exited with code %d. stderr: %s",
            proc_result["returncode"],
            proc_result["stderr"][:500],
        )
        print(f"[AUTOINTERP] AutoCritique agent exited with code {proc_result['returncode']}")

    return proc_result


# ---------------------------------------------------------------------------
# Reading agent outputs
# ---------------------------------------------------------------------------

def read_autocritique_outputs(project_dir: Path) -> Dict[str, Any]:
    """
    Read the files produced by the autocritique agent.

    Returns a dict with keys: ``review_path``, ``log_text``, ``all_files``.
    """
    critique_dir = project_dir / "autocritique"
    outputs: Dict[str, Any] = {
        "review_path": None,
        "log_text": "",
        "all_files": [],
    }

    if not critique_dir.exists():
        return outputs

    for fpath in sorted(critique_dir.iterdir()):
        if not fpath.is_file():
            continue
        outputs["all_files"].append(str(fpath))
        if fpath.name == "AutoCritique_log.md":
            outputs["log_text"] = fpath.read_text(encoding="utf-8", errors="replace")
        elif fpath.name == "AutoCritique_review.md":
            outputs["review_path"] = str(fpath)

    return outputs


# ---------------------------------------------------------------------------
# Prompt template loading helper
# ---------------------------------------------------------------------------

def load_autocritique_prompt_template() -> str:
    """Load the agent autocritique prompt template from prompts/agent_autocritique.yaml."""
    prompt_path = PACKAGE_ROOT / "prompts" / "agent_autocritique.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"AutoCritique prompt template not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("prompt_template", "")
