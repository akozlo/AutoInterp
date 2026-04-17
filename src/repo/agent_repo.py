"""
Run an external AI agent (claude CLI or codex CLI) to assemble finalized
project files into a clean, publishable repo structure.
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

def _get_repo_agent_command(
    provider: str,
    prompt_text: str,
    project_dir: Path,
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
        return cmd, {"cwd": str(project_dir)}

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
        return cmd, {"cwd": str(project_dir)}

    return None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_repo_prompt(prompt_template: str) -> str:
    """
    Substitute placeholders in the prompt template.

    Currently the repo prompt has no dynamic placeholders — the agent
    discovers everything it needs by reading the filesystem.
    """
    return prepend_persona(prompt_template, "agent_repo.yaml")


# ---------------------------------------------------------------------------
# Repo directory scaffolding
# ---------------------------------------------------------------------------

REPO_SUBDIRS = ["data", "scripts", "notebooks", "results", "paper"]


def ensure_repo_structure(project_dir: Path) -> Path:
    """Create the ``repo/`` directory and its subdirectories, return the repo path."""
    repo_dir = project_dir / "repo"
    for subdir in REPO_SUBDIRS:
        (repo_dir / subdir).mkdir(parents=True, exist_ok=True)
    return repo_dir


# ---------------------------------------------------------------------------
# Agent subprocess execution
# ---------------------------------------------------------------------------

def run_repo_agent(
    provider: str,
    project_dir: Path,
    prompt_text: str,
    timeout: int = 900,
    on_progress: Optional[Callable[[str], None]] = None,
    model: str = "",
    sandbox_bypass: bool = False,
) -> Dict[str, Any]:
    """
    Launch the CLI agent subprocess for repo assembly and return the result.

    Returns ``{"success": bool, "stdout": str, "stderr": str, "returncode": int}``.
    """
    result = _get_repo_agent_command(provider, prompt_text, project_dir, model=model, sandbox_bypass=sandbox_bypass)
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

    # Scaffold repo directories before the agent starts
    repo_dir = ensure_repo_structure(cwd)

    logger.debug("Running repo agent: %s (timeout=%ds)", cmd[0], timeout)
    print(f"[AUTOINTERP] Running {cmd[0]} repo agent (timeout={timeout}s)...")

    milestone = MilestoneSpec(
        watch_dir=repo_dir,
        patterns=[
            MilestonePattern(
                glob="README.md",
                message_fn=lambda _: "Wrote repo/README.md",
            ),
            MilestonePattern(
                glob="paper/*.md",
                message_fn=lambda p: f"Wrote paper/{Path(p).name}",
            ),
            MilestonePattern(
                glob="scripts/*.py",
                message_fn=lambda p: f"Wrote scripts/{Path(p).name}",
            ),
            MilestonePattern(
                glob="paper/*.png",
                message_fn=lambda p: f"Wrote paper/{Path(p).name}",
            ),
            MilestonePattern(
                glob="paper/*.svg",
                message_fn=lambda p: f"Wrote paper/{Path(p).name}",
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
            "Repo agent exited with code %d. stderr: %s",
            proc_result["returncode"],
            proc_result["stderr"][:500],
        )
        print(f"[AUTOINTERP] Repo agent exited with code {proc_result['returncode']}")

    return proc_result


# ---------------------------------------------------------------------------
# Reading agent outputs
# ---------------------------------------------------------------------------

def read_repo_outputs(project_dir: Path) -> Dict[str, Any]:
    """
    Read the files produced by the repo agent.

    Returns a dict with keys: ``repo_dir``, ``readme_path``, ``paper_files``,
    ``script_files``, ``all_files``.
    """
    repo_dir = project_dir / "repo"
    outputs: Dict[str, Any] = {
        "repo_dir": None,
        "readme_path": None,
        "paper_files": [],
        "script_files": [],
        "all_files": [],
    }

    if not repo_dir.exists():
        return outputs

    outputs["repo_dir"] = str(repo_dir)

    for fpath in sorted(repo_dir.rglob("*")):
        if not fpath.is_file():
            continue
        rel = fpath.relative_to(repo_dir)
        outputs["all_files"].append(str(rel))

        if fpath.name == "README.md" and fpath.parent == repo_dir:
            outputs["readme_path"] = str(fpath)
        elif rel.parts[0] == "paper":
            outputs["paper_files"].append(str(rel))
        elif rel.parts[0] == "scripts":
            outputs["script_files"].append(str(rel))

    return outputs


# ---------------------------------------------------------------------------
# Prompt template loading helper
# ---------------------------------------------------------------------------

def load_repo_prompt_template() -> str:
    """Load the agent repo prompt template from prompts/agent_repo.yaml."""
    prompt_path = PACKAGE_ROOT / "prompts" / "agent_repo.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Repo prompt template not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("prompt_template", "")
