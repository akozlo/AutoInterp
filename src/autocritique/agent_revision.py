"""
Run an external AI agent (claude CLI or codex CLI) to address a single
AutoCritique recommendation via new or revised analysis work.
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
# Agent command construction (mirrors agent_autocritique.py)
# ---------------------------------------------------------------------------

def _get_revision_agent_command(
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

def _build_revision_prompt(
    prompt_template: str,
    round_number: int,
    recommendation_idx: int,
) -> str:
    """
    Substitute placeholders in the prompt template.

    Replaces ``{i}`` with the recommendation index and ``{k}`` with the
    round number.  ``{n}`` is left literal so the agent sees it as-is
    (it refers to analysis iteration numbers the agent discovers at runtime).
    """
    return prepend_persona(
        prompt_template
        .replace("{i}", str(recommendation_idx))
        .replace("{k}", str(round_number)),
        "agent_revision.yaml",
    )


# ---------------------------------------------------------------------------
# Agent subprocess execution
# ---------------------------------------------------------------------------

def run_revision_agent(
    provider: str,
    project_dir: Path,
    prompt_text: str,
    timeout: int = 1800,
    round_number: int = 1,
    recommendation_idx: int = 1,
    on_progress: Optional[Callable[[str], None]] = None,
    model: str = "",
    sandbox_bypass: bool = False,
) -> Dict[str, Any]:
    """
    Launch the CLI agent subprocess to address one recommendation.

    Returns ``{"success": bool, "stdout": str, "stderr": str, "returncode": int}``.
    """
    result = _get_revision_agent_command(provider, prompt_text, project_dir, model=model, sandbox_bypass=sandbox_bypass)
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
    round_dir = cwd / "autocritique" / f"round_{round_number}"
    round_dir.mkdir(parents=True, exist_ok=True)

    idx = recommendation_idx
    logger.debug(
        "Running revision agent round %d rec %d: %s (timeout=%ds)",
        round_number, idx, cmd[0], timeout,
    )
    print(
        f"[AUTOINTERP] Running {cmd[0]} revision agent "
        f"(round {round_number}, recommendation {idx}, timeout={timeout}s)..."
    )

    milestone = MilestoneSpec(
        watch_dir=round_dir,
        patterns=[
            MilestonePattern(
                glob=f"Response_{idx}_log.md",
                message_fn=lambda _p, _idx=idx: f"Wrote Response_{_idx}_log.md",
            ),
            MilestonePattern(
                glob=f"Response_{idx}.md",
                message_fn=lambda _p, _idx=idx: f"Wrote Response_{_idx}.md",
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
            "Revision agent (round %d rec %d) exited with code %d. stderr: %s",
            round_number,
            idx,
            proc_result["returncode"],
            proc_result["stderr"][:500],
        )
        print(
            f"[AUTOINTERP] Revision agent (round {round_number} rec {idx}) "
            f"exited with code {proc_result['returncode']}"
        )

    return proc_result


# ---------------------------------------------------------------------------
# Reading agent outputs
# ---------------------------------------------------------------------------

def read_revision_outputs(
    project_dir: Path,
    round_number: int = 1,
    recommendation_idx: int = 1,
) -> Dict[str, Any]:
    """
    Read the files produced by the revision agent for one recommendation.

    Returns a dict with keys: ``log_path``, ``response_path``, ``log_text``,
    ``response_text``.
    """
    round_dir = project_dir / "autocritique" / f"round_{round_number}"
    idx = recommendation_idx
    outputs: Dict[str, Any] = {
        "log_path": None,
        "response_path": None,
        "log_text": "",
        "response_text": "",
    }

    if not round_dir.exists():
        return outputs

    log_file = round_dir / f"Response_{idx}_log.md"
    response_file = round_dir / f"Response_{idx}.md"

    if log_file.is_file():
        outputs["log_path"] = str(log_file)
        outputs["log_text"] = log_file.read_text(encoding="utf-8", errors="replace")

    if response_file.is_file():
        outputs["response_path"] = str(response_file)
        outputs["response_text"] = response_file.read_text(encoding="utf-8", errors="replace")

    return outputs


# ---------------------------------------------------------------------------
# Prompt template loading helper
# ---------------------------------------------------------------------------

def load_revision_prompt_template() -> str:
    """Load the agent revision prompt template from prompts/agent_revision.yaml."""
    prompt_path = PACKAGE_ROOT / "prompts" / "agent_revision.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Revision prompt template not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("prompt_template", "")
