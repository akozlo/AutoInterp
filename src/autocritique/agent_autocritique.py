"""
Run an external AI agent (claude CLI or codex CLI) to perform an automated
peer review (AutoCritique) of the generated research report.
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
# Agent command construction (mirrors agent_report.py)
# ---------------------------------------------------------------------------

def _get_autocritique_agent_command(
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

def _build_autocritique_prompt(
    prompt_template: str,
    round_number: int = 1,
    revised_report_filename: Optional[str] = None,
) -> str:
    """
    Substitute placeholders in the prompt template.

    Replaces ``{round_dir}`` with the round-specific subdirectory path
    (e.g. ``./autocritique/round_1/``).

    When *revised_report_filename* is provided (rounds 2+), inserts hints
    telling the agent to read the revised report instead of the original.
    """
    round_dir = f"./autocritique/round_{round_number}/"
    text = prompt_template.replace("{round_dir}", round_dir)

    if revised_report_filename:
        report_hint = (
            f"IMPORTANT: This is review round {round_number}. A revised report "
            f"has been produced after the previous round's revisions. The revised "
            f"report is **{revised_report_filename}** in ./reports/. You must "
            f"review the revised report, NOT the original."
        )
        report_instruction = (
            f"Read the revised report **{revised_report_filename}** "
            f"(not the original report or Reporter_log.md)."
        )
    else:
        report_hint = ""
        report_instruction = (
            "Read the main .md file (not Reporter_log.md or any log files)."
        )

    text = text.replace("{report_hint}", report_hint)
    text = text.replace("{report_instruction}", report_instruction)
    return prepend_persona(text, "agent_autocritique.yaml")


# ---------------------------------------------------------------------------
# Agent subprocess execution
# ---------------------------------------------------------------------------

def run_autocritique_agent(
    provider: str,
    project_dir: Path,
    prompt_text: str,
    timeout: int = 600,
    round_number: int = 1,
    on_progress: Optional[Callable[[str], None]] = None,
    model: str = "",
    sandbox_bypass: bool = False,
) -> Dict[str, Any]:
    """
    Launch the CLI agent subprocess for autocritique and return the result.

    *round_number* determines the output subdirectory
    (``autocritique/round_1/``, ``autocritique/round_2/``, etc.).

    Returns ``{"success": bool, "stdout": str, "stderr": str, "returncode": int}``.
    """
    result = _get_autocritique_agent_command(provider, prompt_text, project_dir, model=model, sandbox_bypass=sandbox_bypass)
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

    logger.debug("Running autocritique agent round %d: %s (timeout=%ds)", round_number, cmd[0], timeout)
    print(f"[AUTOINTERP] Running {cmd[0]} autocritique agent round {round_number} (timeout={timeout}s)...")

    milestone = MilestoneSpec(
        watch_dir=round_dir,
        patterns=[
            MilestonePattern(
                glob="AutoCritique_log.md",
                message_fn=lambda _: "Wrote AutoCritique_log.md",
            ),
            MilestonePattern(
                glob="AutoCritique_review.md",
                message_fn=lambda _: "Wrote AutoCritique_review.md",
            ),
            MilestonePattern(
                glob="Recommendation_*.md",
                message_fn=lambda p: f"Wrote {Path(p).name}",
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
            "AutoCritique agent (round %d) exited with code %d. stderr: %s",
            round_number,
            proc_result["returncode"],
            proc_result["stderr"][:500],
        )
        print(f"[AUTOINTERP] AutoCritique agent (round {round_number}) exited with code {proc_result['returncode']}")

    return proc_result


# ---------------------------------------------------------------------------
# Reading agent outputs
# ---------------------------------------------------------------------------

def read_autocritique_outputs(project_dir: Path, round_number: int = 1) -> Dict[str, Any]:
    """
    Read the files produced by the autocritique agent for a given round.

    Returns a dict with keys: ``review_path``, ``log_text``,
    ``recommendations`` (list of paths), ``all_files``.
    """
    round_dir = project_dir / "autocritique" / f"round_{round_number}"
    outputs: Dict[str, Any] = {
        "review_path": None,
        "log_text": "",
        "recommendations": [],
        "all_files": [],
    }

    if not round_dir.exists():
        return outputs

    for fpath in sorted(round_dir.iterdir()):
        if not fpath.is_file():
            continue
        outputs["all_files"].append(str(fpath))
        if fpath.name == "AutoCritique_log.md":
            outputs["log_text"] = fpath.read_text(encoding="utf-8", errors="replace")
        elif fpath.name == "AutoCritique_review.md":
            outputs["review_path"] = str(fpath)
        elif fpath.name.startswith("Recommendation_") and fpath.name.endswith(".md"):
            outputs["recommendations"].append(str(fpath))

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
