"""
Run an external AI agent (claude CLI or codex CLI) to incorporate all revision
work into the research report after per-recommendation revision agents finish.
"""

import logging
import os
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

def _get_report_revision_agent_command(
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

def _build_report_revision_prompt(
    prompt_template: str,
    round_number: int,
) -> str:
    """
    Substitute placeholders in the prompt template.

    Replaces ``{k}`` with the round number.  ``{n}`` is left literal so the
    agent sees it as-is (it refers to analysis iteration numbers the agent
    discovers at runtime).
    """
    return prepend_persona(
        prompt_template.replace("{k}", str(round_number)),
        "agent_report_revision.yaml",
    )


# ---------------------------------------------------------------------------
# Agent subprocess execution
# ---------------------------------------------------------------------------

def run_report_revision_agent(
    provider: str,
    project_dir: Path,
    prompt_text: str,
    timeout: int = 900,
    round_number: int = 1,
    on_progress: Optional[Callable[[str], None]] = None,
    model: str = "",
    sandbox_bypass: bool = False,
) -> Dict[str, Any]:
    """
    Launch the CLI agent subprocess for report revision and return the result.

    Returns ``{"success": bool, "stdout": str, "stderr": str, "returncode": int}``.
    """
    result = _get_report_revision_agent_command(provider, prompt_text, project_dir, model=model, sandbox_bypass=sandbox_bypass)
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
    reports_dir = cwd / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")

    logger.debug(
        "Running report revision agent round %d: %s (timeout=%ds)",
        round_number, cmd[0], timeout,
    )
    print(
        f"[AUTOINTERP] Running {cmd[0]} report revision agent "
        f"(round {round_number}, timeout={timeout}s)..."
    )

    milestone = MilestoneSpec(
        watch_dir=reports_dir,
        patterns=[
            MilestonePattern(
                glob=f"Report_revision_{round_number}.log",
                message_fn=lambda _p, _k=round_number: f"Wrote Report_revision_{_k}.log",
            ),
            MilestonePattern(
                glob=f"*_revision_{round_number}.md",
                message_fn=lambda p: f"Wrote revised report: {Path(p).name}",
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
            "Report revision agent (round %d) exited with code %d. stderr: %s",
            round_number,
            proc_result["returncode"],
            proc_result["stderr"][:500],
        )
        print(
            f"[AUTOINTERP] Report revision agent (round {round_number}) "
            f"exited with code {proc_result['returncode']}"
        )

    return proc_result


# ---------------------------------------------------------------------------
# Reading agent outputs
# ---------------------------------------------------------------------------

def read_report_revision_outputs(
    project_dir: Path,
    round_number: int = 1,
) -> Dict[str, Any]:
    """
    Read the files produced by the report revision agent.

    Returns a dict with keys: ``revised_report_path``, ``log_path``,
    ``log_text``.
    """
    reports_dir = project_dir / "reports"
    outputs: Dict[str, Any] = {
        "revised_report_path": None,
        "log_path": None,
        "log_text": "",
    }

    if not reports_dir.exists():
        return outputs

    log_file = reports_dir / f"Report_revision_{round_number}.log"
    if log_file.is_file():
        outputs["log_path"] = str(log_file)
        outputs["log_text"] = log_file.read_text(encoding="utf-8", errors="replace")

    # Find the revised report: *_revision_{k}.md
    revision_suffix = f"_revision_{round_number}.md"
    candidates = [
        f for f in sorted(reports_dir.iterdir())
        if f.is_file() and f.name.endswith(revision_suffix)
    ]
    if candidates:
        # Prefer the newest by mtime if multiple matches
        candidates.sort(key=lambda p: p.stat().st_mtime)
        outputs["revised_report_path"] = str(candidates[-1])

    return outputs


# ---------------------------------------------------------------------------
# Prompt template loading helper
# ---------------------------------------------------------------------------

def load_report_revision_prompt_template() -> str:
    """Load the agent report revision prompt template from prompts/agent_report_revision.yaml."""
    prompt_path = PACKAGE_ROOT / "prompts" / "agent_report_revision.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Report revision prompt template not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("prompt_template", "")
