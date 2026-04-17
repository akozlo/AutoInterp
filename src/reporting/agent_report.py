"""
Run an external AI agent (claude CLI or codex CLI) to generate the final
research report autonomously within one subprocess invocation.
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
from AutoInterp.src.core.utils import PathResolver, PACKAGE_ROOT, prepend_persona

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent command construction (mirrors agent_analysis.py / agent_questions.py)
# ---------------------------------------------------------------------------

def _get_report_agent_command(
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

def _build_report_prompt(prompt_template: str) -> str:
    """
    Substitute placeholders in the prompt template.

    Currently the report prompt has no dynamic placeholders — the agent
    discovers everything it needs by reading the filesystem.  This function
    exists for forward-compatibility with future placeholders.
    """
    return prepend_persona(prompt_template, "agent_report.yaml")


# ---------------------------------------------------------------------------
# Agent subprocess execution
# ---------------------------------------------------------------------------

def run_report_agent(
    provider: str,
    project_dir: Path,
    prompt_text: str,
    timeout: int = 900,
    on_progress: Optional[Callable[[str], None]] = None,
    model: str = "",
    sandbox_bypass: bool = False,
) -> Dict[str, Any]:
    """
    Launch the CLI agent subprocess for report generation and return the result.

    Returns ``{"success": bool, "stdout": str, "stderr": str, "returncode": int}``.
    """
    result = _get_report_agent_command(provider, prompt_text, project_dir, model=model, sandbox_bypass=sandbox_bypass)
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

    logger.debug("Running report agent: %s (timeout=%ds)", cmd[0], timeout)
    print(f"[AUTOINTERP] Running {cmd[0]} report agent (timeout={timeout}s)...")

    milestone = MilestoneSpec(
        watch_dir=reports_dir,
        patterns=[
            MilestonePattern(
                glob="Reporter_log.md",
                message_fn=lambda _: "Wrote Reporter_log.md",
            ),
            MilestonePattern(
                glob="*.md",
                message_fn=lambda fname: f"Wrote report file: {fname}",
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
            "Report agent exited with code %d. stderr: %s",
            proc_result["returncode"],
            proc_result["stderr"][:500],
        )
        print(f"[AUTOINTERP] Report agent exited with code {proc_result['returncode']}")

    return proc_result


# ---------------------------------------------------------------------------
# Reading agent outputs
# ---------------------------------------------------------------------------

def read_report_outputs(project_dir: Path) -> Dict[str, Any]:
    """
    Read the files produced by the report agent.

    Returns a dict with keys: ``report_path``, ``reporter_log``, ``all_files``.
    """
    reports_dir = project_dir / "reports"
    outputs: Dict[str, Any] = {
        "report_path": None,
        "reporter_log": "",
        "all_files": [],
    }

    if not reports_dir.exists():
        return outputs

    md_files: List[Path] = []
    for fpath in sorted(reports_dir.iterdir()):
        if not fpath.is_file():
            continue
        outputs["all_files"].append(str(fpath))
        if fpath.name == "Reporter_log.md":
            outputs["reporter_log"] = fpath.read_text(encoding="utf-8", errors="replace")
        elif fpath.suffix == ".md" and fpath.name != "Reporter_log.md":
            md_files.append(fpath)

    # The report is the .md file that is not the log (prefer the newest one)
    if md_files:
        # Sort by modification time, newest last
        md_files.sort(key=lambda p: p.stat().st_mtime)
        outputs["report_path"] = str(md_files[-1])

    return outputs


# ---------------------------------------------------------------------------
# Prompt template loading helper
# ---------------------------------------------------------------------------

def load_report_prompt_template() -> str:
    """Load the agent report prompt template from prompts/agent_report.yaml."""
    prompt_path = PACKAGE_ROOT / "prompts" / "agent_report.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Report prompt template not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("prompt_template", "")
