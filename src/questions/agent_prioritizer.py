"""
Run an external AI agent (claude CLI or codex CLI) to prioritize research
questions in a single subprocess invocation.
"""

import logging
import re
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
# Agent command construction (mirrors questions/agent_questions.py)
# ---------------------------------------------------------------------------

def _get_prioritizer_agent_command(
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

def _build_prioritizer_prompt(prompt_template: str, question_text: str) -> str:
    """
    Substitute ``{question_text}`` in the prompt template.
    """
    return prepend_persona(
        prompt_template.replace("{question_text}", question_text or ""),
        "agent_prioritizer.yaml",
    )


# ---------------------------------------------------------------------------
# Agent subprocess execution
# ---------------------------------------------------------------------------

def run_prioritizer_agent(
    provider: str,
    project_dir: Path,
    prompt_text: str,
    timeout: int = 300,
    on_progress: Optional[Callable[[str], None]] = None,
    model: str = "",
    sandbox_bypass: bool = False,
) -> Dict[str, Any]:
    """
    Launch the CLI agent subprocess for question prioritization.

    Returns ``{"success": bool, "stdout": str, "stderr": str, "returncode": int}``.
    """
    result = _get_prioritizer_agent_command(provider, prompt_text, project_dir, model=model, sandbox_bypass=sandbox_bypass)
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
    questions_dir = cwd / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Running prioritizer agent: %s (timeout=%ds)", cmd[0], timeout)
    print(f"[AUTOINTERP] Running {cmd[0]} prioritizer agent (timeout={timeout}s)...")

    milestone = MilestoneSpec(
        watch_dir=questions_dir,
        patterns=[
            MilestonePattern(
                glob="prioritized_question.txt",
                message_fn=lambda _: "Wrote prioritized_question.txt",
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
            "Prioritizer agent exited with code %d. stderr: %s",
            proc_result["returncode"],
            proc_result["stderr"][:500],
        )
        print(f"[AUTOINTERP] Prioritizer agent exited with code {proc_result['returncode']}")

    return proc_result


# ---------------------------------------------------------------------------
# Reading agent outputs
# ---------------------------------------------------------------------------

def read_prioritizer_outputs(project_dir: Path) -> Dict[str, Any]:
    """
    Read the files produced by the prioritizer agent.

    Returns a dict with keys:
    ``prioritized_text``, ``has_prioritized``, ``title``.
    """
    questions_dir = project_dir / "questions"
    outputs: Dict[str, Any] = {
        "prioritized_text": "",
        "has_prioritized": False,
        "title": "",
    }

    if not questions_dir.exists():
        return outputs

    prioritized_file = questions_dir / "prioritized_question.txt"

    if prioritized_file.exists():
        text = prioritized_file.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            outputs["prioritized_text"] = text
            outputs["has_prioritized"] = True
            # Extract TITLE via regex
            title_match = re.search(r'TITLE:\s*(.*?)(?:\n|$)', text, re.IGNORECASE)
            if title_match:
                outputs["title"] = title_match.group(1).strip()

    return outputs


# ---------------------------------------------------------------------------
# Prompt template loading helper
# ---------------------------------------------------------------------------

def load_prioritizer_prompt_template() -> str:
    """Load the agent prioritizer prompt template from prompts/agent_prioritizer.yaml."""
    prompt_path = PACKAGE_ROOT / "prompts" / "agent_prioritizer.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prioritizer prompt template not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("prompt_template", "")
