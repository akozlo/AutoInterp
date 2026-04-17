"""
Run an external AI agent (claude CLI or codex CLI) to generate research
questions and select the best one, all within one subprocess invocation.
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
# Agent command construction (mirrors reporting/agent_report.py)
# ---------------------------------------------------------------------------

def _get_questions_agent_command(
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

def _build_questions_prompt(prompt_template: str, task_description: str) -> str:
    """
    Substitute ``{task_description}`` in the prompt template.

    When *task_description* is empty or a generic placeholder the agent is
    instructed (via the template itself) to also generate a creative topic.
    """
    return prepend_persona(
        prompt_template.replace("{task_description}", task_description or ""),
        "agent_questions.yaml",
    )


# ---------------------------------------------------------------------------
# Agent subprocess execution
# ---------------------------------------------------------------------------

def run_questions_agent(
    provider: str,
    project_dir: Path,
    prompt_text: str,
    timeout: int = 300,
    on_progress: Optional[Callable[[str], None]] = None,
    model: str = "",
    sandbox_bypass: bool = False,
) -> Dict[str, Any]:
    """
    Launch the CLI agent subprocess for question generation + prioritization.

    Returns ``{"success": bool, "stdout": str, "stderr": str, "returncode": int}``.
    """
    result = _get_questions_agent_command(provider, prompt_text, project_dir, model=model, sandbox_bypass=sandbox_bypass)
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

    logger.debug("Running questions agent: %s (timeout=%ds)", cmd[0], timeout)
    print(f"[AUTOINTERP] Running {cmd[0]} questions agent (timeout={timeout}s)...")

    milestone = MilestoneSpec(
        watch_dir=questions_dir,
        patterns=[
            MilestonePattern(
                glob="questions.txt",
                message_fn=lambda _: "Wrote questions.txt",
            ),
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
            "Questions agent exited with code %d. stderr: %s",
            proc_result["returncode"],
            proc_result["stderr"][:500],
        )
        print(f"[AUTOINTERP] Questions agent exited with code {proc_result['returncode']}")

    return proc_result


# ---------------------------------------------------------------------------
# Reading agent outputs
# ---------------------------------------------------------------------------

def read_questions_outputs(project_dir: Path) -> Dict[str, Any]:
    """
    Read the files produced by the questions agent.

    Returns a dict with keys:
    ``questions_text``, ``prioritized_text``, ``has_questions``,
    ``has_prioritized``, ``title``.
    """
    questions_dir = project_dir / "questions"
    outputs: Dict[str, Any] = {
        "questions_text": "",
        "prioritized_text": "",
        "has_questions": False,
        "has_prioritized": False,
        "title": "",
    }

    if not questions_dir.exists():
        return outputs

    questions_file = questions_dir / "questions.txt"
    prioritized_file = questions_dir / "prioritized_question.txt"

    if questions_file.exists():
        text = questions_file.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            outputs["questions_text"] = text
            outputs["has_questions"] = True

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

def load_questions_prompt_template() -> str:
    """Load the agent questions prompt template from prompts/agent_questions.yaml."""
    prompt_path = PACKAGE_ROOT / "prompts" / "agent_questions.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Questions prompt template not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("prompt_template", "")
