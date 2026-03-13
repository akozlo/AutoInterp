"""
Run an external AI agent (claude CLI or codex CLI) to perform a single
analysis iteration: plan, write code, execute, debug, evaluate, and
update confidence — all autonomously within one subprocess invocation.
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from AutoInterp.core.agent_subprocess import (
    MilestonePattern,
    MilestoneSpec,
    run_agent_with_polling,
)
from AutoInterp.core.utils import PathResolver, PACKAGE_ROOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------

def setup_analysis_workspace(
    path_resolver: PathResolver,
    research_question: str,
    iteration_n: int,
) -> Path:
    """
    Prepare the analysis workspace for iteration *n*.

    On the first call (n == 1) this creates ``analysis/background/`` and
    writes ``Research_Question.md`` and an initial ``confidence.json``.
    Every call creates ``analysis/analysis_{n}/``.

    Returns the path to ``analysis/analysis_{n}/``.
    """
    bg_dir = path_resolver.ensure_analysis_background_dir()

    # Write research question (always overwrite — it's the same question)
    rq_path = bg_dir / "Research_Question.md"
    rq_path.write_text(research_question, encoding="utf-8")

    # Initialise confidence tracker if it doesn't exist
    conf_path = bg_dir / "confidence.json"
    if not conf_path.exists():
        conf_path.write_text(
            json.dumps({"current_confidence": 0.0, "history": []}, indent=2),
            encoding="utf-8",
        )

    # Create iteration directory
    iter_dir = path_resolver.ensure_analysis_iteration_dir(iteration_n)
    return iter_dir


# ---------------------------------------------------------------------------
# Agent command construction (mirrors agent_questions.py pattern)
# ---------------------------------------------------------------------------

def _get_analysis_agent_command(
    provider: str,
    prompt_text: str,
    analysis_dir: Path,
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
        return cmd, {"cwd": str(analysis_dir)}

    if provider_lower == "openai":
        cli = "codex"
        if not shutil.which(cli):
            return None
        cmd = [cli, "exec", "-s", "workspace-write", prompt_text]
        return cmd, {"cwd": str(analysis_dir)}

    return None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_analysis_prompt(
    iteration_n: int,
    analysis_root: Path,
    prompt_template: str,
    model_name: str,
    model_path: str,
) -> str:
    """
    Substitute placeholders in the prompt template.

    Placeholders handled:
      {n}  — current iteration number
      {model_name}  — HuggingFace model name
      {model_path}  — HuggingFace model path (same as name for HF models)
      {prior_analyses_listing}  — directory listing of prior iterations
      {prior_analyses_instructions}  — instructions to review prior work (or empty)
    """
    # Build prior-analyses listing
    prior_listing = ""
    prior_instructions = ""
    if iteration_n > 1:
        lines: List[str] = []
        for prev in range(1, iteration_n):
            prev_dir = analysis_root / f"analysis_{prev}"
            if prev_dir.exists():
                files = sorted(f.name for f in prev_dir.iterdir() if f.is_file())
                lines.append(f"- analysis_{prev}/: {', '.join(files) if files else '(empty)'}")
        if lines:
            prior_listing = (
                "Review the prior analyses. Their directories and files are:\n"
                + "\n".join(lines)
                + "\n\nRead at least the EVALUATION files from prior iterations to understand what has already been tried and learned."
            )
            prior_instructions = (
                "Prior analysis iterations exist in sibling directories. "
                "Review them (especially the EVALUATION files) before planning this iteration, "
                "so you build on previous findings rather than repeating work."
            )
    else:
        prior_listing = "This is the first analysis iteration. No prior work to review."
        prior_instructions = ""

    prompt = prompt_template.replace("{n}", str(iteration_n))
    prompt = prompt.replace("{model_name}", model_name)
    prompt = prompt.replace("{model_path}", model_path)
    prompt = prompt.replace("{prior_analyses_listing}", prior_listing)
    prompt = prompt.replace("{prior_analyses_instructions}", prior_instructions)
    return prompt


# ---------------------------------------------------------------------------
# Agent subprocess execution
# ---------------------------------------------------------------------------

def run_analysis_agent(
    provider: str,
    analysis_dir: Path,
    prompt_text: str,
    timeout: int = 1800,
    on_progress: Optional[Callable[[str], None]] = None,
    iteration_n: int = 1,
) -> Dict[str, Any]:
    """
    Launch the CLI agent subprocess and return the result.

    Returns ``{"success": bool, "stdout": str, "stderr": str, "returncode": int}``.
    """
    result = _get_analysis_agent_command(provider, prompt_text, analysis_dir)
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
    logger.debug("Running analysis agent: %s (timeout=%ds)", cmd[0], timeout)
    print(f"[AUTOINTERP] Running {cmd[0]} analysis agent (timeout={timeout}s)...")

    n = iteration_n
    milestone = MilestoneSpec(
        watch_dir=analysis_dir,
        patterns=[
            MilestonePattern(
                glob=f"ANALYSIS_{n}_PLAN.md",
                message_fn=lambda _: "Wrote analysis plan",
            ),
            MilestonePattern(
                glob="*.py",
                message_fn=lambda fname: f"Wrote script: {fname}",
            ),
            MilestonePattern(
                glob="*.png",
                message_fn=lambda fname: f"Generated figure: {fname}",
            ),
            MilestonePattern(
                glob="*.jpg",
                message_fn=lambda fname: f"Generated figure: {fname}",
            ),
            MilestonePattern(
                glob="*.svg",
                message_fn=lambda fname: f"Generated figure: {fname}",
            ),
            MilestonePattern(
                glob=f"ANALYSIS_{n}_EVALUATION.md",
                message_fn=lambda _: "Wrote evaluation",
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
            "Analysis agent exited with code %d. stderr: %s",
            proc_result["returncode"],
            proc_result["stderr"][:500],
        )
        print(f"[AUTOINTERP] Analysis agent exited with code {proc_result['returncode']}")

    return proc_result


# ---------------------------------------------------------------------------
# Reading agent outputs
# ---------------------------------------------------------------------------

def read_agent_outputs(analysis_root: Path, iteration_n: int) -> Dict[str, Any]:
    """
    Read the files produced by the agent for iteration *n*.

    Returns a dict with keys: ``plan``, ``evaluation``, ``scripts``,
    ``data_files``, ``figures``.
    """
    iter_dir = analysis_root / f"analysis_{iteration_n}"
    outputs: Dict[str, Any] = {
        "plan": "",
        "evaluation": "",
        "scripts": [],
        "data_files": [],
        "figures": [],
    }

    if not iter_dir.exists():
        return outputs

    for fpath in sorted(iter_dir.iterdir()):
        if not fpath.is_file():
            continue
        name = fpath.name

        if name == f"ANALYSIS_{iteration_n}_PLAN.md":
            outputs["plan"] = fpath.read_text(encoding="utf-8", errors="replace")
        elif name == f"ANALYSIS_{iteration_n}_EVALUATION.md":
            outputs["evaluation"] = fpath.read_text(encoding="utf-8", errors="replace")
        elif name.endswith(".py"):
            outputs["scripts"].append(
                {"name": name, "content": fpath.read_text(encoding="utf-8", errors="replace")}
            )
        elif name.endswith(".png") or name.endswith(".jpg") or name.endswith(".svg"):
            outputs["figures"].append(str(fpath))
        else:
            outputs["data_files"].append(str(fpath))

    return outputs


def read_confidence(analysis_root: Path) -> Dict[str, Any]:
    """
    Read ``background/confidence.json`` and return the parsed dict.

    Returns a safe default if the file is missing or corrupt.
    """
    conf_path = analysis_root / "background" / "confidence.json"
    try:
        if conf_path.exists():
            data = json.loads(conf_path.read_text(encoding="utf-8"))
            # Validate essential fields
            if "current_confidence" in data and isinstance(data.get("history"), list):
                return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read confidence.json: %s", exc)

    return {"current_confidence": 0.0, "history": []}


# ---------------------------------------------------------------------------
# Prompt template loading helper
# ---------------------------------------------------------------------------

def load_analysis_prompt_template() -> str:
    """Load the agent analysis prompt template from prompts/agent_analysis.yaml."""
    prompt_path = PACKAGE_ROOT / "prompts" / "agent_analysis.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Analysis prompt template not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("prompt_template", "")
