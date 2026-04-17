"""
Run an external AI agent (claude CLI or codex CLI) to generate all
visualizations autonomously within one subprocess invocation.
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

def _get_visualization_agent_command(
    provider: str,
    prompt_text: str,
    viz_dir: Path,
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
        return cmd, {"cwd": str(viz_dir)}

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
        return cmd, {"cwd": str(viz_dir)}

    return None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_visualization_prompt(
    prompt_template: str,
    analysis_root: Path,
) -> str:
    """
    Substitute ``{analysis_listing}`` in the prompt template with a directory
    listing of the analysis background and iteration directories.

    The ``{n}`` references in the prompt are literal text (the agent's naming
    convention), not Python placeholders.
    """
    lines: List[str] = []

    # List background directory
    bg_dir = analysis_root / "background"
    if bg_dir.exists():
        files = sorted(f.name for f in bg_dir.iterdir() if f.is_file())
        lines.append(f"- background/: {', '.join(files) if files else '(empty)'}")

    # List each analysis_N directory
    iter_dirs = sorted(
        (d for d in analysis_root.iterdir() if d.is_dir() and d.name.startswith("analysis_")),
        key=lambda d: d.name,
    )
    for d in iter_dirs:
        files = sorted(f.name for f in d.iterdir() if f.is_file())
        lines.append(f"- {d.name}/: {', '.join(files) if files else '(empty)'}")

    analysis_listing = "\n".join(lines) if lines else "(no analysis directories found)"

    return prepend_persona(
        prompt_template.replace("{analysis_listing}", analysis_listing),
        "agent_visualization.yaml",
    )


# ---------------------------------------------------------------------------
# Agent subprocess execution
# ---------------------------------------------------------------------------

def run_visualization_agent(
    provider: str,
    viz_dir: Path,
    prompt_text: str,
    timeout: int = 900,
    on_progress: Optional[Callable[[str], None]] = None,
    model: str = "",
    sandbox_bypass: bool = False,
) -> Dict[str, Any]:
    """
    Launch the CLI agent subprocess for visualization generation and return
    the result.

    Returns ``{"success": bool, "stdout": str, "stderr": str, "returncode": int}``.
    """
    result = _get_visualization_agent_command(provider, prompt_text, viz_dir, model=model, sandbox_bypass=sandbox_bypass)
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
    cwd.mkdir(parents=True, exist_ok=True)

    logger.debug("Running visualization agent: %s (timeout=%ds)", cmd[0], timeout)
    print(f"[AUTOINTERP] Running {cmd[0]} visualization agent (timeout={timeout}s)...")

    milestone = MilestoneSpec(
        watch_dir=cwd,
        patterns=[
            MilestonePattern(
                glob="Visualization_log.md",
                message_fn=lambda _: "Wrote Visualization_log.md",
            ),
            MilestonePattern(
                glob="figure_*.py",
                message_fn=lambda fname: f"Wrote script: {fname}",
            ),
            MilestonePattern(
                glob="figure_*.png",
                message_fn=lambda fname: f"Generated figure: {fname}",
            ),
            MilestonePattern(
                glob="figure_*.jpg",
                message_fn=lambda fname: f"Generated figure: {fname}",
            ),
            MilestonePattern(
                glob="figure_*.svg",
                message_fn=lambda fname: f"Generated figure: {fname}",
            ),
            MilestonePattern(
                glob="caption_*.txt",
                message_fn=lambda fname: f"Wrote caption: {fname}",
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
            "Visualization agent exited with code %d. stderr: %s",
            proc_result["returncode"],
            proc_result["stderr"][:500],
        )
        print(f"[AUTOINTERP] Visualization agent exited with code {proc_result['returncode']}")

    return proc_result


# ---------------------------------------------------------------------------
# Reading agent outputs
# ---------------------------------------------------------------------------

def read_visualization_outputs(viz_dir: Path) -> Dict[str, str]:
    """
    Read the files produced by the visualization agent.

    Returns a dict mapping display names to file paths (compatible with the
    ``generate_visualizations()`` return type used by the legacy pipeline).

    Display names are derived from ``caption_N.txt`` files when present,
    otherwise from the figure filename.
    """
    outputs: Dict[str, str] = {}

    if not viz_dir.exists():
        return outputs

    # Collect all figure image files
    image_extensions = (".png", ".jpg", ".jpeg", ".svg")
    figure_files = sorted(
        f for f in viz_dir.iterdir()
        if f.is_file() and f.name.startswith("figure_") and f.suffix.lower() in image_extensions
    )

    for fig_path in figure_files:
        # Try to find a matching caption file: figure_1.png -> caption_1.txt
        stem = fig_path.stem  # e.g. "figure_1"
        num_part = stem.replace("figure_", "", 1)  # e.g. "1"
        caption_path = viz_dir / f"caption_{num_part}.txt"

        if caption_path.exists():
            try:
                display_name = caption_path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                display_name = fig_path.name
        else:
            display_name = fig_path.name

        if not display_name:
            display_name = fig_path.name

        outputs[display_name] = str(fig_path)

    return outputs


# ---------------------------------------------------------------------------
# Prompt template loading helper
# ---------------------------------------------------------------------------

def load_visualization_prompt_template() -> str:
    """Load the agent visualization prompt template from prompts/agent_visualization.yaml."""
    prompt_path = PACKAGE_ROOT / "prompts" / "agent_visualization.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Visualization prompt template not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("prompt_template", "")
