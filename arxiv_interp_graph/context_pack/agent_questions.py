"""
Run an external AI agent (claude CLI or codex CLI) to generate research questions
from downloaded PDFs, producing Research_Questions.txt.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _build_prompt(pdf_dir: Path, template: str) -> str:
    """Replace (dir) placeholder in the prompt template with the absolute PDF directory path."""
    return template.replace("(dir)", str(pdf_dir.resolve()))


def _get_agent_command(provider: str, prompt_text: str, literature_dir: Path):
    """
    Return (cmd_list, subprocess_kwargs) for the selected provider's CLI agent,
    or None if the provider doesn't have a supported CLI agent.

    - anthropic -> claude -p --dangerously-skip-permissions <prompt>
    - openai    -> codex exec -s workspace-write <prompt>
    """
    provider_lower = (provider or "").lower()

    if provider_lower == "anthropic":
        cli = "claude"
        if not shutil.which(cli):
            return None
        cmd = [cli, "-p", "--dangerously-skip-permissions", prompt_text]
        return cmd, {"cwd": str(literature_dir)}

    if provider_lower == "openai":
        cli = "codex"
        if not shutil.which(cli):
            return None
        cmd = [cli, "exec", "-s", "workspace-write", prompt_text]
        return cmd, {"cwd": str(literature_dir)}

    return None


def run_agent_question_generation(
    provider: str,
    literature_dir: str | Path,
    prompt_template: str,
    timeout: int = 600,
) -> Optional[str]:
    """
    Invoke a CLI agent (claude or codex) to read PDFs and write Research_Questions.txt.

    Returns the content of Research_Questions.txt on success, or None on any failure.
    """
    literature_dir = Path(literature_dir)
    pdf_dir = literature_dir / "pdfs"

    # Check PDFs exist
    pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    if not pdf_files:
        logger.warning("No PDFs found in %s; skipping agent question generation.", pdf_dir)
        return None

    prompt_text = _build_prompt(pdf_dir, prompt_template)

    result = _get_agent_command(provider, prompt_text, literature_dir)
    if result is None:
        cli_name = "claude" if provider.lower() == "anthropic" else "codex"
        logger.warning(
            "Agent CLI '%s' not found or provider '%s' unsupported; skipping agent generation.",
            cli_name, provider,
        )
        return None

    cmd, kwargs = result
    logger.debug("Running agent question generation: %s (timeout=%ds)", cmd[0], timeout)
    print(f"[AUTOINTERP] Running {cmd[0]} agent to generate research questions (timeout={timeout}s)...")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            **kwargs,
        )
        if proc.returncode != 0:
            logger.warning("Agent exited with code %d. stderr: %s", proc.returncode, proc.stderr[:500])
            print(f"[AUTOINTERP] Agent exited with code {proc.returncode}")
    except subprocess.TimeoutExpired:
        logger.warning("Agent timed out after %ds.", timeout)
        print(f"[AUTOINTERP] Agent timed out after {timeout}s.")
        return None
    except FileNotFoundError as e:
        logger.warning("Agent CLI not found: %s", e)
        return None
    except Exception as e:
        logger.warning("Agent subprocess failed: %s", e)
        return None

    # Read Research_Questions.txt
    rq_path = literature_dir / "Research_Questions.txt"
    if not rq_path.exists():
        logger.warning("Agent did not produce Research_Questions.txt in %s", literature_dir)
        print("[AUTOINTERP] Agent did not produce Research_Questions.txt; will fall back to LLM.")
        return None

    content = rq_path.read_text(encoding="utf-8").strip()
    if not content:
        logger.warning("Research_Questions.txt is empty.")
        return None

    logger.info("Agent produced Research_Questions.txt (%d chars).", len(content))
    print(f"[AUTOINTERP] Agent produced Research_Questions.txt ({len(content)} chars).")
    return content
