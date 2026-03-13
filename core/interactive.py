"""
Interactive mode helpers — checkpoint loops that pause after each pipeline
stage, display output, and optionally revise via LLM based on user feedback.
"""

import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

import yaml

from AutoInterp.core.utils import PACKAGE_ROOT

logger = logging.getLogger(__name__)

# Cache loaded prompts
_interactive_prompts: Optional[Dict[str, Any]] = None


def _load_interactive_prompts() -> Dict[str, Any]:
    """Load and cache prompts/interactive.yaml."""
    global _interactive_prompts
    if _interactive_prompts is not None:
        return _interactive_prompts
    prompt_path = PACKAGE_ROOT / "prompts" / "interactive.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Interactive prompt template not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        _interactive_prompts = yaml.safe_load(f) or {}
    return _interactive_prompts


def is_interactive(config: Dict[str, Any]) -> bool:
    """Return True if interactive mode is enabled in *config*."""
    return config.get("interactive_mode", False)


def format_stage_output(text: str, stage_name: str, max_lines: int = 80) -> str:
    """Format pipeline stage output for terminal display with header/separator."""
    lines = text.splitlines()
    if len(lines) > max_lines:
        truncated = lines[:max_lines]
        truncated.append(f"\n... ({len(lines) - max_lines} more lines, showing first {max_lines}) ...")
        display_text = "\n".join(truncated)
    else:
        display_text = text

    header = f"[INTERACTIVE] === {stage_name} ==="
    separator = "=" * len(header)
    return f"\n{separator}\n{header}\n{separator}\n{display_text}\n{separator}"


async def make_revision_call(
    llm_interface: Any,
    agent_name: str,
    original_output: str,
    feedback: str,
    stage_key: str,
) -> str:
    """
    Call the LLM to revise *original_output* given user *feedback*.

    *stage_key* selects the revision prompt from ``prompts/interactive.yaml``
    (e.g. ``"question_generation"``, ``"report"``).
    """
    prompts = _load_interactive_prompts()
    system_message = prompts.get("revision_system_message", "")
    stage_prompts = prompts.get(stage_key, {})
    prompt_template = stage_prompts.get("revision_prompt", "")

    if not prompt_template:
        logger.warning("No revision prompt found for stage_key=%s; returning original", stage_key)
        return original_output

    prompt = prompt_template.replace("{original_output}", original_output)
    prompt = prompt.replace("{user_feedback}", feedback)

    revised = await llm_interface.generate(
        prompt=prompt,
        system_message=system_message,
        agent_name=agent_name,
    )
    return revised


async def interactive_checkpoint(
    stage_name: str,
    output_text: str,
    revise_fn: Callable[[str, str], Awaitable[str]],
    save_fn: Callable[[str], None],
    config: Dict[str, Any],
) -> str:
    """
    Core reusable feedback loop for interactive mode.

    If interactive mode is disabled, returns *output_text* immediately.

    Otherwise, displays the output and prompts the user for feedback.
    Empty input (Enter) means "continue".  Any other input triggers a
    revision via *revise_fn(current_output, feedback)*, saves via
    *save_fn(revised_output)*, and re-displays.

    Returns the (possibly revised) output text.
    """
    if not is_interactive(config):
        return output_text

    current = output_text

    while True:
        # Display current output
        print(format_stage_output(current, stage_name))

        try:
            feedback = input("[INTERACTIVE] Provide feedback, or press Enter to continue: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Non-interactive stdin or Ctrl-C — treat as "continue"
            print()
            break

        if not feedback:
            break

        print(f"[INTERACTIVE] Revising {stage_name} based on your feedback...")
        current = await revise_fn(current, feedback)
        save_fn(current)
        print(f"[INTERACTIVE] Revision complete. Showing updated output.")

    return current
