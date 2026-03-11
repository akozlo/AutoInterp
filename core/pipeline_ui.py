"""
PipelineUI — Rich terminal output and HTML dashboard for AutoInterp pipeline.

Provides structured, scannable terminal output using the ``rich`` library and
a self-contained, auto-refreshing HTML dashboard written to disk after every
LLM interaction.
"""

import os
import time
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .dashboard_template import (
    DASHBOARD_TEMPLATE,
    render_tab_buttons,
    render_tab_content,
    STEP_CONFIG,
)

# Try to import rich — gracefully degrade if unavailable
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LLMInteraction:
    """Record of a single LLM API call."""
    agent_name: str
    display_name: str
    prompt: str
    system_message: Optional[str]
    response: str
    model: str
    provider: str
    temperature: float
    max_tokens: int
    duration_seconds: float
    timestamp: datetime
    step_id: str
    iteration_number: Optional[int] = None


@dataclass
class PipelineStep:
    """A single pipeline step (e.g. question_generation, iterative_analysis)."""
    step_id: str
    display_name: str
    status: str = "pending"  # pending | running | completed | failed | skipped
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    summary: Optional[str] = None
    llm_interactions: List[LLMInteraction] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent name → display name mapping
# ---------------------------------------------------------------------------

AGENT_DISPLAY_NAMES = {
    "question_generator": "Question Generator",
    "question_prioritizer": "Question Prioritizer",
    "analysis_planner": "Analysis Planner",
    "analysis_generator": "Analysis Generator",
    "evaluator": "Evaluator",
    "visualization_planner": "Visualization Planner",
    "visualization_generator": "Visualization Generator",
    "visualization_evaluator": "Visualization Evaluator",
    "reporter": "Reporter",
    "title_generator": "Title Generator",
}


def _agent_display_name(agent_name: str) -> str:
    return AGENT_DISPLAY_NAMES.get(agent_name, agent_name)


# ---------------------------------------------------------------------------
# Agent name → step_id mapping
# ---------------------------------------------------------------------------

AGENT_STEP_MAP = {
    "question_generator": "question_generation",
    "question_prioritizer": "question_prioritization",
    "analysis_planner": "iterative_analysis",
    "analysis_generator": "iterative_analysis",
    "evaluator": "iterative_analysis",
    "visualization_planner": "visualization",
    "visualization_generator": "visualization",
    "visualization_evaluator": "visualization",
    "reporter": "report_generation",
    "title_generator": "report_generation",
}


def infer_step_id(agent_name: str) -> str:
    """Map an agent name to the pipeline step it belongs to."""
    return AGENT_STEP_MAP.get(agent_name, "iterative_analysis")


# ---------------------------------------------------------------------------
# PipelineUI
# ---------------------------------------------------------------------------

# Ordered list of step_ids and their display names
PIPELINE_STEPS = [
    ("question_generation", "Question Generation"),
    ("question_prioritization", "Question Prioritization"),
    ("iterative_analysis", "Iterative Analysis"),
    ("visualization", "Visualization"),
    ("report_generation", "Report Generation"),
]


class PipelineUI:
    """
    Manages Rich terminal output and an HTML dashboard for the pipeline.

    Usage:
        ui = PipelineUI(project_dir, config)
        ui.pipeline_start("my task")
        ui.step_start("question_generation")
        ui.llm_call_start("question_generator", ...)
        ui.llm_call_complete("question_generator", prompt, ..., response, ...)
        ui.step_complete("question_generation", summary="3 questions")
        ...
        ui.pipeline_complete(result)
    """

    def __init__(self, project_dir: Path, config: Dict[str, Any]):
        self.project_dir = Path(project_dir)
        self.config = config

        # Read UI config
        ui_config = config.get("ui", {}) or {}
        self.rich_enabled = ui_config.get("rich_terminal", True) and RICH_AVAILABLE
        self.dashboard_enabled = ui_config.get("html_dashboard", True)
        self.dashboard_refresh = ui_config.get("dashboard_refresh", 5)
        self.auto_open_browser = ui_config.get("auto_open_browser", True)

        # Rich console
        if self.rich_enabled:
            self.console = Console()
        else:
            self.console = None

        # Pipeline state
        self.task_name: str = ""
        self.pipeline_start_time: Optional[datetime] = None
        self.steps: Dict[str, PipelineStep] = {}
        self._active_status = None  # rich console.status context manager

        # Initialize steps
        for step_id, display_name in PIPELINE_STEPS:
            self.steps[step_id] = PipelineStep(step_id=step_id, display_name=display_name)

        # Dashboard path
        self.dashboard_path = self.project_dir / "dashboard.html"

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------

    def pipeline_start(self, task_name: str) -> None:
        """Called at the start of the pipeline."""
        self.task_name = task_name
        self.pipeline_start_time = datetime.now()

        if self.rich_enabled:
            self.console.print()
            self.console.print(
                Panel(
                    f"[bold gold1]AutoInterp Pipeline[/bold gold1]\n[dim]{task_name}[/dim]",
                    border_style="gold1",
                    padding=(1, 2),
                )
            )

        if self.dashboard_enabled:
            self._write_dashboard()
            if self.auto_open_browser:
                self._open_browser()

    def pipeline_complete(self, result: Dict[str, Any]) -> None:
        """Called when the pipeline finishes successfully."""
        # Mark any still-running steps as completed
        for step in self.steps.values():
            if step.status == "running":
                step.status = "completed"
                step.end_time = datetime.now()

        if self.rich_enabled:
            elapsed = ""
            if self.pipeline_start_time:
                dt = (datetime.now() - self.pipeline_start_time).total_seconds()
                elapsed = f"  ({self._fmt_duration(dt)})"

            self.console.print()
            self.console.print(
                Panel(
                    f"[bold green]Pipeline Complete[/bold green]{elapsed}\n"
                    f"[dim]Conclusion: {result.get('conclusion', 'N/A')}  |  "
                    f"Confidence: {result.get('final_confidence', 0):.2f}[/dim]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            self._print_step_table()

        if self.dashboard_enabled:
            self._write_dashboard(final=True)

    def pipeline_failed(self, error: str) -> None:
        """Called when the pipeline fails."""
        for step in self.steps.values():
            if step.status == "running":
                step.status = "failed"
                step.end_time = datetime.now()

        if self.rich_enabled:
            self.console.print()
            self.console.print(
                Panel(
                    f"[bold red]Pipeline Failed[/bold red]\n[dim]{error[:200]}[/dim]",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        if self.dashboard_enabled:
            self._write_dashboard(final=True)

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def step_start(self, step_id: str) -> None:
        """Mark a step as running."""
        step = self.steps.get(step_id)
        if not step:
            return
        step.status = "running"
        step.start_time = datetime.now()

        if self.rich_enabled:
            self.console.print(
                f"\n[bold gold1]{step.display_name}[/bold gold1] [dim]started[/dim]"
            )

        if self.dashboard_enabled:
            self._write_dashboard()

    def step_complete(self, step_id: str, summary: str = "") -> None:
        """Mark a step as completed."""
        step = self.steps.get(step_id)
        if not step:
            return
        step.status = "completed"
        step.end_time = datetime.now()
        if summary:
            step.summary = summary

        if self.rich_enabled:
            elapsed = ""
            if step.start_time:
                elapsed = f" ({self._fmt_duration((step.end_time - step.start_time).total_seconds())})"
            summary_str = f"  [dim]{summary}[/dim]" if summary else ""
            self.console.print(
                f"  [green]\\[done][/green] {step.display_name}{elapsed}{summary_str}"
            )

        if self.dashboard_enabled:
            self._write_dashboard()

    def step_failed(self, step_id: str, error: str = "") -> None:
        """Mark a step as failed."""
        step = self.steps.get(step_id)
        if not step:
            return
        step.status = "failed"
        step.end_time = datetime.now()
        if error:
            step.summary = error[:120]

        if self.rich_enabled:
            self.console.print(
                f"  [red]\\[failed][/red] {step.display_name}: {error[:100]}"
            )

        if self.dashboard_enabled:
            self._write_dashboard()

    def step_skipped(self, step_id: str, reason: str = "") -> None:
        """Mark a step as skipped."""
        step = self.steps.get(step_id)
        if not step:
            return
        step.status = "skipped"
        if reason:
            step.summary = reason

        if self.rich_enabled:
            self.console.print(
                f"  [dim]\\[skipped][/dim] {step.display_name}"
                + (f" [dim]({reason})[/dim]" if reason else "")
            )

        if self.dashboard_enabled:
            self._write_dashboard()

    # ------------------------------------------------------------------
    # LLM call tracking
    # ------------------------------------------------------------------

    def llm_call_start(
        self,
        agent_name: str,
        display_name: str,
        model: str,
        provider: str,
        iteration_number: Optional[int] = None,
    ) -> None:
        """Record that an LLM call has started (terminal output handled by LLMInterface)."""
        pass  # Verbose colored output is printed by LLMInterface directly

    def llm_call_complete(
        self,
        agent_name: str,
        display_name: str,
        prompt: str,
        system_message: Optional[str],
        response: str,
        model: str,
        provider: str,
        temperature: float,
        max_tokens: int,
        duration_seconds: float,
        step_id: Optional[str] = None,
        iteration_number: Optional[int] = None,
    ) -> None:
        """Record a completed LLM call and update the HTML dashboard."""
        resolved_step_id = step_id or infer_step_id(agent_name)

        interaction = LLMInteraction(
            agent_name=agent_name,
            display_name=display_name,
            prompt=prompt,
            system_message=system_message,
            response=response,
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            duration_seconds=duration_seconds,
            timestamp=datetime.now(),
            step_id=resolved_step_id,
            iteration_number=iteration_number,
        )

        # Append to the appropriate step
        step = self.steps.get(resolved_step_id)
        if step:
            step.llm_interactions.append(interaction)

        # Terminal output handled by LLMInterface's colored verbose prints

        if self.dashboard_enabled:
            self._write_dashboard()

    # ------------------------------------------------------------------
    # Utility outputs
    # ------------------------------------------------------------------

    def show_result(self, label: str, value: str) -> None:
        """Display a key result (e.g. selected question, confidence score)."""
        if self.rich_enabled:
            self.console.print(
                Panel(
                    f"[bold]{label}[/bold]\n{value}",
                    border_style="cyan",
                    padding=(0, 2),
                )
            )

    def update_project_dir(self, new_dir: Path) -> None:
        """Handle project directory rename mid-pipeline."""
        self.project_dir = Path(new_dir)
        self.dashboard_path = self.project_dir / "dashboard.html"
        if self.dashboard_enabled:
            self._write_dashboard()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stop_active_status(self) -> None:
        """Stop the current Rich spinner if one is active."""
        if self._active_status:
            try:
                self._active_status.stop()
            except Exception:
                pass
            self._active_status = None

    def _print_step_table(self) -> None:
        """Print a compact step-summary table to the terminal."""
        if not self.rich_enabled:
            return
        table = Table(show_header=True, header_style="bold", border_style="dim")
        table.add_column("Step", style="bold")
        table.add_column("Status")
        table.add_column("LLM Calls", justify="right")
        table.add_column("Time", justify="right")
        table.add_column("Summary")

        for step_id, _ in PIPELINE_STEPS:
            step = self.steps[step_id]
            status_style = {
                "completed": "green",
                "failed": "red",
                "running": "yellow",
                "skipped": "dim",
                "pending": "dim",
            }.get(step.status, "")
            n_calls = len(step.llm_interactions)
            elapsed = ""
            if step.start_time and step.end_time:
                elapsed = self._fmt_duration((step.end_time - step.start_time).total_seconds())
            table.add_row(
                step.display_name,
                f"[{status_style}]{step.status}[/{status_style}]",
                str(n_calls),
                elapsed,
                step.summary or "",
            )
        self.console.print(table)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        """Format a duration in seconds to a human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"

    # ------------------------------------------------------------------
    # Dashboard rendering
    # ------------------------------------------------------------------

    def _steps_as_dicts(self) -> list:
        """Convert steps to list of dicts ordered by tab_order for template rendering."""
        result = []
        for step_id, _ in PIPELINE_STEPS:
            step = self.steps[step_id]
            result.append({
                "step_id": step.step_id,
                "display_name": step.display_name,
                "status": step.status,
                "start_time": step.start_time,
                "end_time": step.end_time,
                "summary": step.summary,
                "llm_interactions": [
                    {
                        "agent_name": i.agent_name,
                        "display_name": i.display_name,
                        "prompt": i.prompt,
                        "system_message": i.system_message,
                        "response": i.response,
                        "model": i.model,
                        "provider": i.provider,
                        "temperature": i.temperature,
                        "max_tokens": i.max_tokens,
                        "duration_seconds": i.duration_seconds,
                        "timestamp": i.timestamp,
                        "step_id": i.step_id,
                        "iteration_number": i.iteration_number,
                    }
                    for i in step.llm_interactions
                ],
            })
        return result

    def _write_dashboard(self, final: bool = False) -> None:
        """Write (or rewrite) the HTML dashboard file atomically."""
        try:
            self.project_dir.mkdir(parents=True, exist_ok=True)
            steps_data = self._steps_as_dicts()

            refresh_interval_ms = self.dashboard_refresh * 1000 if not final else 0

            html = DASHBOARD_TEMPLATE.format(
                refresh_interval_ms=refresh_interval_ms,
                is_final="true" if final else "false",
                task_name=self.task_name or "AutoInterp Pipeline",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tab_buttons=render_tab_buttons(steps_data),
                tab_content=render_tab_content(steps_data, self.task_name),
            )

            # Atomic write: write to temp file then rename
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.project_dir), suffix=".tmp", prefix=".dashboard_"
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    f.write(html)
                os.replace(tmp_path, str(self.dashboard_path))
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        except Exception as e:
            # Dashboard write failure should never crash the pipeline
            if self.rich_enabled:
                self.console.print(f"  [dim][warning] Dashboard write failed: {e}[/dim]")

    def _open_browser(self) -> None:
        """Attempt to open the dashboard in the default browser."""
        try:
            import webbrowser
            webbrowser.open(self.dashboard_path.as_uri())
        except Exception:
            pass  # Silently ignore — not critical
