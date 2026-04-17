"""
Shared Popen + filesystem-polling runner for CLI agent subprocesses.

Replaces blocking ``subprocess.run(capture_output=True)`` with a polling loop
that detects milestone files as the agent creates them, emitting progress
callbacks to the terminal and HTML dashboard.
"""

import glob as _glob
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import psutil

logger = logging.getLogger(__name__)

POLL_INTERVAL = 3.0  # seconds between filesystem polls
HEARTBEAT_INTERVAL = 120.0  # seconds between heartbeat messages when idle


@dataclass
class MilestonePattern:
    """A glob pattern and a function that turns a matched filename into a message."""
    glob: str
    message_fn: Callable[[str], str]


@dataclass
class MilestoneSpec:
    """Where to watch and what patterns to look for."""
    watch_dir: Path
    patterns: List[MilestonePattern] = field(default_factory=list)


def _snapshot_all_files(watch_dir: Path) -> Set[str]:
    """Return the set of all regular files currently in the watch directory."""
    if not watch_dir.exists():
        return set()
    return {str(f) for f in watch_dir.iterdir() if f.is_file()}


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds into a readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins:02d}m"


def _has_active_children(pid: int) -> bool:
    """Return True if the process *pid* has any running child processes."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        return any(c.is_running() for c in children)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def run_agent_with_polling(
    cmd: List[str],
    cwd: Path,
    timeout: int,
    milestone: Optional[MilestoneSpec] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Launch an agent subprocess and poll the filesystem for milestone files.

    Parameters
    ----------
    cmd : list of str
        The command to execute.
    cwd : Path
        Working directory for the subprocess.
    timeout : int
        Maximum *agent-thinking* seconds before the process is killed.
        Time spent waiting on child processes (scripts, python, bash, etc.)
        is excluded from this budget.
    milestone : MilestoneSpec, optional
        Filesystem patterns to watch for progress updates.
    on_progress : callable, optional
        Called with a message string each time a new milestone file appears,
        a previously unknown file is detected, or a heartbeat interval elapses.

    Returns
    -------
    dict with keys ``success``, ``stdout``, ``stderr``, ``returncode``.
    """
    # Snapshot ALL existing files in watch_dir before launch so we can
    # detect any new file the agent creates — not just milestone matches.
    seen_files: Set[str] = set()
    if milestone:
        seen_files = _snapshot_all_files(milestone.watch_dir)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd),
        )
    except FileNotFoundError as exc:
        logger.warning("Agent CLI not found: %s", exc)
        return {"success": False, "stdout": "", "stderr": str(exc), "returncode": -1}
    except Exception as exc:
        logger.warning("Failed to start agent subprocess: %s", exc)
        return {"success": False, "stdout": "", "stderr": str(exc), "returncode": -1}

    start_time = time.monotonic()
    last_poll_time = start_time
    last_event_time = start_time  # tracks last milestone, file event, or heartbeat
    agent_thinking_time = 0.0  # accumulates only when no child processes are running

    # Poll loop: wait for process to finish while checking milestones
    while True:
        now = time.monotonic()
        elapsed = now - start_time  # wall-clock (for display / heartbeat)
        delta = now - last_poll_time

        # Accumulate agent-only thinking time (exclude child-process wait)
        has_children = _has_active_children(proc.pid)
        if not has_children:
            agent_thinking_time += delta
        last_poll_time = now

        # Check agent-thinking timeout
        if agent_thinking_time >= timeout:
            wall = time.monotonic() - start_time
            logger.warning(
                "Agent thinking time exceeded %ds (wall clock: %s), killing process.",
                timeout, _fmt_elapsed(wall),
            )
            # Kill child processes first, then the agent
            try:
                parent = psutil.Process(proc.pid)
                for child in parent.children(recursive=True):
                    child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                stdout, stderr = "", ""
            msg = (
                f"Agent thinking time exceeded {timeout}s "
                f"(wall clock: {_fmt_elapsed(wall)})"
            )
            if on_progress:
                on_progress(msg)
            return {
                "success": False,
                "stdout": stdout or "",
                "stderr": stderr or msg,
                "returncode": -9,
            }

        # Wait a polling interval for the process to finish
        try:
            stdout, stderr = proc.communicate(timeout=POLL_INTERVAL)
            # Process finished
            total = time.monotonic() - start_time
            if on_progress:
                on_progress(f"Agent finished ({_fmt_elapsed(total)})")

            # Detect bwrap sandbox failures (old bwrap on RHEL 8, etc.)
            _stderr = stderr or ""
            if proc.returncode != 0 and "bwrap" in _stderr.lower():
                bwrap_msg = (
                    "Codex sandbox (bwrap) failed. If your system has an "
                    "old bwrap version, set codex.sandbox_bypass: true in "
                    "config.yaml"
                )
                print(f"[AUTOINTERP] {bwrap_msg}")
                logger.warning(bwrap_msg)
                if on_progress:
                    on_progress(bwrap_msg)

            return {
                "success": proc.returncode == 0,
                "stdout": stdout or "",
                "stderr": _stderr,
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            # Process still running — poll for milestones
            pass

        # Check for ANY new files in the watch directory
        if milestone and on_progress:
            current_files = _snapshot_all_files(milestone.watch_dir)
            new_files = current_files - seen_files
            for fpath in sorted(new_files):
                fname = Path(fpath).name
                # Check if it matches a milestone pattern for a specific message
                matched = False
                for mp in milestone.patterns:
                    if _glob.fnmatch.fnmatch(fname, mp.glob):
                        try:
                            msg = mp.message_fn(fname)
                            on_progress(msg)
                        except Exception:
                            on_progress(f"New file: {fname}")
                        matched = True
                        break
                if not matched:
                    on_progress(f"New file: {fname}")
                last_event_time = time.monotonic()
            seen_files = current_files

        # Heartbeat: emit a "still running" message when no events for a while
        if on_progress:
            since_event = time.monotonic() - last_event_time
            if since_event >= HEARTBEAT_INTERVAL:
                on_progress(f"Still running... {_fmt_elapsed(elapsed)} elapsed")
                last_event_time = time.monotonic()
