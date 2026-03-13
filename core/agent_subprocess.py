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

logger = logging.getLogger(__name__)

POLL_INTERVAL = 3.0  # seconds between filesystem polls


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


def _snapshot_files(spec: MilestoneSpec) -> Set[str]:
    """Return the set of files currently matching any milestone pattern."""
    seen: Set[str] = set()
    if not spec.watch_dir.exists():
        return seen
    for mp in spec.patterns:
        for match in _glob.glob(str(spec.watch_dir / mp.glob)):
            seen.add(match)
    return seen


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
        Maximum wall-clock seconds before the process is killed.
    milestone : MilestoneSpec, optional
        Filesystem patterns to watch for progress updates.
    on_progress : callable, optional
        Called with a message string each time a new milestone file appears.

    Returns
    -------
    dict with keys ``success``, ``stdout``, ``stderr``, ``returncode``.
    """
    # Snapshot existing files before launch
    seen_files: Set[str] = set()
    if milestone:
        seen_files = _snapshot_files(milestone)

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

    # Poll loop: wait for process to finish while checking milestones
    while True:
        elapsed = time.monotonic() - start_time

        # Check overall timeout
        if elapsed >= timeout:
            logger.warning("Agent timed out after %ds, killing process.", timeout)
            proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                stdout, stderr = "", ""
            if on_progress:
                on_progress(f"Timed out after {timeout}s")
            return {
                "success": False,
                "stdout": stdout or "",
                "stderr": stderr or f"Timed out after {timeout}s",
                "returncode": -9,
            }

        # Wait a polling interval for the process to finish
        remaining = min(POLL_INTERVAL, timeout - elapsed)
        try:
            stdout, stderr = proc.communicate(timeout=remaining)
            # Process finished
            return {
                "success": proc.returncode == 0,
                "stdout": stdout or "",
                "stderr": stderr or "",
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            # Process still running — poll for milestones
            pass

        # Check for new milestone files
        if milestone and on_progress:
            current_files = _snapshot_files(milestone)
            new_files = current_files - seen_files
            for fpath in sorted(new_files):
                fname = Path(fpath).name
                # Find which pattern matched and get the message
                for mp in milestone.patterns:
                    if _glob.fnmatch.fnmatch(fname, mp.glob):
                        try:
                            msg = mp.message_fn(fname)
                            on_progress(msg)
                        except Exception:
                            on_progress(f"New file: {fname}")
                        break
                else:
                    on_progress(f"New file: {fname}")
            seen_files = current_files
