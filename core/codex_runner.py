"""
Codex CLI runner for AutoInterp.
Invokes `codex exec` subprocess for agentic tasks (e.g. Jupyter notebook generation).
"""

import subprocess
import threading
from pathlib import Path
from typing import Optional


def run_codex_exec(
    prompt: str,
    cwd: Path,
    sandbox: str = "workspace-write",
    timeout: int = 600,
    skip_git_repo_check: bool = True,
    stream_output: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run Codex CLI in non-interactive mode with the given prompt.

    Args:
        prompt: Instruction for the Codex agent.
        cwd: Working directory for the agent (project root).
        sandbox: Sandbox policy: "workspace-write", "read-only", or "danger-full-access".
        timeout: Timeout in seconds for the subprocess.
        skip_git_repo_check: If True, pass --skip-git-repo-check to allow running outside a Git repo.
        stream_output: If True, stream Codex stdout/stderr to the terminal in real time.

    Returns:
        subprocess.CompletedProcess with stdout, stderr, returncode.
    """
    # Use --full-auto (workspace-write + on-request) unless danger-full-access is needed.
    # danger-full-access requires -a never for non-interactive; --full-auto would override sandbox.
    use_full_auto = sandbox == "workspace-write"
    cmd = ["codex", "exec", "-C", str(cwd.resolve())]
    if use_full_auto:
        cmd.extend(["--full-auto"])
    else:
        cmd.extend(["--sandbox", sandbox, "-a", "never"])
    if skip_git_repo_check:
        cmd.append("--skip-git-repo-check")
    cmd.append(prompt)

    if stream_output:
        return _run_with_streaming(cmd, cwd, timeout)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _run_with_streaming(
    cmd: list, cwd: Path, timeout: int
) -> subprocess.CompletedProcess:
    """Run subprocess and stream stdout/stderr to terminal in real time."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def read_stream(stream, chunks: list):
        for line in iter(stream.readline, ""):
            chunks.append(line)
            print(f"[Codex] {line.rstrip()}", flush=True)

    t_out = threading.Thread(target=read_stream, args=(proc.stdout, stdout_chunks))
    t_err = threading.Thread(target=read_stream, args=(proc.stderr, stderr_chunks))
    t_out.daemon = True
    t_err.daemon = True
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    finally:
        proc.stdout.close()
        proc.stderr.close()
        t_out.join(timeout=1)
        t_err.join(timeout=1)

    return subprocess.CompletedProcess(
        cmd,
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )
