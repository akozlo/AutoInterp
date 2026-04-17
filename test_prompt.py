#!/usr/bin/env python3
"""
Prompt testing harness for AutoInterp agent stages.

Replays individual agent stages against completed project runs so prompts
can be iterated in minutes instead of the ~2-hour full pipeline.

Usage:
    # See assembled prompt without running
    python test_prompt.py viz --project <completed_run> --dry-run

    # Run visualization stage with default prompt
    python test_prompt.py viz --project <completed_run>

    # Run with a modified prompt, labeled for comparison
    python test_prompt.py viz --project <completed_run> --prompt my_viz_v2.yaml --label "shorter-captions"

    # Question generation with a specific topic
    python test_prompt.py questions --project <completed_run> --task-description "How do attention heads specialize?"

    # Report with different model
    python test_prompt.py report --project <completed_run> --provider anthropic --model claude-opus-4-6

    # Notebook generation from finalized repo
    python test_prompt.py notebook --project <completed_run>
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure package imports resolve when running this file directly
if __package__ is None or __package__ == "":
    pkg_root = Path(__file__).resolve().parent.parent
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

import yaml

from AutoInterp.src.questions.agent_questions import (
    load_questions_prompt_template,
    _build_questions_prompt,
    run_questions_agent,
    read_questions_outputs,
)
from AutoInterp.src.visualization.agent_visualization import (
    load_visualization_prompt_template,
    _build_visualization_prompt,
    run_visualization_agent,
    read_visualization_outputs,
)
from AutoInterp.src.reporting.agent_report import (
    load_report_prompt_template,
    _build_report_prompt,
    run_report_agent,
    read_report_outputs,
)
from AutoInterp.src.notebook.agent_notebook import (
    load_notebook_prompt_template,
    _build_notebook_prompt,
    run_notebook_agent,
    read_notebook_outputs,
)
from AutoInterp.src.literature_review.agent_literature_review import (
    load_literature_review_prompt_template,
    _build_literature_review_prompt,
    run_literature_review_agent,
    read_literature_review_outputs,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECTS_DIR = PACKAGE_ROOT / "projects"
TEST_RUNS_DIR = PACKAGE_ROOT / "test_runs"

# Stage definitions: output dir name, required input symlinks, optional input symlinks
STAGE_DEFS = {
    "questions": {
        "output_dirs": ["questions"],
        "required_inputs": [],
        "optional_inputs": ["literature"],
        "default_timeout": 300,
    },
    "litreview": {
        "output_dirs": ["literature"],
        "required_inputs": ["questions"],
        "optional_inputs": ["literature"],
        "default_timeout": 1800,
    },
    "viz": {
        "output_dirs": ["visualizations"],
        "required_inputs": ["analysis"],
        "optional_inputs": ["questions"],
        "default_timeout": 900,
    },
    "report": {
        "output_dirs": ["reports"],
        "required_inputs": ["analysis", "visualizations"],
        "optional_inputs": ["questions"],
        "default_timeout": 900,
    },
    "notebook": {
        "output_dirs": ["repo/notebooks"],
        "required_inputs": ["repo"],
        "optional_inputs": ["analysis", "visualizations", "reports"],
        "default_timeout": 900,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_provider_model(args: argparse.Namespace) -> Tuple[str, str]:
    """Resolve provider/model from CLI args, falling back to .last_llm.json."""
    provider = args.provider
    model = args.model

    if provider and model:
        return provider, model

    last_llm_path = PACKAGE_ROOT / ".last_llm.json"
    if last_llm_path.exists():
        try:
            data = json.loads(last_llm_path.read_text(encoding="utf-8"))
            if not provider:
                provider = data.get("provider", "")
            if not model:
                model = data.get("model", "")
        except (json.JSONDecodeError, OSError):
            pass

    if not provider:
        print("Error: No provider specified and .last_llm.json not found.")
        print("Use --provider anthropic/openai or run the full pipeline once first.")
        sys.exit(1)

    return provider, model


def resolve_label(stage: str, label: Optional[str]) -> str:
    """Generate a label for this test run, deduplicating if needed."""
    if not label:
        label = datetime.now().strftime("%Y%m%d_%H%M%S")

    stage_dir = TEST_RUNS_DIR / stage
    candidate = label
    counter = 2
    while (stage_dir / candidate).exists():
        candidate = f"{label}_{counter}"
        counter += 1
    return candidate


def load_prompt_template(stage: str, prompt_path: Optional[str]) -> str:
    """Load prompt template from override file or default."""
    if prompt_path:
        p = Path(prompt_path)
        if not p.exists():
            print(f"Error: Prompt file not found: {p}")
            sys.exit(1)
        text = p.read_text(encoding="utf-8")
        # If it's YAML, extract prompt_template key
        if p.suffix in (".yaml", ".yml"):
            try:
                data = yaml.safe_load(text)
                if isinstance(data, dict) and "prompt_template" in data:
                    return data["prompt_template"]
            except yaml.YAMLError:
                pass
        return text

    # Default templates
    if stage == "questions":
        return load_questions_prompt_template()
    elif stage == "litreview":
        return load_literature_review_prompt_template()
    elif stage == "viz":
        return load_visualization_prompt_template()
    elif stage == "report":
        return load_report_prompt_template()
    elif stage == "notebook":
        return load_notebook_prompt_template()
    else:
        raise ValueError(f"Unknown stage: {stage}")


def resolve_project(project_arg: str) -> Path:
    """Resolve --project to an absolute path."""
    p = Path(project_arg)
    if p.is_absolute() and p.is_dir():
        return p
    # Try as a name under projects/
    candidate = PROJECTS_DIR / project_arg
    if candidate.is_dir():
        return candidate
    # Try as relative path
    if p.is_dir():
        return p.resolve()
    print(f"Error: Project directory not found: {project_arg}")
    print(f"  Tried: {p} and {candidate}")
    sys.exit(1)


def create_test_run_dir(stage: str, label: str, source_project: Path) -> Path:
    """Create the test run directory with symlinks for inputs and real output dirs.

    For the ``notebook`` stage, ``repo/`` is handled specially: individual
    subdirs (paper/, scripts/, data/, results/, README.md) are symlinked
    inside a real ``repo/`` directory, while ``repo/notebooks/`` is created
    as a real directory so the agent can write there.
    """
    stage_def = STAGE_DEFS[stage]
    run_dir = TEST_RUNS_DIR / stage / label
    run_dir.mkdir(parents=True, exist_ok=True)

    # Validate required inputs exist in source project
    for inp in stage_def["required_inputs"]:
        src = source_project / inp
        if not src.exists():
            print(f"Error: Required input '{inp}' not found in project: {source_project}")
            sys.exit(1)

    if stage == "notebook":
        # Special handling: symlink individual repo/ contents, create real notebooks/
        _setup_notebook_test_dir(run_dir, source_project)
    else:
        # Create symlinks for inputs
        for inp in stage_def["required_inputs"] + stage_def["optional_inputs"]:
            src = source_project / inp
            link = run_dir / inp
            if link.exists() or link.is_symlink():
                continue  # Don't overwrite existing links
            if src.exists():
                link.symlink_to(src.resolve())

    # Create real output directories
    for out in stage_def["output_dirs"]:
        (run_dir / out).mkdir(parents=True, exist_ok=True)

    return run_dir


def _setup_notebook_test_dir(run_dir: Path, source_project: Path) -> None:
    """Set up the notebook test directory with repo/ subdirs as symlinks.

    Symlinks each child of source_project/repo/ (except notebooks/) into
    run_dir/repo/, so the agent can read them. ``repo/notebooks/`` is
    created as a real writable directory.
    """
    repo_src = source_project / "repo"
    repo_dst = run_dir / "repo"
    repo_dst.mkdir(parents=True, exist_ok=True)

    for child in sorted(repo_src.iterdir()):
        if child.name == "notebooks":
            continue  # will be a real dir (output)
        link = repo_dst / child.name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(child.resolve())

    # Also symlink optional top-level inputs (analysis, visualizations, reports)
    for inp in STAGE_DEFS["notebook"]["optional_inputs"]:
        src = source_project / inp
        link = run_dir / inp
        if link.exists() or link.is_symlink():
            continue
        if src.exists():
            link.symlink_to(src.resolve())


def build_prompt(
    stage: str,
    template: str,
    run_dir: Path,
    task_description: str = "",
) -> str:
    """Build the final prompt text by dispatching to existing builder functions."""
    if stage == "questions":
        return _build_questions_prompt(template, task_description)
    elif stage == "litreview":
        # Read the prioritized question from the source project
        pq_path = run_dir / "questions" / "prioritized_question.txt"
        if pq_path.exists():
            prioritized_question = pq_path.read_text(encoding="utf-8").strip()
        else:
            prioritized_question = task_description or "(no prioritized question found)"
        return _build_literature_review_prompt(template, prioritized_question)
    elif stage == "viz":
        analysis_root = run_dir / "analysis"
        return _build_visualization_prompt(template, analysis_root)
    elif stage == "report":
        return _build_report_prompt(template)
    elif stage == "notebook":
        return _build_notebook_prompt(template, run_dir)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def run_stage(
    stage: str,
    run_dir: Path,
    prompt_text: str,
    provider: str,
    model: str,
    timeout: int,
    sandbox_bypass: bool = False,
) -> Dict[str, Any]:
    """Dispatch to the appropriate agent runner."""
    if stage == "questions":
        return run_questions_agent(
            provider=provider,
            project_dir=run_dir,
            prompt_text=prompt_text,
            timeout=timeout,
            model=model,
            sandbox_bypass=sandbox_bypass,
        )
    elif stage == "litreview":
        literature_dir = run_dir / "literature"
        return run_literature_review_agent(
            provider=provider,
            literature_dir=literature_dir,
            prompt_text=prompt_text,
            timeout=timeout,
            model=model,
            sandbox_bypass=sandbox_bypass,
        )
    elif stage == "viz":
        viz_dir = run_dir / "visualizations"
        return run_visualization_agent(
            provider=provider,
            viz_dir=viz_dir,
            prompt_text=prompt_text,
            timeout=timeout,
            model=model,
            sandbox_bypass=sandbox_bypass,
        )
    elif stage == "report":
        return run_report_agent(
            provider=provider,
            project_dir=run_dir,
            prompt_text=prompt_text,
            timeout=timeout,
            model=model,
            sandbox_bypass=sandbox_bypass,
        )
    elif stage == "notebook":
        return run_notebook_agent(
            provider=provider,
            project_dir=run_dir,
            prompt_text=prompt_text,
            timeout=timeout,
            model=model,
            sandbox_bypass=sandbox_bypass,
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")


def print_summary(stage: str, run_dir: Path, result: Dict[str, Any]) -> None:
    """Read outputs and print a summary."""
    print("\n" + "=" * 60)
    print(f"  RESULT: {'SUCCESS' if result['success'] else 'FAILED'} (exit code {result['returncode']})")
    print("=" * 60)

    if stage == "questions":
        outputs = read_questions_outputs(run_dir)
        if outputs["has_questions"]:
            print(f"\n  questions.txt: {len(outputs['questions_text'])} chars")
        if outputs["has_prioritized"]:
            print(f"  prioritized_question.txt: {len(outputs['prioritized_text'])} chars")
            if outputs["title"]:
                print(f"  Extracted title: {outputs['title']}")
        if not outputs["has_questions"] and not outputs["has_prioritized"]:
            print("\n  (no output files found)")

    elif stage == "litreview":
        literature_dir = run_dir / "literature"
        outputs = read_literature_review_outputs(literature_dir)
        if outputs["has_final_question"]:
            print(f"\n  Final_Research_Question.txt: {len(outputs['final_question_text'])} chars")
        if outputs["has_literature_review"]:
            print(f"  Literature_Review.md: {len(outputs['literature_review_text'])} chars")
        if outputs["paper_summaries"]:
            print(f"  Paper summaries: {len(outputs['paper_summaries'])}")
            for fname, _ in outputs["paper_summaries"]:
                print(f"    - {fname}")
        if outputs["downloaded_pdfs"]:
            print(f"  Downloaded PDFs: {len(outputs['downloaded_pdfs'])}")
        if not outputs["has_final_question"] and not outputs["has_literature_review"]:
            print("\n  (no output files found)")

    elif stage == "viz":
        viz_dir = run_dir / "visualizations"
        outputs = read_visualization_outputs(viz_dir)
        if outputs:
            print(f"\n  Generated {len(outputs)} figure(s):")
            for name, path in outputs.items():
                print(f"    - {Path(path).name}: {name}")
        else:
            print("\n  (no figures generated)")

    elif stage == "report":
        outputs = read_report_outputs(run_dir)
        if outputs["report_path"]:
            report_path = Path(outputs["report_path"])
            size = report_path.stat().st_size
            print(f"\n  Report: {report_path.name} ({size:,} bytes)")
        else:
            print("\n  (no report generated)")
        if outputs["all_files"]:
            print(f"  All files: {', '.join(Path(f).name for f in outputs['all_files'])}")

    elif stage == "notebook":
        outputs = read_notebook_outputs(run_dir)
        if outputs["notebook_path"]:
            nb_path = Path(outputs["notebook_path"])
            size = nb_path.stat().st_size
            print(f"\n  Notebook: {nb_path.name} ({size:,} bytes)")
        else:
            print("\n  (no notebook generated)")
        if outputs["all_files"]:
            print(f"  All notebooks: {', '.join(Path(f).name for f in outputs['all_files'])}")

    print(f"\n  Output dir: {run_dir}")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test AutoInterp agent prompts against completed project runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s viz --project linear_representation_limits_diagnostic_2026-03-12T18-14-54 --dry-run
  %(prog)s viz --project linear_representation_limits_diagnostic_2026-03-12T18-14-54 --label "test-v2"
  %(prog)s questions --project <name> --task-description "How do attention heads specialize?"
  %(prog)s report --project <name> --prompt my_report_v2.yaml
  %(prog)s notebook --project <name>
""",
    )
    parser.add_argument(
        "stage",
        choices=["questions", "litreview", "viz", "report", "notebook"],
        help="Which agent stage to run",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Completed project dir (absolute path or name under projects/)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="YAML or plain text prompt override file",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Label for this run (default: timestamp)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Override provider (default: from .last_llm.json)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model ID",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Override timeout in seconds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print assembled prompt without running the agent",
    )
    parser.add_argument(
        "--task-description",
        default="",
        help="Topic/description for questions stage",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    stage = args.stage
    stage_def = STAGE_DEFS[stage]

    # Resolve project directory
    source_project = resolve_project(args.project)
    print(f"Source project: {source_project}")

    # Load provider/model
    provider, model = load_provider_model(args)
    print(f"Provider: {provider}, Model: {model or '(default)'}")

    # Load prompt template
    template = load_prompt_template(stage, args.prompt)
    if args.prompt:
        print(f"Prompt override: {args.prompt}")

    # Resolve label and create test run directory
    label = resolve_label(stage, args.label)
    run_dir = create_test_run_dir(stage, label, source_project)
    print(f"Test run dir: {run_dir}")

    # Build the final prompt
    prompt_text = build_prompt(stage, template, run_dir, args.task_description)

    # Dry run: just print the prompt
    if args.dry_run:
        print("\n" + "=" * 60)
        print("  DRY RUN — assembled prompt:")
        print("=" * 60 + "\n")
        print(prompt_text)
        print("\n" + "=" * 60)
        print(f"  Prompt length: {len(prompt_text)} chars")
        print(f"  Symlinks in {run_dir}:")
        for child in sorted(run_dir.iterdir()):
            if child.is_symlink():
                print(f"    {child.name} -> {child.resolve()}")
            elif child.is_dir():
                # Show repo/ contents for notebook stage
                print(f"    {child.name}/ (real dir)")
                if stage == "notebook" and child.name == "repo":
                    for repo_child in sorted(child.iterdir()):
                        if repo_child.is_symlink():
                            print(f"      {repo_child.name} -> {repo_child.resolve()}")
                        elif repo_child.is_dir():
                            print(f"      {repo_child.name}/ (real dir)")
        print("=" * 60)
        return

    # Run the agent
    timeout = args.timeout or stage_def["default_timeout"]
    print(f"Timeout: {timeout}s")
    print()

    # Load codex sandbox_bypass from config.yaml
    _cfg_path = Path(__file__).resolve().parent / "config.yaml"
    _codex_sb = False
    if _cfg_path.exists():
        try:
            with open(_cfg_path) as _f:
                _cfg = yaml.safe_load(_f) or {}
            _codex_sb = _cfg.get("codex", {}).get("sandbox_bypass", False)
        except Exception:
            pass

    result = run_stage(stage, run_dir, prompt_text, provider, model, timeout, sandbox_bypass=_codex_sb)
    print_summary(stage, run_dir, result)


if __name__ == "__main__":
    main()
