#!/usr/bin/env python3
"""
Regenerate the Jupyter notebook for an existing AutoInterp project without rerunning
the full pipeline. Reconstructs analysis results, evaluations, and visualizations
from disk and calls the notebook generator.

Usage (from the AutoInterp directory):
    python scripts/regenerate_notebook.py --project "cross_lingual_causal_tracing_factual_recall_2026-03-04T15-07-08"
    python scripts/regenerate_notebook.py --project /path/to/project

The project can be:
  - Project ID (uses projects/ directory from config)
  - Full path to project directory
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

# Load .env (OPENROUTER_API_KEY, HF_TOKEN, etc.) before any AutoInterp imports
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

# Ensure package imports resolve when run as script (python scripts/regenerate_notebook.py)
# Script lives at AutoInterp/scripts/regenerate_notebook.py; imports need parent of AutoInterp in path
_script_dir = Path(__file__).resolve().parent
_pkg_root = _script_dir.parent  # AutoInterp directory
_repo_root = _pkg_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from AutoInterp.core.utils import load_yaml, PathResolver
from AutoInterp.reporting.report_generator import ReportGenerator


def _resolve_project_path(project_arg: str, config: dict) -> tuple[Path, str, Path]:
    """
    Resolve project path and project_id from the --project argument.
    Returns (project_dir, project_id, projects_base) so config can be updated
    when an absolute path is used (projects_base may differ from default).
    """
    default_base = Path(config.get("paths", {}).get("projects", "projects")).expanduser()
    if not default_base.is_absolute():
        pkg_root = Path(__file__).resolve().parent.parent
        default_base = (pkg_root / default_base).resolve()

    arg_path = Path(project_arg).expanduser()
    if arg_path.is_absolute() and arg_path.exists():
        return arg_path, arg_path.name, arg_path.parent
    if arg_path.is_absolute():
        raise FileNotFoundError(f"Project path does not exist: {arg_path}")

    project_dir = default_base / project_arg
    if not project_dir.exists():
        raise FileNotFoundError(f"Project not found: {project_dir}")
    return project_dir, project_arg, default_base


def _reconstruct_analyses_from_disk(project_dir: Path) -> list[dict]:
    """Reconstruct all_analyses list from analysis_scripts directory."""
    analyses = []
    scripts_dir = project_dir / "analysis_scripts"
    if not scripts_dir.exists():
        return analyses

    analysis_dirs = []
    for item in scripts_dir.iterdir():
        if item.is_dir() and item.name.startswith("analysis_"):
            try:
                n = int(item.name.split("_")[1])
                analysis_dirs.append((n, item))
            except (ValueError, IndexError):
                continue

    analysis_dirs.sort(key=lambda x: x[0])

    for analysis_num, analysis_dir in analysis_dirs:
        attempt_dirs = []
        for item in analysis_dir.iterdir():
            if item.is_dir() and item.name.startswith("attempt_"):
                try:
                    attempt_num = int(item.name.split("_")[1])
                    attempt_dirs.append((attempt_num, item))
                except (ValueError, IndexError):
                    continue
        if attempt_dirs:
            attempt_dirs.sort(key=lambda x: x[0])
            script_search_dir = attempt_dirs[-1][1]
        else:
            script_search_dir = analysis_dir  # Flat structure (Codex)

        script_path = None
        for f in script_search_dir.iterdir():
            if f.is_file() and f.name.startswith("analysis_") and f.suffix == ".py":
                script_path = f
                break
        if not script_path:
            continue

        stdout_path = script_search_dir / "stdout.txt"
        stderr_path = script_search_dir / "stderr.txt"
        stdout_content = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
        stderr_content = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""

        # Infer success from stderr: Traceback or "Error:" suggests execution failed
        inferred_success = True
        if stderr_content:
            if "Traceback (most recent call last)" in stderr_content or "Error:" in stderr_content:
                inferred_success = False

        analyses.append({
            "success": inferred_success,
            "script_path": str(script_path),
            "results": {"stdout": stdout_content, "stderr": stderr_content},
        })

    return analyses


def _reconstruct_evaluations_from_disk(project_dir: Path) -> list[dict]:
    """Reconstruct all_evaluations list from evaluation_results directory."""
    evaluations = []
    eval_dir = project_dir / "evaluation_results"
    if not eval_dir.exists():
        return evaluations

    eval_files = sorted(eval_dir.glob("*.txt"))
    for f in eval_files:
        try:
            content = f.read_text(encoding="utf-8")
            new_confidence = 0.0
            m = re.search(r"NEW_CONFIDENCE:\s*([\d.]+)", content, re.IGNORECASE)
            if m:
                new_confidence = float(m.group(1))
            evaluations.append({
                "raw_evaluation": content,
                "new_confidence": new_confidence,
            })
        except Exception:
            continue

    return evaluations


def _load_question(project_dir: Path) -> str:
    """Load the prioritized question from disk."""
    prioritized = project_dir / "questions" / "prioritized_question.txt"
    if prioritized.exists():
        return prioritized.read_text(encoding="utf-8").strip()
    return "No question available"


def _build_visualizations_dict(project_dir: Path) -> dict[str, str]:
    """
    Build visualizations dict (analysis_name -> path) from visualizations directory.
    The notebook generator only needs the keys to know which analyses have viz.
    """
    viz_dir = project_dir / "visualizations"
    if not viz_dir.exists():
        return {}

    result = {}
    for f in viz_dir.glob("visualization_analysis_*.py"):
        m = re.match(r"visualization_(analysis_\d+)_", f.name)
        if m:
            result[m.group(1)] = str(f)
    return result


async def regenerate_notebook(
    project_path: str,
    config_path: str | None = None,
) -> str | None:
    """
    Regenerate the Jupyter notebook for an existing project.

    Returns:
        Path to the generated notebook, or None on failure.
    """
    # Load config
    config = load_yaml(_pkg_root / "config.yaml")
    if config_path:
        override = load_yaml(Path(config_path))
        for k, v in override.items():
            if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                config[k] = {**config[k], **v}
            else:
                config[k] = v

    project_dir, project_id, projects_base = _resolve_project_path(project_path, config)

    # Load prompts
    from AutoInterp.core.utils import load_prompts
    config["prompts"] = load_prompts(_pkg_root / "prompts")

    # Override paths so PathResolver points to this project
    config["project_id"] = project_id
    if "paths" not in config:
        config["paths"] = {}
    config["paths"]["projects"] = str(projects_base)

    # Use OpenRouter when OPENROUTER_API_KEY is set (config default is Anthropic)
    import os
    if os.environ.get("OPENROUTER_API_KEY"):
        if "agents" not in config:
            config["agents"] = {}
        if "reporter" not in config["agents"]:
            config["agents"]["reporter"] = {}
        if "llm" not in config["agents"]["reporter"]:
            config["agents"]["reporter"]["llm"] = {}
        config["agents"]["reporter"]["llm"]["provider"] = "openrouter"
        config["agents"]["reporter"]["llm"]["model"] = "anthropic/claude-opus-4.5"
    else:
        # Anthropic: use Claude Opus 4.5 for reporter
        if "agents" not in config:
            config["agents"] = {}
        if "reporter" not in config["agents"]:
            config["agents"]["reporter"] = {}
        if "llm" not in config["agents"]["reporter"]:
            config["agents"]["reporter"]["llm"] = {}
        config["agents"]["reporter"]["llm"]["model"] = "claude-opus-4-5-20251101"

    # Initialize minimal framework (PathResolver, LLM, ReportGenerator)
    path_resolver = PathResolver(config)

    from AutoInterp.core.llm_interface import LLMInterface
    llm_interface = LLMInterface(config, agent_name="reporter")

    report_generator = ReportGenerator(config=config, llm_interface=llm_interface)

    # Reconstruct data from disk
    all_analyses = _reconstruct_analyses_from_disk(project_dir)
    all_evaluations = _reconstruct_evaluations_from_disk(project_dir)
    question = _load_question(project_dir)
    visualizations = _build_visualizations_dict(project_dir)

    if not all_analyses:
        print("[AUTOINTERP] No analyses found. Ensure analysis_scripts/ contains completed runs.")
        return None

    # Build combined structures
    final_confidence = all_evaluations[-1].get("new_confidence", 0.0) if all_evaluations else 0.0
    combined_analysis_result = {
        "analyses": all_analyses,
        "latest_analysis": all_analyses[-1],
        "analysis_count": len(all_analyses),
        "final_confidence": final_confidence,
    }
    combined_evaluation_result = {
        "evaluations": all_evaluations,
        "latest_evaluation": all_evaluations[-1] if all_evaluations else {},
        "conclusion": "concluded" if final_confidence >= 0.8 else "inconclusive",
        "final_confidence": final_confidence,
        "reasoning": all_evaluations[-1].get("raw_evaluation", "")[:2000] if all_evaluations else "",
    }

    # Determine output path (same naming as full pipeline)
    reports_dir = project_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    # Use existing notebook path to overwrite, or derive from markdown report, else project_id
    existing_notebooks = list(reports_dir.glob("*_notebook.ipynb"))
    if existing_notebooks:
        notebook_path = existing_notebooks[0]
    else:
        md_reports = list(reports_dir.glob("*.md"))
        stem = md_reports[0].stem if md_reports else project_id.replace("_20", "_")[:60]
        notebook_path = reports_dir / f"{stem}_notebook.ipynb"

    print(f"[AUTOINTERP] Regenerating notebook at {notebook_path}...")
    print(f"[AUTOINTERP] Project: {project_id}")
    print(f"[AUTOINTERP] Analyses: {len(all_analyses)}, Evaluations: {len(all_evaluations)}")

    await report_generator.generate_jupyter_notebook(
        question=question,
        analysis_results=combined_analysis_result,
        evaluation_results=combined_evaluation_result,
        visualizations=visualizations,
        task_config=config,
        output_path=notebook_path,
    )

    print(f"[AUTOINTERP] Notebook saved: {notebook_path}")
    return str(notebook_path)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate Jupyter notebook for an existing AutoInterp project"
    )
    parser.add_argument(
        "--project", "-p",
        required=True,
        help="Project path or ID (e.g. /path/to/project or my_study_2026-03-04T12-00-00)",
    )
    parser.add_argument(
        "--config",
        help="Optional path to config override",
        default=None,
    )
    args = parser.parse_args()

    try:
        result = asyncio.run(regenerate_notebook(args.project, args.config))
        sys.exit(0 if result else 1)
    except FileNotFoundError as e:
        print(f"[AUTOINTERP] Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"[AUTOINTERP] Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
