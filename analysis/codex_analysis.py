"""
Codex-based analysis module for AutoInterp.
Replaces sequential LLM calls (planner → generator → executor → evaluator) with a single
Codex invocation per analysis. Codex designs the analysis, iterates on the script until
it runs successfully, and writes a structured interpretation.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.utils import PathResolver, get_timestamp


def build_codex_analysis_prompt(
    question: str,
    iteration_number: int,
    previous_interpretations: List[str],
    config: Dict[str, Any],
    path_resolver: PathResolver,
) -> str:
    """
    Build the prompt for Codex to perform a single analysis.

    Args:
        question: The research question (from prioritized_question.txt)
        iteration_number: 1-based analysis iteration number
        previous_interpretations: Raw evaluation text from prior Codex analyses
        config: Full configuration dictionary
        path_resolver: PathResolver for project paths

    Returns:
        The full prompt string for Codex
    """
    project_dir = path_resolver.get_project_dir().resolve()
    analysis_scripts_dir = path_resolver.ensure_path("analysis_scripts")
    evaluation_dir = path_resolver.ensure_path("evaluation_results")
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    model_info = config.get("model", {})
    model_name = model_info.get("name", "gpt2")
    model_path = model_info.get("path", model_name)

    analysis_dir = analysis_scripts_dir / f"analysis_{iteration_number}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    timestamp = get_timestamp().replace(" ", "_").replace(":", "-")
    script_path = analysis_dir / f"analysis_{timestamp}.py"
    stdout_path = analysis_dir / "stdout.txt"
    stderr_path = analysis_dir / "stderr.txt"
    eval_timestamp = get_timestamp().replace(" ", "_").replace(":", "-")
    interpretation_path = evaluation_dir / f"a{iteration_number}_evaluation_{eval_timestamp}.txt"

    prev_section = ""
    if previous_interpretations:
        prev_section = """
## Previous Analyses and Interpretations
The following interpretations from prior analyses inform your next step. Use them to design a follow-up or complementary analysis.
"""
        for i, interp in enumerate(previous_interpretations, 1):
            prev_section += f"\n### Analysis {i} Interpretation\n{interp}\n"

    return f"""You are in an AutoInterp project at {project_dir}. Your working directory is the project root. You can read and write files within the project directory.

## Task
Perform analysis #{iteration_number} to investigate the research question. You must:
1. Design an analysis that addresses the question (or a follow-up if prior analyses exist)
2. Write a Python script and run it until it executes successfully
3. Write a structured interpretation of the results

## Research Question
{question}

## Model Configuration
- Model name: {model_name}
- Model path: {model_path}
{prev_section}

## Paths (all relative to project root {project_dir})
- Script output: {script_path.relative_to(project_dir)}
- Stdout capture: {stdout_path.relative_to(project_dir)}
- Interpretation output: {interpretation_path.relative_to(project_dir)}
- Data directory (for saving intermediate files): {data_dir.relative_to(project_dir)}/

## Requirements

### 1. Write the analysis script
Create a Python script at: {script_path}

The script must:
- Use transformer_lens to load and analyze the model
- Print all results to stdout (the system captures printed output)
- Save any data files to {data_dir}/
- NOT perform visualizations (just print results)
- Include standard imports: numpy, torch, transformer_lens, pandas, etc.

### 2. Run the script and capture output
Run the script and save output to {stdout_path.relative_to(project_dir)} and stderr to {stderr_path.relative_to(project_dir)}:
```
cd {project_dir} && python {script_path.relative_to(project_dir)} > {stdout_path.relative_to(project_dir)} 2> {stderr_path.relative_to(project_dir)}
```

If the script fails (syntax error, runtime error, or suspicious results like all zeros):
- Read the error from stderr
- Edit and overwrite the SAME script at {script_path.relative_to(project_dir)} (do NOT create new attempt directories or new script files)
- Run again, overwriting stdout.txt and stderr.txt
- Repeat until the script completes successfully and produces valid, interpretable results

You must iterate on this single set of artifacts (script, stdout.txt, stderr.txt) in place. DO NOT create attempt_1, attempt_2, etc.
DO NOT stop until the script runs successfully.

### 3. Write the interpretation
Once the script has run successfully, write a structured interpretation to: {interpretation_path}

The file MUST start with "RAW EVALUATION:" and include these sections in this exact format:

RAW EVALUATION:
## SUMMARY OF PREVIOUS ANALYSES
A detailed description of all successful analyses conducted so far (including this one). For analysis 1, describe only this analysis.

## DATA GENERATED
A detailed description of any data files generated and saved. Include paths and how to load them. If none: "NO DATA GENERATED".

## WHAT WE HAVE LEARNED
A detailed description of what we learned from the results. What do the outputs tell us about the research question?

## CURRENT CONFIDENCE
NEW_CONFIDENCE: [value]

Where [value] is a number between 0 and 1 indicating confidence that we have answered the research question. Base this on the strength of evidence from this and prior analyses.

## NEXT ANALYSIS
One specific follow-up analysis to run next (or "NONE" if the question is sufficiently answered). Should be executable as a single Python script.

## Deliverables
1. A working Python script at {script_path}
2. Captured stdout at {stdout_path}
3. A complete interpretation file at {interpretation_path}

You must not exit until all three deliverables exist and the script has run successfully.
"""


def parse_codex_analysis_output(
    project_dir: Path,
    iteration_number: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Parse the output from a Codex analysis run into analysis_result and evaluation_result dicts.

    Args:
        project_dir: Project directory path
        iteration_number: The analysis iteration number (1-based)

    Returns:
        Tuple of (analysis_result, evaluation_result) in the same format as the
        current analyze_question + evaluate_analysis flow.
    """
    analysis_scripts_dir = project_dir / "analysis_scripts"
    evaluation_dir = project_dir / "evaluation_results"

    analysis_dir = analysis_scripts_dir / f"analysis_{iteration_number}"
    if not analysis_dir.exists():
        return (
            {"success": False, "error": f"Analysis directory not found: {analysis_dir}"},
            {"success": False, "raw_evaluation": "", "new_confidence": 0.0},
        )

    # Find the script and stdout directly in analysis_dir (one set of artifacts per analysis)
    scripts = list(analysis_dir.glob("analysis_*.py"))
    script_path = max(scripts, key=lambda p: p.stat().st_mtime) if scripts else None
    stdout_file = analysis_dir / "stdout.txt"
    stdout_content = stdout_file.read_text(encoding="utf-8", errors="replace") if stdout_file.exists() else ""

    if script_path is None:
        return (
            {"success": False, "error": f"No analysis script found in {analysis_dir}"},
            {"success": False, "raw_evaluation": "", "new_confidence": 0.0},
        )

    # Find the interpretation file
    eval_pattern = f"a{iteration_number}_evaluation_*.txt"
    eval_files = list(evaluation_dir.glob(eval_pattern))
    interpretation_content = ""
    if eval_files:
        latest_eval = max(eval_files, key=lambda p: p.stat().st_mtime)
        interpretation_content = latest_eval.read_text(encoding="utf-8", errors="replace")

    # Parse NEW_CONFIDENCE from interpretation
    new_confidence = 0.5
    if interpretation_content:
        confidence_match = re.search(
            r"NEW_CONFIDENCE:\s*([-+]?\d*\.?\d+)", interpretation_content
        )
        if confidence_match:
            try:
                new_confidence = float(confidence_match.group(1))
                new_confidence = max(0.0, min(1.0, new_confidence))
            except ValueError:
                pass

    analysis_result = {
        "success": True,
        "stdout": stdout_content,
        "script_path": str(script_path),
        "results": {"stdout": stdout_content},
        "output": stdout_content,
    }

    evaluation_result = {
        "success": True,
        "raw_evaluation": interpretation_content,
        "new_confidence": new_confidence,
    }

    return analysis_result, evaluation_result
