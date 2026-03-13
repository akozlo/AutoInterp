#!/usr/bin/env python3
"""
Main entry point for the AutoInterp Agent Framework.
Implements a streamlined pipeline for automated interpretability research.
"""

import os
import re
import sys
import time
import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Ensure package imports resolve when running this file directly
if __package__ is None or __package__ == "":
    pkg_root = Path(__file__).resolve().parent.parent
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

from AutoInterp.core.utils import setup_logging, load_yaml, ensure_directory, get_timestamp, load_prompts, PathResolver, log_to_comprehensive_log, clean_code_content, PACKAGE_ROOT
from AutoInterp.core.llm_interface import LLMInterface
from AutoInterp.questions.question_manager import QuestionManager
from AutoInterp.analysis.analysis_generator import AnalysisGenerator
from AutoInterp.analysis.analysis_executor import AnalysisExecutor
from AutoInterp.analysis.analysis_planner import AnalysisPlanner
from AutoInterp.analysis.evaluator import Evaluator
from AutoInterp.analysis.visualization_evaluator import VisualizationEvaluator
from AutoInterp.analysis.agent_analysis import (
    setup_analysis_workspace,
    load_analysis_prompt_template,
    _build_analysis_prompt,
    run_analysis_agent,
    read_agent_outputs,
    read_confidence,
)
from AutoInterp.reporting.report_generator import ReportGenerator
from AutoInterp.reporting.agent_report import (
    load_report_prompt_template,
    _build_report_prompt,
    run_report_agent,
    read_report_outputs,
)
from AutoInterp.core.interactive import (
    interactive_checkpoint,
    make_revision_call,
    is_interactive,
)
from AutoInterp.autocritique.agent_autocritique import (
    load_autocritique_prompt_template,
    _build_autocritique_prompt,
    run_autocritique_agent,
    read_autocritique_outputs,
)

def select_provider_and_model() -> Tuple[str, str]:
    """
    Prompt user to select LLM provider and model for the current run.

    Returns:
        Tuple of (provider, model_id)
    """
    # Model mappings for each provider
    model_mappings = {
        "anthropic": {
            "Claude Sonnet 4.6": "claude-sonnet-4-6",
            "Claude Opus 4.6": "claude-opus-4-6",
            "Claude Sonnet 4.5": "claude-sonnet-4-5",
        },
        "openai": {
            "GPT-5.4": "gpt-5.4",
            "GPT-5": "gpt-5-2025-08-07",
            "GPT-5-mini": "gpt-5-mini-2025-08-07"
        },
        "openrouter": {
            "Claude Sonnet 4.6": "anthropic/claude-sonnet-4.6",
            "Claude Opus 4.6": "anthropic/claude-opus-4.6",
            "GPT-5.4": "openai/gpt-5.4",
            "Claude Sonnet 4.5": "anthropic/claude-sonnet-4.5",
            "GPT-5": "openai/gpt-5",
            "GPT-5-mini": "openai/gpt-5-mini",
            "Kimi K2": "moonshotai/kimi-k2-0905",
            "Qwen3 235B-A22B": "qwen/qwen3-235b-a22b",
            "DeepSeek V3.2": "deepseek/deepseek-v3.2"
        }
    }

    print("\n" + "="*50)
    print("Select model provider:")
    print("="*50)

    # Provider options - show all options regardless of API keys
    provider_options = ["anthropic", "openai", "openrouter", "manual"]
    print("[1] Anthropic")
    print("[2] OpenAI")
    print("[3] OpenRouter")
    print("[4] Manual Configuration (use config.yaml)")

    # Get provider selection
    while True:
        try:
            choice = input(f"\nSelect provider [1-{len(provider_options)}]: ").strip()
            provider_idx = int(choice) - 1
            if 0 <= provider_idx < len(provider_options):
                selected_provider = provider_options[provider_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(provider_options)}")
        except (ValueError, KeyboardInterrupt):
            print(f"Please enter a valid number between 1 and {len(provider_options)}")

    # If manual configuration, return None to skip overrides
    if selected_provider == "manual":
        print("Using manual configuration from config.yaml")
        return "manual", ""

    print(f"\nSelected provider: {selected_provider.upper()}")

    # Model selection
    print(f"\nSelect default model:")
    available_models = list(model_mappings[selected_provider].keys())

    for i, model_name in enumerate(available_models, 1):
        print(f"[{i}] {model_name}")

    # Get model selection
    while True:
        try:
            choice = input(f"\nSelect model [1-{len(available_models)}]: ").strip()
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(available_models):
                selected_model_name = available_models[model_idx]
                selected_model_id = model_mappings[selected_provider][selected_model_name]
                break
            else:
                print(f"Please enter a number between 1 and {len(available_models)}")
        except (ValueError, KeyboardInterrupt):
            print(f"Please enter a valid number between 1 and {len(available_models)}")

    print(f"Selected model: {selected_model_name}")
    print("="*50)

    return selected_provider, selected_model_id

def apply_provider_model_override(config: Dict[str, Any], provider: str, model_id: str) -> Dict[str, Any]:
    """
    Apply provider and model selection to all agents in the config.

    Args:
        config: Configuration dictionary to modify
        provider: Selected provider (anthropic, openai, openrouter)
        model_id: Selected model ID

    Returns:
        Modified configuration dictionary
    """
    if provider == "manual":
        return config

    # Update all agent configurations
    for agent_name, agent_config in config.get("agents", {}).items():
        if "llm" in agent_config:
            agent_config["llm"]["provider"] = provider
            agent_config["llm"]["model"] = model_id

    # Also update the default LLM config if it exists
    if "llm" in config:
        config["llm"]["provider"] = provider
        config["llm"]["model"] = model_id

    return config


# ---------------------------------------------------------------------------
# Options menu — interactive settings override at startup
# ---------------------------------------------------------------------------

OPTIONS_SETTINGS = [
    {"key": "analysis.max_iterations",       "label": "Max analysis iterations",       "type": "int",   "help": "Maximum analysis cycles per question"},
    {"key": "analysis.confidence_threshold",  "label": "Confidence threshold",           "type": "float", "help": "Stop analysis above this confidence (0-100%)", "display_pct": True},
    {"key": "analysis.use_agent",            "label": "Use CLI agent for analysis",     "type": "bool",  "help": "true = CLI agent, false = legacy pipeline"},
    {"key": "reporting.use_agent",           "label": "Use CLI agent for report",       "type": "bool",  "help": "true = CLI agent, false = legacy pipeline"},
    {"key": "context_pack.enabled",          "label": "Context pack (literature sampling)", "type": "bool", "help": "Sample papers and build literature context"},
    {"key": "visualization.default_format",  "label": "Visualization format",           "type": "str",   "choices": ["png", "svg", "pdf"]},
    {"key": "visualization.dpi",             "label": "Visualization DPI",              "type": "int",   "help": "Dots per inch for raster output"},
    {"key": "ui.html_dashboard",             "label": "HTML dashboard",                 "type": "bool",  "help": "Generate auto-refreshing HTML dashboard"},
    {"key": "ui.auto_open_browser",          "label": "Auto-open browser",              "type": "bool",  "help": "Open dashboard in browser on pipeline start"},
    {"key": "interactive_mode",              "label": "Interactive mode (feedback loops)", "type": "bool", "help": "Pause after each stage for user review and revision"},
    {"key": "autocritique.enabled",          "label": "AutoCritique (peer review)",         "type": "bool", "help": "Run automated peer review after report generation"},
]


def _get_config_value(config: Dict[str, Any], dotted_key: str) -> Any:
    """Read a nested config value using a dotted key like 'analysis.max_iterations'."""
    parts = dotted_key.split(".")
    node = config
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _set_config_value(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Write a nested config value using a dotted key."""
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_user_options(config: Dict[str, Any]) -> None:
    """Load .user_options.json and apply saved overrides to *config* in-place."""
    options_path = Path(__file__).parent / ".user_options.json"
    if not options_path.exists():
        return
    try:
        with open(options_path, "r") as f:
            saved = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    valid_keys = {s["key"] for s in OPTIONS_SETTINGS}
    for key, value in saved.items():
        if key in valid_keys:
            _set_config_value(config, key, value)


def save_user_options(changed: Dict[str, Any]) -> None:
    """Merge *changed* settings into .user_options.json."""
    options_path = Path(__file__).parent / ".user_options.json"
    existing: Dict[str, Any] = {}
    if options_path.exists():
        try:
            with open(options_path, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    existing.update(changed)
    with open(options_path, "w") as f:
        json.dump(existing, f, indent=2)


def _format_value(setting: Dict[str, Any], value: Any) -> str:
    """Format a config value for display."""
    if setting.get("display_pct") and isinstance(value, (int, float)):
        return f"{int(value * 100)}%"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_input(setting: Dict[str, Any], raw: str) -> Any:
    """Parse and validate user input for a setting. Returns the parsed value or raises ValueError."""
    stype = setting["type"]
    raw = raw.strip()
    if stype == "int":
        v = int(raw)
        if v <= 0:
            raise ValueError("Must be a positive integer")
        return v
    elif stype == "float":
        if setting.get("display_pct"):
            v = float(raw.rstrip("%"))
            if not (0 < v <= 100):
                raise ValueError("Must be between 1 and 100")
            return v / 100.0
        v = float(raw)
        if v <= 0:
            raise ValueError("Must be positive")
        return v
    elif stype == "bool":
        if raw.lower() in ("true", "t", "yes", "y", "1"):
            return True
        if raw.lower() in ("false", "f", "no", "n", "0"):
            return False
        raise ValueError("Enter true or false")
    elif stype == "str":
        choices = setting.get("choices")
        if choices and raw.lower() not in choices:
            raise ValueError(f"Choose from: {', '.join(choices)}")
        return raw.lower()
    return raw


def show_options_menu(config: Dict[str, Any]) -> None:
    """Interactive options menu. Modifies *config* in-place and optionally persists."""
    changed: Dict[str, Any] = {}

    while True:
        print("\n" + "=" * 50)
        print("Options")
        print("=" * 50)
        for i, s in enumerate(OPTIONS_SETTINGS, 1):
            val = _get_config_value(config, s["key"])
            display = _format_value(s, val)
            dots = "." * (40 - len(s["label"]))
            print(f"[{i}] {s['label']} {dots} {display}")

        choice = input("\nEnter number to edit, or press Enter to finish: ").strip()
        if not choice:
            break
        try:
            idx = int(choice) - 1
            if not (0 <= idx < len(OPTIONS_SETTINGS)):
                print("Invalid number.")
                continue
        except ValueError:
            print("Invalid input.")
            continue

        setting = OPTIONS_SETTINGS[idx]
        current = _get_config_value(config, setting["key"])
        display = _format_value(setting, current)
        hint = ""
        if setting.get("choices"):
            hint = f" ({'/'.join(setting['choices'])})"
        elif setting["type"] == "bool":
            hint = " (true/false)"
        elif setting.get("display_pct"):
            hint = " (%)"

        raw = input(f"{setting['label']}{hint} [{display}]: ").strip()
        if not raw:
            continue

        try:
            new_val = _parse_input(setting, raw)
        except ValueError as e:
            print(f"  Invalid: {e}")
            continue

        _set_config_value(config, setting["key"], new_val)
        changed[setting["key"]] = new_val

    if not changed:
        return

    print("\nSave changes:")
    print("[1] Just this time")
    print("[2] Make default (save to .user_options.json)")
    save_choice = input("Choice [1]: ").strip()
    if save_choice == "2":
        save_user_options(changed)
        print("Options saved.")
    else:
        print("Options applied for this run.")


async def initialize_framework(
    config_path: Optional[str] = None,
    venv_path: Optional[str] = None,
    projects_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Initialize the framework with the given configuration.
    
    Args:
        config_path: Optional path to an override configuration file
        venv_path: Optional path to existing virtual environment to use
        projects_dir: Optional root directory for generated projects
        
    Returns:
        Dictionary with initialized components
    """
    package_root = Path(__file__).resolve().parent

    # Load global configuration
    config = load_yaml(Path(__file__).parent / "config.yaml")
    
    # Load override configuration if provided
    if config_path:
        override_config = load_yaml(config_path)
        # Merge configurations (override config takes precedence)
        for key, value in override_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                # Deep merge for dictionary values
                config[key] = {**config[key], **value}
            else:
                # Simple override for other values
                config[key] = value

    # Resolve the projects directory, defaulting to the package projects folder
    paths_config = config.setdefault("paths", {})
    configured_projects = projects_dir or paths_config.get("projects", "projects")
    resolved_projects_path = Path(configured_projects).expanduser()
    if not resolved_projects_path.is_absolute():
        resolved_projects_path = (package_root / resolved_projects_path).resolve()
    paths_config["projects"] = str(resolved_projects_path)
    
    # Handle virtual environment path override
    if venv_path:
        # Enable existing venv usage and set the path
        if "execution" not in config:
            config["execution"] = {}
        config["execution"]["use_existing_venv"] = True
        config["execution"]["existing_venv_path"] = venv_path
        # Disable clean venv when using existing venv
        config["execution"]["force_clean_venv"] = False
        print(f"[AUTOINTERP] Using existing virtual environment: {venv_path}")
    
    # Validate execution settings for conflicts
    if config.get("execution", {}).get("use_existing_venv", False):
        # When using existing venv, force_clean_venv should be disabled
        if config.get("execution", {}).get("force_clean_venv", False):
            config["execution"]["force_clean_venv"] = False
            print(f"[AUTOINTERP] Disabled force_clean_venv since using existing virtual environment")
                
    # Validate required configuration sections and values
    required_sections = {
        "framework": ["version", "log_level"],
        "paths": ["projects"],
        "task": ["description"],
        "analysis": ["max_iterations", "confidence_threshold"],
        "execution": ["max_retries"],
        "llm": ["provider", "model", "temperature"]
    }

    # Required component configurations
    required_component_configs = {
        "paths": {
            "required_fields": ["projects", "data", "models"],
            "required_settings": ["create_missing", "cleanup_old"]
        },
        "llm": {
            "required_fields": ["provider", "model"],
            "required_settings": ["temperature", "max_tokens", "timeout"]
        },
        "analysis": {
            "required_fields": ["executor_type", "output_format"],
            "required_settings": ["max_iterations", "timeout_per_analysis"]
        }
    }

    missing_configs = []
    
    # First validate basic required sections and fields
    for section, fields in required_sections.items():
        if section not in config:
            missing_configs.append(f"Missing required configuration section: {section}")
            continue
        
        for field in fields:
            if field not in config[section]:
                missing_configs.append(f"Missing required configuration field: {section}.{field}")
            elif config[section][field] is None:
                missing_configs.append(f"Configuration field cannot be null: {section}.{field}")

    # Then validate component-specific configurations
    for component, requirements in required_component_configs.items():
        if component not in config:
            missing_configs.append(f"Missing required component configuration: {component}")
            continue

        component_config = config[component]
        
        # Check required fields
        for field in requirements["required_fields"]:
            if field not in component_config:
                missing_configs.append(f"Missing required field in {component} configuration: {field}")
            elif component_config[field] is None:
                missing_configs.append(f"Field in {component} configuration cannot be null: {field}")

        # Check required settings
        for setting in requirements["required_settings"]:
            if setting not in component_config:
                missing_configs.append(f"Missing required setting in {component} configuration: {setting}")
            elif component_config[setting] is None:
                missing_configs.append(f"Setting in {component} configuration cannot be null: {setting}")

    if missing_configs:
        error_msg = "Configuration validation failed:\n" + "\n".join(missing_configs)
        print(f"[AUTOINTERP] ERROR: {error_msg}")
        
        # Print the current configuration for debugging
        print(f"[AUTOINTERP] Current configuration:")
        for section, values in config.items():
            print(f"[AUTOINTERP] {section}: {values}")
            
        raise ValueError(error_msg)
                
    # Load prompt configuration 
    try:
        prompts_dir = Path(__file__).parent / "prompts"
        prompts = load_prompts(prompts_dir)
        # Add prompts to config
        config["prompts"] = prompts
    except Exception as e:
        import traceback
        full_traceback = traceback.format_exc()
        print(f"ERROR: Failed to load prompts from {prompts_dir}")
        print(f"Reason: {e}")
        print(f"Full traceback:\n{full_traceback}")
        print("Prompts are required for operation. Exiting...")
        sys.exit(1)
    
    # STEP 1: Always use a fixed working project ID at startup. Name will be updated after question selection
    project_id = f"working_project_{get_timestamp('%Y%m%dT%H%M%S')}"
    
    # Set the project_id in config so it's immediately available everywhere
    config["project_id"] = project_id
    
    # Set up logging with log file in logs directory
    log_level = config["framework"]["log_level"]
    log_file_name = config["framework"].get("log_file")  # This one can be optional
    
    # Initialize logger - only show warnings and errors in console, but log everything to file
    logger = setup_logging(log_level=log_level, console_level="WARNING")
    
    # Initialize the central path resolver with the config
    path_resolver = PathResolver(config)
    logger.info(f"Initialized path resolver with project_id: {project_id}")
    
    # Setup project directories using the path resolver
    path_resolver.ensure_path("")  # Create project root directory
    path_resolver.ensure_path("reports")
    path_resolver.ensure_path("evaluation_results")
    path_resolver.ensure_path("questions")
    path_resolver.ensure_path("logs")
    path_resolver.ensure_path("data")
    
    # Get paths for components to use
    project_dir = path_resolver.get_project_dir()
    
    
    # Now update the logger with the log file path if needed
    if log_file_name:
        logs_dir = path_resolver.get_path("logs")
        log_file_path = logs_dir / log_file_name
        logger = setup_logging(log_level, str(log_file_path))
        logger.info(f"Log file path set to: {log_file_path}")
    
    logger.info(f"Initializing AutoInterp Agent Framework v{config['framework']['version']}")
    
    # Initialize LLM interface with validated config
    llm_interface = LLMInterface(config, agent_name="question_generator")  # Use question_generator as the default agent
    logger.info(f"LLM interface initialized with provider: {config['agents']['question_generator']['llm']['provider']} and model: {config['agents']['question_generator']['llm']['model']}")
    
    # Initialize question manager
    question_manager = QuestionManager(
        llm_interface=llm_interface,
        config=config
    )
    logger.info("Question manager initialized")
    
    # Initialize analysis components with validated config
    analysis_generator = AnalysisGenerator(
        llm_interface=llm_interface,
        config=config
    )
    logger.info("Analysis generator initialized")
    
    analysis_executor = AnalysisExecutor(config=config)
    logger.info("Analysis executor initialized")
    
    analysis_planner = AnalysisPlanner(
        llm_interface=llm_interface,
        path_resolver=path_resolver
    )
    logger.info("Analysis planner initialized")
    
    evaluator = Evaluator(
        question_manager=question_manager,
        llm_interface=llm_interface,
        config=config
    )
    logger.info("Evaluator initialized")

    visualization_evaluator = VisualizationEvaluator(
        llm_interface=llm_interface,
        config=config
    )
    logger.info("Visualization Evaluator initialized")
    
    # Initialize reporting components
    
    report_generator = ReportGenerator(config=config, llm_interface=llm_interface)
    logger.info("Report generator initialized")
    
    # Return framework components including path resolver
    return {
        "config": config,
        "logger": logger,
        "path_resolver": path_resolver,
        "llm_interface": llm_interface,
        "question_manager": question_manager,
        "analysis_generator": analysis_generator,
        "analysis_executor": analysis_executor,
        "analysis_planner": analysis_planner,
        "evaluator": evaluator,
        "visualization_evaluator": visualization_evaluator,
        "report_generator": report_generator
    }

async def generate_questions(
    llm_interface: LLMInterface,
    question_manager: QuestionManager,
    config: Dict[str, Any],
    logger: Any
) -> List[Dict[str, Any]]:
    """
    Generate initial questions using the question_generator agent.
    
    Args:
        llm_interface: LLM interface for interacting with language models
        question_manager: Manager for question tracking
        config: Configuration dictionary
        logger: Logging instance
        
    Returns:
        List of generated questions
    """
    logger.info("Generating questions...")
    print("[AUTOINTERP] PHASE 1/4: Question Generation")
    
    # Get task details for questions generation
    task_config = config.get("task")
    if not task_config or "description" not in task_config:
        error_msg = "Missing required task configuration. Task 'description' must be specified in the task config."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Generate a task name from the description for logging
    task_description = task_config["description"]
    task_name = task_description[:50] + "..." if len(task_description) > 50 else task_description
    task_description = task_config["description"]
    
    if not task_description.strip():
        error_msg = "Task description cannot be empty. A detailed description is required for question generation."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Log task starting
    logger.info(f"Starting task: {task_name}")
    logger.info(f"Task description: {task_description}")
    
    # Generate questions using the question_manager
    # Now it just saves to a text file and returns empty list
    await question_manager.generate_questions(
        task_description=task_description,
        count=3
    )
    
    # Read the raw questions from file and print them directly
    raw_questions_path = question_manager.storage_dir / "questions.txt"
    if raw_questions_path.exists():
        try:
            with open(raw_questions_path, 'r') as f:
                raw_questions = f.read()
            print("\n============= QUESTION GENERATOR OUTPUT =============")
            print(raw_questions)
            print("======================================================\n")
        except Exception as e:
            print(f"[AUTOINTERP] Error reading raw questions: {e}")
    
    # Interactive checkpoint: let user revise questions before proceeding
    if raw_questions_path.exists() and is_interactive(config):
        try:
            with open(raw_questions_path, 'r') as f:
                current_questions = f.read()

            async def _revise_questions(current: str, feedback: str) -> str:
                return await make_revision_call(
                    llm_interface, "question_generator", current, feedback, "question_generation"
                )

            def _save_questions(text: str) -> None:
                with open(raw_questions_path, 'w') as f:
                    f.write(text)

            await interactive_checkpoint(
                "Question Generation", current_questions, _revise_questions, _save_questions, config
            )
        except Exception as e:
            logger.warning(f"Interactive checkpoint failed for question generation: {e}")

    # For logging - we don't have structured questions anymore, just carry on
    logger.info("Generated initial questions - see raw text output in console")

    # Return empty list since we now use raw text files instead of structured questions
    return []

async def prioritize_questions(
    question_manager: QuestionManager,
    llm_interface: LLMInterface,
    config: Dict[str, Any],
    logger: Any,
    evaluator: Optional[Evaluator] = None
) -> Dict[str, Any]:
    """
    Prioritize questions using the question_prioritizer agent.
    
    Args:
        question_manager: Manager for question tracking
        llm_interface: LLM interface for interacting with language models
        config: Configuration dictionary
        logger: Logging instance
        evaluator: Optional evaluator instance for updating after project rename
        
    Returns:
        Selected question to investigate
    """
    logger.info("Prioritizing questions...")
    print("[AUTOINTERP] PHASE 2/4: Question Prioritization")
    
    # Get task details for context
    # Generate a task name from the description for logging
    task_description = config.get("task", {}).get("description", "")
    task_name = task_description[:50] + "..." if len(task_description) > 50 else task_description or "Unnamed Task"
    task_description = config.get("task", {}).get("description", "")
    
    # Call prioritize_questions to have the question_prioritizer agent select a question
    # It now saves the output to prioritized_question.txt and returns empty list
    await question_manager.prioritize_questions()
    
    # Read the prioritized question from file and print it directly
    prioritized_path = question_manager.storage_dir / "prioritized_question.txt"
    if not prioritized_path.exists():
        logger.error("No prioritized_question.txt file was generated.")
        print("\n[AUTOINTERP] No prioritized question file was generated. Using raw question text instead.\n")
        # Try to use raw questions instead
        raw_questions_path = question_manager.storage_dir / "questions.txt"
        if raw_questions_path.exists():
            # Copy the raw questions to prioritized question path
            import shutil
            shutil.copy(raw_questions_path, prioritized_path)
            print(f"[AUTOINTERP] Copied raw questions to {prioritized_path}")

    # Read and print prioritized question
    if prioritized_path.exists():
        try:
            with open(prioritized_path, 'r') as f:
                prioritized_text = f.read()
            print("\n============= QUESTION PRIORITIZER OUTPUT =============")
            print(prioritized_text)
            print("========================================================\n")

            # Interactive checkpoint: let user revise prioritized question before title extraction
            if is_interactive(config):
                try:
                    async def _revise_prioritized(current: str, feedback: str) -> str:
                        return await make_revision_call(
                            llm_interface, "question_prioritizer", current, feedback, "question_prioritization"
                        )

                    def _save_prioritized(text: str) -> None:
                        with open(prioritized_path, 'w') as f:
                            f.write(text)

                    prioritized_text = await interactive_checkpoint(
                        "Question Prioritization", prioritized_text, _revise_prioritized, _save_prioritized, config
                    )
                except Exception as e:
                    logger.warning(f"Interactive checkpoint failed for question prioritization: {e}")

            # Extract TITLE from the prioritized text if available
            import re  # Make sure re is imported in this scope
            title_match = re.search(r'TITLE:\s*(.*?)(?:\n|$)', prioritized_text, re.IGNORECASE)
            if title_match:
                extracted_title = title_match.group(1).strip()
                print(f"[AUTOINTERP] Extracted title: {extracted_title}")
                
                # Update project_id with this title immediately
                if extracted_title:
                    # Sanitize the title for use as directory name
                    sanitized_title = re.sub(r'[^\w\-\.]', '_', extracted_title).lower()
                    sanitized_title = re.sub(r'_+', '_', sanitized_title)
                    
                    # Get timestamp for uniqueness
                    timestamp = get_timestamp("%Y-%m-%dT%H-%M-%S")
                    
                    # Create new project_id
                    new_project_id = f"{sanitized_title}_{timestamp}"
                    
                    # Store the original title for later use
                    config["title"] = extracted_title
                    
                    print(f"[AUTOINTERP] Setting new project ID: {new_project_id}")

                    # Immediately rename the project (we'll do it properly later)
                    old_project_id = config.get("project_id", "working_project")
                    config["project_id"] = new_project_id

        except Exception as e:
            print(f"[AUTOINTERP] Error reading prioritized question: {e}")
            raise ValueError("Failed to read prioritized question")
    else:
        raise ValueError("No prioritized question was found")
    
    # Pass the raw text directly as the active question
    logger.info(f"Selected question")
    active_question = prioritized_text
    selected_hyp = prioritized_text  # We still need this for compatibility with the return value
    
    # HERE'S THE KEY CHANGE:
    # Now that we've selected a question, let's rename the project with a meaningful name
    
    # If we already have a title from the prioritized question, use that
    # Otherwise, fall back to the task name
    if "title" in config:
        # We already set new_project_id in config when we extracted the title
        new_project_id = config["project_id"]
        print(f"[AUTOINTERP] Using extracted title for project ID: {new_project_id}")
    else:
        # Fall back to using the task name with timestamp
        default_timestamp = get_timestamp("%Y-%m-%dT%H-%M-%S")
        
        # Use a default project name when no title is extracted
        task_name = "interpretability_project"
        
        # Convert task name to a valid directory name
        safe_task_name = re.sub(r'[^\w]', '_', task_name).lower().strip('_')
        
        # Generate the new project ID
        new_project_id = f"{safe_task_name}_{default_timestamp}"
        print(f"[AUTOINTERP] No title found in prioritized question, using task name: {new_project_id}")
    
    # Get the old and new project paths using the path resolver
    path_resolver = PathResolver()  # Get the singleton instance
    old_project_dir = path_resolver.get_project_dir()
    
    # Need this for the os.rename operation
    configured_projects = config.get("paths", {}).get("projects")
    if configured_projects:
        projects_dir = Path(configured_projects)
    else:
        projects_dir = path_resolver.base_project_dir

    if not projects_dir.is_absolute():
        projects_dir = path_resolver.base_project_dir / projects_dir

    new_project_dir = projects_dir / new_project_id
    
    # ONLY rename if working_project exists and the new project doesn't
    if old_project_dir.exists() and not new_project_dir.exists():
        try:
            # Log before renaming
            logger.info(f"Renaming project directory from '{path_resolver.project_id}' to '{new_project_id}'")
            print(f"[AUTOINTERP] Renaming project from '{path_resolver.project_id}' to '{new_project_id}'...")
            
            # Rename the directory (this moves all files from old to new location)
            import os
            os.rename(old_project_dir, new_project_dir)
            
            # Update the project_id in config
            config["project_id"] = new_project_id
            
            # Update the path resolver with the new project_id
            # This ensures all future path resolutions will use the new project_id
            path_resolver.update_project_id(new_project_id)

            # Update the question manager's storage directory
            question_manager.update_storage_dir()
            
            # Update the evaluator's output directory if provided
            if evaluator:
                evaluator.output_dir = path_resolver.ensure_path("evaluation_results")
                logger.info(f"Updated evaluator's output directory to: {evaluator.output_dir}")
            else:
                logger.warning("Evaluator not available, skipping update of evaluation_results directory")
            logger.info(f"Updated question manager's storage directory")
            
            # Update PipelineUI with new project directory (if stored in config)
            _pipeline_ui = config.get("_pipeline_ui_ref")
            if _pipeline_ui:
                _pipeline_ui.update_project_dir(new_project_dir)

            logger.info(f"Successfully renamed project directory to '{new_project_id}'")
            print(f"[AUTOINTERP] Project directory renamed to: {new_project_id}")

            # Console logging was already set up at the start of the pipeline
            # The rename operation moved the entire directory including console.log
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to rename project directory: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"[AUTOINTERP] Warning: Could not rename project directory: {str(e)}")
    else:
        # Project already exists or rename didn't happen
        # Console logging was already set up at the start of the pipeline
        pass
    
    # Log the decision
    logger.info("Selected question using question_prioritizer - see raw text output in console")
    
    # Return our simple dict with raw_text instead of structured question
    return selected_hyp

async def analyze_question(
    active_question: Union[str, Dict[str, Any]],  # Can be raw text or dict
    analysis_generator: AnalysisGenerator,
    analysis_executor: AnalysisExecutor,
    analysis_planner: AnalysisPlanner,
    question_manager: QuestionManager,
    config: Dict[str, Any],
    logger: Any,
    iteration_number: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze the active question using the analysis_planner, analysis_generator, and analysis_executor.
    
    Args:
        active_question: The question to analyze
        analysis_generator: Generator for analysis code
        analysis_executor: Executor for analysis code
        analysis_planner: Planner for analysis strategy
        question_manager: Manager for question tracking
        config: Configuration dictionary
        logger: Logging instance
        
    Returns:
        Analysis results
    """
    logger.info("Starting analysis phase...")
    print(f"[AUTOINTERP] PHASE 3/4: Analysis of Question")
    print(f"[AUTOINTERP] Using raw question text for analysis\n")
    
    # First, plan the analysis
    logger.info("Planning analysis approach...")
    print(f"[AUTOINTERP] Planning analysis approach...")
    
    try:
        # Generate the analysis plan
        plan_path, analysis_plan = await analysis_planner.plan_analysis(
            active_question=active_question,
            config=config,
            iteration_number=iteration_number
        )
        
        logger.info(f"Generated analysis plan at {plan_path}")
        print(f"[AUTOINTERP] Generated analysis plan at {plan_path}")

        # Interactive checkpoint: let user revise analysis plan before code generation
        if is_interactive(config):
            try:
                _planner_llm = analysis_planner.llm_interface

                async def _revise_plan(current: str, feedback: str) -> str:
                    return await make_revision_call(
                        _planner_llm, "analysis_planner", current, feedback, "analysis_plan"
                    )

                def _save_plan(text: str) -> None:
                    nonlocal analysis_plan
                    analysis_plan = text
                    if plan_path and Path(plan_path).exists():
                        with open(plan_path, 'w') as f:
                            f.write(text)

                analysis_plan = await interactive_checkpoint(
                    "Analysis Plan", analysis_plan, _revise_plan, _save_plan, config
                )
            except Exception as e:
                logger.warning(f"Interactive checkpoint failed for analysis plan: {e}")

        # Next, generate analysis code based on the plan
        logger.info("Generating analysis script from plan...")
        print(f"[AUTOINTERP] Generating analysis script from plan...")
        
        # Pass the analysis plan directly - it will be formatted in the template
        script_path, analysis_code = await analysis_generator.generate_analysis(
            question=active_question,
            task_config=config,
            analysis_plan=analysis_plan,
            iteration_number=iteration_number
        )
        
        logger.info(f"Generated analysis script at {script_path}")
        print(f"[AUTOINTERP] Generated analysis script at {script_path}")
        
        # Execute the analysis with retry for script errors
        logger.info("Executing analysis...")
        print(f"[AUTOINTERP] Executing analysis script on question... (this may take a while)")
        
        max_retries = config.get("execution", {}).get("max_retries", 1)
        max_total_attempts = max_retries + 1  # Total attempts = initial attempt + retries
        attempt_number = 1
        
        while attempt_number <= max_total_attempts:
            execution_result = await analysis_executor.execute_analysis(
                script_path=script_path,
                question=active_question,
                parameters=config.get("analysis_parameters", {})
            )
            
            if execution_result.get("success", False):
                # Script ran successfully, break the retry loop
                break
            
            error_msg = execution_result.get("error", "Unknown error")
            error_traceback = execution_result.get("traceback", "")
            
            # Log detailed error to logger
            logger.error(f"Analysis execution failed (attempt {attempt_number}/{max_total_attempts}): {error_msg}")
            if error_traceback:
                logger.error(f"Traceback: {error_traceback}")
            
            # Print error details to console for user visibility
            print(f"[AUTOINTERP] Analysis execution failed (attempt {attempt_number}/{max_total_attempts})")
            print(f"[AUTOINTERP] Error: {error_msg}")
            if error_traceback:
                print(f"[AUTOINTERP] Traceback:\n{error_traceback}")
            
            if attempt_number < max_total_attempts:
                # Try to regenerate the script with the error information
                logger.info(f"Regenerating analysis script (attempt {attempt_number + 1}/{max_total_attempts})")
                print(f"[AUTOINTERP] Regenerating script (attempt {attempt_number + 1}/{max_total_attempts})...")
                
                # Increment the attempt counter in the analysis generator
                analysis_generator.increment_attempt()
                
                # Read stderr directly if possible
                stderr_file = Path(execution_result.get("execution_dir", "")) / "stderr.txt"
                stderr_content = ""
                
                if stderr_file.exists():
                    try:
                        with open(stderr_file, 'r') as f:
                            stderr_content = f.read()
                        logger.info(f"Read stderr.txt content for error context")
                    except Exception as e:
                        logger.error(f"Error reading stderr.txt: {e}")
                
                # Format the traceback nicely
                error_traceback_formatted = ""
                if stderr_content:
                    error_traceback_formatted = f"TRACEBACK:\n{stderr_content}"
                elif error_traceback:
                    error_traceback_formatted = f"TRACEBACK:\n{error_traceback}"

                # Capture the current script path before generating a new one
                previous_script_path = script_path

                # Generate a modified analysis script with error context
                script_path, analysis_code = await analysis_generator.generate_analysis(
                    question=active_question,
                    task_config=config,
                    error_context={
                        "error": error_msg,
                        "traceback": error_traceback_formatted,
                        "previous_script": previous_script_path
                    },
                    iteration_number=iteration_number
                )
                
                logger.info(f"Regenerated analysis script at {script_path}")
                print(f"[AUTOINTERP] Regenerated analysis script at {script_path}")
                
                attempt_number += 1
            else:
                # We've exhausted our attempts
                print(f"[AUTOINTERP] Analysis execution failed after {max_total_attempts} attempts.")
                
                # Check if we should shutdown or continue
                fail_on_max_retries = config.get("execution", {}).get("fail_on_max_retries", False)
                
                if fail_on_max_retries:
                    # Shutdown the entire system
                    logger.critical(f"System shutdown due to max retries exceeded (fail_on_max_retries=true)")
                    print(f"[AUTOINTERP] SYSTEM SHUTDOWN: Analysis failed after {max_total_attempts} attempts (fail_on_max_retries=true)")
                    raise SystemExit(f"Analysis execution failed: {error_msg}\nTraceback: {error_traceback}")
                else:
                    # Continue to next analysis (existing behavior)
                    logger.warning(f"Continuing to next analysis despite failures (fail_on_max_retries=false)")
                    print(f"[AUTOINTERP] Continuing to next analysis despite execution failures...")
                    # Raise error with full details for the iterative_analysis function to handle
                    raise ValueError(f"Analysis execution failed: {error_msg}\nTraceback: {error_traceback}")
        
        logger.info(f"Analysis execution completed in {execution_result.get('execution_time_formatted', 'unknown time')}")
        print(f"[AUTOINTERP] Analysis execution completed in {execution_result.get('execution_time_formatted', 'unknown time')}")
        
        # We'll handle moving to the next analysis in the iterative_analysis function
        # Don't call move_to_next_analysis() here
        
        # Record success in logs
        logger.info(f"Analysis completed successfully in {execution_result.get('execution_time_formatted', 'unknown time')}. Produced results of size {len(str(execution_result.get('results', {})))} characters.")

        # Return both analysis results and the plan
        return execution_result, analysis_plan
        
    except Exception as e:
        import traceback
        full_traceback = traceback.format_exc()
        
        # Log the full error details and also print to console
        logger.error(f"Error in analysis: {str(e)}")
        logger.error(full_traceback)
        
        # Print full error details to console
        print(f"[AUTOINTERP] Error in analysis: {str(e)}")
        print(f"[AUTOINTERP] Full traceback:\n{full_traceback}")
        
        # Record failure in logs
        logger.error(f"Analysis attempt failed: {str(e)}")
        logger.error(f"Traceback: {full_traceback}")
        
        # Re-raise the exception for the caller to handle
        raise e

async def evaluate_analysis(
    active_question: Union[str, Dict[str, Any]],  # Can be raw text or dict
    analysis_results: Dict[str, Any],
    evaluator: Evaluator,
    question_manager: QuestionManager,
    config: Dict[str, Any],
    logger: Any,
    current_confidence: float = 0.0,
    iteration_number: Optional[int] = None,
    attempt_number: Optional[int] = None,
    analysis_plan: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate analysis results against the question.
    
    Args:
        active_question: The question being tested
        analysis_results: Results from the analysis
        evaluator: Evaluator component
        question_manager: Manager for question tracking
        config: Configuration dictionary
        logger: Logging instance
        
    Returns:
        Evaluation results
    """
    logger.info("Starting evaluation phase...")
    print(f"[AUTOINTERP] Evaluating analysis results against question")
    
    # Use the evaluator to assess results
    # For raw text questions, we need to generate a dummy ID
    if isinstance(active_question, str):
        # Create a simple dummy ID
        dummy_id = "txt_question_1"
        evaluation_result = await evaluator.evaluate_analysis(
            analysis_results=analysis_results,
            question_id=dummy_id,
            current_confidence=current_confidence,
            iteration_number=iteration_number,
            attempt_number=attempt_number,
            analysis_plan=analysis_plan
        )
    else:
        # For dict questions, use the ID if available
        evaluation_result = await evaluator.evaluate_analysis(
            analysis_results=analysis_results,
            question_id=active_question.get("id", "txt_question_1"),
            current_confidence=current_confidence,
            iteration_number=iteration_number,
            attempt_number=attempt_number,
            analysis_plan=analysis_plan
        )
    
    # Log the evaluation results
    supports = evaluation_result.get("supports_question", None)
    confidence_impact = evaluation_result.get("confidence_impact", 0.0)
    explanation = evaluation_result.get("explanation", "")
    
    # Ensure confidence_impact is a valid number for formatting
    try:
        confidence_impact = float(confidence_impact) if confidence_impact is not None else 0.0
    except (ValueError, TypeError):
        confidence_impact = 0.0
    
    supports_text = "supports" if supports else "does not support" if supports is False else "is inconclusive regarding"
    logger.info(f"Evaluation complete. Evidence {supports_text} the question. Confidence impact: {confidence_impact:+.2f}")
    print(f"[AUTOINTERP] Evaluation complete")
    print(f"[AUTOINTERP] Evidence {supports_text} the question")
    
    # Print key insights
    if "key_insights" in evaluation_result and evaluation_result["key_insights"]:
        print(f"[AUTOINTERP] Key insights:")
        for i, insight in enumerate(evaluation_result["key_insights"][:3]):
            print(f"[AUTOINTERP]   {i+1}. {insight}")
    
    # Log evaluation results
    logger.info(f"Evaluated analysis results against question. Evidence {supports_text} the question. Confidence {confidence_impact:+.2f}.")
    if "key_insights" in evaluation_result and evaluation_result["key_insights"]:
        logger.info(f"Key insight: {evaluation_result['key_insights'][0]}")
    
    return evaluation_result

async def iterative_analysis(
    active_question: Union[str, Dict[str, Any]],  # Can be raw text or dict
    analysis_generator: AnalysisGenerator,
    analysis_executor: AnalysisExecutor,
    analysis_planner: AnalysisPlanner,
    evaluator: Evaluator,
    question_manager: QuestionManager,
    config: Dict[str, Any],
    logger: Any,
    max_iterations: int = 6,
    confidence_threshold: float = 0.8
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Perform iterative analysis and evaluation until confidence threshold is reached.
    
    Args:
        active_question: The question to analyze
        analysis_generator: Generator for analysis code
        analysis_executor: Executor for analysis code
        evaluator: Evaluator for analysis results
        question_manager: Manager for question tracking
        config: Configuration dictionary
        logger: Logging instance
        max_iterations: Maximum number of analysis iterations
        confidence_threshold: Confidence threshold to stop iteration
        
    Returns:
        Tuple of (all analyses, all evaluations)
    """
    logger.info(f"Starting iterative analysis cycle. Max iterations: {max_iterations}, Confidence threshold: {confidence_threshold}")
    print(f"[AUTOINTERP] Beginning iterative analysis cycle")
    print(f"[AUTOINTERP] Will continue analyzing until confidence is ≥ {confidence_threshold}, or until {max_iterations} analyses are completed")
    
    all_analyses = []
    all_evaluations = []
    
    # Get initial confidence - use default since we're using raw text
    current_confidence = 0.0
    
    for iteration in range(max_iterations):
        print(f"\n[AUTOINTERP] === ANALYSIS CYCLE {iteration+1}/{max_iterations} ===")
        print(f"[AUTOINTERP] Current confidence: {current_confidence:.2f}")
        
        # Make sure we start a new analysis for this iteration (not a retry attempt)
        # but only if this isn't the first iteration
        if iteration > 0:
            # Reset the analysis generator to start a new analysis
            analysis_generator.move_to_next_analysis()
        
        # Run analysis
        try:
            analysis_result, analysis_plan = await analyze_question(
                active_question=active_question,
                analysis_generator=analysis_generator,
                analysis_executor=analysis_executor,
                analysis_planner=analysis_planner,
                question_manager=question_manager,
                config=config,
                logger=logger,
                iteration_number=iteration + 1
            )
            all_analyses.append(analysis_result)
            
            # Evaluate results
            evaluation_result = await evaluate_analysis(
                active_question=active_question,
                analysis_results=analysis_result,
                evaluator=evaluator,
                question_manager=question_manager,
                config=config,
                logger=logger,
                current_confidence=current_confidence,
                iteration_number=iteration + 1,
                attempt_number=analysis_generator.current_attempt,
                analysis_plan=analysis_plan
            )
            
            # Check if evaluator detected failed analysis (similar to compilation failure handling)
            raw_evaluation = evaluation_result.get("raw_evaluation", "")
            if "ANALYSIS_FAILED" in raw_evaluation:
                logger.warning(f"Evaluator detected failed analysis in iteration {iteration+1}")
                print(f"[AUTOINTERP] Evaluator detected problematic analysis results")
                
                # Extract the reason for failure from the evaluation
                failure_reason = "Analysis produced uninformative or erroneous results"
                
                # Try to extract more specific failure reason from the evaluation
                if "ANALYSIS_FAILED" in raw_evaluation:
                    # Look for text after ANALYSIS_FAILED to get the reason
                    import re
                    failure_match = re.search(r'ANALYSIS_FAILED[:\s]*(.+?)(?:\n|$)', raw_evaluation, re.IGNORECASE)
                    if failure_match:
                        failure_reason = failure_match.group(1).strip()
                
                # Retry with both execution output and evaluator feedback
                max_retries = config.get("execution", {}).get("max_retries", 1)
                max_total_attempts = max_retries + 1
                attempt_number = 1
                
                analysis_failed_handled = False
                while attempt_number <= max_total_attempts and not analysis_failed_handled:
                    logger.info(f"Retrying analysis due to ANALYSIS_FAILED (attempt {attempt_number}/{max_total_attempts})")
                    print(f"[AUTOINTERP] Retrying analysis (attempt {attempt_number}/{max_total_attempts})...")
                    
                    # Increment the attempt counter in the analysis generator
                    analysis_generator.increment_attempt()
                    
                    # Create comprehensive error context with both execution and evaluation info
                    comprehensive_error_context = {
                        "error": failure_reason,
                        "traceback": f"EVALUATOR FEEDBACK:\n{raw_evaluation}\n\nEXECUTION OUTPUT:\n{analysis_result.get('stdout', '')}",
                        "previous_script": analysis_result.get("script_path", ""),
                        "analysis_failed": True  # Flag to indicate this is an ANALYSIS_FAILED retry
                    }
                    
                    # Generate a new analysis script with both execution results and evaluator feedback
                    new_script_path, new_analysis_code = await analysis_generator.generate_analysis(
                        question=active_question,
                        task_config=config,
                        error_context=comprehensive_error_context,
                        iteration_number=iteration + 1
                    )
                    
                    logger.info(f"Generated new analysis script at {new_script_path}")
                    print(f"[AUTOINTERP] Generated new analysis script at {new_script_path}")
                    
                    # Execute the new analysis
                    new_analysis_result = await analysis_executor.execute_analysis(
                        script_path=new_script_path,
                        question=active_question,
                        parameters=config.get("analysis_parameters", {})
                    )
                    
                    if not new_analysis_result.get("success", False):
                        # New script failed to execute, use normal compilation error handling
                        error_msg = new_analysis_result.get("error", "Unknown error")
                        logger.error(f"Retry analysis execution failed (attempt {attempt_number}/{max_total_attempts}): {error_msg}")
                        print(f"[AUTOINTERP] Retry analysis execution failed (attempt {attempt_number}/{max_total_attempts}): {error_msg}")
                        
                        if attempt_number < max_total_attempts:
                            attempt_number += 1
                            continue
                        else:
                            # Exhausted retries, check configuration for behavior
                            fail_on_max_retries = config.get("execution", {}).get("fail_on_max_retries", False)
                            
                            if fail_on_max_retries:
                                # Shutdown the entire system
                                logger.critical(f"System shutdown due to max retries exceeded in ANALYSIS_FAILED retry loop")
                                print(f"[AUTOINTERP] SYSTEM SHUTDOWN: All retry attempts failed (fail_on_max_retries=true)")
                                raise SystemExit("All retry attempts failed, shutting down system")
                            else:
                                # Continue with original failed analysis (existing behavior)
                                logger.error(f"All retry attempts failed, continuing with original analysis")
                                print(f"[AUTOINTERP] All retry attempts failed, continuing with original analysis")
                                break
                    else:
                        # New analysis succeeded, evaluate it
                        new_evaluation_result = await evaluate_analysis(
                            active_question=active_question,
                            analysis_results=new_analysis_result,
                            evaluator=evaluator,
                            question_manager=question_manager,
                            config=config,
                            logger=logger,
                            current_confidence=current_confidence,
                            iteration_number=iteration + 1,
                            attempt_number=analysis_generator.current_attempt,
                            analysis_plan=analysis_plan
                        )
                        
                        # Check if the new analysis still failed
                        new_raw_evaluation = new_evaluation_result.get("raw_evaluation", "")
                        if "ANALYSIS_FAILED" in new_raw_evaluation:
                            logger.warning(f"Retry analysis still failed evaluation (attempt {attempt_number}/{max_total_attempts})")
                            print(f"[AUTOINTERP] Retry analysis still produced problematic results (attempt {attempt_number}/{max_total_attempts})")
                            
                            if attempt_number < max_total_attempts:
                                attempt_number += 1
                                continue
                            else:
                                # Use the new results even though they failed evaluation
                                logger.info(f"Using final retry results despite evaluation failure")
                                print(f"[AUTOINTERP] Using final retry results despite evaluation concerns")
                                analysis_result = new_analysis_result
                                evaluation_result = new_evaluation_result
                                # Mark that this evaluation should not trigger early termination
                                evaluation_result["skip_confidence_check"] = True
                                analysis_failed_handled = True
                        else:
                            # New analysis passed evaluation
                            logger.info(f"Retry analysis succeeded after {attempt_number} attempts")
                            print(f"[AUTOINTERP] Retry analysis succeeded after {attempt_number} attempts")
                            analysis_result = new_analysis_result
                            evaluation_result = new_evaluation_result
                            analysis_failed_handled = True
                            break
                
                # Update the analysis result in the list
                if all_analyses:
                    all_analyses[-1] = analysis_result
                else:
                    all_analyses.append(analysis_result)
            
            all_evaluations.append(evaluation_result)
            
            # Log successful analysis and evaluation to comprehensive log
            # Only log if the evaluation doesn't contain ANALYSIS_FAILED
            if "ANALYSIS_FAILED" not in evaluation_result.get("raw_evaluation", ""):
                # Get the project directory for logging
                path_resolver = PathResolver()
                project_dir = path_resolver.get_project_dir()
                
                # Log the analysis script content
                script_path = analysis_result.get("script_path", "")
                if script_path and Path(script_path).exists():
                    try:
                        with open(script_path, 'r') as f:
                            script_content = f.read()
                        log_to_comprehensive_log(project_dir, script_content, f"ANALYSIS SCRIPT (Iteration {iteration + 1})")
                    except Exception as e:
                        logger.warning(f"Failed to read script for comprehensive log: {e}")
                
                # Log the execution results (stdout)
                stdout_content = analysis_result.get("stdout", "")
                if stdout_content:
                    log_to_comprehensive_log(project_dir, stdout_content, f"ANALYSIS RESULTS (Iteration {iteration + 1})")
                
                # Log the evaluation
                evaluation_content = evaluation_result.get("raw_evaluation", "")
                if evaluation_content:
                    log_to_comprehensive_log(project_dir, evaluation_content, f"EVALUATION (Iteration {iteration + 1})")
            
            # With raw text, no need to update the active question
            # Just keep using the same text
            
            # Update confidence with the new value from evaluation
            current_confidence = evaluation_result.get("new_confidence", current_confidence)

            # Interactive feedback: show evaluation and collect guidance for next iteration
            if is_interactive(config) and iteration < max_iterations - 1:
                eval_summary = evaluation_result.get("raw_evaluation", "")
                if eval_summary:
                    print(f"\n[INTERACTIVE] === Evaluation (Iteration {iteration + 1}) ===")
                    print(eval_summary[:2000])
                    if len(eval_summary) > 2000:
                        print(f"... ({len(eval_summary) - 2000} more characters)")
                try:
                    user_guidance = input("[INTERACTIVE] Feedback for next analysis iteration (Enter to continue): ").strip()
                    if user_guidance:
                        # Store guidance so the planner can incorporate it
                        config.setdefault("_interactive_guidance", []).append(
                            f"User feedback after iteration {iteration + 1}: {user_guidance}"
                        )
                        print(f"[INTERACTIVE] Guidance noted for next iteration.")
                except (EOFError, KeyboardInterrupt):
                    pass

            # Check if we've reached our confidence threshold
            # Skip confidence check if all ANALYSIS_FAILED retries were exhausted
            # Only stop when confidence is HIGH (>= threshold), not when low
            if not evaluation_result.get("skip_confidence_check", False):
                if current_confidence >= confidence_threshold:
                    logger.info(f"Reached confidence threshold: {current_confidence:.2f}")
                    conclusion = "SUPPORT"
                    print(f"[AUTOINTERP] Reached confidence threshold: {current_confidence:.2f} - {conclusion}")
                    print(f"[AUTOINTERP] Concluding analysis phase after {iteration+1} iterations")
                    break
            else:
                logger.info(f"Skipping confidence check due to exhausted ANALYSIS_FAILED retries")
                print(f"[AUTOINTERP] Skipping confidence check due to failed analysis - continuing to next iteration")
            
            # If more iterations to go, continue
            if iteration < max_iterations - 1:
                if "follow_up_recommendations" in evaluation_result and evaluation_result["follow_up_recommendations"]:
                    next_analysis = evaluation_result["follow_up_recommendations"][0]
                    print(f"[AUTOINTERP] Proceeding with next analysis: {next_analysis}")
                else:
                    print(f"[AUTOINTERP] Proceeding with next analysis: Confidence still inconclusive ({current_confidence:.2f})")
        
        except ValueError as e:
            # Show full error details including traceback if available
            error_message = str(e)
            
            # Extract traceback if available in the error message
            traceback_start = error_message.find("Traceback:")
            if traceback_start > -1:
                error_text = error_message[:traceback_start].strip()
                traceback_text = error_message[traceback_start:].strip()
                # Print detailed error to console
                print(f"\n[AUTOINTERP] Analysis failed: {error_text}")
                print(f"[AUTOINTERP] {traceback_text}")
            else:
                # Print detailed error to console
                print(f"\n[AUTOINTERP] Analysis failed: {error_message}")
            
            # Log the full error for debugging
            logger.error(f"Error in analysis cycle {iteration+1}: {str(e)}")
            
            # Record the error in logs
            logger.error(f"Analysis attempt in cycle {iteration+1} failed: {error_message}")
            
            # Continue to next iteration
        except Exception as e:
            # Show full error message and traceback
            import traceback
            full_traceback = traceback.format_exc()
            error_message = str(e)
            
            # Log detailed error in logger
            logger.error(f"Error in analysis cycle {iteration+1}: {str(e)}")
            logger.error(full_traceback)
            
            # Print full error details to console
            print(f"\n[AUTOINTERP] Analysis failed: {error_message}")
            print(f"[AUTOINTERP] Full traceback:\n{full_traceback}")
            
            # Continue to next iteration
            
    return all_analyses, all_evaluations


async def iterative_analysis_agent(
    active_question: str,
    config: Dict[str, Any],
    logger: Any,
    max_iterations: int = 3,
    confidence_threshold: float = 0.85,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Perform iterative analysis using a CLI agent subprocess (claude or codex).

    Each iteration launches a single agent invocation that autonomously plans,
    writes code, executes, debugs, evaluates, and updates confidence.

    Returns:
        Tuple of (all_analyses, all_evaluations) — same shape as
        ``iterative_analysis()`` for downstream compatibility.
    """
    logger.info(
        "Starting agent-based iterative analysis. max_iterations=%d, threshold=%.2f",
        max_iterations,
        confidence_threshold,
    )
    print(f"[AUTOINTERP] Beginning agent-based iterative analysis")
    print(
        f"[AUTOINTERP] Will continue until confidence >= {confidence_threshold} "
        f"or {max_iterations} iterations completed"
    )

    path_resolver = PathResolver()
    analysis_root = path_resolver.ensure_analysis_dir()

    # Determine provider
    llm_config = config.get("llm", {})
    provider = (llm_config.get("provider") or "").lower()

    # Model info from config
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "meta-llama/Llama-3.2-1B-Instruct")
    model_path = model_cfg.get("tokenizer", model_name)

    # Load prompt template
    try:
        prompt_template = load_analysis_prompt_template()
    except FileNotFoundError as exc:
        logger.error("Could not load analysis prompt template: %s", exc)
        print(f"[AUTOINTERP] ERROR: {exc}")
        return [], []

    # Agent timeout
    agent_timeout = config.get("analysis", {}).get(
        "agent_timeout",
        config.get("agents", {}).get("analysis_agent", {}).get("timeout", 1800),
    )

    all_analyses: List[Dict[str, Any]] = []
    all_evaluations: List[Dict[str, Any]] = []
    consecutive_failures = 0

    # Get optional PipelineUI
    pipeline_ui = config.get("_pipeline_ui_ref")

    for iteration in range(1, max_iterations + 1):
        # Read current confidence
        conf_data = read_confidence(analysis_root)
        current_confidence = conf_data.get("current_confidence", 0.0)

        print(f"\n[AUTOINTERP] === ANALYSIS AGENT ITERATION {iteration}/{max_iterations} ===")
        print(f"[AUTOINTERP] Current confidence: {current_confidence:.2f}")

        # Setup workspace
        iter_dir = setup_analysis_workspace(path_resolver, active_question, iteration)
        logger.info("Workspace ready: %s", iter_dir)

        # Build prompt
        prompt_text = _build_analysis_prompt(
            iteration_n=iteration,
            analysis_root=analysis_root,
            prompt_template=prompt_template,
            model_name=model_name,
            model_path=model_path,
        )

        # Run agent
        import time as _time
        _agent_start = _time.time()
        _analysis_progress_cb = None
        if pipeline_ui:
            _iter = iteration  # capture for closure
            def _analysis_progress_cb(msg, _iter=_iter):
                pipeline_ui.step_progress("iterative_analysis", f"[Iter {_iter}] {msg}")
        agent_result = run_analysis_agent(
            provider=provider,
            analysis_dir=iter_dir,
            prompt_text=prompt_text,
            timeout=agent_timeout,
            on_progress=_analysis_progress_cb,
            iteration_n=iteration,
        )
        _agent_duration = _time.time() - _agent_start

        # Read outputs
        outputs = read_agent_outputs(analysis_root, iteration)
        new_conf = read_confidence(analysis_root)
        new_confidence = new_conf.get("current_confidence", current_confidence)

        # Build analysis dict (compatible with downstream)
        script_content = "\n\n".join(s["content"] for s in outputs["scripts"]) if outputs["scripts"] else ""
        analysis_dict: Dict[str, Any] = {
            "success": agent_result["success"],
            "stdout": agent_result.get("stdout", ""),
            "stderr": agent_result.get("stderr", ""),
            "script_path": str(iter_dir),
            "script_content": script_content,
            "execution_dir": str(iter_dir),
            "execution_time_formatted": f"{_agent_duration:.0f}s",
            "agent_mode": True,
        }
        all_analyses.append(analysis_dict)

        # Build evaluation dict (compatible with downstream)
        evaluation_dict: Dict[str, Any] = {
            "raw_evaluation": outputs.get("evaluation", ""),
            "new_confidence": new_confidence,
            "supports_question": new_confidence >= 0.5,
            "confidence_impact": new_confidence - current_confidence,
            "key_insights": [],
            "follow_up_recommendations": [],
            "agent_mode": True,
        }
        all_evaluations.append(evaluation_dict)

        # Record in dashboard
        if pipeline_ui:
            pipeline_ui.llm_call_complete(
                agent_name="analysis_agent",
                display_name=f"Analysis Agent (Iteration {iteration})",
                prompt=prompt_text[:2000] + ("..." if len(prompt_text) > 2000 else ""),
                system_message=None,
                response=outputs.get("evaluation", agent_result.get("stdout", ""))[:3000],
                model=f"{provider} CLI agent",
                provider=provider,
                temperature=0,
                max_tokens=0,
                duration_seconds=_agent_duration,
                step_id="iterative_analysis",
                iteration_number=iteration,
            )

        # Log to comprehensive log
        project_dir = path_resolver.get_project_dir()
        if outputs.get("evaluation"):
            log_to_comprehensive_log(
                project_dir,
                outputs["evaluation"],
                f"ANALYSIS AGENT EVALUATION (Iteration {iteration})",
            )
        if script_content:
            log_to_comprehensive_log(
                project_dir,
                script_content,
                f"ANALYSIS AGENT SCRIPTS (Iteration {iteration})",
            )

        # Track consecutive failures for fallback
        if not agent_result["success"] and not outputs.get("evaluation"):
            consecutive_failures += 1
            logger.warning(
                "Agent iteration %d failed (consecutive failures: %d)",
                iteration,
                consecutive_failures,
            )
            print(f"[AUTOINTERP] Agent iteration {iteration} failed")
            if consecutive_failures >= 2:
                logger.warning("2 consecutive agent failures; stopping agent analysis.")
                print("[AUTOINTERP] 2 consecutive agent failures. Stopping analysis.")
                break
        else:
            consecutive_failures = 0
            print(f"[AUTOINTERP] Agent iteration {iteration} complete. Confidence: {new_confidence:.2f}")

        # Interactive feedback: show evaluation and collect guidance for next agent iteration
        if is_interactive(config) and iteration < max_iterations:
            eval_text = outputs.get("evaluation", "")
            if eval_text:
                print(f"\n[INTERACTIVE] === Agent Evaluation (Iteration {iteration}) ===")
                print(eval_text[:2000])
                if len(eval_text) > 2000:
                    print(f"... ({len(eval_text) - 2000} more characters)")
            try:
                user_feedback = input("[INTERACTIVE] Feedback for next iteration (Enter to continue): ").strip()
                if user_feedback:
                    # Write feedback to background/user_feedback.md for next iteration's prompt
                    feedback_path = analysis_root / "background" / "user_feedback.md"
                    with open(feedback_path, "a", encoding="utf-8") as f:
                        f.write(f"\n## User Feedback After Iteration {iteration}\n{user_feedback}\n")
                    print(f"[INTERACTIVE] Feedback saved for next iteration.")
            except (EOFError, KeyboardInterrupt):
                pass

        # Check confidence threshold
        if new_confidence >= confidence_threshold:
            logger.info("Reached confidence threshold: %.2f", new_confidence)
            print(f"[AUTOINTERP] Reached confidence threshold: {new_confidence:.2f}")
            print(f"[AUTOINTERP] Concluding analysis phase after {iteration} iterations")
            break

    return all_analyses, all_evaluations


def _find_agent_analyses(path_resolver: PathResolver) -> List[Tuple[str, str, str]]:
    """
    Find analyses produced by the agent-based pipeline (analysis/analysis_N/ layout).

    Returns:
        List of tuples: (script_content, evaluation_content, analysis_name)
        or empty list if the agent layout does not exist.
    """
    import logging
    logger = logging.getLogger(__name__)

    analysis_root = path_resolver.get_analysis_dir()
    if not analysis_root.exists():
        return []

    results = []
    # Discover analysis_N directories
    iter_dirs = []
    for item in analysis_root.iterdir():
        if item.is_dir() and item.name.startswith("analysis_"):
            try:
                n = int(item.name.split("_")[1])
                iter_dirs.append((n, item))
            except (ValueError, IndexError):
                continue
    iter_dirs.sort(key=lambda x: x[0])

    for n, d in iter_dirs:
        # Collect .py script content
        scripts: List[str] = []
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix == ".py":
                try:
                    scripts.append(f.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    pass

        # Read evaluation file
        eval_path = d / f"ANALYSIS_{n}_EVALUATION.md"
        eval_content = ""
        if eval_path.exists():
            try:
                eval_content = eval_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

        if scripts and eval_content:
            combined_script = "\n\n".join(scripts)
            results.append((combined_script, eval_content, f"analysis_{n}"))
            logger.info(f"Agent layout: found analysis_{n} ({len(scripts)} scripts)")

    if results:
        logger.info(f"Agent layout: {len(results)} successful analyses")
    return results


def find_successful_analyses(path_resolver: PathResolver) -> List[Tuple[str, str, str]]:
    """
    Find all successful analyses and extract their script and output files.

    Checks the agent layout (analysis/analysis_N/) first; falls back to the
    legacy layout (analysis/analysis_N/attempt_N/) if the agent
    layout doesn't exist or is empty.

    Args:
        path_resolver: PathResolver for getting project paths

    Returns:
        List of tuples: (analysis_script_content, analysis_output_content, analysis_name)
    """
    import logging
    logger = logging.getLogger(__name__)

    # Try agent layout first
    agent_results = _find_agent_analyses(path_resolver)
    if agent_results:
        return agent_results

    # Legacy layout (scripts now also stored under analysis/)
    successful_analyses = []
    analysis_dir = path_resolver.get_path("analysis")

    if not analysis_dir.exists():
        logger.warning(f"Analysis directory not found: {analysis_dir}")
        return successful_analyses

    # Find all analysis_N directories
    analysis_dirs = []
    for item in analysis_dir.iterdir():
        if item.is_dir() and item.name.startswith("analysis_"):
            try:
                # Extract the analysis number
                analysis_num = int(item.name.split("_")[1])
                analysis_dirs.append((analysis_num, item))
            except (ValueError, IndexError):
                logger.warning(f"Skipping malformed analysis directory: {item.name}")
                continue

    # Sort by analysis number
    analysis_dirs.sort(key=lambda x: x[0])

    logger.info(f"Found {len(analysis_dirs)} analysis directories")

    # For each analysis directory, find the highest attempt
    for analysis_num, analysis_dir in analysis_dirs:
        logger.info(f"Processing analysis_{analysis_num}")

        # Find all attempt_N directories within this analysis
        attempt_dirs = []
        for item in analysis_dir.iterdir():
            if item.is_dir() and item.name.startswith("attempt_"):
                try:
                    # Extract the attempt number
                    attempt_num = int(item.name.split("_")[1])
                    attempt_dirs.append((attempt_num, item))
                except (ValueError, IndexError):
                    logger.warning(f"Skipping malformed attempt directory: {item.name}")
                    continue

        if not attempt_dirs:
            logger.warning(f"No attempt directories found in {analysis_dir}")
            continue

        # Sort by attempt number and get the highest one
        attempt_dirs.sort(key=lambda x: x[0])
        highest_attempt_num, highest_attempt_dir = attempt_dirs[-1]

        logger.info(f"Using highest attempt: attempt_{highest_attempt_num} for analysis_{analysis_num}")

        # Look for the required files in the highest attempt directory
        analysis_script_content = None
        analysis_output_content = None

        # Find analysis_generator_*.txt file
        for file_path in highest_attempt_dir.iterdir():
            if file_path.is_file() and file_path.name.startswith("analysis_generator_") and file_path.suffix == ".txt":
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        analysis_script_content = f.read()
                    logger.info(f"Found analysis script file: {file_path.name}")
                    break
                except Exception as e:
                    logger.warning(f"Error reading analysis script file {file_path}: {e}")

        # Find stdout.txt file
        stdout_file = highest_attempt_dir / "stdout.txt"
        if stdout_file.exists():
            try:
                with open(stdout_file, 'r', encoding='utf-8') as f:
                    analysis_output_content = f.read()
                logger.info(f"Found stdout.txt file")
            except Exception as e:
                logger.warning(f"Error reading stdout.txt: {e}")
        else:
            logger.warning(f"stdout.txt not found in {highest_attempt_dir}")

        # Only add if we found both required files
        if analysis_script_content and analysis_output_content:
            analysis_name = f"analysis_{analysis_num}"
            successful_analyses.append((analysis_script_content, analysis_output_content, analysis_name))
            logger.info(f"Successfully extracted files for {analysis_name}")
        else:
            logger.warning(f"Missing required files for analysis_{analysis_num} (script: {analysis_script_content is not None}, output: {analysis_output_content is not None})")

    logger.info(f"Found {len(successful_analyses)} successful analyses with complete files")
    return successful_analyses


async def evaluate_and_retry_visualization(
    analysis_name: str,
    analysis_number: int,
    viz_code: str,
    framework: Dict[str, Any],
    logger: Any,
    max_retries: int = 2
):
    """
    Evaluate a visualization and retry generation if issues are found.

    Args:
        analysis_name: Name of the analysis
        analysis_number: Analysis iteration number
        viz_code: The visualization code that was generated
        framework: Framework components
        logger: Logging instance
        max_retries: Maximum number of retry attempts
    """
    visualization_evaluator = framework["visualization_evaluator"]

    for attempt in range(max_retries + 1):
        logger.info(f"Evaluating visualization for {analysis_name} (attempt {attempt + 1})")
        print(f"[AUTOINTERP] Evaluating visualization for {analysis_name}...")

        try:
            # Evaluate the visualization
            evaluation_response = await visualization_evaluator.evaluate_visualization(
                visualization_code=viz_code,
                analysis_number=analysis_number,
                iteration_number=analysis_number,  # Analysis iteration (a1, a2, a3)
                attempt_number=attempt + 1         # Visualization retry attempt (1, 2, 3)
            )

            # Check if the response contains VISUALIZATION_ERROR
            if "VISUALIZATION_ERROR" in evaluation_response:
                # Visualization has issues
                logger.warning(f"Visualization evaluation found issues for {analysis_name}: {evaluation_response}")
                print(f"[AUTOINTERP] ⚠ Visualization issues found for {analysis_name}")
                print(f"[AUTOINTERP] Issue: {evaluation_response}")

                if attempt < max_retries:
                    logger.info(f"Retrying visualization generation for {analysis_name} (attempt {attempt + 2})")
                    print(f"[AUTOINTERP] Retrying visualization generation for {analysis_name}...")

                    # Get framework components needed for retry
                    analysis_executor = framework["analysis_executor"]
                    path_resolver = framework["path_resolver"]
                    config = framework["config"]

                    # Initialize visualization components
                    from AutoInterp.visualization.visualization_generator import VisualizationGenerator
                    llm_interface = framework["llm_interface"]
                    viz_generator = VisualizationGenerator(llm_interface, path_resolver)

                    # Increment attempt counter
                    viz_generator.current_attempt = attempt + 2

                    # Create error context with evaluation feedback (standardized format)
                    error_context = {
                        "evaluation_feedback": evaluation_response,
                        "previous_code": viz_code,
                        "visualization_error": True,
                        "error_type": "evaluation_failure"
                    }

                    # Need to get the original analysis script and output for regeneration
                    # This should be passed to this function - for now, use placeholder
                    try:
                        # Get successful analyses to find the matching one
                        successful_analyses = find_successful_analyses(path_resolver)

                        # Find the matching analysis
                        matching_analysis = None
                        for script, output, name in successful_analyses:
                            if name == analysis_name:
                                matching_analysis = (script, output, name)
                                break

                        if not matching_analysis:
                            logger.error(f"Could not find matching analysis for {analysis_name}")
                            print(f"[AUTOINTERP] ✗ Could not find analysis data for retry")
                            break

                        analysis_script, analysis_output, _ = matching_analysis

                        # Regenerate visualization code with error context
                        logger.info(f"Regenerating visualization code for {analysis_name}")
                        new_viz_code = await viz_generator.generate_visualization(
                            analysis_script=analysis_script,
                            analysis_output=analysis_output,
                            error_context=error_context
                        )

                        # Save and execute the new visualization script
                        viz_dir = path_resolver.ensure_path("visualizations")
                        from AutoInterp.core.utils import get_timestamp
                        timestamp = get_timestamp().replace(" ", "_").replace(":", "-")
                        viz_script_name = f"visualization_{analysis_name}_{timestamp}_retry{attempt + 1}.py"
                        viz_script_path = viz_dir / viz_script_name

                        # Clean the new visualization code to remove markdown code fences
                        cleaned_new_viz_code = clean_code_content(new_viz_code)

                        # Write the new visualization script
                        with open(viz_script_path, 'w') as f:
                            f.write(cleaned_new_viz_code)

                        logger.info(f"Saved retry visualization script to: {viz_script_path}")

                        # Execute the new visualization script
                        execution_result = await analysis_executor.execute_analysis(
                            script_path=viz_script_path,
                            question={"raw_text": f"Retry visualization for {analysis_name}"},
                            parameters=config.get("analysis_parameters", {}),
                            execution_type="visualization"
                        )

                        if execution_result.get("success", False):
                            # Move the new visualization files
                            viz_files = execution_result.get("visualization_files", [])
                            if viz_files:
                                # Extract analysis number for prefix
                                analysis_number_str = analysis_name.split("_")[1] if "_" in analysis_name else "unknown"

                                for viz_file_path in viz_files:
                                    try:
                                        original_path = Path(viz_file_path)
                                        if original_path.exists():
                                            original_name = original_path.name
                                            new_filename = f"a{analysis_number_str}_{original_name}"
                                            new_path = viz_dir / new_filename

                                            # Move the file, replacing the previous one
                                            import shutil
                                            shutil.move(str(original_path), str(new_path))
                                            logger.info(f"Moved retry visualization file: {original_path} → {new_path}")
                                            print(f"[AUTOINTERP] ✓ Regenerated visualization: {new_filename}")

                                            # Update viz_code for next evaluation iteration
                                            viz_code = new_viz_code
                                            break
                                    except Exception as e:
                                        logger.warning(f"Failed to move retry visualization file {viz_file_path}: {e}")
                        else:
                            error_msg = execution_result.get("error", "Unknown error")
                            logger.error(f"Failed to execute retry visualization script for {analysis_name}: {error_msg}")
                            print(f"[AUTOINTERP] ✗ Retry visualization execution failed: {error_msg}")
                            continue  # Continue to next retry attempt

                    except Exception as e:
                        logger.error(f"Error during visualization retry for {analysis_name}: {str(e)}")
                        print(f"[AUTOINTERP] ✗ Error during visualization retry: {str(e)}")
                        continue  # Continue to next retry attempt
                else:
                    logger.warning(f"Max retries reached for visualization {analysis_name} - continuing with current visualization")
                    print(f"[AUTOINTERP] ⚠ Max retries reached for visualization {analysis_name}")
                    break
            else:
                # Visualization is good - evaluation contains caption
                logger.info(f"Visualization evaluation passed for {analysis_name}: {evaluation_response}")
                print(f"[AUTOINTERP] ✓ Visualization evaluation passed for {analysis_name}")
                print(f"[AUTOINTERP] Caption: {evaluation_response}")

                # Log the successful visualization caption to comprehensive log
                try:
                    path_resolver = framework["path_resolver"]
                    project_dir = path_resolver.get_project_dir()
                    log_to_comprehensive_log(project_dir, evaluation_response, f"Figure {analysis_number}")
                    logger.info(f"Added visualization caption for analysis {analysis_number} to comprehensive log")
                except Exception as e:
                    logger.warning(f"Failed to log visualization caption to comprehensive log: {e}")

                break

        except Exception as e:
            logger.error(f"Error during visualization evaluation for {analysis_name}: {str(e)}")
            print(f"[AUTOINTERP] ✗ Error evaluating visualization for {analysis_name}: {str(e)}")
            break


async def generate_visualizations(
    successful_analyses: List[Tuple[str, str, str]],
    framework: Dict[str, Any]
) -> Dict[str, str]:
    """
    Generate visualizations for successful analyses using the visualization pipeline.
    
    Args:
        successful_analyses: List of (script_content, output_content, analysis_name) tuples
        framework: Framework components (llm_interface, path_resolver, etc.)
        
    Returns:
        Dictionary mapping analysis names to generated visualization file paths
    """
    import logging
    from AutoInterp.visualization.visualization_planner import VisualizationPlanner
    from AutoInterp.visualization.visualization_generator import VisualizationGenerator
    
    logger = logging.getLogger(__name__)
    
    if not successful_analyses:
        logger.info("No successful analyses found, skipping visualization generation")
        return {}
    
    logger.info(f"Generating visualizations for {len(successful_analyses)} analyses")
    
    # Get framework components
    llm_interface = framework["llm_interface"]
    path_resolver = framework["path_resolver"]
    analysis_executor = framework["analysis_executor"]
    config = framework["config"]
    
    # Initialize visualization components
    viz_planner = VisualizationPlanner(llm_interface, path_resolver)
    viz_generator = VisualizationGenerator(llm_interface, path_resolver)
    
    visualization_files = {}
    
    # Ensure visualization directory exists
    viz_dir = path_resolver.ensure_path("visualizations")
    logger.info(f"Visualization directory: {viz_dir}")
    
    # Process each successful analysis
    for i, (analysis_script, analysis_output, analysis_name) in enumerate(successful_analyses):
        logger.info(f"Processing visualization for {analysis_name} ({i+1}/{len(successful_analyses)})")
        print(f"[AUTOINTERP] Generating visualization for {analysis_name}...")
        
        try:
            # Step 1: Plan the visualization
            logger.info(f"Planning visualization for {analysis_name}")
            viz_plan = await viz_planner.plan_visualization(analysis_script, analysis_output)
            
            # Step 2: Generate visualization code
            logger.info(f"Generating visualization code for {analysis_name}")
            viz_code = await viz_generator.generate_visualization(analysis_script, analysis_output)
            
            # Step 3: Save and execute the visualization script
            timestamp = get_timestamp().replace(" ", "_").replace(":", "-")
            viz_script_name = f"visualization_{analysis_name}_{timestamp}.py"
            viz_script_path = viz_dir / viz_script_name
            
            # Clean the visualization code to remove markdown code fences
            cleaned_viz_code = clean_code_content(viz_code)

            # Write the visualization script to file
            with open(viz_script_path, 'w') as f:
                f.write(cleaned_viz_code)
            
            logger.info(f"Saved visualization script to: {viz_script_path}")
            
            # Step 4: Execute the visualization script
            logger.info(f"Executing visualization script for {analysis_name}")
            execution_result = await analysis_executor.execute_analysis(
                script_path=viz_script_path,
                question={"raw_text": f"Visualization for {analysis_name}"},  # Dummy question for executor
                parameters=config.get("analysis_parameters", {}),
                execution_type="visualization"
            )
            
            if execution_result.get("success", False):
                # Check if visualization files were generated
                viz_files = execution_result.get("visualization_files", [])
                if viz_files:
                    # Extract analysis number for prefix
                    analysis_number = analysis_name.split("_")[1] if "_" in analysis_name else "unknown"
                    
                    # Move and rename visualization files to /visualizations/ directory with prefix
                    moved_files = []
                    for viz_file_path in viz_files:
                        try:
                            original_path = Path(viz_file_path)
                            if original_path.exists():
                                original_name = original_path.name
                                
                                # Only add prefix to files that don't already have an analysis prefix
                                import re
                                if re.match(r'^a\d+_', original_name):
                                    # File already has a prefix, skip it (it's from a previous analysis)
                                    logger.info(f"Skipping file with existing prefix: {original_name}")
                                    continue
                                
                                # Add prefix to new files (those without existing prefix)
                                new_filename = f"a{analysis_number}_{original_name}"
                                new_path = viz_dir / new_filename
                                
                                # Move the file to visualizations directory  
                                import shutil
                                shutil.move(str(original_path), str(new_path))
                                moved_files.append(str(new_path))
                                logger.info(f"Moved visualization file: {original_path} → {new_path}")
                        except Exception as e:
                            logger.warning(f"Failed to move visualization file {viz_file_path}: {e}")
                    
                    if moved_files:
                        # Use the first moved visualization file
                        visualization_files[analysis_name] = moved_files[0]
                        logger.info(f"Successfully generated visualization for {analysis_name}: {moved_files[0]}")
                        print(f"[AUTOINTERP] ✓ Visualization generated for {analysis_name}: a{analysis_number}_{Path(moved_files[0]).name}")

                        # Evaluate the visualization
                        await evaluate_and_retry_visualization(
                            analysis_name=analysis_name,
                            analysis_number=int(analysis_number) if analysis_number.isdigit() else 1,
                            viz_code=viz_code,
                            framework=framework,
                            logger=logger
                        )
                    else:
                        logger.warning(f"Generated visualization files but failed to move them for {analysis_name}")
                        print(f"[AUTOINTERP] ⚠ Visualization generated but failed to organize files for {analysis_name}")
                else:
                    logger.warning(f"Visualization script executed successfully but no image files were generated for {analysis_name}")
                    print(f"[AUTOINTERP] ⚠ Visualization script ran but no images generated for {analysis_name}")

                    # Treat this as a failure that needs retry - script ran but produced no visualizations
                    error_msg = "Visualization script executed but no image files were generated"

                    # Implement retry logic for no-output failures
                    max_viz_retries = config.get("execution", {}).get("max_retries", 1)
                    max_total_viz_attempts = max_viz_retries + 1
                    viz_attempt = 2  # We already tried once
                    last_error = error_msg

                    while viz_attempt <= max_total_viz_attempts:
                        logger.info(f"Retrying visualization generation for {analysis_name} due to no output (attempt {viz_attempt}/{max_total_viz_attempts})")
                        print(f"[AUTOINTERP] Retrying visualization generation for {analysis_name} - no images produced (attempt {viz_attempt}/{max_total_viz_attempts})...")

                        try:
                            # Increment attempt counter
                            viz_generator.increment_attempt()

                            # Create error context for no-output case (standardized format)
                            error_context = {
                                "error": error_msg,
                                "stderr": execution_result.get("stderr", ""),
                                "stdout": execution_result.get("stdout", ""),
                                "previous_script": str(viz_script_path),
                                "visualization_failed": True,
                                "no_images_generated": True,
                                "error_type": "execution_failure"
                            }

                            # Generate new visualization code with error context
                            retry_viz_code = await viz_generator.generate_visualization(
                                analysis_script,
                                analysis_output,
                                error_context=error_context
                            )

                            # Save and execute retry script (same logic as above)
                            retry_timestamp = get_timestamp().replace(" ", "_").replace(":", "-")
                            retry_viz_script_name = f"visualization_{analysis_name}_{retry_timestamp}_retry{viz_attempt-1}.py"
                            retry_viz_script_path = viz_dir / retry_viz_script_name

                            cleaned_retry_viz_code = clean_code_content(retry_viz_code)
                            with open(retry_viz_script_path, 'w') as f:
                                f.write(cleaned_retry_viz_code)

                            logger.info(f"Saved no-output retry visualization script to: {retry_viz_script_path}")

                            # Execute retry
                            retry_execution_result = await analysis_executor.execute_analysis(
                                script_path=retry_viz_script_path,
                                question={"raw_text": f"Visualization retry for {analysis_name}"},
                                parameters=config.get("analysis_parameters", {}),
                                execution_type="visualization"
                            )

                            if retry_execution_result.get("success", False):
                                retry_viz_files = retry_execution_result.get("visualization_files", [])
                                if retry_viz_files:
                                    # Success! Same processing logic as main retry
                                    analysis_number = analysis_name.split("_")[1] if "_" in analysis_name else "unknown"
                                    moved_files = []
                                    for viz_file_path in retry_viz_files:
                                        try:
                                            original_path = Path(viz_file_path)
                                            if original_path.exists():
                                                original_name = original_path.name
                                                new_filename = f"a{analysis_number}_{original_name}"
                                                new_path = viz_dir / new_filename
                                                import shutil
                                                shutil.move(str(original_path), str(new_path))
                                                moved_files.append(str(new_path))
                                                logger.info(f"Moved no-output retry visualization file: {original_path} → {new_path}")
                                        except Exception as e:
                                            logger.warning(f"Failed to move no-output retry visualization file {viz_file_path}: {e}")

                                    if moved_files:
                                        visualization_files[analysis_name] = moved_files[0]
                                        logger.info(f"Successfully generated no-output retry visualization for {analysis_name}: {moved_files[0]}")
                                        print(f"[AUTOINTERP] ✓ No-output retry visualization succeeded for {analysis_name}")

                                        # Evaluate the successful retry
                                        await evaluate_and_retry_visualization(
                                            analysis_name=analysis_name,
                                            analysis_number=int(analysis_number) if analysis_number.isdigit() else 1,
                                            viz_code=retry_viz_code,
                                            framework=framework,
                                            logger=logger
                                        )
                                        break  # Success - exit retry loop
                                else:
                                    logger.warning(f"No-output retry still produced no images for {analysis_name}")
                            else:
                                retry_error_msg = retry_execution_result.get("error", "Unknown error")
                                logger.error(f"No-output retry visualization failed for {analysis_name}: {retry_error_msg}")
                                print(f"[AUTOINTERP] ✗ No-output retry {viz_attempt} failed for {analysis_name}: {retry_error_msg}")

                            viz_attempt += 1

                        except Exception as e:
                            logger.error(f"Error during no-output visualization retry for {analysis_name}: {str(e)}")
                            print(f"[AUTOINTERP] ✗ Error during no-output visualization retry: {str(e)}")
                            viz_attempt += 1
                            continue

                    if viz_attempt > max_total_viz_attempts:
                        logger.warning(f"Max retries reached for no-output visualization {analysis_name} - skipping")
                        print(f"[AUTOINTERP] ⚠ Max retries reached for no-output visualization {analysis_name} - skipping")
            else:
                error_msg = execution_result.get("error", "Unknown error")
                logger.error(f"Failed to execute visualization script for {analysis_name}: {error_msg}")
                print(f"[AUTOINTERP] ✗ Visualization failed for {analysis_name}: {error_msg}")

                # Implement retry logic for visualization failures similar to analysis retries
                max_viz_retries = config.get("execution", {}).get("max_retries", 1)
                max_total_viz_attempts = max_viz_retries + 1
                viz_attempt = 2  # We already tried once
                last_error = error_msg  # Track last error to detect repeating failures

                while viz_attempt <= max_total_viz_attempts:
                    logger.info(f"Retrying visualization generation for {analysis_name} (attempt {viz_attempt}/{max_total_viz_attempts})")
                    print(f"[AUTOINTERP] Retrying visualization generation for {analysis_name} (attempt {viz_attempt}/{max_total_viz_attempts})...")

                    try:
                        # Increment attempt counter in viz generator
                        viz_generator.increment_attempt()

                        # Create error context with execution failure details (standardized format)
                        error_context = {
                            "error": error_msg,
                            "stderr": execution_result.get("stderr", ""),
                            "stdout": execution_result.get("stdout", ""),
                            "previous_script": str(viz_script_path),
                            "visualization_failed": True,
                            "error_type": "execution_failure"
                        }

                        # Generate new visualization code with error context
                        retry_viz_code = await viz_generator.generate_visualization(
                            analysis_script,
                            analysis_output,
                            error_context=error_context
                        )

                        # Save retry visualization script
                        retry_timestamp = get_timestamp().replace(" ", "_").replace(":", "-")
                        retry_viz_script_name = f"visualization_{analysis_name}_{retry_timestamp}_retry{viz_attempt-1}.py"
                        retry_viz_script_path = viz_dir / retry_viz_script_name

                        # Clean and write the retry visualization code
                        cleaned_retry_viz_code = clean_code_content(retry_viz_code)
                        with open(retry_viz_script_path, 'w') as f:
                            f.write(cleaned_retry_viz_code)

                        logger.info(f"Saved retry visualization script to: {retry_viz_script_path}")

                        # Execute retry visualization script
                        retry_execution_result = await analysis_executor.execute_analysis(
                            script_path=retry_viz_script_path,
                            question={"raw_text": f"Visualization retry for {analysis_name}"},
                            parameters=config.get("analysis_parameters", {}),
                            execution_type="visualization"
                        )

                        if retry_execution_result.get("success", False):
                            # Retry succeeded - process the results
                            retry_viz_files = retry_execution_result.get("visualization_files", [])
                            if retry_viz_files:
                                analysis_number = analysis_name.split("_")[1] if "_" in analysis_name else "unknown"

                                # Move and rename visualization files
                                moved_files = []
                                for viz_file_path in retry_viz_files:
                                    try:
                                        original_path = Path(viz_file_path)
                                        if original_path.exists():
                                            original_name = original_path.name

                                            # Add prefix to new files
                                            new_filename = f"a{analysis_number}_{original_name}"
                                            new_path = viz_dir / new_filename

                                            # Move the file
                                            import shutil
                                            shutil.move(str(original_path), str(new_path))
                                            moved_files.append(str(new_path))
                                            logger.info(f"Moved retry visualization file: {original_path} → {new_path}")
                                    except Exception as e:
                                        logger.warning(f"Failed to move retry visualization file {viz_file_path}: {e}")

                                if moved_files:
                                    # Success! Update the visualization files dict
                                    visualization_files[analysis_name] = moved_files[0]
                                    logger.info(f"Successfully generated retry visualization for {analysis_name}: {moved_files[0]}")
                                    print(f"[AUTOINTERP] ✓ Retry visualization succeeded for {analysis_name}")

                                    # Evaluate the successful retry visualization
                                    await evaluate_and_retry_visualization(
                                        analysis_name=analysis_name,
                                        analysis_number=int(analysis_number) if analysis_number.isdigit() else 1,
                                        viz_code=retry_viz_code,
                                        framework=framework,
                                        logger=logger
                                    )
                                    break  # Success - exit retry loop
                                else:
                                    logger.warning(f"Retry visualization generated files but failed to move them for {analysis_name}")
                            else:
                                logger.warning(f"Retry visualization script executed but no image files generated for {analysis_name}")
                        else:
                            # Retry failed too
                            retry_error_msg = retry_execution_result.get("error", "Unknown error")
                            logger.error(f"Retry visualization failed for {analysis_name}: {retry_error_msg}")
                            print(f"[AUTOINTERP] ✗ Retry {viz_attempt} failed for {analysis_name}: {retry_error_msg}")

                            # Check if we're getting the same error repeatedly
                            if retry_error_msg == last_error:
                                logger.warning(f"Same error repeated for {analysis_name}, may need different approach")
                            last_error = retry_error_msg

                        viz_attempt += 1

                    except Exception as e:
                        logger.error(f"Error during visualization retry for {analysis_name}: {str(e)}")
                        print(f"[AUTOINTERP] ✗ Error during visualization retry: {str(e)}")
                        viz_attempt += 1
                        continue

                if viz_attempt > max_total_viz_attempts:
                    logger.warning(f"Max retries reached for visualization {analysis_name} - skipping")
                    print(f"[AUTOINTERP] ⚠ Max retries reached for visualization {analysis_name} - skipping")
                
        except Exception as e:
            logger.error(f"Error generating visualization for {analysis_name}: {str(e)}")
            print(f"[AUTOINTERP] ✗ Error generating visualization for {analysis_name}: {str(e)}")
            continue
    
    logger.info(f"Generated {len(visualization_files)} visualizations successfully")
    print(f"[AUTOINTERP] Generated {len(visualization_files)}/{len(successful_analyses)} visualizations successfully")
    
    return visualization_files


async def generate_report(
    active_question: Union[str, Dict[str, Any]],  # Can be raw text or dict
    all_analyses: List[Dict[str, Any]],
    all_evaluations: List[Dict[str, Any]],
    report_generator: ReportGenerator,
    config: Dict[str, Any],
    logger: Any,
    framework: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a final report based on all analyses and evaluations.
    
    Args:
        active_question: The question that was tested
        all_analyses: List of all analysis results
        all_evaluations: List of all evaluation results
        report_generator: Report generator component
        config: Configuration dictionary
        logger: Logging instance
        framework: Framework components for visualization generation
        
    Returns:
        Report generation results
    """
    logger.info("Starting report generation...")
    print(f"[AUTOINTERP] PHASE 4/4: Report Generation")
    
    # Get final confidence from the latest evaluation result
    final_confidence = 0.0  # Default value
    if all_evaluations and len(all_evaluations) > 0:
        # Get the latest evaluation's confidence
        final_confidence = all_evaluations[-1].get("new_confidence", 0.0)
        logger.info(f"Using final confidence {final_confidence} from latest evaluation")
    
    # Determine conclusion type based on confidence
    conclusion_type = "inconclusive"
    conclusion_text = "INCONCLUSIVE"
    
    # If confidence is high enough to answer the question
    confidence_threshold = config.get("analysis", {}).get("confidence_threshold", 0.8)
    if final_confidence >= confidence_threshold:
        conclusion_type = "concluded"
        conclusion_text = "CONCLUDED"


    print(f"[AUTOINTERP] Final conclusion: Investigation is {conclusion_text}")
    print(f"[AUTOINTERP] Final confidence: {final_confidence:.2f}")
    
    # Prepare combined analysis results for the report
    combined_analysis_result = {
        "analyses": all_analyses,
        "latest_analysis": all_analyses[-1] if all_analyses else {},
        "analysis_count": len(all_analyses),
        "final_confidence": final_confidence
    }
    
    combined_evaluation_result = {
        "evaluations": all_evaluations,
        "latest_evaluation": all_evaluations[-1] if all_evaluations else {},
        "conclusion": conclusion_type,
        "final_confidence": final_confidence
    }
    
    # Include confidence statement
    confidence_statement = f"After {len(all_analyses)} analyses, we have {final_confidence:.2f} confidence that the question is {conclusion_text}."
    combined_evaluation_result["confidence_statement"] = confidence_statement
    
    # Generate visualizations for successful analyses
    pipeline_ui = framework.get("pipeline_ui")
    if pipeline_ui:
        pipeline_ui.step_start("visualization")
    print(f"[AUTOINTERP] Generating visualizations for completed analyses...")
    logger.info("Starting visualization generation phase")
    
    # Find successful analyses and extract their files
    path_resolver = framework["path_resolver"]
    successful_analyses = find_successful_analyses(path_resolver)
    
    # Generate visualizations using the full pipeline
    visualizations = await generate_visualizations(successful_analyses, framework)

    # Interactive feedback on visualizations
    _viz_feedback = ""
    if is_interactive(config) and visualizations:
        viz_summary = f"Generated {len(visualizations)} visualizations:\n"
        for name, path in visualizations.items():
            viz_summary += f"  - {name}: {path}\n"
        print(f"\n[INTERACTIVE] === Visualizations ===\n{viz_summary}")
        try:
            _viz_feedback = input("[INTERACTIVE] Feedback on visualizations (Enter to continue): ").strip()
            if _viz_feedback:
                print(f"[INTERACTIVE] Visualization feedback noted — will be passed to report generation.")
        except (EOFError, KeyboardInterrupt):
            pass
    # Store viz feedback in config so report agent/generator can use it
    if _viz_feedback:
        config["_visualization_feedback"] = _viz_feedback

    if pipeline_ui:
        pipeline_ui.step_complete("visualization", summary=f"{len(visualizations)} visualizations")
        pipeline_ui.step_start("report_generation")

    # ------------------------------------------------------------------
    # Decide whether to use agent or legacy report generation
    # ------------------------------------------------------------------
    import shutil as _shutil
    reporting_cfg = config.get("reporting", {})
    _report_use_agent = reporting_cfg.get("use_agent", False)
    _provider = (config.get("llm", {}).get("provider") or "").lower()
    _report_agent_available = (
        _report_use_agent
        and _provider in ("anthropic", "openai")
        and _shutil.which("claude" if _provider == "anthropic" else "codex") is not None
    )

    if _report_agent_available:
        # --- Agent-based report generation ---
        print("[AUTOINTERP] Using CLI agent for report generation")
        logger.info("Using CLI agent for report generation (provider=%s)", _provider)

        try:
            prompt_template = load_report_prompt_template()
        except FileNotFoundError as exc:
            logger.error("Could not load report prompt template: %s", exc)
            print(f"[AUTOINTERP] ERROR: {exc}; falling back to legacy reporter")
            _report_agent_available = False  # fall through to legacy below

        if _report_agent_available:
            prompt_text = _build_report_prompt(prompt_template)
            agent_timeout = reporting_cfg.get("agent_timeout", 900)

            _report_progress_cb = None
            if pipeline_ui:
                def _report_progress_cb(msg):
                    pipeline_ui.step_progress("report_generation", msg)

            agent_result = run_report_agent(
                provider=_provider,
                project_dir=path_resolver.get_project_dir(),
                prompt_text=prompt_text,
                timeout=agent_timeout,
                on_progress=_report_progress_cb,
            )

            outputs = read_report_outputs(path_resolver.get_project_dir())
            report_path = outputs.get("report_path")

            if report_path:
                logger.info("Agent generated report at %s", report_path)
                print(f"[AUTOINTERP] Agent generated report at {report_path}")

                # Interactive checkpoint on the agent-generated report
                if is_interactive(config):
                    try:
                        report_content = Path(report_path).read_text(encoding="utf-8")
                        _report_llm = framework.get("llm_interface") or LLMInterface(config, agent_name="reporter")

                        async def _revise_report_agent(current: str, feedback: str) -> str:
                            return await make_revision_call(
                                _report_llm, "reporter", current, feedback, "report"
                            )

                        def _save_report_agent(text: str) -> None:
                            Path(report_path).write_text(text, encoding="utf-8")

                        await interactive_checkpoint(
                            "Report", report_content, _revise_report_agent, _save_report_agent, config
                        )
                    except Exception as e:
                        logger.warning(f"Interactive checkpoint failed for agent report: {e}")

                if pipeline_ui:
                    pipeline_ui.step_complete("report_generation", summary=str(report_path))
                return {
                    "report_path": report_path,
                    "conclusion": conclusion_type,
                    "final_confidence": final_confidence,
                }
            else:
                logger.warning("Report agent finished but no report .md found; falling back to legacy reporter")
                print("[AUTOINTERP] Report agent did not produce a report; falling back to legacy reporter")

    # --- Legacy report generation (or fallback from agent failure) ---
    if _report_use_agent and not _report_agent_available:
        if _provider in ("anthropic", "openai"):
            cli_name = "claude" if _provider == "anthropic" else "codex"
            print(f"[AUTOINTERP] Agent mode enabled but '{cli_name}' CLI not found; falling back to legacy reporter")
        elif _provider:
            print(f"[AUTOINTERP] Agent mode not supported for provider '{_provider}'; using legacy reporter")

    try:
        task_description = config.get("task", {}).get("description", "")
        task_name = task_description[:50] + "..." if len(task_description) > 50 else task_description or "Unnamed Task"
        # Pass the active_question as plain text directly
        report_path = await report_generator.generate_report(
            question=active_question,  # Plain text, not a dictionary
            analysis_results=combined_analysis_result,
            evaluation_results=combined_evaluation_result,
            visualizations=visualizations,
            title=None  # Let report generator generate the title
        )

        logger.info(f"Generated report at {report_path}")
        print(f"[AUTOINTERP] Generated comprehensive report at {report_path}")

        # Interactive checkpoint on the legacy-generated report
        if is_interactive(config) and report_path and Path(report_path).exists():
            try:
                report_content = Path(report_path).read_text(encoding="utf-8")
                _legacy_llm = framework.get("llm_interface") or LLMInterface(config, agent_name="reporter")

                async def _revise_report_legacy(current: str, feedback: str) -> str:
                    return await make_revision_call(
                        _legacy_llm, "reporter", current, feedback, "report"
                    )

                def _save_report_legacy(text: str) -> None:
                    Path(report_path).write_text(text, encoding="utf-8")

                await interactive_checkpoint(
                    "Report", report_content, _revise_report_legacy, _save_report_legacy, config
                )
            except Exception as e:
                logger.warning(f"Interactive checkpoint failed for legacy report: {e}")

        if pipeline_ui:
            pipeline_ui.step_complete("report_generation", summary=str(report_path))

        return {
            "report_path": report_path,
            "conclusion": conclusion_type,
            "final_confidence": final_confidence
        }
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        print(f"[AUTOINTERP] Error generating report: {str(e)}")
        if pipeline_ui:
            pipeline_ui.step_failed("report_generation", error=str(e))

        # Create a simple markdown report as fallback
        simple_report = f"# Interpretability Report: {task_name}\n\n"
        simple_report += f"## Question\n"

        simple_report += f"{active_question}\n\n"

        simple_report += f"## Conclusion\nThe question is {conclusion_text} with {final_confidence:.2f} confidence.\n\n"
        simple_report += f"## Analysis Summary\nPerformed {len(all_analyses)} analyses.\n\n"

        if all_evaluations and "key_insights" in all_evaluations[-1]:
            simple_report += "## Key Insights\n"
            for insight in all_evaluations[-1]["key_insights"]:
                simple_report += f"- {insight}\n"

        # Get the latest report path from the path resolver
        path_resolver = PathResolver()
        reports_dir = path_resolver.ensure_path("reports")
        report_path = reports_dir / "simple_report.md"

        # Ensure the directory exists
        reports_dir.mkdir(exist_ok=True, parents=True)

        # Save simple report
        with open(report_path, "w") as f:
            f.write(simple_report)

        logger.info(f"Generated simple report at {report_path}")
        print(f"[AUTOINTERP] Generated simple report at {report_path}")

        return {
            "report_path": str(report_path),
            "conclusion": conclusion_type,
            "final_confidence": final_confidence,
            "is_fallback": True
        }

async def streamlined_pipeline(framework: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the streamlined pipeline for interpretability research.
    
    Args:
        framework: Dictionary with framework components
        
    Returns:
        Dictionary with pipeline results
    """
    logger = framework["logger"]
    config = framework["config"]
    
    # Setup console output logging immediately to capture all pipeline output
    path_resolver = framework["path_resolver"]
    project_dir = path_resolver.get_project_dir()
    from AutoInterp.core.utils import setup_console_logging_to_file
    setup_console_logging_to_file(project_dir)
    
    # Generate a task name from the description for logging
    task_description = config.get('task', {}).get('description', '')
    task_name = task_description[:50] + "..." if len(task_description) > 50 else task_description or 'Unnamed Task'
    logger.info(f"Starting streamlined pipeline for task: {task_name}")

    # Get optional PipelineUI
    pipeline_ui = framework.get("pipeline_ui")
    # Store in config so prioritize_questions (which renames the project) can update it
    if pipeline_ui:
        config["_pipeline_ui_ref"] = pipeline_ui

    if pipeline_ui:
        pipeline_ui.pipeline_start(task_name)
    else:
        print("\n" + "="*80)
        print(f"[AUTOINTERP] Starting task execution - {get_timestamp()}")
        print(f"[AUTOINTERP] Task: {task_name}")
        print("="*80 + "\n")

    use_context_pack_question = False

    try:
        # 1. Question Generation (via context pack or direct LLM)
        if pipeline_ui:
            pipeline_ui.step_start("question_generation")

        ctx_cfg = config.get("context_pack", {}) or {}
        if ctx_cfg.get("enabled", False):
            print("[AUTOINTERP] Context pack enabled: building 3-paper pack and generating question...")
            _ctx_start = time.time() if pipeline_ui else None
            _ctx_prompt_used = ""
            try:
                arxiv_interp_root = PACKAGE_ROOT / "arxiv_interp_graph"
                if str(arxiv_interp_root) not in sys.path:
                    sys.path.insert(0, str(arxiv_interp_root))
                graph_path = ctx_cfg.get("graph") or (PACKAGE_ROOT / "arxiv_interp_graph" / "output" / "graph_state.json")
                graph_path = Path(graph_path)
                if not graph_path.is_absolute():
                    graph_path = (PACKAGE_ROOT / graph_path).resolve()
                questions_dir = path_resolver.ensure_path("questions")
                literature_dir = path_resolver.ensure_path("literature")
                if graph_path.exists():
                    from context_pack.sampling import build_context_pack
                    from context_pack.download import download_context_pack_pdfs, write_manifest
                    from context_pack.agent_questions import run_agent_question_generation
                    from context_pack.run import _generate_question_llm
                    from api_client import SemanticScholarClient
                    from context_pack.llm_client import get_llm_generate_fn
                    from context_pack.run import _load_graph

                    s2_client = SemanticScholarClient()

                    # Load graph and sample 3 papers
                    G = _load_graph(graph_path)
                    papers = build_context_pack(
                        G,
                        seed_id=ctx_cfg.get("seed_id"),
                        s2_client=s2_client,
                        seed=ctx_cfg.get("seed"),
                    )

                    question_text = ""
                    if len(papers) >= 3:
                        # Download PDFs to literature/pdfs/
                        papers = download_context_pack_pdfs(papers, literature_dir, s2_client)
                        write_manifest(papers, literature_dir)

                        # Determine provider
                        llm_config = config.get("llm") or {}
                        if not llm_config and (PACKAGE_ROOT / ".last_llm.json").exists():
                            with open(PACKAGE_ROOT / ".last_llm.json") as f:
                                llm_config = json.load(f)
                        provider = (llm_config.get("provider") or "").lower()

                        # Try agent-based generation if enabled and provider supports it
                        use_agent = ctx_cfg.get("use_agent", True)
                        agent_timeout = ctx_cfg.get("agent_timeout", 600)

                        if use_agent and provider in ("anthropic", "openai"):
                            prompt_template = config.get("prompts", {}).get("question_manager", {}).get("agent_question_generator", {}).get("prompt_template", "")
                            if not prompt_template:
                                prompt_template = "There are scientific articles in the directory (dir). Read them and devise three research questions about LLM interpretability. Write them to Research_Questions.txt"
                            _ctx_prompt_used = prompt_template.replace("(dir)", str((literature_dir / "pdfs").resolve()))
                            _qgen_progress_cb = None
                            if pipeline_ui:
                                def _qgen_progress_cb(msg):
                                    pipeline_ui.step_progress("question_generation", msg)
                            question_text = run_agent_question_generation(
                                provider=provider,
                                literature_dir=literature_dir,
                                prompt_template=prompt_template,
                                timeout=agent_timeout,
                                on_progress=_qgen_progress_cb,
                            ) or ""

                        # Fallback to LLM API call if agent didn't produce output
                        if not question_text:
                            if provider in ("anthropic", "openai"):
                                print("[AUTOINTERP] Agent generation failed; falling back to LLM API call.")
                            llm_generate_fn = None
                            if llm_config and (llm_config.get("provider") or llm_config.get("model")):
                                llm_generate_fn = get_llm_generate_fn(
                                    provider=llm_config.get("provider"),
                                    model=llm_config.get("model"),
                                )
                            if llm_generate_fn:
                                _ctx_prompt_used = _ctx_prompt_used or "(LLM API fallback with paper excerpts)"
                                question_text = _generate_question_llm(papers, llm_generate_fn) or ""

                    if question_text:
                        (questions_dir / "questions.txt").write_text(question_text, encoding="utf-8")
                        use_context_pack_question = True
                        print(f"[AUTOINTERP] Context pack done: 3 papers, manifest + PDFs in literature/, question in questions/questions.txt")

                        # Interactive checkpoint: let user revise context-pack questions
                        if is_interactive(config):
                            try:
                                _cp_llm = framework["llm_interface"]
                                _cp_questions_file = questions_dir / "questions.txt"

                                async def _revise_cp_questions(current: str, feedback: str) -> str:
                                    return await make_revision_call(
                                        _cp_llm, "question_generator", current, feedback, "question_generation"
                                    )

                                def _save_cp_questions(text: str) -> None:
                                    nonlocal question_text
                                    question_text = text
                                    _cp_questions_file.write_text(text, encoding="utf-8")

                                question_text = await interactive_checkpoint(
                                    "Context Pack Questions", question_text, _revise_cp_questions, _save_cp_questions, config
                                )
                            except Exception as e:
                                logger.warning(f"Interactive checkpoint failed for context pack questions: {e}")

                        # Record the context pack interaction in the dashboard
                        if pipeline_ui:
                            _ctx_duration = time.time() - _ctx_start if _ctx_start else 0
                            _ctx_model = f"{provider} CLI agent" if use_agent and provider in ("anthropic", "openai") else llm_config.get("model", "unknown")
                            pipeline_ui.llm_call_complete(
                                agent_name="context_pack_agent",
                                display_name="Context Pack (Literature → Questions)",
                                prompt=_ctx_prompt_used or "(agent prompt — see literature/pdfs/)",
                                system_message=None,
                                response=question_text,
                                model=_ctx_model,
                                provider=provider or "unknown",
                                temperature=0,
                                max_tokens=0,
                                duration_seconds=_ctx_duration,
                                step_id="question_generation",
                            )
                    else:
                        print("[AUTOINTERP] Context pack produced no question; falling back to normal question generation.")
                else:
                    print(f"[AUTOINTERP] Context pack skipped: graph not found at {graph_path}")
            except Exception as e:
                logger.warning(f"Context pack failed: {e}")
                import traceback
                traceback.print_exc()
                print("[AUTOINTERP] Falling back to normal question generation.")

        # If context pack didn't produce a question, use standard LLM question generation
        if not use_context_pack_question:
            await generate_questions(
                llm_interface=framework["llm_interface"],
                question_manager=framework["question_manager"],
                config=config,
                logger=logger
            )

        if pipeline_ui:
            qg_summary = "from literature (context pack)" if use_context_pack_question else "LLM-generated"
            pipeline_ui.step_complete("question_generation", summary=qg_summary)

        # 2. Question Prioritization (always runs)
        if pipeline_ui:
            pipeline_ui.step_start("question_prioritization")
        active_question = await prioritize_questions(
            question_manager=framework["question_manager"],
            llm_interface=framework["llm_interface"],
            config=config,
            logger=logger,
            evaluator=framework["evaluator"]
        )
        if pipeline_ui:
            # Show first 80 chars of the selected question as summary
            summary = active_question[:80] + "..." if len(active_question) > 80 else active_question
            pipeline_ui.step_complete("question_prioritization", summary=summary)

        # 3. Iterative Analysis and Evaluation
        if pipeline_ui:
            pipeline_ui.step_start("iterative_analysis")

        # Decide: agent mode vs legacy pipeline
        import shutil as _shutil
        analysis_cfg = config.get("analysis", {})
        _use_agent = analysis_cfg.get("use_agent", False)
        _provider = (config.get("llm", {}).get("provider") or "").lower()
        _agent_available = (
            _use_agent
            and _provider in ("anthropic", "openai")
            and _shutil.which("claude" if _provider == "anthropic" else "codex") is not None
        )

        if _agent_available:
            print("[AUTOINTERP] Using CLI agent for analysis")
            all_analyses, all_evaluations = await iterative_analysis_agent(
                active_question=active_question,
                config=config,
                logger=logger,
                max_iterations=analysis_cfg.get("max_iterations", 3),
                confidence_threshold=analysis_cfg.get("confidence_threshold", 0.85),
            )
        else:
            if _use_agent and _provider in ("anthropic", "openai"):
                cli_name = "claude" if _provider == "anthropic" else "codex"
                print(f"[AUTOINTERP] Agent mode enabled but '{cli_name}' CLI not found; falling back to legacy pipeline")
            elif _use_agent:
                print(f"[AUTOINTERP] Agent mode not supported for provider '{_provider}'; using legacy pipeline")
            all_analyses, all_evaluations = await iterative_analysis(
                active_question=active_question,
                analysis_generator=framework["analysis_generator"],
                analysis_executor=framework["analysis_executor"],
                analysis_planner=framework["analysis_planner"],
                evaluator=framework["evaluator"],
                question_manager=framework["question_manager"],
                config=config,
                logger=logger,
                max_iterations=analysis_cfg.get("max_iterations", 3),
                confidence_threshold=analysis_cfg.get("confidence_threshold", 0.8),
            )
        if pipeline_ui:
            final_conf = all_evaluations[-1].get("new_confidence", 0.0) if all_evaluations else 0.0
            pipeline_ui.step_complete(
                "iterative_analysis",
                summary=f"{len(all_analyses)} analyses, confidence: {final_conf:.2f}"
            )

        # 4. Report Generation
        # In the TXT-based approach, the active question is simply the prioritized_question.txt file
        logger.info("Loading prioritized question text file as active question")
        prioritized_path = framework["question_manager"].storage_dir / "prioritized_question.txt"
        if prioritized_path.exists():
            try:
                with open(prioritized_path, 'r') as f:
                    active_question = f.read()
                logger.info(f"Loaded prioritized question from {prioritized_path}")
            except Exception as e:
                logger.error(f"Error loading prioritized question: {e}")
                active_question = "No question available"
        else:
            logger.warning(f"Prioritized question file not found at {prioritized_path}")
            active_question = "No question available"
        
        report_result = await generate_report(
            active_question=active_question,
            all_analyses=all_analyses,
            all_evaluations=all_evaluations,
            report_generator=framework["report_generator"],
            config=config,
            logger=logger,
            framework=framework
        )

        # ------------------------------------------------------------------
        # AutoCritique — optional automated peer review
        # ------------------------------------------------------------------
        import shutil as _shutil_ac
        autocritique_cfg = config.get("autocritique", {})
        _ac_enabled = autocritique_cfg.get("enabled", False)
        _ac_use_agent = autocritique_cfg.get("use_agent", True)
        _ac_provider = (config.get("llm", {}).get("provider") or "").lower()
        _ac_review_path = None

        if _ac_enabled and _ac_use_agent:
            _ac_cli_name = "claude" if _ac_provider == "anthropic" else "codex"
            _ac_agent_available = (
                _ac_provider in ("anthropic", "openai")
                and _shutil_ac.which(_ac_cli_name) is not None
            )

            if _ac_agent_available:
                print("[AUTOINTERP] PHASE 5: AutoCritique (automated peer review)")
                logger.info("Starting AutoCritique (provider=%s)", _ac_provider)
                if pipeline_ui:
                    pipeline_ui.step_start("autocritique")

                try:
                    _ac_prompt_template = load_autocritique_prompt_template()
                    _ac_prompt_text = _build_autocritique_prompt(_ac_prompt_template)
                    _ac_timeout = autocritique_cfg.get("agent_timeout", 600)

                    _ac_progress_cb = None
                    if pipeline_ui:
                        def _ac_progress_cb(msg):
                            pipeline_ui.step_progress("autocritique", msg)

                    _ac_result = run_autocritique_agent(
                        provider=_ac_provider,
                        project_dir=path_resolver.get_project_dir(),
                        prompt_text=_ac_prompt_text,
                        timeout=_ac_timeout,
                        on_progress=_ac_progress_cb,
                    )

                    _ac_outputs = read_autocritique_outputs(path_resolver.get_project_dir())
                    _ac_review_path = _ac_outputs.get("review_path")

                    if _ac_review_path:
                        logger.info("AutoCritique review at %s", _ac_review_path)
                        print(f"[AUTOINTERP] AutoCritique review at {_ac_review_path}")
                        if pipeline_ui:
                            pipeline_ui.step_complete("autocritique", summary=str(_ac_review_path))
                    else:
                        logger.warning("AutoCritique agent finished but no review file found")
                        print("[AUTOINTERP] AutoCritique agent did not produce a review")
                        if pipeline_ui:
                            pipeline_ui.step_failed("autocritique", error="No review file produced")

                except Exception as _ac_exc:
                    logger.error("AutoCritique failed: %s", _ac_exc)
                    print(f"[AUTOINTERP] AutoCritique failed: {_ac_exc}")
                    if pipeline_ui:
                        pipeline_ui.step_failed("autocritique", error=str(_ac_exc))
            else:
                if _ac_provider in ("anthropic", "openai"):
                    print(f"[AUTOINTERP] AutoCritique: '{_ac_cli_name}' CLI not found; skipping")
                else:
                    print(f"[AUTOINTERP] AutoCritique: provider '{_ac_provider}' not supported; skipping")
                if pipeline_ui:
                    pipeline_ui.step_skipped("autocritique", reason=f"CLI not available for {_ac_provider}")
        elif _ac_enabled and not _ac_use_agent:
            print("[AUTOINTERP] AutoCritique: agent mode disabled; skipping")
            if pipeline_ui:
                pipeline_ui.step_skipped("autocritique", reason="agent mode disabled")
        else:
            if pipeline_ui:
                pipeline_ui.step_skipped("autocritique", reason="disabled")

        # Print task completion message
        result = {
            "status": "completed",
            "task_name": task_name,
            "report_path": report_result.get("report_path"),
            "autocritique_review_path": _ac_review_path,
            "conclusion": report_result.get("conclusion"),
            "final_confidence": report_result.get("final_confidence")
        }

        if pipeline_ui:
            pipeline_ui.pipeline_complete(result)
        else:
            print("\n" + "="*80)
            print(f"[AUTOINTERP] Task execution completed - {get_timestamp()}")
            if "report_path" in report_result and report_result["report_path"]:
                print(f"[AUTOINTERP] Report generated: {report_result['report_path']}")
            print("="*80 + "\n")

        # Final logging
        logger.info(f"Completed interpretability research pipeline. Final conclusion: Question is {report_result.get('conclusion', 'inconclusive').upper()} with {report_result.get('final_confidence', 0.5):.2f} confidence.")

        return result
    
    except Exception as e:
        # Get full error message including traceback
        import traceback
        full_traceback = traceback.format_exc()
        error_message = str(e)

        # Log detailed error information to logger
        logger.error(f"Error in streamlined pipeline: {str(e)}")
        logger.error(full_traceback)

        # Notify PipelineUI
        if pipeline_ui:
            pipeline_ui.pipeline_failed(error_message)
        else:
            # Print full error message and traceback to console
            print(f"\n[AUTOINTERP] ERROR: Task execution failed")
            print(f"[AUTOINTERP] Error details: {error_message}")
            print(f"[AUTOINTERP] Full traceback:\n{full_traceback}")

        # Log the pipeline failure
        logger.error(f"Pipeline execution failed: {error_message}")
        logger.error(f"Full traceback: {full_traceback}")

        return {
            "status": "failed",
            "error": error_message,
            "full_error": full_traceback,
            "task_name": task_name
        }

def build_argument_parser() -> argparse.ArgumentParser:
    """
    Create the CLI argument parser for the AutoInterp framework.
    """
    parser = argparse.ArgumentParser(description="AutoInterp Agent Framework")
    parser.add_argument("--config", help="Path to override configuration file", default=None)
    parser.add_argument("--venv", help="Path to existing virtual environment to use", default=None)
    parser.add_argument(
        "--projects-dir",
        help="Root directory for generated projects (defaults to the package projects folder)",
        default=None,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run (default: run)")
    # Default: run the full interpretability pipeline
    run_parser = subparsers.add_parser("run", help="Run the full interpretability research pipeline (default)")
    run_parser.add_argument("--config", help="Path to override configuration file", default=None)
    run_parser.add_argument("--venv", help="Path to existing virtual environment to use", default=None)
    run_parser.add_argument("--projects-dir", help="Root directory for projects", default=None)
    run_parser.set_defaults(command="run")

    # context-pack: seed + forward/backward -> 3 papers -> PDFs + manifest -> optional LLM question
    ctx_parser = subparsers.add_parser("context-pack", help="Build 3-paper context pack (seed + citing + cited), PDFs + manifest, optional research question")
    ctx_parser.add_argument("--output-dir", default=None, help="Output directory; default: auto from generated question (projects/<slug>_<timestamp>/questions)")
    ctx_parser.add_argument("--graph", default=None, help="Path to graph_state.json (default: arxiv_interp_graph/output/graph_state.json)")
    ctx_parser.add_argument("--seed-id", default=None, help="Seed paper ID (default: random)")
    ctx_parser.add_argument("--seed", type=int, default=None, help="Random seed (default: different each run; set e.g. 42 for reproducibility)")
    ctx_parser.add_argument("--no-download", action="store_true", help="Do not download PDFs")
    ctx_parser.add_argument("--no-llm", action="store_true", help="Do not generate research question")
    ctx_parser.set_defaults(command="context-pack")

    parser.set_defaults(command="run")
    return parser


def run_context_pack_cmd(args: argparse.Namespace) -> None:
    """Build 3-paper context pack (seed + forward/backward), PDFs + manifest, optional LLM question."""
    pkg_root = Path(__file__).resolve().parent
    arxiv_interp_root = pkg_root / "arxiv_interp_graph"
    if str(arxiv_interp_root) not in sys.path:
        sys.path.insert(0, str(arxiv_interp_root))
    try:
        from context_pack.run import run_context_pack
        from api_client import SemanticScholarClient
    except ImportError as e:
        print(f"[AUTOINTERP] ERROR: Cannot load context_pack (is arxiv_interp_graph present?): {e}")
        sys.exit(1)
    graph_path = args.graph
    if not graph_path:
        graph_path = pkg_root / "arxiv_interp_graph" / "output" / "graph_state.json"
    graph_path = Path(graph_path)
    if not graph_path.is_absolute():
        graph_path = (pkg_root / graph_path).resolve()
    if not graph_path.exists():
        print(f"[AUTOINTERP] ERROR: Graph not found: {graph_path}")
        sys.exit(1)
    if args.output_dir is None:
        ts = get_timestamp("%Y-%m-%dT%H-%M-%S")
        output_dir = (PACKAGE_ROOT / "projects" / f"context_pack_{ts}" / "questions").resolve()
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[AUTOINTERP] Context pack output: {output_dir}")
    client = SemanticScholarClient()
    llm_generate_fn = None
    if getattr(args, "no_llm", False):
        print("[AUTOINTERP] Skipping question generation (--no-llm). No context_pack_question.txt will be written.")
    if not getattr(args, "no_llm", False):
        try:
            last_llm = pkg_root / ".last_llm.json"
            cfg_path = pkg_root / "config.yaml"
            llm_config = {}
            if last_llm.exists():
                with open(last_llm) as f:
                    llm_config = json.load(f)
            elif cfg_path.exists():
                import yaml
                with open(cfg_path) as f:
                    root = yaml.safe_load(f) or {}
                llm_config = root.get("llm") or {}
            if llm_config and (llm_config.get("provider") or llm_config.get("model")):
                from context_pack.llm_client import get_llm_generate_fn
                llm_generate_fn = get_llm_generate_fn(
                    provider=llm_config.get("provider"),
                    model=llm_config.get("model"),
                )
                print("[AUTOINTERP] Using LLM for research question: {} / {}".format(
                    llm_config.get("provider"), llm_config.get("model")))
        except Exception as e:
            print("[AUTOINTERP] LLM config load failed:", e)
    if llm_generate_fn is None and not getattr(args, "no_llm", False):
        print("[AUTOINTERP] No LLM config (.last_llm.json or config.yaml llm). No context_pack_question.txt will be written.")
    result = run_context_pack(
        graph_path=graph_path,
        output_dir=output_dir,
        seed_id=args.seed_id,
        s2_client=client,
        seed=getattr(args, "seed", None),
        download_pdfs=not getattr(args, "no_download", False),
        llm_generate_fn=llm_generate_fn,
    )
    papers = result.get("papers", [])
    print(f"[AUTOINTERP] Context pack: {len(papers)} papers")
    for p in papers:
        print(f"  [{p.get('relation')}] {p.get('paperId', '')[:12]}... {p.get('title', '')[:50]}...")
    if result.get("manifest_path"):
        print(f"[AUTOINTERP] manifest: {result['manifest_path']}")
    if result.get("question_path"):
        print(f"[AUTOINTERP] question: {result['question_path']}")
    elif llm_generate_fn is not None and not result.get("question_text"):
        print("[AUTOINTERP] LLM was called but returned empty; context_pack_question.txt was not written.")
    # Auto-rename project folder from context_pack_<ts> to <slug>_<ts> using QUESTION line
    if args.output_dir is None and result.get("question_text"):
        qtext = result["question_text"]
        q_match = re.search(r"QUESTION:\s*(.+?)(?:\n|$)", qtext, re.IGNORECASE | re.DOTALL)
        if q_match:
            title = q_match.group(1).strip()
            slug = re.sub(r"[^\w\s\-]", "", title).strip()
            slug = re.sub(r"\s+", "_", slug).lower()[:50].strip("_")
            if slug:
                old_parent = output_dir.parent
                new_name = f"{slug}_{ts}"
                projects_base = PACKAGE_ROOT / "projects"
                new_parent = projects_base / new_name
                if old_parent != new_parent and old_parent.exists() and not new_parent.exists():
                    try:
                        os.rename(old_parent, new_parent)
                        print(f"[AUTOINTERP] Renamed project to: {new_parent / 'questions'}")
                    except Exception as e:
                        print(f"[AUTOINTERP] Could not rename to {new_name}: {e}")


async def async_main(args: argparse.Namespace) -> None:
    """
    Execute the AutoInterp workflow with parsed CLI arguments.
    """
    # Print ASCII art from title.txt in color #be8d13 (golden/amber)
    title_path = Path(__file__).parent / "misc" / "title.txt"
    if title_path.exists():
        try:
            with open(title_path, 'r') as f:
                title_art = f.read()
            # ANSI escape sequence for RGB color
            # Color #be8d13 in RGB is 190,141,19
            color_start = "\033[38;2;190;141;19m"
            color_end = "\033[0m"
            # Add a line break before the ASCII art
            print(f"\n{color_start}{title_art}{color_end}")
        except Exception as e:
            print(f"Error displaying title art: {e}")
    
    try:
        # Initialize framework
        framework = await initialize_framework(args.config, args.venv, args.projects_dir)

        # Apply any saved user option defaults before anything reads config
        load_user_options(framework["config"])

        # Provider and model selection
        selected_provider, selected_model = select_provider_and_model()

        # Apply provider/model override to config
        framework["config"] = apply_provider_model_override(
            framework["config"],
            selected_provider,
            selected_model
        )

        # Save selected provider/model so topic-package uses the same model when calling LLM to generate topic
        if selected_provider and selected_provider != "manual" and selected_model:
            try:
                last_llm_path = Path(__file__).parent / ".last_llm.json"
                with open(last_llm_path, "w") as f:
                    json.dump({"provider": selected_provider, "model": selected_model}, f, indent=0)
            except Exception:
                pass

        # Offer interactive options menu
        opt_choice = input("\nPress [O] for Options, or Enter to continue: ").strip().lower()
        if opt_choice == "o":
            show_options_menu(framework["config"])

        # Re-initialize components with updated config if provider was changed
        if selected_provider != "manual":

            # Get the updated config
            updated_config = framework["config"]
            logger = framework["logger"]
            path_resolver = framework["path_resolver"]

            # Re-initialize LLM interface with updated config
            llm_interface = LLMInterface(updated_config, agent_name="question_generator")
            logger.info(f"LLM interface re-initialized with provider: {updated_config['agents']['question_generator']['llm']['provider']} and model: {updated_config['agents']['question_generator']['llm']['model']}")

            # Re-initialize question manager
            question_manager = QuestionManager(
                llm_interface=llm_interface,
                config=updated_config
            )
            logger.info("Question manager re-initialized")

            # Re-initialize analysis components with updated config
            analysis_generator = AnalysisGenerator(
                llm_interface=llm_interface,
                config=updated_config
            )
            logger.info("Analysis generator re-initialized")

            analysis_executor = AnalysisExecutor(config=updated_config)
            logger.info("Analysis executor re-initialized")

            analysis_planner = AnalysisPlanner(
                llm_interface=llm_interface,
                path_resolver=path_resolver
            )
            logger.info("Analysis planner re-initialized")

            evaluator = Evaluator(
                question_manager=question_manager,
                llm_interface=llm_interface,
                config=updated_config
            )
            logger.info("Evaluator re-initialized")

            visualization_evaluator = VisualizationEvaluator(
                llm_interface=llm_interface,
                config=updated_config
            )
            logger.info("Visualization Evaluator re-initialized")

            # Re-initialize reporting components
            report_generator = ReportGenerator(config=updated_config, llm_interface=llm_interface)
            logger.info("Report generator re-initialized")

            # Update framework components
            framework.update({
                "llm_interface": llm_interface,
                "question_manager": question_manager,
                "analysis_generator": analysis_generator,
                "analysis_executor": analysis_executor,
                "analysis_planner": analysis_planner,
                "evaluator": evaluator,
                "visualization_evaluator": visualization_evaluator,
                "report_generator": report_generator
            })

        # Get user input for task description
        # Use same golden/amber color as ASCII art title (#be8d13 = RGB 190,141,19)
        color_start = "\033[38;2;190;141;19m"
        color_end = "\033[0m"
        print(f"\n{color_start}" + "="*60)
        print("Welcome to the AutoInterp Agent Framework!")
        print("="*60 + f"{color_end}")
        context_pack_enabled = framework["config"].get("context_pack", {}).get("enabled", False)
        if context_pack_enabled:
            user_input = input("\nEnter a topic for LLM interpretability research (press Enter to generate one from the literature): ").strip()
        else:
            user_input = input("\nEnter a topic for LLM interpretability research (press Enter to generate one with LLM): ").strip()

        if not user_input:
            if context_pack_enabled:
                # Context pack will sample papers and generate a grounded question in the pipeline
                print("[AUTOINTERP] No topic provided. The context pack will generate a research question from the literature.")
                framework["config"]["task"]["description"] = "LLM interpretability research"
            else:
                print("[AUTOINTERP] No topic provided. Generating a topic with LLM...")
                try:
                    generated_topic = await framework["question_manager"].generate_topic()
                    print(f"[AUTOINTERP] Generated topic: {generated_topic}")
                    framework["config"]["task"]["description"] = generated_topic
                except Exception as e:
                    print(f"[AUTOINTERP] Error generating topic: {e}")
                    print("[AUTOINTERP] No task description available. Please provide a topic manually.")
        else:
            print(f"[AUTOINTERP] Using your topic: {user_input}")
            framework["config"]["task"]["description"] = user_input
        
        # Get project directory
        projects_dir = Path(framework["config"].get("paths", {}).get("projects", "projects"))
        project_id = framework["config"].get("project_id", "")
        
        if project_id:
            project_dir = projects_dir / project_id
            print(f"Using project directory at: {project_dir}")
        
        # Initialize PipelineUI if rich terminal or HTML dashboard is enabled
        try:
            from AutoInterp.core.pipeline_ui import PipelineUI
            pipeline_ui = PipelineUI(
                project_dir=framework["path_resolver"].get_project_dir(),
                config=framework["config"],
            )
            framework["pipeline_ui"] = pipeline_ui
            framework["llm_interface"].pipeline_ui = pipeline_ui
        except Exception as e:
            # PipelineUI is non-critical — fall back to legacy output
            print(f"[AUTOINTERP] Warning: PipelineUI init failed ({e}), using legacy output")
            framework["pipeline_ui"] = None

        # Execute streamlined pipeline
        await streamlined_pipeline(framework)
    
    except Exception as e:
        import traceback
        full_traceback = traceback.format_exc()
        print(f"Error in AutoInterp Agent Framework: {e}")
        print(f"Full traceback:\n{full_traceback}")
        raise


def main(argv: Optional[List[str]] = None) -> None:
    """
    Entry point used by console scripts and `python -m AutoInterp`.
    """
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    command = getattr(args, "command", "run")
    if command == "context-pack":
        try:
            run_context_pack_cmd(args)
        except Exception as e:
            import traceback
            print(f"[AUTOINTERP] Context pack failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        return
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\n[AUTOINTERP] Run cancelled by user.")
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
