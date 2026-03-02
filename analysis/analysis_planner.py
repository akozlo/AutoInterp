"""
Analysis Planner Module - Designs the next step of analysis given the question and accumulated evidence.
"""

import os
import asyncio
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from AutoInterp.core.llm_interface import LLMInterface
from AutoInterp.core.utils import PathResolver, save_file, log_to_comprehensive_log

logger = logging.getLogger(__name__)

class AnalysisPlanner:
    """
    Plans the next analysis steps based on the current question and accumulated evidence.
    """
    
    def __init__(self, llm_interface: LLMInterface, path_resolver: PathResolver):
        """
        Initialize the Analysis Planner.

        Args:
            llm_interface: Interface to interact with LLM
            path_resolver: Utility to resolve file paths
        """
        self.llm_interface = llm_interface
        self.path_resolver = path_resolver
        # We don't need to initialize question_manager here, it's not used in this class
    
    async def plan_analysis(self, 
                          active_question: str,
                          config: Dict[str, Any],
                          iteration_number: Optional[int] = None) -> Tuple[str, str]:
        """
        Create a plan for the next analysis based on the current question and evaluations.

        Args:
            active_question: The question to focus on (raw question text)
            config: Configuration for the analysis planning

        Returns:
            Tuple[str, str]: The path to the saved plan file and the plan content
        """
        logger.info(f"Planning analysis for question")
        
        # Load the prioritized question
        question_text = ""
        question_file_path = self.path_resolver.get_prioritized_question_path()
        if question_file_path.exists():
            with open(question_file_path, "r") as f:
                question_text = f.read()
        
        # Get all evaluation files, but only the final ones (no ANALYSIS_FAILED)
        eval_dir = self.path_resolver.get_evaluation_dir()
        evaluation_files = []
        if eval_dir.exists():
            # Get all evaluation files
            all_eval_files = list(eval_dir.glob("*evaluation_*.txt"))
            
            # Group files by iteration number and find the highest attempt for each
            iteration_files = {}
            for f in all_eval_files:
                # Parse filename: a{iteration}.{attempt}_evaluation_{timestamp}.txt or a{iteration}_evaluation_{timestamp}.txt
                filename = f.name
                if filename.startswith("a") and "_evaluation_" in filename:
                    # Extract iteration and attempt from filename
                    prefix = filename.split("_evaluation_")[0]  # e.g., "a1.2" or "a1"
                    iteration_part = prefix[1:]  # Remove the 'a'
                    
                    if "." in iteration_part:
                        # Format: a{iteration}.{attempt}
                        iteration_str, attempt_str = iteration_part.split(".", 1)
                        try:
                            iteration_num = int(iteration_str)
                            attempt_num = int(attempt_str)
                        except ValueError:
                            continue  # Skip malformed filenames
                    else:
                        # Format: a{iteration} (legacy format, assume attempt 1)
                        try:
                            iteration_num = int(iteration_part)
                            attempt_num = 1
                        except ValueError:
                            continue  # Skip malformed filenames
                    
                    # Check if this file contains ANALYSIS_FAILED
                    try:
                        with open(f, 'r') as file:
                            content = file.read()
                            if "ANALYSIS_FAILED" in content:
                                continue  # Skip failed evaluations
                    except Exception:
                        continue  # Skip files we can't read
                    
                    # Keep track of the highest attempt for each iteration
                    # But exclude the current iteration since it hasn't been evaluated yet
                    if iteration_number is None or iteration_num < iteration_number:
                        if iteration_num not in iteration_files or attempt_num > iteration_files[iteration_num][1]:
                            iteration_files[iteration_num] = (f, attempt_num)

            # Extract the final evaluation files, sorted by iteration number
            evaluation_files = [iteration_files[i][0] for i in sorted(iteration_files.keys())]
        
        # Load all evaluations
        evaluations_text = ""
        logger.info(f"Analysis planner found {len(evaluation_files)} evaluation files for iteration {iteration_number}")
        for eval_file in evaluation_files:
            logger.info(f"Reading evaluation file: {eval_file.name}")
            with open(eval_file, "r") as f:
                evaluations_text += f"=== {eval_file.name} ===\n{f.read()}\n\n"

        if not evaluations_text.strip():
            evaluations_text = "No previous evaluations available - this appears to be the first analysis iteration."
            logger.info("No previous evaluations found - this is the first iteration")
        
        # Prepare the prompt with the question and evaluations
        input_data = {
            "task_config": config,
            "question": question_text,
            "evaluations": evaluations_text
        }
        
        # Log the prompt to a file
        log_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        
        # Create log filename with iteration prefix if available
        if iteration_number is not None:
            log_filename = f"a{iteration_number}_analysis_planner_{log_timestamp}_log.txt"
        else:
            log_filename = f"analysis_planner_{log_timestamp}_log.txt"
        log_dir = self.path_resolver.get_path("logs")
        
        if not log_dir.exists():
            os.makedirs(log_dir)
        
        # Get the prompts from the config
        prompts_config = self.llm_interface.config.get("prompts", {}).get("analysis_planner", {})
        system_prompt = prompts_config.get("system", "You are an analysis planner for an AI interpretability research project.")
        user_prompt_template = prompts_config.get("user", "Plan the next steps of analysis based on this question: {{question}} and the following evaluations: {{evaluations}}")
        
        # Format the user prompt with the actual input data
        user_prompt = user_prompt_template
        if "{{question}}" in user_prompt:
            user_prompt = user_prompt.replace("{{question}}", input_data.get("question", ""))
        if "{{evaluations}}" in user_prompt:
            user_prompt = user_prompt.replace("{{evaluations}}", input_data.get("evaluations", ""))
        
        # Generate the analysis plan with the correct parameters
        plan_response = await self.llm_interface.generate(
            prompt=user_prompt,
            system_message=system_prompt,
            agent_name="analysis_planner"
        )
        
        # Create the log path
        log_path = log_dir / log_filename
        
        # Create the log content
        log_content = "======= ANALYSIS PLANNER LOG =======\n\n"
        log_content += "=== SYSTEM PROMPT ===\n"
        log_content += system_prompt + "\n\n"
        log_content += "=== USER PROMPT ===\n"
        log_content += user_prompt + "\n\n"
        log_content += "=== MODEL RESPONSE ===\n"
        log_content += plan_response
        
        # Write the log file
        with open(log_path, "w") as f:
            f.write(log_content)
            
        logger.info(f"Analysis planner log saved to: {log_path}")
        
        # Save the analysis plan (just the LLM output)
        plan_timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        
        # Create filename with iteration prefix if available
        if iteration_number is not None:
            plan_filename = f"a{iteration_number}_analysis_plan_{plan_timestamp}.txt"
        else:
            plan_filename = f"analysis_plan_{plan_timestamp}.txt"
        plan_dir = self.path_resolver.get_analysis_plans_dir()
        
        # Create the directory if it doesn't exist
        if not plan_dir.exists():
            os.makedirs(plan_dir)
        
        plan_path = plan_dir / plan_filename
        # Use regular synchronous call to save_file - just the response
        save_file(plan_response, plan_path)
        
        # Log to comprehensive log
        project_dir = self.path_resolver.get_project_dir()
        log_to_comprehensive_log(project_dir, plan_response, "ANALYSIS PLAN")
        
        # Log the planning completion
        logger.info(f"Generated analysis plan at {plan_path}")
        return str(plan_path), plan_response