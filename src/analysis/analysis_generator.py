"""
Analysis Generator module for the AutoInterp Agent Framework.
Dynamically generates Python code for interpretability analyses based on questions.
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from ..core.llm_interface import LLMInterface, CodeGeneration
from ..core.utils import get_timestamp, load_yaml, ensure_directory, save_file, PathResolver, clean_code_content
from ..analysis.analysis_executor import AnalysisExecutor

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

class AnalysisGenerator:
    """
    Generates Python code for interpretability analyses.
    
    Responsibilities:
    - Generate analysis code based on questions and task requirements
    - Create different types of analyses (attention, neuron activation, etc.)
    - Adapt to different models and question types
    - Provide structured output for later evaluation
    """
    
    
    def __init__(self, 
                llm_interface: LLMInterface,
                config: Dict[str, Any]):
        """
        Initialize the Analysis Generator.
        
        Args:
            llm_interface: LLM interface for code generation
            config: Configuration dictionary
        """
        self.llm = llm_interface
        self.config = config
        self.code_generator = CodeGeneration(llm_interface)
        
        # Get the PathResolver singleton
        self.path_resolver = PathResolver(config)
        
        # Track the current analysis number and attempt
        self.current_analysis = 1
        self.current_attempt = 1
        
        # Log initialization with path resolver
        logger = logging.getLogger("autointerp.analysis_generator")
        logger.info(f"AnalysisGenerator initialized using path resolver with project_id: {self.path_resolver.project_id}")
        
        # Ensure analysis directory exists
        self.path_resolver.ensure_path("analysis")

        # Log the path being used
        logger.debug(f"Analysis directory: {self.path_resolver.get_path('analysis')}")
    
# Template functionality removed completely

    def increment_attempt(self):
        """
        Increment the current attempt counter.
        Should be called when an analysis fails and needs to be retried.
        """
        self.current_attempt += 1
        logger = logging.getLogger("autointerp.analysis_generator")
        logger.info(f"Incrementing attempt counter to {self.current_attempt} for analysis {self.current_analysis}")
        
    def move_to_next_analysis(self):
        """
        Move to the next analysis by incrementing the analysis counter and resetting the attempt counter.
        Should be called when an analysis is successful or the maximum number of attempts is reached.
        """
        self.current_analysis += 1
        self.current_attempt = 1
        logger = logging.getLogger("autointerp.analysis_generator")
        logger.info(f"Moving to next analysis: {self.current_analysis}, attempt: {self.current_attempt}")
        
    async def generate_analysis(self,
                             question: Dict[str, Any],
                             task_config: Dict[str, Any],
                             analysis_plan: Optional[str] = None,
                             error_context: Optional[Dict[str, Any]] = None,
                             iteration_number: Optional[int] = None) -> Tuple[str, str]:
        """
        Generate analysis code for a question.
        
        Args:
            question: Question data - could be a dictionary or raw text from prioritized_question.txt
            task_config: Task-specific configuration
            analysis_plan: Optional analysis plan from the analysis_planner agent
            error_context: Optional error information from a previous run to help fix issues
            
        Returns:
            Tuple of (analysis_script_path, analysis_code)
        """
        # Extract model information from task config
        model_info = task_config.get("model", {})
        model_name = model_info.get("name", "gpt2")
        model_path = model_info.get("path", model_name)
        vision_model = model_info.get("vision_model", "")
        reasoning_model = model_info.get("reasoning_model", "")
        
        # Extract analysis parameters from task config
        analysis_params = task_config.get("analysis_parameters", {})
        
        # Ensure question is a string (load from file if needed)
        if not isinstance(question, str):
            try:
                # Load prioritized question from the file
                prioritized_path = self.path_resolver.get_path("questions", "prioritized_question.txt")
                if os.path.exists(prioritized_path):
                    with open(prioritized_path, 'r') as f:
                        question = f.read()
                else:
                    # Convert to string as fallback
                    question = str(question)
            except Exception as e:
                # Convert to string as fallback
                question = str(question)
        
        # Previous insights no longer gathered from notebooks for elegance
        previous_insights = ""
        
        # Update the generation spec with previous insights and error context
        # We're not passing analysis_plan separately anymore since it's hardcoded into the template
        generation_spec = self._build_generation_spec(
            question=question,
            model_name=model_name,
            model_path=model_path,
            vision_model=vision_model,
            reasoning_model=reasoning_model,
            analysis_params=analysis_params,
            previous_insights=previous_insights,
            analysis_plan=analysis_plan,  # Still passing it for backward compatibility
            error_context=error_context
        )
        
        # Get system message from prompts config
        system_message = self.config.get("prompts", {}).get("analysis_generator", {}).get("system_message")
        
        if system_message is None:
            raise ValueError("ANALYSIS GENERATOR SYSTEM MESSAGE MISSING FROM CONFIG. WE NEED A SYSTEM MESSAGE!")
        
        # Generate the analysis code using the agent's configured max_tokens
        response = await self.llm.generate(
            prompt=f"Generate Python code for the following specification:\n\n{generation_spec}",
            system_message=system_message,
            agent_name="analysis_generator",
            iteration_number=iteration_number
        )
        
        # Extract code block if needed
        code_block = self.code_generator._extract_code_block(response, "python")
        analysis_code = code_block if code_block else response

        # Clean the analysis code to remove markdown code fences
        analysis_code = clean_code_content(analysis_code)

        # Add missing imports if not present
        if "np.ndarray" in analysis_code and "import numpy as np" not in analysis_code:
            analysis_code = "import numpy as np\n" + analysis_code
        
        # Add other common imports that might be missing
        if "torch" in analysis_code and "import torch" not in analysis_code:
            analysis_code = "import torch\n" + analysis_code
            
        if "plt." in analysis_code and "import matplotlib.pyplot as plt" not in analysis_code:
            analysis_code = "import matplotlib.pyplot as plt\n" + analysis_code
            
        # Save the generated code using the new directory structure
        timestamp = get_timestamp().replace(" ", "_").replace(":", "-")
        
        # Create the analysis directory structure
        analysis_dir_name = f"analysis_{self.current_analysis}"
        attempt_dir_name = f"attempt_{self.current_attempt}"
        
        # Use PathResolver to get absolute path and ensure directory exists
        try:
            # Create analysis directory if it doesn't exist
            analysis_dir = self.path_resolver.ensure_path("analysis", analysis_dir_name)
            
            # Create attempt directory within the analysis directory
            attempt_dir = ensure_directory(analysis_dir / attempt_dir_name)
            
            # Create script filename
            filename = f"analysis_{timestamp}.py"
            script_path = attempt_dir / filename
            
            # Save the generator text as well
            generator_txt_filename = f"analysis_generator_{timestamp}.txt"
            generator_txt_path = attempt_dir / generator_txt_filename
            
            # Log the file path
            logger = logging.getLogger("autointerp.analysis_generator")
            logger.info(f"Saving analysis script to: {script_path}")
            print(f"[AUTOINTERP] Saving analysis script to: {script_path}")
            
            # Add standard imports to the beginning of the script
            final_code = AnalysisExecutor.STANDARD_IMPORTS + analysis_code
            
            # Write the file with standard imports
            with open(script_path, "w") as f:
                f.write(final_code)
                
            # Write the generator prompt and response for debugging
            with open(generator_txt_path, "w") as f:
                f.write(generation_spec + "\n\n" + response)
                
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            print(f"[AUTOINTERP] Error saving analysis script: {e}")
            print(f"[AUTOINTERP] Full traceback:\n{full_traceback}")
            
            # Try a fallback location
            fallback_dir = PACKAGE_ROOT / "src" / "analysis" / analysis_dir_name / attempt_dir_name
            ensure_directory(fallback_dir)
            fallback_path = fallback_dir / filename
            print(f"[AUTOINTERP] Using fallback path: {fallback_path}")
            
            # Use the same final_code with standard imports
            final_code = AnalysisExecutor.STANDARD_IMPORTS + analysis_code
            
            with open(fallback_path, "w") as f:
                f.write(final_code)
                
            # Also save the generator text in the fallback location
            generator_fallback_path = fallback_dir / generator_txt_filename
            with open(generator_fallback_path, "w") as f:
                f.write(generation_spec + "\n\n" + response)
                
            script_path = fallback_path
        
        return str(script_path), final_code
    
    
    def _build_generation_spec(self,
                             question: Union[str, Dict[str, Any]],
                             model_name: str,
                             model_path: str,
                             vision_model: str = "",
                             reasoning_model: str = "",
                             analysis_params: Dict[str, Any] = None,
                             previous_insights: str = "",
                             analysis_plan: Optional[str] = None,
                             error_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a specification for code generation.

        Args:
            question: The raw question text or a dictionary (will be converted to string)
            model_name: Name of the model
            model_path: Path to the model
            vision_model: HuggingFace model for vision tasks
            reasoning_model: HuggingFace model for reasoning tasks
            analysis_params: Additional analysis parameters
            previous_insights: Insights from previous analyses in notebooks
            analysis_plan: Optional analysis plan from the analysis_planner agent
            error_context: Optional error information from a previous failed run

        Returns:
            Specification string for code generation
        """
        if analysis_params is None:
            analysis_params = {}
        # Use question text directly (should already be a string at this point)
        question_text = question if question else "No question text available"
        
        # Check if we have information from previous failed attempts
        retry_info = []
        if analysis_params and "retry_info" in analysis_params:
            retry_info = analysis_params.get("retry_info", [])
            # If this came through task_config directly, we need to extract it
        elif "retry_info" in analysis_params.get("task_config", {}):
            retry_info = analysis_params.get("task_config", {}).get("retry_info", [])
            
        # Add current error context to retry info if provided
        if error_context:
            retry_info.append({
                "attempt": len(retry_info) + 1,
                "error_message": error_context.get("error", "Unknown error"),
                "error_traceback": error_context.get("traceback", ""),
                "failed_script": ""
            })
            
            # If we have a path to the previous script, load its content
            previous_script_path = error_context.get("previous_script")
            if previous_script_path and os.path.exists(previous_script_path):
                try:
                    with open(previous_script_path, 'r') as f:
                        retry_info[-1]["failed_script"] = f.read()
                except Exception as e:
                    retry_info[-1]["failed_script"] = f"Error reading previous script: {str(e)}"
        
        # Get the prompt template from config, if available
        agent_config = self.config.get("agents", {}).get("analysis_generator", {})
        prompt_template = agent_config.get("prompt_template", "")
        
        # Use the template if available, otherwise try to get it from prompts config, then fallback to default
        if prompt_template:
            spec = prompt_template.format(
                question_text=question_text,
                model_name=model_name,
                model_path=model_path,
                vision_model=vision_model,
                reasoning_model=reasoning_model,
                analysis_plan=analysis_plan if analysis_plan else "",  # Use empty string if no plan
                project_dir=str(self.path_resolver.get_project_dir())  # Add project directory path
            )
        else:
            # Try to get base template from prompts config
            base_template = self.config.get("prompts", {}).get("analysis_generator", {}).get("base_template")

            if base_template:
                # Use the template from prompts config
                # Include analysis_plan in the template formatting
                spec = base_template.format(
                    question_text=question_text,
                    model_name=model_name,
                    model_path=model_path,
                    vision_model=vision_model,
                    reasoning_model=reasoning_model,
                    analysis_plan=analysis_plan if analysis_plan else "",  # Use empty string if no plan
                    project_dir=str(self.path_resolver.get_project_dir())  # Add project directory path
                )
            else:
                # If not found, use a default template
                logger = logging.getLogger("autointerp.analysis_generator")
                logger.warning("Base template missing from configuration. Using default template.")
                
                # Default template that matches the one from analysis_generator.yaml
                default_template = """# Analysis Specification:
  
{question_text}
  
Model: {model_name}
Model Path: {model_path}

{analysis_plan}
  
Write a Python script to answer this question. Implement the analysis exactly according to the ANALYSIS PLAN above.

IMPORTANT: Your response MUST be ONLY executable Python code with no surrounding text or explanations.
Do not prefix your response with phrases like "Here's a Python script..." or "This code will...".
Any explanations or notes must be included as comments within the code.

The script should:

1. Load and analyze the model
2. Return structured results as a dictionary
3. Be well-documented with clear comments, both in the code and alongside printed output
4. PRINT the results to stdout - do not save them as a variable - the system captures all printed output
5. Don't do any visualizations - just print the results to stdout
6. IMPORTANT: Include all necessary imports at the top of the file
"""
                
                spec = default_template.format(
                    question_text=question_text,
                    model_name=model_name,
                    model_path=model_path,
                    vision_model=vision_model,
                    reasoning_model=reasoning_model,
                    analysis_plan=analysis_plan if analysis_plan else "",
                    project_dir=str(self.path_resolver.get_project_dir())  # Add project directory path
                )
        
        # We no longer dynamically add the analysis plan here since it's now part of the base template
        # The plan will be inserted using the {analysis_plan} template variable
        
        # Add previous insights if available to inform the analysis
        if previous_insights:
            # Try to get insights template from prompts config
            insights_template = self.config.get("prompts", {}).get("analysis_generator", {}).get("insights_template")
            
            if insights_template:
                spec += insights_template.format(previous_insights=previous_insights)
            else:
                # Log that insights template is missing from configuration
                logger = logging.getLogger("autointerp.analysis_generator")
                logger.warning("Insights template missing from configuration")
        
        
        # Add previous failed attempts information if available
        if retry_info:
            # Get retry templates from prompts config
            retry_template = self.config.get("prompts", {}).get("analysis_generator", {}).get("retry_template")
            retry_instructions = self.config.get("prompts", {}).get("analysis_generator", {}).get("retry_instructions")
            
            if not retry_template:
                # Log that the retry_template is missing from config
                logger = logging.getLogger("autointerp.analysis_generator")
                logger.warning("Retry template missing from configuration")
            
            spec += "\n# PREVIOUS FAILED ATTEMPTS - Learn from these errors\n"
            # Include up to the last 3 failed attempts to avoid excessive context
            for attempt in retry_info[-3:]:
                spec += retry_template.format(
                    attempt=attempt.get('attempt'),
                    error_message=attempt.get('error_message', 'Unknown error'),
                    error_traceback=attempt.get('error_traceback', ''),
                    failed_script=attempt.get('failed_script', 'No script available')
                )
            
            # Add retry instructions if available
            if retry_instructions:
                spec += retry_instructions
            else:
                # Log that retry_instructions is missing from config
                logger = logging.getLogger("autointerp.analysis_generator")
                logger.warning("Retry instructions missing from configuration")
        
      
        return spec
