"""
Visualization Generator Module - Generates visualization code for analysis results.
"""

import os
import asyncio
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from AutoInterp.src.core.llm_interface import LLMInterface
from AutoInterp.src.core.utils import PathResolver, save_file, clean_code_content

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """
    Generates visualization code for successful analysis results.
    """
    
    def __init__(self, llm_interface: LLMInterface, path_resolver: PathResolver):
        """
        Initialize the Visualization Generator.

        Args:
            llm_interface: Interface to interact with LLM
            path_resolver: Utility to resolve file paths
        """
        self.llm = llm_interface
        self.path_resolver = path_resolver
        self.current_attempt = 1

    def increment_attempt(self):
        """
        Increment the current attempt counter.
        Should be called when a visualization fails evaluation and needs to be retried.
        """
        self.current_attempt += 1
    
    async def generate_visualization(self, analysis_script: str, analysis_output: str, error_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate visualization code based on analysis script and output.

        Args:
            analysis_script: The analysis script content
            analysis_output: The output from the analysis
            error_context: Optional error information from a previous visualization attempt

        Returns:
            str: The generated visualization code
        """
        logger.info("Generating visualization code for analysis results")
        
        # Get the correct output directory for visualizations
        viz_output_dir = self.path_resolver.ensure_path("visualizations")
        logger.info(f"Visualization output directory: {viz_output_dir}")
        
        # Get the prompts from the config
        prompts_config = self.llm.config.get("prompts", {}).get("visualization_generator", {})
        system_prompt = prompts_config.get("system", "You are a visualization generator for AI interpretability research.")
        user_prompt_template = prompts_config.get("user", "Generate visualization code for this analysis: {{analysis_script}} with output: {{analysis_output}}")
        
        # Modify system prompt to include output directory instructions
        enhanced_system_prompt = system_prompt + f"""

CRITICAL: All visualization files MUST be saved to this specific directory: {viz_output_dir}
- Use plt.savefig("{viz_output_dir}/filename.png") for matplotlib figures
- Use fig.write_html("{viz_output_dir}/filename.html") for plotly figures
- Always use absolute paths starting with {viz_output_dir}/
- Do not save files to the current working directory or relative paths
- Give your output files descriptive names that relate to what they visualize

- You MUST set the matplotlib backend to 'Agg' BEFORE importing pyplot. This prevents GUI errors on headless systems.
- Start your script EXACTLY like this:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
"""
        
        # Format the user prompt with the actual input data
        user_prompt = user_prompt_template.replace("{{analysis_script}}", analysis_script)
        user_prompt = user_prompt.replace("{{analysis_output}}", analysis_output)
        
        # Add output directory instruction to user prompt as well
        enhanced_user_prompt = user_prompt + f"""

IMPORTANT: Save all visualization files to: {viz_output_dir}/
Use absolute paths like: plt.savefig("{viz_output_dir}/my_visualization.png")
"""

        # Add comprehensive error context if this is a retry attempt (similar to analysis_generator.py)
        if error_context:
            # Handle both evaluation and execution error contexts
            retry_info = []

            # Check if this is an evaluation error (has evaluation_feedback)
            if "evaluation_feedback" in error_context:
                retry_info.append({
                    "attempt": self.current_attempt - 1,
                    "error_type": "evaluation_failure",
                    "evaluation_feedback": error_context.get("evaluation_feedback", "Previous visualization had evaluation issues"),
                    "previous_code": error_context.get("previous_code", ""),
                    "failed_script": ""
                })

            # Check if this is an execution error (has error/stderr/stdout)
            if "error" in error_context:
                retry_entry = {
                    "attempt": self.current_attempt - 1,
                    "error_type": "execution_failure",
                    "error_message": error_context.get("error", "Unknown error"),
                    "error_traceback": error_context.get("stderr", ""),
                    "stdout": error_context.get("stdout", ""),
                    "failed_script": ""
                }

                # Load failed script content if path is provided (like analysis_generator.py does)
                previous_script_path = error_context.get("previous_script")
                if previous_script_path and os.path.exists(previous_script_path):
                    try:
                        with open(previous_script_path, 'r') as f:
                            retry_entry["failed_script"] = f.read()
                    except Exception as e:
                        retry_entry["failed_script"] = f"Error reading previous script: {str(e)}"

                retry_info.append(retry_entry)

            # Add comprehensive retry context to the prompt
            if retry_info:
                enhanced_user_prompt += "\n\n# PREVIOUS FAILED ATTEMPTS - Learn from these errors\n"

                for attempt_info in retry_info:
                    if attempt_info["error_type"] == "evaluation_failure":
                        enhanced_user_prompt += f"""
## ATTEMPT {attempt_info['attempt']} - EVALUATION FAILURE:
Evaluation Feedback: {attempt_info['evaluation_feedback']}

Previous visualization code that had evaluation issues:
```python
{attempt_info['previous_code']}
```

Please analyze the evaluation feedback and fix the visualization issues mentioned above.
"""
                    elif attempt_info["error_type"] == "execution_failure":
                        enhanced_user_prompt += f"""
## ATTEMPT {attempt_info['attempt']} - EXECUTION FAILURE:
Error: {attempt_info['error_message']}

Error Output:
```
{attempt_info['error_traceback']}
```

Standard Output:
```
{attempt_info['stdout']}
```

Failed visualization script:
```python
{attempt_info['failed_script']}
```

Please analyze the error and fix the issues in the visualization code.
"""

                enhanced_user_prompt += """
RETRY INSTRUCTIONS:
- Fix the specific errors mentioned above
- Ensure all imports are correct and available
- Verify file paths are accessible
- Test that the visualization saves files to the correct directory
- Make sure the code runs without exceptions
"""
        
        # Generate the visualization code
        visualization_response = await self.llm.generate(
            prompt=enhanced_user_prompt,
            system_message=enhanced_system_prompt,
            agent_name="visualization_generator"
        )

        # Clean the visualization code to remove markdown code fences
        cleaned_visualization_code = clean_code_content(visualization_response)

        logger.info("Generated visualization code")
        return cleaned_visualization_code