"""
Visualization Planner Module - Plans visualizations for analysis results.
"""

import os
import asyncio
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from AutoInterp.core.llm_interface import LLMInterface
from AutoInterp.core.utils import PathResolver, save_file

logger = logging.getLogger(__name__)

class VisualizationPlanner:
    """
    Plans visualizations for successful analysis results.
    """
    
    def __init__(self, llm_interface: LLMInterface, path_resolver: PathResolver):
        """
        Initialize the Visualization Planner.

        Args:
            llm_interface: Interface to interact with LLM
            path_resolver: Utility to resolve file paths
        """
        self.llm = llm_interface
        self.path_resolver = path_resolver
    
    async def plan_visualization(self, analysis_script: str, analysis_output: str) -> str:
        """
        Create a visualization plan based on analysis script and output.

        Args:
            analysis_script: The analysis script content
            analysis_output: The output from the analysis

        Returns:
            str: The visualization plan content
        """
        logger.info("Planning visualization for analysis results")
        
        # Get the prompts from the config
        prompts_config = self.llm.config.get("prompts", {}).get("visualization_planner", {})
        system_prompt = prompts_config.get("system", "You are a visualization planner for AI interpretability research.")
        user_prompt_template = prompts_config.get("user", "Create a visualization plan for this analysis: {{analysis_script}} with output: {{analysis_output}}")
        
        # Format the user prompt with the actual input data
        user_prompt = user_prompt_template.replace("{{analysis_script}}", analysis_script)
        user_prompt = user_prompt.replace("{{analysis_output}}", analysis_output)
        
        # Generate the visualization plan
        plan_response = await self.llm.generate(
            prompt=user_prompt,
            system_message=system_prompt,
            agent_name="visualization_planner"
        )
        
        logger.info("Generated visualization plan")
        return plan_response