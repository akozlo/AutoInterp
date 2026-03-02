"""
Visualization Evaluator module for the AutoInterp Agent Framework.
Evaluates generated visualizations for issues and creates captions.
"""

import os
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from ..core.llm_interface import LLMInterface
from ..core.utils import get_timestamp, ensure_directory, PathResolver, save_file

class VisualizationEvaluator:
    """
    Evaluates visualization results and determines if they have issues.

    Responsibilities:
    - Assess visualization images and code for problems
    - Generate captions for successful visualizations
    - Detect visualization errors for retry logic
    - Record evaluation results for later reference
    """

    def __init__(self,
                 llm_interface: LLMInterface,
                 config: Dict[str, Any]):
        """
        Initialize the VisualizationEvaluator.

        Args:
            llm_interface: LLM interface for visualization evaluation
            config: Configuration dictionary
        """
        self.llm = llm_interface
        self.config = config

        # Use the PathResolver singleton
        self.path_resolver = PathResolver(config)

        # Log initialization
        import logging
        logger = logging.getLogger("autointerp.visualization_evaluator")
        logger.info(f"VisualizationEvaluator initialized using path resolver with project_id: {self.path_resolver.project_id}")

        # Don't cache the output directory - resolve it dynamically
        # This ensures we get the correct path even if project_id changes after initialization

        # Log the current project_id
        logger.debug(f"VisualizationEvaluator project_id at initialization: {self.path_resolver.project_id}")

    def _encode_image_to_base64(self, image_path: Path) -> tuple:
        """
        Encode an image file to base64 for API transmission.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (base64_data, media_type)
        """
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')

            # Determine media type from file extension
            extension = image_path.suffix.lower()
            if extension == '.png':
                media_type = 'image/png'
            elif extension in ['.jpg', '.jpeg']:
                media_type = 'image/jpeg'
            elif extension == '.svg':
                media_type = 'image/svg+xml'
            elif extension == '.pdf':
                media_type = 'application/pdf'
            else:
                media_type = 'image/png'  # Default fallback

            return base64_data, media_type

        except Exception as e:
            import logging
            logger = logging.getLogger("autointerp.visualization_evaluator")
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    def _find_visualization_file(self, analysis_number: int) -> Optional[Path]:
        """
        Find the visualization file for a given analysis number.

        Args:
            analysis_number: The analysis iteration number

        Returns:
            Path to the visualization file, or None if not found
        """
        try:
            visualizations_dir = self.path_resolver.get_path("visualizations")
            if not visualizations_dir.exists():
                return None

            # Look for files with the correct prefix
            prefix = f"a{analysis_number}_"

            # Common image extensions
            extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']

            for ext in extensions:
                for file_path in visualizations_dir.glob(f"{prefix}*{ext}"):
                    return file_path

            return None

        except Exception as e:
            import logging
            logger = logging.getLogger("autointerp.visualization_evaluator")
            logger.error(f"Failed to find visualization file for analysis {analysis_number}: {e}")
            return None

    async def evaluate_visualization(self,
                                   visualization_code: str,
                                   analysis_number: int,
                                   question_id: Optional[str] = None,
                                   iteration_number: Optional[int] = None,
                                   attempt_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a visualization against quality standards.

        Args:
            visualization_code: The Python code that generated the visualization
            analysis_number: The analysis iteration number
            question_id: ID of the question being analyzed
            iteration_number: Optional iteration number for file naming
            attempt_number: Optional attempt number for file naming

        Returns:
            Dictionary with evaluation results
        """
        # Find the corresponding visualization file
        visualization_path = self._find_visualization_file(analysis_number)

        if not visualization_path:
            error_message = f"No visualization file found for analysis {analysis_number}"
            await self._save_evaluation_result(error_message, iteration_number, attempt_number)
            return error_message

        try:
            # Encode the image for API transmission
            base64_data, media_type = self._encode_image_to_base64(visualization_path)

            # Evaluate the visualization using LLM with image
            evaluation_response = await self._evaluate_with_llm(
                visualization_code=visualization_code,
                image_base64=base64_data,
                media_type=media_type,
                image_filename=visualization_path.name,
                iteration_number=iteration_number
            )

            # Save the evaluation result (just the raw response)
            await self._save_evaluation_result(evaluation_response, iteration_number, attempt_number)

            return evaluation_response

        except Exception as e:
            import logging
            logger = logging.getLogger("autointerp.visualization_evaluator")
            logger.error(f"Error evaluating visualization: {e}")

            error_message = f"Visualization evaluation failed: {str(e)}"
            await self._save_evaluation_result(error_message, iteration_number, attempt_number)
            return error_message

    async def _evaluate_with_llm(self,
                               visualization_code: str,
                               image_base64: str,
                               media_type: str,
                               image_filename: str,
                               iteration_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Use LLM to evaluate visualization with both code and image.

        Args:
            visualization_code: The Python code
            image_base64: Base64-encoded image data
            media_type: MIME type of the image
            image_filename: Name of the image file
            iteration_number: Optional iteration number

        Returns:
            Dictionary with evaluation results
        """
        # Get prompt template from config
        prompt_config = self.config.get("prompts", {}).get("visualization_evaluator", {})

        if not prompt_config:
            raise ValueError("Visualization evaluator prompt configuration not found")

        system_message = prompt_config.get("system")
        user_prompt_template = prompt_config.get("user")

        if not system_message or not user_prompt_template:
            raise ValueError("Visualization evaluator system message or user prompt template not found")

        # Format the user prompt with the visualization code
        user_prompt = user_prompt_template.format(
            visualization_code=f"{visualization_code}"
        )

        # Create message content with both image and text
        message_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_base64
                }
            },
            {
                "type": "text",
                "text": user_prompt
            }
        ]

        # Generate evaluation using LLM with multimodal input
        response = await self.llm.generate_with_images(
            message_content=message_content,
            system_message=system_message,
            agent_name="visualization_evaluator",
            iteration_number=iteration_number
        )

        # Just return the raw LLM response - no structured data needed
        return response.strip()

    async def _save_evaluation_result(self,
                                    evaluation_text: str,
                                    iteration_number: Optional[int] = None,
                                    attempt_number: Optional[int] = None):
        """
        Save the visualization evaluation result to a text file.

        Args:
            evaluation_text: The raw evaluation text from the LLM
            iteration_number: Optional iteration number for file naming
            attempt_number: Optional attempt number for file naming
        """
        try:
            timestamp = get_timestamp().replace(':', '-').replace(' ', '_')

            # Create filename with iteration and attempt prefix if available
            if iteration_number is not None and attempt_number is not None:
                eval_filename = f"a{iteration_number}.{attempt_number}_viz_evaluation_{timestamp}.txt"
            elif iteration_number is not None:
                eval_filename = f"a{iteration_number}_viz_evaluation_{timestamp}.txt"
            else:
                eval_filename = f"viz_evaluation_{timestamp}.txt"

            # Resolve output directory dynamically to ensure correct path after project renames
            output_dir = self.path_resolver.ensure_path("logs")
            eval_file_path = output_dir / eval_filename

            # Just save the raw evaluation text - no extra structure
            save_file(evaluation_text, eval_file_path)

        except Exception as e:
            import logging
            logger = logging.getLogger("autointerp.visualization_evaluator")
            logger.error(f"Failed to save visualization evaluation result: {e}")