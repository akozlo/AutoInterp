"""
Evaluator module for the AutoInterp Agent Framework.
Evaluates analysis outcomes against questions, updates confidence,
extracts insights, and suggests improvements.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from ..core.llm_interface import LLMInterface
from ..questions.question_manager import QuestionManager
from ..core.utils import get_timestamp, ensure_directory, PathResolver, save_file

class Evaluator:
    """
    Evaluates analysis results against interpretability questions.
    
    Responsibilities:
    - Assess analysis results against the original question
    - Update confidence that question has been answered based on evidence
    - Extract key insights and findings from analysis
    - Recommend follow-up analyses or question refinements
    - Record evaluation results for later reference
    """
    
    def __init__(self, 
                question_manager: QuestionManager,
                llm_interface: LLMInterface,
                config: Dict[str, Any]):
        """
        Initialize the Evaluator.
        
        Args:
            question_manager: Question manager for tracking questions
            llm_interface: LLM interface for analysis evaluation
            config: Configuration dictionary
        """
        self.question_manager = question_manager
        self.llm = llm_interface
        self.config = config
        
        # Use the PathResolver singleton instead of storing project_id directly
        self.path_resolver = PathResolver(config)
        
        # Log initialization with path resolver
        import logging
        logger = logging.getLogger("autointerp.evaluator")
        logger.info(f"Evaluator initialized using path resolver with project_id: {self.path_resolver.project_id}")
        
        # Ensure evaluation_results directory exists and set output_dir
        self.output_dir = self.path_resolver.ensure_path("evaluation_results")
        
        # Log the output directory
        logger.debug(f"Evaluation results directory: {self.output_dir}")
    
    async def evaluate_analysis(self,
                             analysis_results: Dict[str, Any],
                             question_id: Optional[str] = None,
                             current_confidence: Optional[float] = None,
                             iteration_number: Optional[int] = None,
                             attempt_number: Optional[int] = None,
                             analysis_plan: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate analysis results against a question.
        
        Args:
            analysis_results: Results from analysis execution
            question_id: ID of the question to evaluate against
                           (will use active question if not provided)
            
        Returns:
            Dictionary with evaluation results
        """
        # Check if we're working with raw text question
        if question_id == "txt_question_1":
            import logging
            logger = logging.getLogger("autointerp.evaluator")
            logger.debug("Using raw text question from file")
            
            # Try to load from prioritized_question.txt
            prioritized_path = self.path_resolver.get_path("questions", "prioritized_question.txt")
            raw_question_text = ""
            if os.path.exists(prioritized_path):
                try:
                    with open(prioritized_path, 'r') as f:
                        raw_question_text = f.read()
                    logger.debug(f"Loaded raw question for evaluation from {prioritized_path}")
                except Exception as e:
                    logger.warning(f"Error loading raw question text: {e}")
                    
            if not raw_question_text:
                # Try to load from regular questions.txt
                questions_path = self.path_resolver.get_path("questions", "questions.txt")
                if os.path.exists(questions_path):
                    try:
                        with open(questions_path, 'r') as f:
                            raw_question_text = f.read()
                        logger.debug(f"Loaded raw questions from {questions_path}")
                    except Exception as e:
                        logger.warning(f"Error loading raw questions text: {e}")
            
            # Create a synthetic question dictionary with text
            question = {
                "id": "txt_question_1",
                "statement": raw_question_text,
                "rationale": "Raw text question with no structured rationale",
                "confidence": current_confidence if current_confidence is not None else 0.5
            }
        else:
            # Try to get structured question the old way
            question = None
            if question_id:
                # Find the question with the given ID
                all_questions = await self.question_manager.get_all_questions()
                matching = [h for h in all_questions if h["id"] == question_id]
                if matching:
                    question = matching[0]
            
            # If no question_id or couldn't find it, use active question
            if not question:
                question = await self.question_manager.get_active_question()
                
                # If active_question is a string (raw text), convert to dict format
                if isinstance(question, str):
                    question = {
                        "id": "txt_question_1",
                        "statement": question,
                        "rationale": "Raw text question with no structured rationale",
                        "confidence": 0.5  # Default confidence
                    }
                elif not question:
                    return {
                        "success": False,
                        "error": "No active question to evaluate against"
                    }
        
        # Create a unique ID for this evaluation - use current timestamp for raw text
        # This ensures we always have a unique ID
        timestamp = get_timestamp().replace(':', '-').replace(' ', '_')
        evaluation_id = f"eval_{question['id']}_{timestamp}"
        
        # We no longer save metadata.json as a separate file
        
        # Check if analysis was successful
        if not analysis_results.get("success", False):
            error_message = analysis_results.get('error', 'Unknown error')
            
            error_evaluation = {
                "success": False,
                "error": f"Analysis failed: {error_message}",
                "raw_evaluation": f"Analysis failed: {error_message}",
                "question": question,
                "evaluation_id": evaluation_id,
                "timestamp": get_timestamp()
            }
            
            # Save error as text file instead of JSON
            timestamp = get_timestamp().replace(':', '-').replace(' ', '_')
            
            # Create filename with iteration prefix if available
            if iteration_number is not None:
                eval_filename = f"a{iteration_number}_evaluation_{timestamp}.txt"
            else:
                eval_filename = f"evaluation_{timestamp}.txt"
            
            eval_txt_path = self.output_dir / eval_filename

            # Debug logging to verify paths
            print(f"[AUTOINTERP] Error evaluation output_dir: {self.output_dir}")
            print(f"[AUTOINTERP] Error evaluation file path: {eval_txt_path}")

            error_text = f"""RAW EVALUATION:
Analysis failed: {error_message}

Error: {error_message}
Timestamp: {get_timestamp()}"""
            
            # Save the error text file with error handling
            try:
                save_file(error_text, eval_txt_path)
                print(f"[AUTOINTERP] Saved error evaluation to: {eval_txt_path}")
                import logging
                logger = logging.getLogger("autointerp.evaluator")
                logger.info(f"Successfully saved error evaluation file: {eval_txt_path}")
            except Exception as e:
                print(f"[AUTOINTERP] ERROR: Failed to save error evaluation file {eval_txt_path}: {e}")
                import logging
                logger = logging.getLogger("autointerp.evaluator")
                logger.error(f"Failed to save error evaluation file {eval_txt_path}: {e}")
                # Don't fail the entire evaluation if file save fails
                pass
            
            return error_evaluation
        
        # Evaluate the results against the question
        evaluation_result = await self._evaluate_results(analysis_results, question, iteration_number, analysis_plan)
        
        # Update confidence that question has been answered based on evaluation
        if evaluation_result.get("supports_question") is not None:
            supports = evaluation_result["supports_question"]
            new_confidence = evaluation_result.get("new_confidence", question["confidence"])
            evidence = evaluation_result.get("explanation", "No explanation provided")
            
            # Simply track confidence in the evaluation result
            print(f"[AUTOINTERP] Setting confidence to {new_confidence}")
            
            # Update the evaluation result
            evaluation_result["updated_question"] = question  # Just use the same question
            evaluation_result["old_confidence"] = question["confidence"]
            evaluation_result["new_confidence"] = new_confidence
        
        # Save the evaluation result as a structured text file
        evaluation_result["evaluation_id"] = evaluation_id
        evaluation_result["timestamp"] = get_timestamp()
        
        # Create structured text content
        timestamp = get_timestamp().replace(':', '-').replace(' ', '_')
        
        # Create filename with iteration and attempt prefix if available
        if iteration_number is not None and attempt_number is not None:
            eval_filename = f"a{iteration_number}.{attempt_number}_evaluation_{timestamp}.txt"
        elif iteration_number is not None:
            eval_filename = f"a{iteration_number}_evaluation_{timestamp}.txt"
        else:
            eval_filename = f"evaluation_{timestamp}.txt"
        
        eval_txt_path = self.output_dir / eval_filename

        # Debug logging to verify paths
        print(f"[AUTOINTERP] Evaluation output_dir: {self.output_dir}")
        print(f"[AUTOINTERP] Evaluation file path: {eval_txt_path}")

        # Format the evaluation text with just raw evaluation
        eval_text = f"""RAW EVALUATION:
{evaluation_result.get('raw_evaluation', 'No raw evaluation available')}"""
        
        # Save the text file with error handling
        try:
            save_file(eval_text, eval_txt_path)
            print(f"[AUTOINTERP] Saved evaluation to: {eval_txt_path}")
            import logging
            logger = logging.getLogger("autointerp.evaluator")
            logger.info(f"Successfully saved evaluation file: {eval_txt_path}")
        except Exception as e:
            print(f"[AUTOINTERP] ERROR: Failed to save evaluation file {eval_txt_path}: {e}")
            import logging
            logger = logging.getLogger("autointerp.evaluator")
            logger.error(f"Failed to save evaluation file {eval_txt_path}: {e}")
            # Don't fail the entire evaluation if file save fails
            pass
        
        return evaluation_result
    
    async def _evaluate_results(self,
                            analysis_results: Dict[str, Any],
                            question: Dict[str, Any],
                            iteration_number: Optional[int] = None,
                            analysis_plan: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM to evaluate analysis results against a question.
        
        Args:
            analysis_results: Results from analysis execution
            question: Question to evaluate against
            
        Returns:
            Dictionary with structured evaluation
        """
        # Extract key elements for evaluation and ensure they're not None
        question_statement = question.get("statement", "No question statement provided")
        question_rationale = question.get("rationale", "No rationale provided")
        analysis_output = analysis_results.get("results", {})
        
        # Get agent config
        agent_config = self.config.get("agents", {}).get("evaluator", {})
        
        # Get the execution outputs
        analysis_script = analysis_results.get("script_path", "No script available")
        # Read the script content if path is provided
        script_content = ""
        if analysis_script and os.path.exists(analysis_script):
            try:
                with open(analysis_script, 'r') as f:
                    script_content = f.read()
            except Exception as e:
                script_content = f"Error reading script: {str(e)}"
                
        # Extract stdout and stderr
        stdout = analysis_results.get("stdout", "No output")
        stderr = analysis_results.get("stderr", "")
        
        # Get prompt template from prompts config
        prompt_template = self.config.get("prompts", {}).get("evaluator", {}).get("prompt_template")
        
        if prompt_template is None:
            raise ValueError("EVALUATOR PROMPT TEMPLATE MISSING FROM CONFIG! WE NEED A PROMPT TEMPLATE")
        
        # Prepare stderr section if it exists
        stderr_section = f"ERROR OUTPUT:\n```\n{stderr}\n```" if stderr else ""
        
        if prompt_template is None:
            # If we get here, something is very wrong - our earlier check should have caught this
            raise ValueError("EVALUATOR PROMPT TEMPLATE IS STILL MISSING")
            
        # Get confidence with a default value
        question_confidence = question.get("confidence", 0.0)  # Default to 0.0 if not specified

        # Use the analysis plan or a default message
        analysis_plan_text = analysis_plan if analysis_plan else "No analysis plan available."

        prompt = prompt_template.format(
            question_statement=question_statement,
            question_rationale=question_rationale,
            question_confidence=question_confidence,
            script_content=script_content,
            stdout=stdout,
            stderr_section=stderr_section,
            analysis_plan=analysis_plan_text
        )
        
        # Get system message from prompts config
        system_message = self.config.get("prompts", {}).get("evaluator", {}).get("system_message")
        
        if system_message is None:
            raise ValueError("EVALUATOR SYSTEM MESSAGE MISSING FROM CONFIG")
        
        # Get LLM settings from config
        llm_config = agent_config.get("llm", {})
        temperature = llm_config.get("temperature", 0.2)  # Default to 0.2 for more consistent responses
        
        # Generate evaluation using LLM
        response = await self.llm.generate(
            prompt=prompt,
            system_message=system_message,
            agent_name="evaluator",  # Use evaluator agent
            iteration_number=iteration_number
        )
        
        # Create simple evaluation result with raw LLM output
        evaluation = {
            "success": True,
            "raw_evaluation": response
        }
        
        # Extract only the critical new confidence value if present
        confidence_match = re.search(r'NEW_CONFIDENCE:\s*([-+]?\d*\.\d+|\d+)', response)
        if confidence_match:
            try:
                evaluation["new_confidence"] = float(confidence_match.group(1))
                # Ensure in range 0.0 to 1.0
                evaluation["new_confidence"] = max(0.0, min(1.0, evaluation["new_confidence"]))
            except ValueError:
                pass
        
        return evaluation
    
    async def _get_previous_evaluations_text(self, question_id: str) -> str:
        """
        Gather and format previous evaluations for the given question.
        
        Args:
            question_id: ID of the question to get evaluations for
            
        Returns:
            Formatted text of previous evaluations
        """
        try:
            # Get all evaluations for this question
            evaluations = await self.list_evaluations(question_id)
            
            if not evaluations:
                return "No previous evaluations available."
            
            # Sort by timestamp to get chronological order
            evaluations.sort(key=lambda x: x.get("timestamp", ""), reverse=False)
            
            formatted_evaluations = []
            
            for i, eval_meta in enumerate(evaluations, 1):
                evaluation_id = eval_meta.get("evaluation_id", "unknown")
                
                # Get the full evaluation content
                full_eval = await self.get_evaluation_by_id(evaluation_id)
                if not full_eval:
                    continue
                
                # Only include evaluations that actually have content
                # (This excludes the current evaluation which may be empty)
                raw_evaluation = full_eval.get("raw_evaluation", "")
                if not raw_evaluation or "No evaluation content available" in raw_evaluation:
                    continue
                    
                # Format as a numbered evaluation
                eval_text = f"=== ANALYSIS {i} ===\n{raw_evaluation.strip()}\n"
                formatted_evaluations.append(eval_text)
            
            if not formatted_evaluations:
                return "No previous evaluations available."
                
            return "\n".join(formatted_evaluations)
            
        except Exception as e:
            # Log the error but don't fail the evaluation
            import logging
            logger = logging.getLogger("autointerp.evaluator")
            logger.warning(f"Error gathering previous evaluations: {e}")
            return "No previous evaluations available."
    
    # Notebook storage removed for elegance
    
    async def get_evaluation_by_id(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific evaluation by ID.
        
        Args:
            evaluation_id: ID of the evaluation to retrieve
            
        Returns:
            Evaluation dictionary or None if not found
        """
        # Look for evaluation files directly in the output directory
        # Files can have format: evaluation_*.txt or a{n}_evaluation_*.txt
        evaluation_files = list(self.output_dir.glob("*evaluation_*.txt"))
        
        # Filter files that match our evaluation_id pattern
        matching_files = []
        for file in evaluation_files:
            # Extract timestamp from filename to match with evaluation_id
            filename = file.name
            if "evaluation_" in filename:
                # For evaluation_id format: eval_{question_id}_{timestamp}
                # Extract timestamp from evaluation_id
                if evaluation_id.startswith("eval_"):
                    expected_timestamp = evaluation_id.split('_')[-1]
                    if expected_timestamp in filename:
                        matching_files.append(file)
        
        if not matching_files:
            return None
        
        # Use the most recent evaluation file
        evaluation_file = sorted(matching_files)[-1]
        
        try:
            # Read the full text file content
            with open(evaluation_file, 'r') as f:
                content = f.read()
                
            # Create a simple result dictionary with raw content
            result = {
                "evaluation_id": evaluation_id,
                "success": True,  # Assume success if file exists
                "raw_evaluation": content,
                "timestamp": get_timestamp()  # Use current timestamp if not found in file
            }
            
            # Extract only the critical metadata - timestamp and errors
            import re
            
            # Extract timestamp if available
            timestamp_match = re.search(r'Timestamp: (.*)', content)
            if timestamp_match:
                result["timestamp"] = timestamp_match.group(1).strip()
                
            # Extract any error
            error_match = re.search(r'Error: (.*)', content)
            if error_match:
                result["error"] = error_match.group(1)
                result["success"] = False
                
            # Extract new confidence if available (important for tracking)
            confidence_match = re.search(r'NEW_CONFIDENCE:\s*([-+]?\d*\.\d+|\d+)', content)
            if confidence_match:
                try:
                    result["new_confidence"] = float(confidence_match.group(1))
                except ValueError:
                    pass
                
            return result
        except Exception as e:
            print(f"[AUTOINTERP] Error reading evaluation file: {e}")
            return None
    
    async def list_evaluations(self, question_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all evaluations with minimal metadata.
        
        Args:
            question_id: Optional filter for a specific question
            
        Returns:
            List of evaluation metadata dictionaries with basic info
        """
        evaluations = []
        
        # Look for evaluation files directly in output directory
        # Files can have format: evaluation_*.txt or a{n}_evaluation_*.txt
        eval_files = list(self.output_dir.glob("*evaluation_*.txt"))
        
        for eval_file in eval_files:
            try:
                filename = eval_file.name
                
                # Extract analysis number if present (a{n}_ prefix)
                analysis_number = None
                display_filename = filename
                if filename.startswith("a") and "_" in filename:
                    try:
                        prefix = filename.split("_")[0]
                        if prefix[1:].isdigit():
                            analysis_number = int(prefix[1:])
                            display_filename = filename[len(prefix)+1:]  # Remove a{n}_ prefix
                    except (ValueError, IndexError):
                        pass
                
                # Create evaluation_id based on filename timestamp
                if "evaluation_" in filename:
                    timestamp_part = display_filename.split("evaluation_")[1].split(".")[0]
                    
                    # For question filtering, we need to determine which question this belongs to
                    # Since we don't store it in the filename anymore, we'll assume txt_question_1 for now
                    # or try to extract from file content if needed
                    current_question_id = "txt_question_1"  # Default assumption
                    
                    # Filter by question_id if provided
                    if question_id and current_question_id != question_id:
                        continue
                    
                    evaluation_id = f"eval_{current_question_id}_{timestamp_part}"
                    
                    # Create a basic metadata dictionary
                    metadata = {
                        "evaluation_id": evaluation_id,
                        "file_path": str(eval_file),
                        "timestamp": timestamp_part,
                        "question_id": current_question_id
                    }
                    
                    # Add analysis number if available
                    if analysis_number is not None:
                        metadata["analysis_number"] = analysis_number
                    
                    # Simple check for errors
                    try:
                        with open(eval_file, 'r') as f:
                            content = f.read(1000)  # Just read the beginning
                            if "Error:" in content or "error" in content.lower():
                                metadata["status"] = "error"
                            else:
                                metadata["status"] = "complete"
                    except Exception:
                        metadata["status"] = "unknown"
                    
                    evaluations.append(metadata)
            except Exception as e:
                print(f"[AUTOINTERP] Error processing evaluation file {eval_file}: {e}")
        
        # Sort by analysis number if available, then by timestamp (newest first)
        evaluations.sort(key=lambda x: (x.get("analysis_number", 0), x.get("timestamp", "")), reverse=True)
        
        return evaluations
    
    # The generate_follow_up_analyses method has been removed
    # since we're using unstructured text directly without extracting
    # structured follow-up analyses