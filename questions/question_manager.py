"""
Question Manager module for the AutoInterp Agent Framework.
Manages the generation, tracking, and evaluation of interpretability questions.
Maintains a prioritized list of questions and their current status.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from ..core.llm_interface import LLMInterface
from ..core.utils import get_timestamp, load_yaml, ensure_directory, PathResolver, log_to_comprehensive_log

class QuestionManager:
    """
    Manages interpretability questions throughout their lifecycle.
    
    Responsibilities:
    - Generate new questions based on context and task description
    - Maintain a prioritized list of questions
    - Track question status and evidence
    - Update question confidence based on evidence
    - Recommend next questions to investigate
    """
    
    def __init__(self, 
                llm_interface: LLMInterface,
                config: Dict[str, Any]):
        """
        Initialize the Question Manager.
        
        Args:
            llm_interface: LLM interface for generating responses
            config: Configuration dictionary
        """
        self.llm = llm_interface
        self.config = config
        
        # Initialize question storage
        self.questions = []
        self.active_question = None
        
        # Use the PathResolver singleton instead of storing project_id directly
        self.path_resolver = PathResolver(config)
        
        # Log initialization with path resolver
        import logging
        logger = logging.getLogger("autointerp.question_manager")
        logger.info(f"QuestionManager initialized using path resolver with project_id: {self.path_resolver.project_id}")
        
        # Ensure questions directory exists and set storage_dir
        self.storage_dir = self.path_resolver.ensure_path("questions")
        
        # Load existing questions if available
        self._load_questions()
    
    def _load_questions(self) -> None:
        """
        Load existing questions from storage if available.
        """
        # Try to load raw questions text if available
        raw_questions_path = self.storage_dir / "questions.txt"
        if raw_questions_path.exists():
            try:
                with open(raw_questions_path, 'r') as f:
                    self.raw_questions_text = f.read()
                print(f"[AUTOINTERP] Loaded raw questions from {raw_questions_path}")
            except Exception as e:
                print(f"[AUTOINTERP] Error loading raw questions from text file: {e}")
                
        # Start with empty questions list
        self.questions = []
    
    def update_storage_dir(self) -> None:
        """
        Update storage directory path based on current project_id in path_resolver.
        Call this method after the project has been renamed.
        """
        old_storage_dir = self.storage_dir
        # Get the updated path from path_resolver
        self.storage_dir = self.path_resolver.ensure_path("questions")
        
        import logging
        logger = logging.getLogger("autointerp.question_manager")
        logger.info(f"Updated questions storage directory from '{old_storage_dir}' to '{self.storage_dir}'")
        
        # If questions exist and old_storage_dir exists but is different from new storage_dir,
        # copy raw text files to the new location
        if old_storage_dir.exists() and str(old_storage_dir) != str(self.storage_dir):
            # If we have questions in memory, save them to the new location
            if self.questions:
                self._save_questions()
            
            # Try to copy over raw text file
            old_questions_file = old_storage_dir / "questions.txt"
            if old_questions_file.exists():
                # Ensure new directory exists
                ensure_directory(self.storage_dir)
                
                # Copy the text file
                try:
                    import shutil
                    shutil.copy2(old_questions_file, self.storage_dir / "questions.txt")
                    logger.info(f"Copied questions from {old_questions_file} to {self.storage_dir / 'questions.txt'}")
                except Exception as e:
                    logger.error(f"Error copying questions from old location: {e}")
    
    def _save_questions(self) -> None:
        """
        No longer saving questions as JSON - using raw text format only.
        """
        print("[AUTOINTERP] Not saving questions as JSON - using raw text format only")
    
    async def generate_questions(self, 
                               task_description: str, 
                               context: Optional[str] = None,
                               count: int = 3) -> List[Dict[str, Any]]:
        """
        Generate new interpretability questions based on the task description and context.
        
        Args:
            task_description: Description of the interpretability task
            context: Additional context or previous findings
            count: Number of questions to generate
            
        Returns:
            List of generated question dictionaries
        """
        # Get prompt template from prompts config
        prompt_template = self.config.get("prompts", {}).get("question_manager", {}).get("generator", {}).get("prompt_template")
        
        if prompt_template is None:
            raise ValueError("GENERATOR PROMPT TEMPLATE MISSING FROM CONFIG. WE NEED A PROMPT TEMPLATE")
        
        # Fill in template
        prompt = prompt_template.format(
            task_description=task_description,
            count=count
        )
        
        if context:
            prompt += f"\nAdditional context and previous findings:\n{context}\n"
        
        prompt += f"\nGenerate {count} numbered questions, each with a clear statement and brief rationale:"
        
        # Get system message from prompts config
        system_message = self.config.get("prompts", {}).get("question_manager", {}).get("generator", {}).get("system_message")
        
        if system_message is None:
            raise ValueError("GENERATOR SYSTEM MESSAGE MISSING FROM CONFIG. WE NEED A SYSTEM MESSAGE!")
        
        # Generate questions using LLM with the question_generator agent name
        response = await self.llm.generate(
            prompt=prompt,
            system_message=system_message,
            agent_name="question_generator"
        )
        
        # Store the raw LLM response for question selection
        self.raw_questions_text = response
        
        # Save raw questions as plain text file
        raw_questions_path = self.storage_dir / "questions.txt"
        try:
            with open(raw_questions_path, 'w') as f:
                f.write(response)
            print(f"[AUTOINTERP] Raw questions saved to {raw_questions_path}")
        except Exception as e:
            print(f"[AUTOINTERP] Error saving raw questions to text file: {e}")
            
        # No more parsing or structured question generation
        print("[AUTOINTERP] Using raw text format only for questions")
        
        # Return empty list as we're not tracking structured questions anymore
        return []
    
    
    
    
    
    async def get_active_question(self) -> Optional[Dict[str, Any]]:
        """
        Get the currently active question.
        
        Returns:
            The active question dictionary or None if no active question
        """
        return self.active_question
    
    async def get_all_questions(self) -> List[Dict[str, Any]]:
        """
        Get all tracked questions.
        
        Returns:
            List of all question dictionaries
        """
        return self.questions
        
    # Method update_confidence has been removed
    
    # Method add_evidence has been removed
    
    
    
    async def prioritize_questions(self) -> List[Dict[str, Any]]:
        """
        Select the best question to investigate.
        
        Returns:
            Empty list as we're using raw text format only
        """
        # Try to load raw questions from text file first
        raw_questions_path = self.storage_dir / "questions.txt"
        if raw_questions_path.exists():
            try:
                with open(raw_questions_path, 'r') as f:
                    self.raw_questions_text = f.read()
                print(f"[AUTOINTERP] Loaded raw questions from {raw_questions_path}")
            except Exception as e:
                print(f"[AUTOINTERP] Error loading raw questions from text file: {e}")
                return []
        else:
            print(f"[AUTOINTERP] No questions.txt found at {raw_questions_path}")
            return []
                
        # If we still don't have the raw text, just return empty list
        if not hasattr(self, 'raw_questions_text') or not self.raw_questions_text:
            return []
        
        # Get prompt template from prompts config
        prompt_template = self.config.get("prompts", {}).get("question_manager", {}).get("prioritizer", {}).get("prompt_template")
        
        if prompt_template is None:
            raise ValueError("PROMPT TEMPLATE MISSING FROM CONFIG. WE NEED A PROMPT TEMPLATE")
            
        prompt = prompt_template.format(
            question_text=self.raw_questions_text
        )
        
        # Get system message from prompts config
        system_message = self.config.get("prompts", {}).get("question_manager", {}).get("prioritizer", {}).get("system_message")
        
        if system_message is None:
            raise ValueError("SYSTEM MESSAGE MISSING FROM CONFIG. WE NEED A SYSTEM MESSAGE")
        
        # Generate selection using LLM
        response = await self.llm.generate(
            prompt=prompt,
            system_message=system_message,
            agent_name="question_prioritizer"
        )
        
        # Save the prioritized question as a text file
        prioritized_path = self.storage_dir / "prioritized_question.txt"
        try:
            with open(prioritized_path, 'w') as f:
                f.write(response)
            print(f"[AUTOINTERP] Prioritized question saved to {prioritized_path}")
            
            # Log to comprehensive log
            project_dir = self.path_resolver.get_project_dir()
            log_to_comprehensive_log(project_dir, response, "QUESTION PRIORITIZER OUTPUT")
            
        except Exception as e:
            print(f"[AUTOINTERP] Error saving prioritized question to text file: {e}")
        
        # Return empty list as we're using raw text format only
        return []
    
    async def generate_topic(self) -> str:
        """
        Generate a topic for interpretability research using the topic_generator agent.
        
        Returns:
            Generated topic description as a string
        """
        # Get prompt template from prompts config
        prompt_template = self.config.get("prompts", {}).get("question_manager", {}).get("topic_generator", {}).get("prompt_template")
        
        if prompt_template is None:
            raise ValueError("TOPIC GENERATOR PROMPT TEMPLATE MISSING FROM CONFIG!")
        
        # Get system message from prompts config
        system_message = self.config.get("prompts", {}).get("question_manager", {}).get("topic_generator", {}).get("system_message")
        
        if system_message is None:
            raise ValueError("TOPIC GENERATOR SYSTEM MESSAGE MISSING FROM CONFIG!")
        
        # Generate topic using LLM
        response = await self.llm.generate(
            prompt=prompt_template,
            system_message=system_message,
            agent_name="question_generator"  # Use existing agent
        )
        
        return response.strip()
    


# Backward compatibility alias
QuestionManager = QuestionManager