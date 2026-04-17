"""
LLM Interface module for the AutoInterp Agent Framework.
Provides interface to various LLM providers, including Claude via the Anthropic API.
"""

import os
import json
import asyncio
import subprocess
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime

from .utils import PathResolver, get_timestamp

class ColorCodes:
    """ANSI color codes for console output."""
    # Define base colors for each agent type
    AGENT_COLORS = {
        'question_generator': '\033[92m',   # Green
        'question_prioritizer': '\033[96m', # Cyan
        # Backward compatibility
        'question_generator': '\033[92m',   # Green
        'question_prioritizer': '\033[96m', # Cyan
        'analysis_planner': '\033[95m',       # Magenta
        'analysis_generator': '\033[93m',     # Yellow
        'evaluator': '\033[91m',              # Red
        'visualization_evaluator': '\033[94m', # Blue
        'CodeAnalyzer': '\033[97m',           # White
        'CodeImprover': '\033[90m',           # Dark Gray
        'default': '\033[37m'                 # Light Gray
    }
    
    # Human-readable names for agent types
    AGENT_DISPLAY_NAMES = {
        'question_generator': 'Question Generator',
        'question_prioritizer': 'Question Prioritizer',
        # Backward compatibility
        'question_generator': 'Question Generator',
        'question_prioritizer': 'Question Prioritizer',
        'analysis_planner': 'Analysis Planner',
        'analysis_generator': 'Analysis Generator',
        'evaluator': 'Evaluator',
        'visualization_evaluator': 'Visualization Evaluator',
        'CodeAnalyzer': 'Code Analyzer',
        'CodeImprover': 'Code Improver',
        'default': 'Unknown Agent'
    }
    
    # Darker versions for completions (using darker color codes)
    AGENT_COLORS_DARK = {
        'question_generator': '\033[32m',   # Dark Green
        'question_prioritizer': '\033[36m', # Dark Cyan
        # Backward compatibility
        'question_generator': '\033[32m',   # Dark Green
        'question_prioritizer': '\033[36m', # Dark Cyan
        'analysis_planner': '\033[35m',       # Dark Magenta
        'analysis_generator': '\033[33m',     # Dark Yellow
        'evaluator': '\033[31m',              # Dark Red
        'visualization_evaluator': '\033[34m', # Dark Blue
        'CodeAnalyzer': '\033[37m',           # Light Gray
        'CodeImprover': '\033[30m',           # Black
        'default': '\033[90m'                 # Dark Gray
    }
    
    RESET = '\033[0m'                         # Reset to default
    BOLD = '\033[1m'
    
    @classmethod
    def get_agent_color(cls, agent_name: str, is_completion: bool = False) -> str:
        """Get the color code for a specific agent."""
        colors = cls.AGENT_COLORS_DARK if is_completion else cls.AGENT_COLORS
        return colors.get(agent_name, colors['default'])
    
    @classmethod
    def get_agent_display_name(cls, agent_name: str) -> str:
        """Get the human-readable display name for an agent."""
        return cls.AGENT_DISPLAY_NAMES.get(agent_name, cls.AGENT_DISPLAY_NAMES['default'])
    
    @classmethod
    def colorize(cls, text: str, agent_name: str, is_completion: bool = False) -> str:
        """Colorize text with the agent's color."""
        color = cls.get_agent_color(agent_name, is_completion)
        return f"{color}{text}{cls.RESET}"

class LLMInterface:
    """
    Interface for Large Language Models, supporting multiple providers.
    """
    
    def __init__(self, config: Dict[str, Any], agent_name: Optional[str] = None):
        """
        Initialize the LLM interface.
        
        Args:
            config: Configuration dictionary for LLM settings
            agent_name: Optional name of the agent to use specific config
        """
        self.config = config
        self.agent_name = agent_name
        
        # Validate provider configurations exist at top level
        if "providers" not in config:
            raise ValueError("No providers configuration found in config.yaml")
            
        providers_config = config["providers"]
        
        # Require agent name and agent-specific LLM config
        if not agent_name:
            raise ValueError("Agent name must be provided")
        if "agents" not in config:
            raise ValueError("No agents configuration found in config.yaml")
        if agent_name not in config["agents"]:
            raise ValueError(f"Agent '{agent_name}' not found in config.yaml")
        if "llm" not in config["agents"][agent_name]:
            raise ValueError(f"No LLM configuration found for agent '{agent_name}' in config.yaml")
            
        # Get the agent's specific LLM config
        llm_config = config["agents"][agent_name]["llm"]
            
        # Require and set all LLM parameters from agent's config
        if "provider" not in llm_config:
            raise ValueError(f"No LLM provider specified for agent '{agent_name}'")
        self.provider = llm_config["provider"]
        
        # Validate provider exists in top-level config
        if self.provider not in providers_config:
            raise ValueError(f"Provider '{self.provider}' not found in providers configuration")
            
        if "model" not in llm_config:
            raise ValueError(f"No model specified for agent '{agent_name}'")
        self.model = llm_config["model"]
            
        if "temperature" not in llm_config:
            raise ValueError(f"No temperature specified for agent '{agent_name}'")
        self.temperature = llm_config["temperature"]
            
        if "max_tokens" not in llm_config:
            raise ValueError(f"No max_tokens specified for agent '{agent_name}'")
        self.max_tokens = llm_config["max_tokens"]
            
        if "timeout" not in llm_config:
            raise ValueError(f"No timeout specified for agent '{agent_name}'")
        self.timeout = llm_config["timeout"]

        # Pipeline UI (set externally when PipelineUI is active)
        self.pipeline_ui = None

        # Access the central path resolver for consistent logging paths
        self.path_resolver = PathResolver(config)

        # Get provider-specific configurations from top level
        # Always set anthropic_api_version if the provider config exists,
        # since generate() can dispatch to _generate_anthropic() for a
        # different agent even if this instance's default provider isn't anthropic.
        if "anthropic" in providers_config:
            provider_config = providers_config["anthropic"]
            if "api_version" not in provider_config:
                raise ValueError("No API version specified in Anthropic provider config")
            self.anthropic_api_version = provider_config["api_version"]
    
    async def generate(self, 
                      prompt: str, 
                      system_message: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      agent_name: Optional[str] = None,
                      iteration_number: Optional[int] = None) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            system_message: Optional system message for chat models
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            agent_name: Optional name of the agent making the request
            
        Returns:
            Generated text response
        """
        # No parameter overrides allowed - must use agent's configured values
        if temperature is not None or max_tokens is not None:
            raise ValueError("Temperature and max_tokens overrides are not allowed - use values from agent config")
        
        # Get the agent information - either from initialization or from the parameter
        actual_agent_name = agent_name if agent_name is not None else self.agent_name
        
        # Get the agent-specific config if a different agent is requested
        if actual_agent_name != self.agent_name:
            if "agents" not in self.config:
                raise ValueError("No agents configuration found in config")
            if actual_agent_name not in self.config["agents"]:
                raise ValueError(f"Agent '{actual_agent_name}' not found in config")
            if "llm" not in self.config["agents"][actual_agent_name]:
                raise ValueError(f"No LLM configuration found for agent '{actual_agent_name}'")
                
            agent_llm_config = self.config["agents"][actual_agent_name]["llm"]
            provider = agent_llm_config["provider"]
            model = agent_llm_config["model"]
            temp = agent_llm_config["temperature"]
            tokens = agent_llm_config["max_tokens"]
            timeout = agent_llm_config["timeout"]
        else:
            # Use the initialized values
            provider = self.provider
            model = self.model
            temp = self.temperature
            tokens = self.max_tokens
            timeout = self.timeout
        
        # Log detailed information about the LLM call to a text file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        
        if "project_id" not in self.config:
            raise ValueError("No project_id specified in config")
        project_id = self.config["project_id"]
            
        if "paths" not in self.config:
            raise ValueError("No paths configuration found in config")
        if "projects" not in self.config["paths"]:
            raise ValueError("No projects path specified in config")
            
        projects_base = Path(self.config["paths"]["projects"])
        
        # Always save logs to the logs directory
        log_dir = projects_base / project_id / "logs"
        
        # Create the directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with iteration prefix if available
        if iteration_number is not None:
            log_file = log_dir / f"a{iteration_number}_{actual_agent_name}_{timestamp}.txt"
        else:
            log_file = log_dir / f"{actual_agent_name}_{timestamp}.txt"
        
        # Determine the prompt source file for verbose logging
        prompt_source_file = "Unknown"
        try:
            if "prompts" in self.config and actual_agent_name in self.config["prompts"]:
                # Check if this agent has prompts config
                prompt_source_file = f"{actual_agent_name}.yaml (from prompts config)"
            elif hasattr(self, 'config') and 'agents' in self.config and actual_agent_name in self.config['agents']:
                prompt_source_file = f"config.yaml (agent: {actual_agent_name})"
        except:
            prompt_source_file = "Unknown"
        
        # Format the content for request
        log_content = f"""###################################
LLM API Call from agent: {actual_agent_name}
Model: {model}
Provider: {provider}
Temperature: {temp}
Max tokens: {tokens}

System Prompt:
{system_message if system_message else "None"}

User Prompt:
{prompt}
###################################
"""
        
        # VERBOSE LOGGING TO STDOUT WITH COLORS
        agent_display_name = ColorCodes.get_agent_display_name(actual_agent_name)
        call_start_time = time.time()

        agent_color = ColorCodes.get_agent_color(actual_agent_name, is_completion=False)
        print(ColorCodes.colorize("\n" + "="*80, actual_agent_name))
        print(ColorCodes.colorize(f"LLM CALL INITIATED - {agent_display_name}", actual_agent_name))
        print(ColorCodes.colorize("="*80, actual_agent_name))
        print(ColorCodes.colorize(f"PROMPT SOURCE FILE: {prompt_source_file}", actual_agent_name))
        print(ColorCodes.colorize(f"AGENT NAME: {actual_agent_name}", actual_agent_name))
        print(ColorCodes.colorize(f"MODEL: {model}", actual_agent_name))
        print(ColorCodes.colorize(f"PROVIDER: {provider}", actual_agent_name))
        print(ColorCodes.colorize(f"TEMPERATURE: {temp}", actual_agent_name))
        print(ColorCodes.colorize(f"MAX TOKENS: {tokens}", actual_agent_name))
        print(ColorCodes.colorize(f"LOG FILE: {log_file}", actual_agent_name))
        print(ColorCodes.colorize("\n" + "-"*80, actual_agent_name))
        print(ColorCodes.colorize("SYSTEM MESSAGE:", actual_agent_name))
        print(ColorCodes.colorize("-"*80, actual_agent_name))
        if system_message:
            print(ColorCodes.colorize(system_message, actual_agent_name))
        else:
            print(ColorCodes.colorize("(No system message)", actual_agent_name))
        print(ColorCodes.colorize("\n" + "-"*80, actual_agent_name))
        print(ColorCodes.colorize("USER PROMPT:", actual_agent_name))
        print(ColorCodes.colorize("-"*80, actual_agent_name))
        print(ColorCodes.colorize(prompt, actual_agent_name))
        print(ColorCodes.colorize("\n" + "-"*80, actual_agent_name))
        print(ColorCodes.colorize("CALLING LLM... Please wait...", actual_agent_name))
        print(ColorCodes.colorize("-"*80, actual_agent_name))

        # Write to the file
        with open(log_file, "w") as f:
            f.write(log_content)
        
        # Get response based on provider
        response = None
        if provider == "anthropic":
            response = await self._generate_anthropic(
                prompt=prompt,
                system_message=system_message,
                temperature=temp,
                max_tokens=tokens,
                model=model,
                timeout=timeout
            )
        elif provider == "openai":
            response = await self._generate_openai(
                prompt=prompt,
                system_message=system_message,
                temperature=temp,
                max_tokens=tokens,
                model=model,
                timeout=timeout
            )
        elif provider == "openrouter":
            response = await self._generate_openrouter(
                prompt=prompt,
                system_message=system_message,
                temperature=temp,
                max_tokens=tokens,
                model=model,
                timeout=timeout
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        # VERBOSE LOGGING OF RESPONSE TO STDOUT WITH DARKER COLORS
        call_duration = time.time() - call_start_time

        print(ColorCodes.colorize("\n" + "="*80, actual_agent_name, is_completion=True))
        print(ColorCodes.colorize(f"LLM RESPONSE RECEIVED - {agent_display_name}", actual_agent_name, is_completion=True))
        print(ColorCodes.colorize("="*80, actual_agent_name, is_completion=True))
        print(ColorCodes.colorize(f"AGENT: {actual_agent_name}", actual_agent_name, is_completion=True))
        print(ColorCodes.colorize(f"MODEL: {model}", actual_agent_name, is_completion=True))
        print(ColorCodes.colorize("\n" + "-"*80, actual_agent_name, is_completion=True))
        print(ColorCodes.colorize("LLM OUTPUT:", actual_agent_name, is_completion=True))
        print(ColorCodes.colorize("-"*80, actual_agent_name, is_completion=True))
        print(ColorCodes.colorize(response, actual_agent_name, is_completion=True))
        print(ColorCodes.colorize("\n" + "="*80, actual_agent_name, is_completion=True))
        print(ColorCodes.colorize("LLM CALL COMPLETE", actual_agent_name, is_completion=True))
        print(ColorCodes.colorize("="*80 + "\n", actual_agent_name, is_completion=True))

        # Also record in pipeline UI for HTML dashboard
        if self.pipeline_ui:
            self.pipeline_ui.llm_call_complete(
                agent_name=actual_agent_name,
                display_name=agent_display_name,
                prompt=prompt,
                system_message=system_message,
                response=response,
                model=model,
                provider=provider,
                temperature=temp,
                max_tokens=tokens,
                duration_seconds=call_duration,
                iteration_number=iteration_number,
            )
        
        # Append LLM output to the log file
        response_content = f"""
###################################
LLM Response to agent: {actual_agent_name}

Output:
{response}
###################################
"""
        # Append to the file
        with open(log_file, "a") as f:
            f.write(response_content)
        
        return response
    
    async def _generate_anthropic(self, 
                                prompt: str, 
                                system_message: Optional[str] = None,
                                temperature: float = 0.7,
                                max_tokens: int = 1000,
                                model: Optional[str] = None,
                                timeout: Optional[int] = None) -> str:
        """
        Generate text using Anthropic API with retry logic for transient errors.
        
        Args:
            prompt: The prompt to send to Anthropic
            system_message: Optional system message
            temperature: Temperature parameter (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            model: Model to use (default: self.model)
            timeout: Timeout in seconds (default: self.timeout)
            
        Returns:
            Generated text response
        """
        max_retries = 10
        base_delay = 2.0  # 2 seconds base delay
        
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp package required for Anthropic API calls")
            
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": self.anthropic_api_version
        }
        
        # Use provided values or defaults from self
        actual_model = model if model is not None else self.model
        actual_timeout = timeout if timeout is not None else self.timeout
        
        payload = {
            "model": actual_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_message:
            payload["system"] = system_message
        
        # Retry loop for transient errors
        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 (4 total attempts)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.anthropic.com/v1/messages",
                        headers=headers,
                        json=payload,
                        timeout=actual_timeout
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["content"][0]["text"]
                        
                        # Handle different error types
                        error_text = await response.text()
                        
                        # Check if this is a retryable error
                        is_retryable = self._is_retryable_error(response.status, error_text)
                        
                        if not is_retryable or attempt == max_retries:
                            # Non-retryable error or last attempt - raise immediately
                            raise RuntimeError(f"Anthropic API error ({response.status}): {error_text}")
                        
                        # Calculate delay with exponential backoff, capped at 180 seconds
                        delay = min(base_delay * (2 ** attempt), 180.0)
                        
                        print(f"[AUTOINTERP] API error ({response.status}): {error_text}")
                        print(f"[AUTOINTERP] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        
                        await asyncio.sleep(delay)
                        continue
                        
            except asyncio.TimeoutError as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Anthropic API timeout after {max_retries + 1} attempts: {str(e)}")
                
                delay = min(base_delay * (2 ** attempt), 180.0)
                print(f"[AUTOINTERP] API timeout error: {str(e)}")
                print(f"[AUTOINTERP] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
                continue
                
            except aiohttp.ClientError as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Anthropic API client error after {max_retries + 1} attempts: {str(e)}")
                
                delay = min(base_delay * (2 ** attempt), 180.0)
                print(f"[AUTOINTERP] API client error: {str(e)}")
                print(f"[AUTOINTERP] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
                continue
                
            except Exception as e:
                # For non-HTTP errors, check if it's a known transient error pattern
                error_str = str(e).lower()
                if any(pattern in error_str for pattern in ['timeout', 'connection', 'network', 'temporary']):
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), 180.0)
                        print(f"[AUTOINTERP] Transient error: {str(e)}")
                        print(f"[AUTOINTERP] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue
                
                # Non-transient error or last attempt
                raise RuntimeError(f"Error using Anthropic API: {str(e)}")
        
        # Should never reach here
        raise RuntimeError("Unexpected error in retry loop")
    
    def _is_retryable_error(self, status_code: int, error_text: str) -> bool:
        """
        Determine if an API error is retryable based on status code and error content.
        
        Args:
            status_code: HTTP status code
            error_text: Error response text
            
        Returns:
            True if the error should be retried, False otherwise
        """
        # Retryable HTTP status codes
        retryable_status_codes = {
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
            529   # Overloaded (Anthropic specific)
        }
        
        if status_code in retryable_status_codes:
            return True
        
        # Check error text for specific retryable patterns
        error_text_lower = error_text.lower()
        retryable_patterns = [
            'overloaded',
            'rate limit',
            'too many requests',
            'server error',
            'temporarily unavailable',
            'service unavailable',
            'timeout',
            'internal error'
        ]
        
        return any(pattern in error_text_lower for pattern in retryable_patterns)

    async def _generate_openai(self,
                              prompt: str,
                              system_message: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 1000,
                              model: Optional[str] = None,
                              timeout: Optional[int] = None) -> str:
        """
        Generate text using OpenAI API with retry logic for transient errors.

        Args:
            prompt: The prompt to send to OpenAI
            system_message: Optional system message (instructions)
            temperature: Temperature parameter (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            model: Model to use (default: self.model)
            timeout: Timeout in seconds (default: self.timeout)

        Returns:
            Generated text response
        """
        max_retries = 10
        base_delay = 2.0  # 2 seconds base delay

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for OpenAI API calls")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        client = OpenAI(api_key=api_key)

        # Use provided values or defaults from self
        actual_model = model if model is not None else self.model
        actual_timeout = timeout if timeout is not None else self.timeout

        # Construct input for OpenAI responses API format
        input_messages = [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}]
        }]

        # Retry loop for transient errors
        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 (4 total attempts)
            try:
                # Use OpenAI responses API format (GPT-5 models don't accept temperature)
                # Don't set max_output_tokens to let the model manage token allocation for reasoning
                response = client.responses.create(
                    model=actual_model,
                    instructions=system_message if system_message else "You are a helpful assistant.",
                    input=input_messages
                )

                # Check if response has content
                if hasattr(response, 'output_text'):
                    output_text = response.output_text
                    if output_text and output_text.strip():
                        return output_text
                    else:
                        # Empty response - try to get content another way
                        print(f"[DEBUG] OpenAI response.output_text is empty or None")
                        print(f"[DEBUG] Response object: {type(response)}")
                        print(f"[DEBUG] Response attributes: {dir(response)}")

                        # Check for other possible response formats
                        if hasattr(response, 'content'):
                            content = response.content
                            if content and hasattr(content, 'text'):
                                return content.text
                            elif content and isinstance(content, str):
                                return content

                        # If still no content, raise an error
                        raise RuntimeError(f"OpenAI API returned empty response")
                else:
                    # No output_text attribute - check response structure
                    print(f"[DEBUG] OpenAI response has no output_text attribute")
                    print(f"[DEBUG] Response object: {type(response)}")
                    print(f"[DEBUG] Response attributes: {dir(response)}")

                    # Try alternative response structures
                    if hasattr(response, 'choices') and response.choices:
                        if hasattr(response.choices[0], 'message'):
                            return response.choices[0].message.content
                        elif hasattr(response.choices[0], 'text'):
                            return response.choices[0].text

                    raise RuntimeError(f"OpenAI API response structure not recognized")

            except Exception as e:
                error_str = str(e).lower()

                # Check if this is a retryable error
                is_retryable = self._is_openai_retryable_error(e)

                if not is_retryable or attempt == max_retries:
                    # Non-retryable error or last attempt - raise immediately
                    raise RuntimeError(f"OpenAI API error: {str(e)}")

                # Calculate delay with exponential backoff, capped at 180 seconds
                delay = min(base_delay * (2 ** attempt), 180.0)

                print(f"[AUTOINTERP] OpenAI API error: {str(e)}")
                print(f"[AUTOINTERP] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")

                await asyncio.sleep(delay)
                continue

        # Should never reach here
        raise RuntimeError("Unexpected error in OpenAI retry loop")

    def _is_openai_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an OpenAI API error is retryable.

        Args:
            error: Exception from OpenAI API

        Returns:
            True if the error should be retried, False otherwise
        """
        error_str = str(error).lower()
        retryable_patterns = [
            'rate limit',
            'too many requests',
            'server error',
            'service unavailable',
            'timeout',
            'internal error',
            'overloaded',
            'temporarily unavailable',
            'connection',
            'network'
        ]

        return any(pattern in error_str for pattern in retryable_patterns)

    async def _generate_openrouter(self,
                                  prompt: str,
                                  system_message: Optional[str] = None,
                                  temperature: float = 0.7,
                                  max_tokens: int = 1000,
                                  model: Optional[str] = None,
                                  timeout: Optional[int] = None) -> str:
        """
        Generate text using OpenRouter API with retry logic for transient errors.

        Args:
            prompt: The prompt to send to OpenRouter
            system_message: Optional system message (developer role)
            temperature: Temperature parameter (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            model: Model to use (default: self.model)
            timeout: Timeout in seconds (default: self.timeout)

        Returns:
            Generated text response
        """
        max_retries = 10
        base_delay = 2.0  # 2 seconds base delay

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for OpenRouter API calls")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        # Use provided values or defaults from self
        actual_model = model if model is not None else self.model
        actual_timeout = timeout if timeout is not None else self.timeout

        # Construct messages for OpenRouter chat completions format
        messages = []
        if system_message:
            messages.append({"role": "developer", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        # Retry loop for transient errors
        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 (4 total attempts)
            try:
                completion = client.chat.completions.create(
                    model=actual_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                return completion.choices[0].message.content

            except Exception as e:
                # Check if this is a retryable error
                is_retryable = self._is_openrouter_retryable_error(e)

                if not is_retryable or attempt == max_retries:
                    # Non-retryable error or last attempt - raise immediately
                    raise RuntimeError(f"OpenRouter API error: {str(e)}")

                # Calculate delay with exponential backoff, capped at 180 seconds
                delay = min(base_delay * (2 ** attempt), 180.0)

                print(f"[AUTOINTERP] OpenRouter API error: {str(e)}")
                print(f"[AUTOINTERP] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")

                await asyncio.sleep(delay)
                continue

        # Should never reach here
        raise RuntimeError("Unexpected error in OpenRouter retry loop")

    def _is_openrouter_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an OpenRouter API error is retryable.

        Args:
            error: Exception from OpenRouter API

        Returns:
            True if the error should be retried, False otherwise
        """
        error_str = str(error).lower()
        retryable_patterns = [
            'rate limit',
            'too many requests',
            'server error',
            'service unavailable',
            'timeout',
            'internal error',
            'overloaded',
            'temporarily unavailable',
            'connection',
            'network'
        ]

        return any(pattern in error_str for pattern in retryable_patterns)

    async def generate_with_images(self,
                                 message_content: List[Dict[str, Any]],
                                 system_message: Optional[str] = None,
                                 temperature: Optional[float] = None,
                                 max_tokens: Optional[int] = None,
                                 agent_name: Optional[str] = None,
                                 iteration_number: Optional[int] = None) -> str:
        """
        Generate text using multimodal input (text + images) with the configured LLM.

        Args:
            message_content: List of message content blocks (text and image)
            system_message: Optional system message for chat models
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            agent_name: Optional name of the agent making the request
            iteration_number: Optional iteration number for logging

        Returns:
            Generated text response
        """
        # No parameter overrides allowed - must use agent's configured values
        if temperature is not None or max_tokens is not None:
            raise ValueError("Temperature and max_tokens overrides are not allowed - use values from agent config")

        # Get the agent information - either from initialization or from the parameter
        actual_agent_name = agent_name if agent_name is not None else self.agent_name

        # Get the agent-specific config if a different agent is requested
        if actual_agent_name != self.agent_name:
            if "agents" not in self.config:
                raise ValueError("No agents configuration found in config")

            agent_config = self.config["agents"].get(actual_agent_name)
            if not agent_config:
                raise ValueError(f"Agent '{actual_agent_name}' not found in config")

            # Extract LLM settings for this specific agent
            llm_config = agent_config.get("llm", {})
            provider = llm_config.get("provider", self.provider)
            model = llm_config.get("model", self.model)
            temp = llm_config.get("temperature", self.temperature)
            tokens = llm_config.get("max_tokens", self.max_tokens)
            timeout = llm_config.get("timeout", self.timeout)
        else:
            # Use the current instance's configuration
            provider = self.provider
            model = self.model
            temp = self.temperature
            tokens = self.max_tokens
            timeout = self.timeout

        # Print agent information in the header
        display_name = ColorCodes.get_agent_display_name(actual_agent_name)
        call_start_time = time.time()

        colored_name = ColorCodes.colorize(display_name, actual_agent_name, is_completion=False)
        iteration_info = f" (Iteration {iteration_number})" if iteration_number is not None else ""
        print(f"\n{colored_name}{iteration_info}:")

        # Create log entry for multimodal request
        timestamp = get_timestamp()

        # Create prompt preview from text content blocks
        text_blocks = [block["text"] for block in message_content if block["type"] == "text"]
        prompt_preview = " ".join(text_blocks)[:200] + ("..." if len(" ".join(text_blocks)) > 200 else "")

        # Count image blocks
        image_count = len([block for block in message_content if block["type"] == "image"])

        log_content = f"""Timestamp: {timestamp}
Agent: {actual_agent_name}
Provider: {provider}
Model: {model}
Temperature: {temp}
Max Tokens: {tokens}
Timeout: {timeout}
System Message: {system_message or 'None'}
Message Content (Preview): {prompt_preview}
Image Count: {image_count}

"""

        # Write to log file
        if self.path_resolver:
            log_dir = self.path_resolver.ensure_path("logs")
        else:
            log_dir = Path(self.config.get("logging", {}).get("log_dir", "logs"))
            log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{actual_agent_name}_multimodal_log.txt"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(log_content)

        # Get response based on provider
        response = None
        if provider == "anthropic":
            response = await self._generate_anthropic_multimodal(
                message_content=message_content,
                system_message=system_message,
                temperature=temp,
                max_tokens=tokens,
                model=model,
                timeout=timeout
            )
        elif provider == "openai":
            response = await self._generate_openai_multimodal(
                message_content=message_content,
                system_message=system_message,
                temperature=temp,
                max_tokens=tokens,
                model=model,
                timeout=timeout
            )
        elif provider == "openrouter":
            response = await self._generate_openrouter_multimodal(
                message_content=message_content,
                system_message=system_message,
                temperature=temp,
                max_tokens=tokens,
                model=model,
                timeout=timeout
            )
        else:
            raise ValueError(f"Multimodal support not available for provider: {provider}")

        # Print colored completion indicator
        call_duration = time.time() - call_start_time

        colored_completion = ColorCodes.colorize("✓ Complete", actual_agent_name, is_completion=True)
        print(f"{colored_completion}")

        # Also record in pipeline UI for HTML dashboard
        if self.pipeline_ui:
            text_blocks = [block["text"] for block in message_content if block["type"] == "text"]
            prompt_text = "\n".join(text_blocks)
            image_count = len([block for block in message_content if block["type"] == "image"])
            if image_count:
                prompt_text += f"\n\n[{image_count} image(s) attached]"

            self.pipeline_ui.llm_call_complete(
                agent_name=actual_agent_name,
                display_name=display_name,
                prompt=prompt_text,
                system_message=system_message,
                response=response,
                model=model,
                provider=provider,
                temperature=temp,
                max_tokens=tokens,
                duration_seconds=call_duration,
                iteration_number=iteration_number,
            )

        # Append response to log
        response_content = f"""
Response:
{response}

"""

        with open(log_file, "a") as f:
            f.write(response_content)

        return response

    async def _generate_anthropic_multimodal(self,
                                           message_content: List[Dict[str, Any]],
                                           system_message: Optional[str] = None,
                                           temperature: float = 0.7,
                                           max_tokens: int = 1000,
                                           model: Optional[str] = None,
                                           timeout: Optional[int] = None) -> str:
        """
        Generate text using Anthropic API with multimodal input support.

        Args:
            message_content: List of message content blocks (text and image)
            system_message: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Model name to use
            timeout: Request timeout in seconds

        Returns:
            Generated text response
        """
        max_retries = 3
        base_delay = 2.0

        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp package required for Anthropic API calls")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": self.anthropic_api_version
        }

        # Use provided values or defaults from self
        actual_model = model if model is not None else self.model
        actual_timeout = timeout if timeout is not None else self.timeout

        # Construct the message payload for multimodal
        message_payload = {
            "role": "user",
            "content": message_content
        }

        payload = {
            "model": actual_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [message_payload]
        }

        if system_message:
            payload["system"] = system_message

        # Retry loop for transient errors
        for attempt in range(max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.anthropic.com/v1/messages",
                        headers=headers,
                        json=payload,
                        timeout=actual_timeout
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["content"][0]["text"]

                        # Handle different error types
                        error_text = await response.text()

                        # Check if this is a retryable error
                        is_retryable = self._is_retryable_error(response.status, error_text)

                        if not is_retryable or attempt == max_retries:
                            # Non-retryable error or last attempt - raise immediately
                            raise RuntimeError(f"Anthropic API error ({response.status}): {error_text}")

                        # Calculate delay with exponential backoff, capped at 180 seconds
                        delay = min(base_delay * (2 ** attempt), 180.0)

                        print(f"[AUTOINTERP] API error ({response.status}): {error_text}")
                        print(f"[AUTOINTERP] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")

                        await asyncio.sleep(delay)
                        continue

            except asyncio.TimeoutError as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Anthropic API timeout after {max_retries + 1} attempts: {str(e)}")

                delay = min(base_delay * (2 ** attempt), 180.0)
                print(f"[AUTOINTERP] API timeout error: {str(e)}")
                print(f"[AUTOINTERP] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
                continue

            except aiohttp.ClientError as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Anthropic API client error after {max_retries + 1} attempts: {str(e)}")

                delay = min(base_delay * (2 ** attempt), 180.0)
                print(f"[AUTOINTERP] API client error: {str(e)}")
                print(f"[AUTOINTERP] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
                continue

            except Exception as e:
                # For non-HTTP errors, check if it's a known transient error pattern
                error_str = str(e).lower()
                if any(pattern in error_str for pattern in ['timeout', 'connection', 'network', 'temporary']):
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), 180.0)
                        print(f"[AUTOINTERP] Transient error: {str(e)}")
                        print(f"[AUTOINTERP] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue

                # Non-transient error or last attempt
                raise RuntimeError(f"Error using Anthropic API: {str(e)}")

        # Should never reach here
        raise RuntimeError("Unexpected error in retry loop")

    async def _generate_openai_multimodal(self,
                                        message_content: List[Dict[str, Any]],
                                        system_message: Optional[str] = None,
                                        temperature: float = 0.7,
                                        max_tokens: int = 1000,
                                        model: Optional[str] = None,
                                        timeout: Optional[int] = None) -> str:
        """
        Generate text using OpenAI API with multimodal input support.

        Args:
            message_content: List of message content blocks (text and image)
            system_message: Optional system message (instructions)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Model name to use
            timeout: Request timeout in seconds

        Returns:
            Generated text response
        """
        max_retries = 3
        base_delay = 2.0

        try:
            from openai import OpenAI
            import base64
        except ImportError:
            raise ImportError("openai package required for OpenAI API calls")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        client = OpenAI(api_key=api_key)

        # Use provided values or defaults from self
        actual_model = model if model is not None else self.model
        actual_timeout = timeout if timeout is not None else self.timeout

        # Convert message content to OpenAI format
        openai_content = []
        for block in message_content:
            if block["type"] == "text":
                openai_content.append({"type": "input_text", "text": block["text"]})
            elif block["type"] == "image":
                # Handle image content - convert to base64 if it's a file path
                if "source" in block and block["source"]["type"] == "base64":
                    # Already base64 encoded (Anthropic format)
                    media_type = block["source"]["media_type"]
                    base64_data = block["source"]["data"]
                    openai_content.append({
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{base64_data}"
                    })
                elif "image_url" in block:
                    # Direct image URL (OpenAI format)
                    openai_content.append({
                        "type": "input_image",
                        "image_url": block["image_url"]
                    })

        # Construct input for OpenAI responses API format
        input_messages = [{
            "role": "user",
            "content": openai_content
        }]

        # Retry loop for transient errors
        for attempt in range(max_retries + 1):
            try:
                response = client.responses.create(
                    model=actual_model,
                    instructions=system_message if system_message else "You are a helpful assistant.",
                    input=input_messages
                )

                # Check if response has content (same debugging as text-only)
                if hasattr(response, 'output_text'):
                    output_text = response.output_text
                    if output_text and output_text.strip():
                        return output_text
                    else:
                        print(f"[DEBUG] OpenAI multimodal response.output_text is empty or None")
                        print(f"[DEBUG] Response object: {type(response)}")

                        # Check for alternative response formats
                        if hasattr(response, 'content'):
                            content = response.content
                            if content and hasattr(content, 'text'):
                                return content.text
                            elif content and isinstance(content, str):
                                return content

                        raise RuntimeError(f"OpenAI multimodal API returned empty response")
                else:
                    print(f"[DEBUG] OpenAI multimodal response has no output_text attribute")
                    print(f"[DEBUG] Response object: {type(response)}")

                    # Try alternative response structures
                    if hasattr(response, 'choices') and response.choices:
                        if hasattr(response.choices[0], 'message'):
                            return response.choices[0].message.content
                        elif hasattr(response.choices[0], 'text'):
                            return response.choices[0].text

                    raise RuntimeError(f"OpenAI multimodal API response structure not recognized")

            except Exception as e:
                # Check if this is a retryable error
                is_retryable = self._is_openai_retryable_error(e)

                if not is_retryable or attempt == max_retries:
                    # Non-retryable error or last attempt - raise immediately
                    raise RuntimeError(f"OpenAI multimodal API error: {str(e)}")

                # Calculate delay with exponential backoff, capped at 180 seconds
                delay = min(base_delay * (2 ** attempt), 180.0)

                print(f"[AUTOINTERP] OpenAI multimodal API error: {str(e)}")
                print(f"[AUTOINTERP] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")

                await asyncio.sleep(delay)
                continue

        # Should never reach here
        raise RuntimeError("Unexpected error in OpenAI multimodal retry loop")

    async def _generate_openrouter_multimodal(self,
                                            message_content: List[Dict[str, Any]],
                                            system_message: Optional[str] = None,
                                            temperature: float = 0.7,
                                            max_tokens: int = 1000,
                                            model: Optional[str] = None,
                                            timeout: Optional[int] = None) -> str:
        """
        Generate text using OpenRouter API with multimodal input support.

        Args:
            message_content: List of message content blocks (text and image)
            system_message: Optional system message (developer role)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Model name to use
            timeout: Request timeout in seconds

        Returns:
            Generated text response
        """
        max_retries = 3
        base_delay = 2.0

        try:
            from openai import OpenAI
            import base64
        except ImportError:
            raise ImportError("openai package required for OpenRouter API calls")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        # Use provided values or defaults from self
        actual_model = model if model is not None else self.model
        actual_timeout = timeout if timeout is not None else self.timeout

        # Convert message content to OpenRouter format
        user_content = []
        for block in message_content:
            if block["type"] == "text":
                user_content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                # Handle image content - convert to OpenRouter format
                if "source" in block and block["source"]["type"] == "base64":
                    # Already base64 encoded (Anthropic format)
                    media_type = block["source"]["media_type"]
                    base64_data = block["source"]["data"]
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{base64_data}"}
                    })
                elif "image_url" in block:
                    # Direct image URL (OpenAI/OpenRouter format)
                    user_content.append({
                        "type": "image_url",
                        "image_url": block["image_url"]
                    })

        # Construct messages for OpenRouter chat completions format
        messages = []
        if system_message:
            messages.append({"role": "developer", "content": system_message})
        messages.append({"role": "user", "content": user_content})

        # Retry loop for transient errors
        for attempt in range(max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model=actual_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                return completion.choices[0].message.content

            except Exception as e:
                # Check if this is a retryable error
                is_retryable = self._is_openrouter_retryable_error(e)

                if not is_retryable or attempt == max_retries:
                    # Non-retryable error or last attempt - raise immediately
                    raise RuntimeError(f"OpenRouter multimodal API error: {str(e)}")

                # Calculate delay with exponential backoff, capped at 180 seconds
                delay = min(base_delay * (2 ** attempt), 180.0)

                print(f"[AUTOINTERP] OpenRouter multimodal API error: {str(e)}")
                print(f"[AUTOINTERP] Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")

                await asyncio.sleep(delay)
                continue

        # Should never reach here
        raise RuntimeError("Unexpected error in OpenRouter multimodal retry loop")


class CodeGeneration:
    """
    Specialized interface for code generation and analysis tasks.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        """
        Initialize the code generation interface.
        
        Args:
            llm_interface: LLM interface instance to use for generation
        """
        self.llm = llm_interface
    
    def _extract_code_block(self, text: str, language: str = "python") -> str:
        """
        Extract code block from markdown-formatted text.
        
        Args:
            text: Text containing code blocks
            language: Programming language to look for
            
        Returns:
            Extracted code or original text if no code blocks found
        """
        import re
        
        # Look for code blocks with the specified language
        pattern = rf"```{language}\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            # Return the first code block with the specified language
            return matches[0]
        
        # Look for code blocks without a language specifier
        pattern = r"```\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            # Return the first generic code block
            return matches[0]
        
        # If we get here, no code blocks were found
        return text
    
    async def generate_code(self, 
                          specification: str, 
                          language: str = "python",
                          context: Optional[str] = None) -> str:
        """
        Generate code based on a specification.
        
        Args:
            specification: Description of the code to generate
            language: Programming language to use
            context: Optional additional context (existing code, etc.)
            
        Returns:
            Generated code
        """
        # Get the system message from config if available, otherwise use a default
        system_message = self.llm.config.get("prompts", {}).get("code_generation", {}).get("system_message", 
            f"You are an expert {language} programmer. Generate clean, efficient, and well-documented code. Your response MUST contain ONLY executable code with no introduction, explanation, or concluding text outside of code comments. Do not prefix your response with phrases like 'Here's the code' or explain what the code does outside of code comments.")
        
        prompt = f"Generate {language} code for the following specification:\n\n{specification}"
        
        if context:
            prompt += f"\n\nContext/existing code:\n```{language}\n{context}\n```"
        
        # Use LLM's configured temperature for code generation
        response = await self.llm.generate(
            prompt=prompt,
            system_message=system_message,
            agent_name="analysis_generator"
        )
        
        # Extract code block from response if needed
        code_block = self._extract_code_block(response, language)
        return code_block if code_block else response
    
    async def analyze_code(self, code: str, query: str, language: str = "python") -> str:
        """
        Analyze code and provide insights based on a specific query.
        
        Args:
            code: The code to analyze
            query: Specific question or analysis request
            language: Programming language of the code
            
        Returns:
            Analysis results
        """
        system_message = f"You are an expert in code analysis. Provide clear, accurate, and insightful analysis."
        
        prompt = f"Analyze the following {language} code:\n\n```{language}\n{code}\n```\n\nQuestion/Analysis Request: {query}"
        
        return await self.llm.generate(
            prompt=prompt,
            system_message=system_message,
            agent_name="CodeAnalyzer"
        )
    
    async def improve_code(self, 
                         code: str, 
                         improvement_type: str = "general", 
                         language: str = "python") -> Tuple[str, str]:
        """
        Improve code based on a specific improvement type.
        
        Args:
            code: The code to improve
            improvement_type: Type of improvement (general, performance, readability, security)
            language: Programming language of the code
            
        Returns:
            Tuple of (improved code, explanation of changes)
        """
        system_message = f"You are an expert {language} programmer specializing in code improvement."
        
        prompt = f"Improve the following {language} code for {improvement_type}:\n\n```{language}\n{code}\n```\n\n"
        prompt += "Provide both the improved code and a clear explanation of the changes made."
        
        response = await self.llm.generate(
            prompt=prompt,
            system_message=system_message,
            agent_name="CodeImprover"
        )
        
        # Extract improved code and explanation
        improved_code = self._extract_code_block(response, language)
        
        # If no code block found, return the entire response as both code and explanation
        if not improved_code:
            return response, response
        
        # Remove the code block from the explanation
        explanation = response.replace(f"```{language}\n{improved_code}\n```", "").strip()
        
        return improved_code, explanation
