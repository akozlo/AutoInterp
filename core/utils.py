"""
Utility functions for the AutoInterp Agent Framework.
Contains reusable helper functions for logging, timestamps, file handling, and safe command execution.
"""

import os
import sys
import yaml
import time
import logging
import datetime
import subprocess
import tempfile
import shutil
import pkg_resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Set

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Configure logging
def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 console_level: str = "INFO") -> logging.Logger:
    """
    Sets up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console_level: Logging level for console output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Logger object
    """
    level = getattr(logging, log_level.upper())
    console_level_val = getattr(logging, console_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger with console level
    root_logger = logging.getLogger()
    root_logger.setLevel(level)  # Set file logging level
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Add a console handler to the root logger with the specified console level
    root_console_handler = logging.StreamHandler()
    root_console_handler.setLevel(console_level_val)
    root_console_handler.setFormatter(formatter)
    root_logger.addHandler(root_console_handler)
    
    # Setup autointerp logger for file logging
    logger = logging.getLogger("autointerp")
    logger.setLevel(level)  # This is for file logging
    logger.propagate = False  # Don't propagate to root logger
    
    # Clear any existing handlers to prevent duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add file handler if log_file is specified - this will get ALL messages including debug info
    if log_file:
        # Ensure directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)  # Use the full log level for file logging
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Also add file handler to root logger to capture all logs
        root_file_handler = logging.FileHandler(log_file)
        root_file_handler.setLevel(level)
        root_file_handler.setFormatter(formatter)
        root_logger.addHandler(root_file_handler)
    
    return logger

def setup_console_logging_to_file(project_path: Union[str, Path]) -> None:
    """
    Sets up console output capture to a file in the project directory.
    Redirects both stdout and stderr to capture all console output.
    
    Args:
        project_path: Path to the project directory where console.log should be saved
    """
    import sys
    import io
    from pathlib import Path
    
    project_path = Path(project_path)
    console_log_path = project_path / "logs" / "console.log"
    
    # Ensure the logs directory exists
    console_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a tee-like class that writes to both console and file
    class TeeOutput:
        def __init__(self, original, log_file):
            self.original = original
            self.log_file = log_file
            
        def write(self, text):
            # Write to original (console)
            self.original.write(text)
            self.original.flush()
            
            # Write to log file
            try:
                self.log_file.write(text)
                self.log_file.flush()
            except:
                pass  # Ignore errors writing to log file
                
        def flush(self):
            self.original.flush()
            try:
                self.log_file.flush()
            except:
                pass
                
        def fileno(self):
            return self.original.fileno()
    
    # Open the console log file in append mode
    try:
        console_log_file = open(console_log_path, 'a', encoding='utf-8')
        
        # Add a separator to mark new session
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        console_log_file.write(f"\n\n=== NEW SESSION STARTED AT {timestamp} ===\n")
        console_log_file.flush()
        
        # Replace stdout and stderr with tee objects
        sys.stdout = TeeOutput(sys.stdout, console_log_file)
        sys.stderr = TeeOutput(sys.stderr, console_log_file)
        
        print(f"[AUTOINTERP] Console output logging enabled: {console_log_path}")
        
    except Exception as e:
        print(f"[AUTOINTERP] Warning: Could not set up console logging: {e}")

# File handling functions
def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary containing the YAML content
        
    Raises:
        FileNotFoundError: If the file does not exist
        yaml.YAMLError: If the file is not valid YAML
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")


        
def load_txt(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dictionary from a TXT file created by save_txt.
    
    Args:
        file_path: Path to TXT file
        
    Returns:
        Dictionary parsed from the TXT content
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    file_path = Path(file_path)   
    with open(file_path, 'r') as f:
        content = f.read()    
    
    # Parse the TXT content into a dictionary
    result = {}
    current_key = None
    current_value = ""
    multiline_value = False
    indent_level = 0
    
    lines = content.split('\n')
    i = 0
    
    def parse_value(value_str):
        """Parse a string value into appropriate Python type"""
        value_str = value_str.strip()
        if value_str == "None":
            return None
        elif value_str == "True":
            return True
        elif value_str == "False":
            return False
        elif value_str.isdigit():
            return int(value_str)
        elif value_str.replace('.', '', 1).isdigit() and value_str.count('.') == 1:
            return float(value_str)
        else:
            return value_str
    
    # Simple parsing for basic structures
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
            
        # Check for top-level key-value pairs
        if ':' in line and not line.startswith(' '):
            if current_key is not None:
                result[current_key] = parse_value(current_value)
                
            parts = line.split(':', 1)
            current_key = parts[0].strip()
            current_value = parts[1].strip()
            
        # Add to current value
        else:
            current_value += "\n" + line
            
        i += 1
    
    # Add the last key-value pair
    if current_key is not None:
        result[current_key] = parse_value(current_value)
    
    return result

        
def save_txt(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save dictionary to TXT file with a clean, readable format.
    
    Args:
        data: Dictionary to save
        file_path: Path to save TXT file (should end with .txt)
    """
    file_path = Path(file_path)
        
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Format the content in a clean, hierarchical structure
    txt_content = "========================================\n"
    
    def format_value(value, indent_level=0):
        """Format a value with proper indentation and structure"""
        indent = "  " * indent_level
        
        if isinstance(value, dict):
            result = "\n"
            for k, v in value.items():
                result += f"{indent}  {k}: {format_value(v, indent_level + 1)}\n"
            return result
        elif isinstance(value, list):
            if not value:
                return "[]"
            result = "\n"
            for item in value:
                result += f"{indent}  - {format_value(item, indent_level + 1)}\n"
            return result
        elif isinstance(value, bool):
            return "True" if value else "False"
        elif value is None:
            return "None"
        else:
            # Special handling for long strings
            if isinstance(value, str) and len(value) > 80 and "\n" in value:
                return f"\n{indent}  '''\n{indent}  {value}\n{indent}  '''"
            return str(value)
    
    # Add each top-level key with its formatted value
    for key, value in data.items():
        txt_content += f"{key}: {format_value(value)}\n"
    
    # Write the formatted content
    with open(file_path, 'w') as f:
        f.write(txt_content)

def save_file(content: str, file_path: Union[str, Path]) -> None:
    """
    Save string content to a file.
    
    Args:
        content: String content to save
        file_path: Path to save file
    """
    file_path = Path(file_path)
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write(content)

def clean_code_content(code_content: str) -> str:
    """
    Clean code content by removing exact phrases ```python and ```.

    Args:
        code_content: Raw code content that may contain these exact phrases

    Returns:
        Cleaned code content with the exact phrases removed
    """
    if not code_content or not isinstance(code_content, str):
        return code_content

    # Remove ```python first (since it contains ```)
    cleaned_code = code_content.replace('```python', '')

    # Then remove ```
    cleaned_code = cleaned_code.replace('```', '')

    return cleaned_code

# Time-related functions
def get_timestamp(format_string: str = None) -> str:
    """
    Get current timestamp in the specified format.
    
    Args:
        format_string: Optional format string for strftime. If None, returns ISO format.
    
    Returns:
        Formatted timestamp string. Default is ISO format (YYYY-MM-DDTHH:MM:SS)
    """
    now = datetime.datetime.now()
    if format_string:
        return now.strftime(format_string)
    else:
        return now.isoformat(timespec='seconds')

def format_time_elapsed(seconds: float) -> str:
    """
    Format elapsed time in a human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours}h {minutes}m {seconds}s"

# Environment management
def create_virtual_env(base_dir: Union[str, Path] = None) -> Tuple[bool, str, str]:
    """
    Create a new virtual environment for analysis execution.
    Uses a standard location to reuse across a session and avoid accumulation.
    
    Args:
        base_dir: Optional base directory for the virtual environment
        
    Returns:
        Tuple of (success, env_path, error_message)
    """
    try:
        # Use a standard location for the virtual environment
        if base_dir is None:
            base_dir = Path.home() / ".autointerp"
        else:
            base_dir = Path(base_dir)
            
        base_dir.mkdir(parents=True, exist_ok=True)
        env_dir = base_dir / "venv"
        
        # If the environment already exists, return it
        if env_dir.exists():
            # Check if it's a valid venv by looking for the python executable
            python_exec = env_dir / "bin" / "python"
            if sys.platform.startswith('win'):
                python_exec = env_dir / "Scripts" / "python.exe"
                
            if python_exec.exists():
                return True, str(env_dir), ""
        
        # If we get here, either the venv doesn't exist or is invalid,
        # so we should create/recreate it
        if env_dir.exists():
            # Try to remove the existing environment
            try:
                shutil.rmtree(env_dir)
            except Exception as e:
                return False, "", f"Failed to remove existing virtual environment: {str(e)}"
        
        # Create virtual environment
        cmd = [sys.executable, "-m", "venv", str(env_dir)]
        returncode, stdout, stderr = execute_command(cmd, timeout=180)
        
        if returncode != 0:
            return False, "", f"Failed to create virtual environment: {stderr}"
        
        return True, str(env_dir), ""
    except Exception as e:
        return False, "", f"Error creating virtual environment: {str(e)}"

def install_package(env_path: Union[str, Path], package: str) -> Tuple[bool, str]:
    """
    Install a package in the specified virtual environment.
    
    Args:
        env_path: Path to the virtual environment
        package: Package name to install
        
    Returns:
        Tuple of (success, error_message)
    """
    env_path = Path(env_path)
    
    # Determine the pip executable path based on the platform
    if sys.platform.startswith('win'):
        pip_path = env_path / "Scripts" / "pip"
    else:
        pip_path = env_path / "bin" / "pip"
    
    # Ensure pip is up to date
    cmd = [str(pip_path), "install", "--upgrade", "pip"]
    returncode, stdout, stderr = execute_command(cmd, timeout=180)
    
    # Install the package
    cmd = [str(pip_path), "install", package]
    returncode, stdout, stderr = execute_command(cmd, timeout=300)
    
    if returncode != 0:
        return False, f"Failed to install {package}: {stderr}"
        
    return True, ""

def install_requirements(env_path: Union[str, Path], requirements_file: Union[str, Path] = None) -> Tuple[bool, str]:
    """
    Install requirements from a requirements file or a predefined set of interpretability packages.
    
    Args:
        env_path: Path to the virtual environment
        requirements_file: Optional path to requirements.txt file
        
    Returns:
        Tuple of (success, error_message)
    """
    # Default interpretability packages if no requirements file is provided
    # Only install essential packages to speed up initialization
    default_packages = [
        "numpy",
        "matplotlib",
        "torch",
        "transformers",
        "transformer-lens",  # Use hyphen not underscore
        "transformer_lens",  # Also try with underscore for compatibility
        "einops",
        "tqdm",
        "spacy",
        "nltk", 
        "textblob",
        "scikit-learn",
        "pandas",
        "seaborn",
        "scipy"
    ]
    
    env_path = Path(env_path)
    
    # Determine the pip executable path based on the platform
    if sys.platform.startswith('win'):
        pip_path = env_path / "Scripts" / "pip"
    else:
        pip_path = env_path / "bin" / "pip"
    
    # First upgrade pip
    cmd = [str(pip_path), "install", "--upgrade", "pip"]
    execute_command(cmd, timeout=180)
    
    # Log the installation process
    print(f"[AUTOINTERP] Installing packages in virtual environment at {env_path}")
    
    if requirements_file:
        # Install from requirements file
        print(f"[AUTOINTERP] Installing packages from requirements file: {requirements_file}")
        cmd = [str(pip_path), "install", "-r", str(requirements_file)]
        returncode, stdout, stderr = execute_command(cmd, timeout=900)  # Increased timeout
        
        if returncode != 0:
            print(f"[AUTOINTERP] Failed to install requirements: {stderr}")
            return False, f"Failed to install requirements: {stderr}"
        
        print(f"[AUTOINTERP] Successfully installed packages from requirements file")
    else:
        # Install default packages
        print(f"[AUTOINTERP] Installing default packages: {', '.join(default_packages)}")
        
        # Install packages one by one with shorter timeouts
        for package in default_packages:
            print(f"[AUTOINTERP] Installing {package}...")
            cmd = [str(pip_path), "install", package]
            returncode, stdout, stderr = execute_command(cmd, timeout=180)  # Shorter timeout per package
            
            if returncode != 0:
                print(f"[AUTOINTERP] Failed to install {package}: {stderr}")
                # Continue with other packages instead of failing completely
                continue
            else:
                print(f"[AUTOINTERP] Successfully installed {package}")
        
        # Track and report failed packages
        failed_packages = []
        for package in default_packages:
            # Check if the package is actually installed
            if sys.platform.startswith('win'):
                check_cmd = [str(pip_path), "show", package]
            else:
                check_cmd = [str(pip_path), "show", package]
            
            check_code, check_stdout, check_stderr = execute_command(check_cmd, timeout=60)
            if check_code != 0:
                failed_packages.append(package)
                print(f"[AUTOINTERP] Warning: {package} installation verification failed")
        
        if failed_packages:
            print(f"[AUTOINTERP] These packages failed to install and may cause errors: {', '.join(failed_packages)}")
            # Try installing nltk specifically since it's commonly needed
            if "nltk" in failed_packages:
                print(f"[AUTOINTERP] Attempting to install nltk with alternative method...")
                retry_cmd = [str(pip_path), "install", "--force-reinstall", "nltk"]
                retry_code, retry_stdout, retry_stderr = execute_command(retry_cmd, timeout=180)
                if retry_code == 0:
                    print(f"[AUTOINTERP] Successfully installed nltk with alternative method")
                    failed_packages.remove("nltk")
                else:
                    print(f"[AUTOINTERP] Alternative installation of nltk failed: {retry_stderr}")
            
            if failed_packages:
                return False, f"Failed to install these packages: {', '.join(failed_packages)}"
            
        print(f"[AUTOINTERP] Package installation and verification complete")
        return True, "All required packages installed and verified"
    
    # Verify requirements file installation
    print(f"[AUTOINTERP] Verifying package installation...")
    # Check for a few critical packages we know we need
    critical_packages = ["nltk", "transformer_lens", "transformers"]
    failed = []
    
    for package in critical_packages:
        check_cmd = [str(pip_path), "show", package]
        check_code, check_stdout, check_stderr = execute_command(check_cmd, timeout=60)
        if check_code != 0:
            failed.append(package)
            print(f"[AUTOINTERP] Critical package {package} not found after installation")
    
    if failed:
        print(f"[AUTOINTERP] Critical packages missing: {', '.join(failed)}")
        print(f"[AUTOINTERP] Will attempt to install them individually...")
        
        for package in failed:
            print(f"[AUTOINTERP] Installing {package} individually...")
            indiv_cmd = [str(pip_path), "install", "--force-reinstall", package]
            indiv_code, indiv_stdout, indiv_stderr = execute_command(indiv_cmd, timeout=180)
            if indiv_code == 0:
                print(f"[AUTOINTERP] Successfully installed {package}")
                failed.remove(package)
            else:
                print(f"[AUTOINTERP] Failed to install {package}: {indiv_stderr}")
        
        if failed:
            return False, f"Failed to install critical packages: {', '.join(failed)}"
    
    return True, "All required packages installed and verified"

def cleanup_virtual_env(env_path: Union[str, Path]) -> bool:
    """
    Clean up a virtual environment directory.
    This function is now a no-op since we're reusing the environment across a session,
    but kept for API compatibility.
    
    Args:
        env_path: Path to the virtual environment
        
    Returns:
        Boolean indicating success
    """
    # Don't actually remove the environment since we're reusing it
    # Just log that we're skipping cleanup
    logging.getLogger("autointerp.utils").debug(f"Skipping cleanup of persistent environment: {env_path}")
    return True

def execute_in_virtual_env(env_path: Optional[Union[str, Path]], script_path: Union[str, Path], 
                          args: List[str] = None, timeout: int = 300,
                          cwd: Optional[Union[str, Path]] = None) -> Tuple[int, str, str]:
    """
    Execute a script in a virtual environment or using system Python.
    
    Args:
        env_path: Path to the virtual environment, or None to use system Python
        script_path: Path to the script to execute
        args: Additional arguments for the script
        timeout: Timeout in seconds
        cwd: Working directory
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    script_path = Path(script_path)
    
    # If env_path is None, use system Python
    if env_path is None:
        print(f"[AUTOINTERP] Using system Python to execute {script_path}")
        python_path = sys.executable
    else:
        env_path = Path(env_path)
        # Determine the python executable path based on the platform
        if sys.platform.startswith('win'):
            python_path = env_path / "Scripts" / "python"
        else:
            python_path = env_path / "bin" / "python"
    
    # Verify python executable exists
    if not Path(python_path).exists():
        print(f"[AUTOINTERP] ERROR: Python executable not found at {python_path}")
        return 1, "", f"Python executable not found at {python_path}"
    
    # Build command
    cmd = [str(python_path), str(script_path)]
    if args:
        cmd.extend(args)
    
    # Execute command
    return execute_command(cmd, timeout=timeout, cwd=cwd)

def handle_module_error(env_path: Optional[Union[str, Path]], error_msg: str) -> Tuple[bool, str]:
    """
    Handle ModuleNotFoundError by attempting to install the missing module.
    
    Args:
        env_path: Path to the virtual environment, or None if using system Python
        error_msg: Error message from the failed execution
        
    Returns:
        Tuple of (success, message)
    """
    # If env_path is None, we're using system Python and can't install packages
    if env_path is None:
        print(f"[AUTOINTERP] Using system Python - cannot install missing packages automatically.")
        print(f"[AUTOINTERP] Please install any missing packages manually using pip.")
        
        # Try to extract the missing module name for better error reporting
        if "ModuleNotFoundError" in error_msg or "ImportError" in error_msg:
            import re
            # Enhanced pattern matching (matching analysis_executor.py)
            patterns = [
                r"ModuleNotFoundError: No module named '([^']+)'",
                r"ImportError: No module named '([^']+)'",
                r"ImportError: cannot import name '([^']+)'",
                r"ImportError: No module named ([^\s\n]+)",
                r"import ([^\s\n]+).*ImportError",
                r"from ([^\s\.\n]+).*ImportError"
            ]
            
            missing_modules = set()
            
            # Extract all missing modules
            for pattern in patterns:
                matches = re.findall(pattern, error_msg, re.IGNORECASE)
                for match in matches:
                    # Clean up the module name
                    module_name = match.strip().strip("'\"").split('.')[0]  # Get root module
                    if module_name and not module_name.startswith('_'):
                        missing_modules.add(module_name)
            
            if missing_modules:
                # Comprehensive mapping of module names to pip install names (matching analysis_executor.py)
                pip_module_map = {
                    # Machine Learning & AI
                    "transformer_lens": "transformer-lens",
                    "sklearn": "scikit-learn", 
                    "scikit_learn": "scikit-learn",
                    "cv2": "opencv-python",
                    "PIL": "Pillow",
                    "Image": "Pillow",
                    
                    # Data & Visualization
                    "plotly": "plotly",
                    "bokeh": "bokeh",
                    "altair": "altair",
                    
                    # Natural Language Processing
                    "spacy": "spacy",
                    "nltk": "nltk", 
                    "textblob": "textblob",
                    "gensim": "gensim",
                    
                    # Deep Learning
                    "torchvision": "torchvision",
                    "tensorflow": "tensorflow",
                    "tf": "tensorflow",
                    "keras": "keras",
                    
                    # Utilities
                    "requests": "requests",
                    "bs4": "beautifulsoup4",
                    "lxml": "lxml",
                    "openpyxl": "openpyxl",
                    "yaml": "PyYAML",
                    "dateutil": "python-dateutil",
                    "dotenv": "python-dotenv"
                }
                
                for missing_module in missing_modules:
                    pip_module = pip_module_map.get(missing_module, missing_module)
                    print(f"[AUTOINTERP] Missing module: {missing_module}")
                    print(f"[AUTOINTERP] To install it manually, run: pip install {pip_module}")
        
        return False, "Cannot install missing modules when using system Python"
    
    # Enhanced error pattern detection (matching analysis_executor.py)
    import re
    
    module_patterns = [
        r"ModuleNotFoundError: No module named '([^']+)'",
        r"ImportError: No module named '([^']+)'", 
        r"ImportError: cannot import name '([^']+)'",
        r"ImportError: No module named ([^\s\n]+)",
        r"import ([^\s\n]+).*ImportError",
        r"from ([^\s\.\n]+).*ImportError"
    ]
    
    missing_modules = set()
    
    # Extract all missing modules
    for pattern in module_patterns:
        matches = re.findall(pattern, error_msg, re.IGNORECASE)
        for match in matches:
            # Clean up the module name
            module_name = match.strip().strip("'\"").split('.')[0]  # Get root module
            if module_name and not module_name.startswith('_'):
                missing_modules.add(module_name)
    
    if not missing_modules:
        return False, "Could not identify missing module from error message"
    
    # Comprehensive mapping of module names to pip install names (matching analysis_executor.py)
    pip_module_map = {
        # Machine Learning & AI
        "transformer_lens": "transformer-lens",
        "sklearn": "scikit-learn",
        "scikit_learn": "scikit-learn",
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "Image": "Pillow",
        
        # Data & Visualization
        "plotly": "plotly",
        "bokeh": "bokeh",
        "altair": "altair",
        
        # Natural Language Processing
        "spacy": "spacy",
        "nltk": "nltk", 
        "textblob": "textblob",
        "gensim": "gensim",
        
        # Deep Learning
        "torchvision": "torchvision",
        "tensorflow": "tensorflow",
        "tf": "tensorflow",
        "keras": "keras",
        
        # Utilities
        "requests": "requests",
        "bs4": "beautifulsoup4",
        "lxml": "lxml",
        "openpyxl": "openpyxl",
        "yaml": "PyYAML",
        "dateutil": "python-dateutil",
        "dotenv": "python-dotenv"
    }
    
    installed_any = False
    
    for missing_module in missing_modules:
        pip_module = pip_module_map.get(missing_module, missing_module)
        
        print(f"[AUTOINTERP] Attempting to install missing module: {missing_module} -> {pip_module}")
        
        # Try to install the missing module with enhanced retry logic
        success, err_msg = install_package(env_path, pip_module)
        if success:
            print(f"[AUTOINTERP] Successfully installed module: {pip_module}")
            installed_any = True
        else:
            print(f"[AUTOINTERP] Failed to install module {pip_module}: {err_msg}")
            
            # Try common fallbacks for specific packages
            if missing_module in ["transformer_lens", "circuitsvis", "jaxtyping"]:
                # These packages may need specific versions or install methods
                
                # For transformer_lens, first ensure torch is installed
                if missing_module == "transformer_lens":
                    print(f"[AUTOINTERP] Attempting to first install torch dependency")
                    install_package(env_path, "torch")
                    install_package(env_path, "einops")
                    install_package(env_path, "transformers")
                    success, err_msg = install_package(env_path, "transformer-lens")
                    if success:
                        print(f"[AUTOINTERP] Successfully installed transformer-lens after dependencies")
                        installed_any = True
    
    if installed_any:
        return True, f"Installed missing modules: {', '.join(missing_modules)}"
    else:
        return False, f"Failed to install missing modules: {', '.join(missing_modules)}"

# Safe command execution
def execute_command(command: List[str], 
                   timeout: int = 300, 
                   cwd: Optional[Union[str, Path]] = None) -> Tuple[int, str, str]:
    """
    Safely execute a shell command with timeout.
    
    Args:
        command: Command to execute as a list of strings
        timeout: Maximum execution time in seconds
        cwd: Working directory for command execution
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    try:
        # Use subprocess.run for better control and security
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        return process.returncode, process.stdout, process.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"Command timed out after {timeout} seconds: {' '.join(command)}"
    except Exception as e:
        return 1, "", f"Error executing command: {e}"

# Path handling
def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(directory)
    # Check if directory exists before creating it to avoid any automatic renaming
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

# Path management
class PathResolver:
    """
    Central path resolver for the AutoInterp Agent Framework.
    Provides a single source of truth for all path resolution throughout the framework.
    """
    _instance = None
    
    def __new__(cls, config: Dict[str, Any] = None):
        """
        Singleton pattern to ensure only one path resolver exists.
        
        Args:
            config: Configuration dictionary with project paths
        """
        if cls._instance is None:
            cls._instance = super(PathResolver, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the path resolver.
        
        Args:
            config: Configuration dictionary with project paths
        """
        # Only initialize once due to singleton pattern
        if self._initialized:
            return
            
        self.config = config or {}
        self.project_id = self.config.get("project_id", "working_project")

        # Always use absolute paths to avoid working directory issues
        default_projects_path = PACKAGE_ROOT / "projects"
        base_projects_value = self.config.get("paths", {}).get("projects", default_projects_path)
        base_projects_path = Path(base_projects_value).expanduser()

        if not base_projects_path.is_absolute():
            self.base_project_dir = (PACKAGE_ROOT / base_projects_path).resolve()
        else:
            self.base_project_dir = base_projects_path.resolve()

        self._initialized = True
        
        # Log initial configuration
        logger = logging.getLogger("autointerp.paths")
        logger.debug(f"PathResolver initialized with project_id: {self.project_id}")
        logger.debug(f"Base project directory: {self.base_project_dir}")
        logger.debug(f"Base project directory (absolute): {self.base_project_dir.resolve()}")
        logger.debug(f"Current working directory: {Path.cwd()}")
    
    def update_project_id(self, project_id: str) -> None:
        """
        Update the project ID used for path resolution.
        
        Args:
            project_id: New project ID
        """
        old_id = self.project_id
        self.project_id = project_id
        logger = logging.getLogger("autointerp.paths")
        logger.info(f"Project ID updated from '{old_id}' to '{project_id}'")
    
    def get_project_dir(self) -> Path:
        """
        Get the current project directory path.
        
        Returns:
            Path to the project directory
        """
        return self.base_project_dir / self.project_id
    
    def get_path(self, component_type: str, subpath: str = "") -> Path:
        """
        Get path for a specific component within the project directory.
        
        Args:
            component_type: Type of component (e.g., 'analysis_scripts', 'visualizations')
            subpath: Optional subdirectory or file within the component directory
            
        Returns:
            Absolute path to the requested location
        """
        project_dir = self.get_project_dir()
        component_dir = project_dir / component_type
        
        if subpath:
            return component_dir / subpath
        return component_dir
    
    def ensure_path(self, component_type: str, subpath: str = "") -> Path:
        """
        Get path for a component and ensure the directory exists.
        
        Args:
            component_type: Type of component (e.g., 'analysis_scripts', 'visualizations')
            subpath: Optional subdirectory or file within the component directory
            
        Returns:
            Absolute path to the requested location
        """
        path = self.get_path(component_type, subpath)
        if subpath and "." in subpath.split("/")[-1]:  # If subpath looks like a file
            ensure_directory(path.parent)
        else:
            ensure_directory(path)
        return path
        
    def get_analysis_plans_dir(self) -> Path:
        """
        Get the directory path for analysis plans.
        
        Returns:
            Path to the analysis plans directory
        """
        return self.ensure_path("analysis_plans")
        
    def get_prioritized_question_path(self) -> Path:
        """
        Get the path to the prioritized question file.
        
        Returns:
            Path to the prioritized question file
        """
        return self.get_path("questions", "prioritized_question.txt")
    
        
    def get_evaluation_dir(self) -> Path:
        """
        Get the directory path for evaluation results.
        
        Returns:
            Path to the evaluation results directory
        """
        return self.get_path("evaluation_results")

# Configuration validation


def load_prompts(base_path: Union[str, Path], main_file: str = "prompts.yaml") -> Dict[str, Any]:
    """
    Load and merge prompt YAML files.
    
    Args:
        base_path: Base directory containing prompt files
        main_file: Main prompts file that may import other files
        
    Returns:
        Dictionary containing all merged prompts
        
    Raises:
        FileNotFoundError: If the main file or an imported file does not exist
        yaml.YAMLError: If any file is not valid YAML
    """
    # Get logger for consistent handling of debug output
    logger = logging.getLogger("autointerp.prompts")
    
    base_path = Path(base_path)
    prompts_file = base_path / main_file
    
    logger.debug(f"Loading prompts from base path: {base_path}")
    logger.debug(f"Main prompts file: {prompts_file}")
    
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    # Load the main prompts file
    with open(prompts_file, 'r') as f:
        try:
            prompts_config = yaml.safe_load(f) or {}
            logger.debug(f"Main prompts file content: {list(prompts_config.keys())}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing prompts file {prompts_file}: {e}")
    
    # Handle imports
    if "imports" in prompts_config:
        imports = prompts_config.pop("imports", [])
        logger.debug(f"Found imports in main prompts file: {imports}")
        
        for import_file in imports:
            import_path = base_path / import_file
            logger.debug(f"Importing file: {import_path}")
            
            if not import_path.exists():
                logger.error(f"Import file not found: {import_path}")
                raise FileNotFoundError(f"Imported prompts file not found: {import_path}")
            
            with open(import_path, 'r') as f:
                try:
                    imported_config = yaml.safe_load(f) or {}
                    logger.debug(f"Imported config from {import_file}: {list(imported_config.keys())}")
                except yaml.YAMLError as e:
                    raise yaml.YAMLError(f"Error parsing imported prompts file {import_path}: {e}")
            
            # Merge imported config with main config based on file name
            # For question_manager.yaml, use a special structure
            if 'question_manager.yaml' in str(import_path):
                # Create question_manager key if it doesn't exist
                if 'question_manager' not in prompts_config:
                    prompts_config['question_manager'] = {}
                
                # Now add everything from the import under question_manager
                for section, contents in imported_config.items():
                    logger.debug(f"Adding section '{section}' from question_manager.yaml to question_manager key")
                    prompts_config['question_manager'][section] = contents
                
                # Also add backward compatibility under question_manager
                if 'question_manager' not in prompts_config:
                    prompts_config['question_manager'] = {}
                for section, contents in imported_config.items():
                    prompts_config['question_manager'][section] = contents
            
            # For analysis_generator.yaml, structure correctly
            elif 'analysis_generator.yaml' in str(import_path):
                # Create analysis_generator key if it doesn't exist
                if 'analysis_generator' not in prompts_config:
                    prompts_config['analysis_generator'] = {}
                
                # Now directly add all items under analysis_generator
                for key, value in imported_config.items():
                    prompts_config['analysis_generator'][key] = value
                    logger.debug(f"Added key '{key}' under analysis_generator")
            
            # For evaluator.yaml, structure correctly (check exact filename)
            elif import_path.name == 'evaluator.yaml':
                # Create evaluator key if it doesn't exist
                if 'evaluator' not in prompts_config:
                    prompts_config['evaluator'] = {}
                
                # Now directly add all items under evaluator
                for key, value in imported_config.items():
                    prompts_config['evaluator'][key] = value
                    logger.debug(f"Added key '{key}' under evaluator")
                    
            # For analysis_planner.yaml, structure correctly
            elif 'analysis_planner.yaml' in str(import_path):
                # Create analysis_planner key if it doesn't exist
                if 'analysis_planner' not in prompts_config:
                    prompts_config['analysis_planner'] = {}
                
                # Now directly add all items under analysis_planner
                for key, value in imported_config.items():
                    prompts_config['analysis_planner'][key] = value
                    logger.debug(f"Added key '{key}' under analysis_planner")
            
            # For reporter.yaml, structure correctly
            elif 'reporter.yaml' in str(import_path):
                # Create reporter key if it doesn't exist
                if 'reporter' not in prompts_config:
                    prompts_config['reporter'] = {}
                
                # Now directly add all items under reporter
                for key, value in imported_config.items():
                    prompts_config['reporter'][key] = value
                    logger.debug(f"Added key '{key}' under reporter")
            
            # For visualization_planner.yaml, structure correctly
            elif 'visualization_planner.yaml' in str(import_path):
                # Create visualization_planner key if it doesn't exist
                if 'visualization_planner' not in prompts_config:
                    prompts_config['visualization_planner'] = {}
                
                # Now directly add all items under visualization_planner
                for key, value in imported_config.items():
                    prompts_config['visualization_planner'][key] = value
                    logger.debug(f"Added key '{key}' under visualization_planner")
            
            # For visualization_generator.yaml, structure correctly
            elif 'visualization_generator.yaml' in str(import_path):
                # Create visualization_generator key if it doesn't exist
                if 'visualization_generator' not in prompts_config:
                    prompts_config['visualization_generator'] = {}

                # Now directly add all items under visualization_generator
                for key, value in imported_config.items():
                    prompts_config['visualization_generator'][key] = value
                    logger.debug(f"Added key '{key}' under visualization_generator")

            # For visualization_evaluator.yaml, structure correctly
            elif import_path.name == 'visualization_evaluator.yaml':
                # Create visualization_evaluator key if it doesn't exist
                if 'visualization_evaluator' not in prompts_config:
                    prompts_config['visualization_evaluator'] = {}

                # Now directly add all items under visualization_evaluator
                for key, value in imported_config.items():
                    prompts_config['visualization_evaluator'][key] = value
                    logger.debug(f"Added key '{key}' under visualization_evaluator")

            # For other files, use the old way
            else:
                # Merge imported config with main config
                # For each section in the imported config
                for section, contents in imported_config.items():
                    logger.debug(f"Merging section '{section}' from {import_file}")
                    # If this section doesn't exist in the main config, create it
                    if section not in prompts_config:
                        prompts_config[section] = {}
                    
                    # If contents is a dict, merge it with existing section
                    if isinstance(contents, dict):
                        for key, value in contents.items():
                            prompts_config[section][key] = value
                            logger.debug(f"Added key '{key}' to section '{section}'")
                    else:
                        # For non-dict contents, just replace/add the value
                        prompts_config[section] = contents
                        logger.debug(f"Replaced section '{section}' with non-dict content")
    
    # Load TransformerLens_Notes.txt and append to analysis_generator system_message
    transformerlens_notes_path = base_path.parent / "misc" / "TransformerLens_Notes.txt"
    if transformerlens_notes_path.exists():
        try:
            with open(transformerlens_notes_path, 'r') as f:
                notes_content = f.read()
            
            # Append the notes to the analysis_generator system_message
            if 'analysis_generator' in prompts_config and 'system_message' in prompts_config['analysis_generator']:
                current_system_message = prompts_config['analysis_generator']['system_message']
                prompts_config['analysis_generator']['system_message'] = current_system_message + "\n\n" + notes_content
                logger.debug("Successfully appended TransformerLens_Notes.txt to analysis_generator system_message")
            else:
                logger.warning("analysis_generator system_message not found, could not append TransformerLens_Notes.txt")
        except Exception as e:
            logger.warning(f"Could not load TransformerLens_Notes.txt: {e}")
    else:
        logger.debug(f"TransformerLens_Notes.txt not found at {transformerlens_notes_path}")
    
    logger.debug(f"Final config structure: {list(prompts_config.keys())}")
    
    return prompts_config

# Comprehensive logging functionality
def log_to_comprehensive_log(project_path: Union[str, Path], content: str, section_title: str = None) -> None:
    """
    Append content to the comprehensive log file in the project's logs directory.
    
    Args:
        project_path: Path to the project directory
        content: Content to append to the log
        section_title: Optional title for this section (will be formatted with separators)
    """
    project_path = Path(project_path)
    logs_dir = project_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    comprehensive_log_path = logs_dir / "comprehensive_log.txt"
    
    try:
        # Create the file if it doesn't exist
        if not comprehensive_log_path.exists():
            with open(comprehensive_log_path, 'w', encoding='utf-8') as f:
                f.write("=== COMPREHENSIVE EXECUTION LOG ===\n")
                f.write(f"Started: {get_timestamp()}\n")
                f.write("="*80 + "\n\n")
        
        # Append the new content
        with open(comprehensive_log_path, 'a', encoding='utf-8') as f:
            if section_title:
                f.write(f"\n{'='*20} {section_title} {'='*20}\n")
                f.write(f"Timestamp: {get_timestamp()}\n")
                f.write("-"*80 + "\n")
            f.write(content)
            f.write("\n" + "-"*80 + "\n\n")
            
    except Exception as e:
        logger = logging.getLogger("autointerp.comprehensive_log")
        logger.error(f"Failed to write to comprehensive log: {e}")
        print(f"[AUTOINTERP] Warning: Failed to write to comprehensive log: {e}")

def get_comprehensive_log_path(project_path: Union[str, Path]) -> Path:
    """
    Get the path to the comprehensive log file.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Path to the comprehensive_log.txt file
    """
    project_path = Path(project_path)
    return project_path / "logs" / "comprehensive_log.txt"
