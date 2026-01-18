# AutoInterp Agent Framework

The AutoInterp Agent Framework is an automated system designed for mechanistic interpretability research on Large Language Models (LLMs). It adopts a modular approach in which the research process is broken down into steps that are each executed by an API call to an LLM. The system takes as its input a research question and outputs a research report with original analyses,visualizations, and interpretation.

AutoInterp is still in very early stages and may be buggy. Please feel free to submit pull requests or suggest edits; this is intended to be a community project.

## Getting Started

### Prerequisites

- Python 3.8+
- Dependencies from requirements.txt

### Installation

1. Clone the repository:
```bash
git clone https://github.com/akozlo/AutoInterp.git
cd AutoInterp
```

2. Set up a Python environment (choose one option):

**Option A: Use an existing conda/venv environment**

If you already have a conda or venv environment with your desired packages, simply activate it before running AutoInterp. The system will auto-detect and use your active environment:
```bash
conda activate myenv  # or: source myenv/bin/activate
pip install -r requirements.txt
```

**Option B: Create a new virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Option C: Let AutoInterp handle it**

If you don't activate any environment, AutoInterp will either:
- Use Docker sandbox mode (if Docker is installed) - recommended for security
- Create its own venv at `~/.autointerp/venv` and install dependencies automatically

3. (Optional) Install the package to expose the CLI entry point:
```bash
pip install -e .
```

4. Ensure necessary environment variables are set, either in the environment or in a .env file:

```bash
# Only necessary to provide keys for model providers you will use 
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here

# Huggingface token is needed to download some models from transformers
HF_TOKEN=your_key_here
```

### Configuration

Optionally configure `config.yaml` with custom settings:
   - Task details (name, description)
   - Model configuration
   - LLM provider and model 
   - Analysis parameters
   - Visualization settings
   - Resource limits

## Usage

### Running a Task

```bash
# Default run (no install required)
python -m AutoInterp

# Equivalent explicit script call
python main.py

# Override configuration file
python -m AutoInterp --config path/to/override_config.yaml

# Choose where projects are written
python -m AutoInterp --projects-dir /absolute/or/relative/path

# After optional `pip install -e .`
interp-agent --help
```

### Sandboxed Execution with Docker

- Enable the sandbox by setting `analysis.execution.sandbox: true` (default) in `config.yaml`. When enabled, AutoInterp runs generated analysis scripts inside a Docker container instead of directly on the host.
- Configure the container under `execution.docker`:
  - `image`: base image to run (defaults to `python:3.10-slim`; swap for GPU-enabled images such as `pytorch/pytorch:latest` if needed).
  - `use_gpu`: set to `true` to pass through GPUs via `--gpus all` (requires the NVIDIA Container Toolkit).
  - `cache_dir`: persistent host directory where Python packages are installed once and reused across runs.
  - `extra_args`: additional `docker run` flags (for custom networks, resource limits, etc.).
  - `env`: additional environment variables to propagate into the container.
- Hugging Face, pip, and torch caches are mounted automatically so downloads persist between runs. Project artifacts remain on the host because the project directory is bind-mounted read/write.


## Directory Structure
```
.
├── __init__.py                 # Package entry and exports
├── __main__.py                 # Enables `python -m AutoInterp`
├── core/
│   ├── llm_interface.py        # Manages cognitive loop and interactions with LLM
│   └── utils.py                # General utilities and path resolution
│
├── questions/
│   ├── __init__.py             # Module initialization
│   └── question_manager.py     # Manages research questions
│
├── prompts/
│   ├── prompts.yaml            # Main prompts configuration file
│   ├── analysis_generator.yaml # Analysis Generator Prompts
│   ├── analysis_planner.yaml   # Analysis Planning Prompts
│   ├── evaluator.yaml          # Prompts for evaluating analysis results
│   ├── question_manager.yaml   # Prompts for generating and prioritizing questions
│   ├── reporter.yaml           # Prompts for generating final report
│   ├── visualization_planner.yaml      # Prompts for visualization planning
│   ├── visualization_generator.yaml    # Prompts for visualization generation
│   └── visualization_evaluator.yaml    # Prompts for visualization evaluation
│
├── analysis/
│   ├── analysis_executor.py    # Securely executes generated scripts
│   ├── analysis_generator.py   # Dynamically generates analysis scripts
│   ├── analysis_planner.py     # Devise a plan for the next analysis
│   ├── evaluator.py            # Evaluates analysis outcomes
│   └── visualization_evaluator.py      # Evaluates generated visualizations
│
├── visualization/
│   ├── visualization_planner.py# Plans visualizations for analysis results
│   └── visualization_generator.py      # Generates visualization code
│
├── reporting/
│   └── report_generator.py     # Creates reproducible reports with visualizations
│
├── misc/
│   ├── title.txt               # Project title information
│   └── TransformerLens_Notes.txt# Technical notes and documentation
│
├── main.py                     # Main workflow orchestrator
├── config.yaml                 # Configuration parameters (includes task configuration)
└── PROMPTS_README.md           # Documentation for prompt system
```



## AutoInterp System Components

### Question Generation

- **Question Writer**: Writes multiple empirical research questions based on the user's initial input.
- **Question Prioritizer**: Selects one research question based on feasibility, importance, and relevance to the user's request.

### Analysis

- **Planner**: Creates detailed plans for analysis approaches based on questions and previous results
- **Generator**: Creates Python code for analyses based on questions and plans
- **Executor**: Safely runs analysis code in sandboxed environments and captures results
- **Evaluator**: Assesses results and determines if they increase or decrease confidence that the research question has been adequately answered.

### Visualization

- **Visualization Planner**: Plans appropriate visualizations for successful analysis results
- **Visualization Generator**: Creates Python visualization code using matplotlib, seaborn, and other libraries
- **Visualization Evaluator**: Uses multimodal LLMs to assess visualization quality and detect issues

### Reporting

- **Report Generator**: Produces comprehensive reports with findings, visualizations, and insights in multiple formats

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or feedback, please open an issue on GitHub or contact akozlo@uchicago.edu.
