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
   - Context pack settings (`context_pack.use_agent`, `context_pack.agent_timeout`)
   - Reporting settings (`reporting.use_agent`, `reporting.agent_timeout`)
   - Pipeline UI settings (`ui.html_dashboard`, `ui.dashboard_refresh`, `ui.auto_open_browser`)
   - Interactive mode (`interactive_mode: true` to pause after each stage for user feedback)

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

### arxiv_interp_graph (Context Pack)

AutoInterp ships with the `arxiv_interp_graph` module, which builds a citation graph of interpretability papers and can generate a lightweight context pack. The context pack selects three related papers, downloads their full text (PDF or HTML), and generates research questions — either via an external AI agent (Claude CLI or Codex CLI) or via an LLM API call.

```bash
# Run context pack from the repo root
python main.py context-pack
```

**Question generation strategy:** When the context pack is enabled, the system picks a strategy based on the selected LLM provider:

| Provider | Strategy | CLI tool |
|----------|----------|----------|
| Anthropic | Agent subprocess | `claude` |
| OpenAI | Agent subprocess | `codex` |
| OpenRouter / Manual | LLM API call | — |

The agent reads the downloaded articles (PDFs and HTML files) directly and writes `Research_Questions.txt`. If the agent fails (CLI not installed, timeout, etc.), the system falls back to the LLM API call automatically. Set `context_pack.use_agent: false` in `config.yaml` to always use the LLM API fallback.

**Article download pipeline:** Each paper in the pre-built citation graph (1003 papers) stores an `arxiv_id` or `open_access_url` so downloads work without live API calls in most cases. Papers from Distill and the Transformer Circuits Thread are downloaded as HTML files; all others as PDFs. To re-enrich the graph after adding new papers, run `cd arxiv_interp_graph && python enrich_arxiv_ids.py`.

To use agent mode with Anthropic, install and authenticate the Claude CLI:
```bash
curl -fsSL https://claude.ai/install.sh | bash   # install
claude                                             # first run: follow login prompts
```

The generated questions are written to `questions/questions.txt` and then passed through the normal question prioritizer, which selects the best question and extracts a project title.

Key outputs:
- `projects/<project_id>/literature/manifest.json` (paper metadata)
- `projects/<project_id>/literature/pdfs/` (downloaded articles — PDFs and HTML files)
- `projects/<project_id>/literature/Research_Questions.txt` (agent output, if agent was used)
- `projects/<project_id>/questions/questions.txt` (questions for prioritizer)

### Options Menu

After selecting a provider and model, the CLI prompts `Press [O] for Options, or Enter to continue:`. Pressing `O` opens an interactive menu to adjust common settings without editing `config.yaml`:

```
==================================================
Options
==================================================
[1] Max analysis iterations ............ 6
[2] Confidence threshold ............... 85%
[3] Use CLI agent for analysis ......... true
[4] Use CLI agent for report ........... true
[5] Context pack (literature sampling) . true
[6] Visualization format ............... png
[7] Visualization DPI .................. 300
[8] HTML dashboard ..................... true
[9] Auto-open browser .................. true
[10] Interactive mode (feedback loops) . false

Enter number to edit, or press Enter to finish:
```

After editing, you can apply changes for the current run only or save them as persistent defaults in `.user_options.json`. Saved defaults are loaded automatically on future runs and override `config.yaml` values. Press Enter at the prompt to skip the menu entirely.

### Interactive Mode

Set `interactive_mode: true` in `config.yaml` (or toggle via the Options menu) to enable feedback loops. The pipeline pauses after each major stage and displays its output. You can then:

- **Press Enter** to accept the output and continue to the next stage.
- **Type feedback** and press Enter to have the LLM revise the output. The revised version is displayed and you can provide additional feedback or press Enter to continue.

Interactive mode adds checkpoints after: question generation, question prioritization, analysis plans (legacy mode), analysis evaluations (both modes), visualizations, and the final report. In agent mode analysis, user feedback between iterations is saved to `analysis/background/user_feedback.md` and automatically incorporated into the next iteration's prompt.

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
│   ├── agent_subprocess.py     # Shared Popen + filesystem-polling runner for CLI agents
│   ├── interactive.py          # Interactive mode: feedback loops and LLM revision calls
│   ├── llm_interface.py        # Manages cognitive loop and interactions with LLM
│   ├── pipeline_ui.py          # Pipeline UI — step tracking and HTML dashboard
│   ├── dashboard_template.py   # HTML template and render helpers for dashboard
│   └── utils.py                # General utilities and path resolution
│
├── questions/
│   ├── __init__.py             # Module initialization
│   └── question_manager.py     # Manages research questions
│
├── prompts/
│   ├── prompts.yaml            # Main prompts configuration file
│   ├── interactive.yaml        # Revision prompts for interactive mode feedback loops
│   ├── agent_analysis.yaml     # Prompt template for analysis CLI agent
│   ├── agent_report.yaml       # Prompt template for report CLI agent
│   ├── analysis_generator.yaml # Analysis Generator Prompts
│   ├── analysis_planner.yaml   # Analysis Planning Prompts
│   ├── evaluator.yaml          # Prompts for evaluating analysis results
│   ├── question_manager.yaml   # Prompts for generating and prioritizing questions
│   ├── reporter.yaml           # Prompts for generating final report (legacy)
│   ├── visualization_planner.yaml      # Prompts for visualization planning
│   ├── visualization_generator.yaml    # Prompts for visualization generation
│   └── visualization_evaluator.yaml    # Prompts for visualization evaluation
│
├── analysis/
│   ├── agent_analysis.py       # CLI agent analysis: subprocess, output reading
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
│   ├── agent_report.py         # CLI agent report generation: subprocess, output reading
│   └── report_generator.py     # Creates reproducible reports with visualizations (legacy)
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

By default, analysis iterations are handled by a CLI agent subprocess (Claude CLI or Codex CLI) that autonomously plans, writes code, executes, debugs, and evaluates. When agent mode is unavailable, the system falls back to a legacy 4-module pipeline:

- **Planner**: Creates detailed plans for analysis approaches based on questions and previous results
- **Generator**: Creates Python code for analyses based on questions and plans
- **Executor**: Safely runs analysis code in sandboxed environments and captures results
- **Evaluator**: Assesses results and determines if they increase or decrease confidence that the research question has been adequately answered.

Both modes write all output to the `analysis/` subdirectory within the project.

### Visualization

- **Visualization Planner**: Plans appropriate visualizations for successful analysis results
- **Visualization Generator**: Creates Python visualization code using matplotlib, seaborn, and other libraries
- **Visualization Evaluator**: Uses multimodal LLMs to assess visualization quality and detect issues

### Reporting

By default, the final report is generated by a CLI agent subprocess (Claude CLI or Codex CLI) that reads all analysis outputs and visualizations, then writes an academic-style research report autonomously. The agent writes `Reporter_log.md` (working notes) and `{title}.md` (the final report) to the `reports/` directory. When agent mode is unavailable, the system falls back to a legacy multi-call pipeline:

- **Report Generator**: Produces comprehensive reports with findings, visualizations, and insights in multiple formats

Set `reporting.use_agent: false` in `config.yaml` to always use the legacy pipeline.

### Pipeline UI & HTML Dashboard

During each run, AutoInterp writes a self-contained `dashboard.html` file to the project directory. The dashboard provides:

- **Per-step tabs** — Questions, Prioritize, Analysis, Visualization, Report — each showing all LLM prompts and responses for that stage
- **Auto-refresh** — the page polls for updates during the run and preserves your tab state, scroll position, and expanded/collapsed sections
- **Analysis grouping** — iterative analysis calls are grouped by iteration and attempt, with a color gradient from gold to burnt umber
- **Collapsible sections** — system prompts, user prompts, and assistant responses are each collapsible (all collapsed by default)

The dashboard is opened automatically in your browser when the pipeline starts. After the run completes, auto-refresh is removed so the final file is a static snapshot. Configure dashboard behavior in `config.yaml` under the `ui:` section:

```yaml
ui:
  html_dashboard: true     # Write dashboard.html to project dir
  dashboard_refresh: 5     # Refresh interval in seconds
  auto_open_browser: true  # Open in browser on pipeline start
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or feedback, please open an issue on GitHub or contact akozlo@uchicago.edu.
