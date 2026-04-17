# AutoInterp Agent Framework

The AutoInterp Agent Framework is an automated system for mechanistic interpretability research on Large Language Models (LLMs). It breaks the research process into modular stages that are primarily executed by CLI agent subprocesses (Claude CLI or Codex CLI), with LLM API fallbacks for several stages. The system takes a research question as input and produces a research report with original analyses, visualizations, and interpretation.

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

If you already have a conda or venv environment with your desired packages, simply activate it before running AutoInterp. AutoInterp can detect an active `VIRTUAL_ENV` or `CONDA_PREFIX` during analysis execution:
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

If you don't activate any environment, AutoInterp can either:
- Use Docker sandbox mode if enabled in `config.yaml`
- Create or reuse a project-local `venv/` during analysis execution

3. Ensure necessary environment variables are set, either in the environment or in a .env file:

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
   - Visualization settings (`visualization.use_agent`, `visualization.agent_timeout`)
   - Resource limits
   - Literature search settings (`literature_search.enabled`, `literature_search.n_papers`, `literature_search.use_agent`, `literature_search.agent_timeout`)
   - Reporting settings (`reporting.use_agent`, `reporting.agent_timeout`)
   - Pipeline UI settings (`ui.html_dashboard`, `ui.dashboard_refresh`, `ui.auto_open_browser`)
   - Interactive mode (`interactive_mode: true` to pause after each stage for user feedback)
   - AutoCritique settings (`autocritique.enabled`, `autocritique.use_agent`, `autocritique.agent_timeout`)

## Usage

### Running a Task

```bash
# Default run from the repo root
python main.py

# Override configuration file
python main.py --config path/to/override_config.yaml

# Choose where projects are written
python main.py --projects-dir /absolute/or/relative/path

# Use an existing virtual environment explicitly
python main.py --venv /path/to/venv
```

### Headless / Batch Runs

The `--provider`, `--model`, and `--topic` flags bypass the interactive menus, enabling fully non-interactive execution (e.g. SLURM `sbatch` jobs):

```bash
# Fully headless — specify provider, model, and topic
python main.py run --provider anthropic --model claude-sonnet-4-6 --topic "superposition in LLMs"

# Auto-generate topic from literature search or LLM (empty string)
python main.py run --provider openai --model o3 --topic ""
```

- `--provider` and `--model` must be specified together; omitting both falls through to the interactive menu
- `--topic "some topic"` sets the research topic and disables literature search (same as typing a topic interactively)
- `--topic ""` triggers auto-generation from literature search or LLM, depending on config
- Omitting `--topic` entirely falls through to the interactive prompt

### citation_graph (Literature Search)

AutoInterp ships with the `citation_graph` module, which builds a citation graph of interpretability papers and can generate a lightweight literature search. The literature search selects three related papers, downloads their full text (PDF or HTML), and generates research questions — either via an external AI agent (Claude CLI or Codex CLI) or via an LLM API call.

```bash
# Run literature search from the repo root
python main.py literature-search
```

**Question generation strategy:** When the literature search is enabled, the system picks a strategy based on the selected LLM provider:

| Provider | Strategy | CLI tool |
|----------|----------|----------|
| Anthropic | Agent subprocess | `claude` |
| OpenAI | Agent subprocess | `codex` |
| OpenRouter / Manual | LLM API call | — |

The agent reads the downloaded articles (PDFs and HTML files) directly and writes `Research_Questions.txt`. If the agent fails (CLI not installed, timeout, etc.), the system falls back to the LLM API call automatically. Set `literature_search.use_agent: false` in `config.yaml` to always use the LLM API fallback.

Prompt files live under `prompts/`, while optional persona overlays live under `persona_prompts/` with names like `agent_analysis_persona.yaml`. Persona files are loaded separately and prepended to the main prompt templates at runtime.

**Article download pipeline:** Each paper in the pre-built citation graph (563 papers) stores an `arxiv_id` (94.1%) or `open_access_url` (1.8%) so downloads work without live API calls in most cases. The 23 papers (4.1%) with no stored URL are automatically excluded from literature search sampling — they remain in the graph for topology/statistics but won't be selected. If a download fails at runtime (broken URL, timeout), the system automatically retries with a replacement paper from the graph (up to 3 attempts per slot). Papers from Distill and the Transformer Circuits Thread are downloaded as HTML files; all others as PDFs. To re-enrich the graph after adding new papers, run `cd citation_graph && python enrich_arxiv_ids.py`.

To use agent mode with Anthropic, install and authenticate the Claude CLI:
```bash
curl -fsSL https://claude.ai/install.sh | bash   # install
claude                                             # first run: follow login prompts
```

The generated questions are written to `questions/questions.txt` and then passed through the question prioritizer (also agent-backed by default), which selects the best question and extracts a project title. If the agent is unavailable, the system falls back to an LLM API call.

Key outputs:
- `projects/<project_id>/literature/manifest.json` (paper metadata)
- `projects/<project_id>/literature/pdfs/` (downloaded articles — PDFs and HTML files)
- `projects/<project_id>/literature/Research_Questions.txt` (agent output, if agent was used)
- `projects/<project_id>/questions/questions.txt` (questions for prioritizer)

### Options Menu

The provider selection menu includes an `[5] Options` entry alongside the provider choices. Selecting it opens an interactive menu to adjust common settings without editing `config.yaml`. After closing the Options menu, you are returned to the provider selection screen:

```
==================================================
Options
==================================================
[1] Max analysis iterations ............ 6
[2] Confidence threshold ............... 85%
[3] Use CLI agent for analysis ......... true
[4] Use CLI agent for report ........... true
[5] Use CLI agent for visualization .... true
[6] Literature search .................. true
[7] Articles for question gen .......... 3
[8] Visualization format ............... png
[9] Visualization DPI .................. 300
[10] HTML dashboard .................... true
[11] Auto-open browser ................. true
[12] Interactive mode (feedback loops) . false
[13] AutoCritique (peer review) ........ true

Enter number to edit, or press Enter to finish:
```

After editing, you can apply changes for the current run only or save them as persistent defaults in `.user_options.json`. Saved defaults are loaded automatically on future runs and override `config.yaml` values.

### Manual Configuration (Per-Stage Models)

The provider selection menu includes `[4] Manual Configuration` for fine-grained control over which model each pipeline stage uses. Selecting it opens a per-stage model picker:

```
============================================================
Per-Stage Model Configuration
============================================================
[ 1] Question Generator ........... anthropic / claude-sonnet-4-6
[ 2] Question Prioritizer ......... anthropic / claude-sonnet-4-6
[ 3] Analysis Planner ............. anthropic / claude-sonnet-4-6
[ 4] Analysis Generator ........... anthropic / claude-sonnet-4-6
[ 5] Analysis Evaluator ........... anthropic / claude-sonnet-4-6
[ 6] Visualization Planner ........ anthropic / claude-sonnet-4-6
[ 7] Visualization Generator ...... anthropic / claude-sonnet-4-6
[ 8] Visualization Evaluator ...... anthropic / claude-sonnet-4-6
[ 9] Report Generator ............. anthropic / claude-sonnet-4-6
[10] Title Generator .............. anthropic / claude-sonnet-4-6

Enter number to edit, or press Enter to finish:
```

When you select a stage, you are shown a numbered list of all available models across all providers. You can mix and match providers freely (e.g. use Claude for analysis and GPT for reporting). After editing, changes can be applied for the current run only or saved as persistent defaults in `.user_manual_models.json`.

To add a custom model to the picker, add it to the relevant agent's `llm` section in `config.yaml` — any (provider, model) pair found in the config but not in the built-in catalogue will appear automatically.

### Interactive Mode

Set `interactive_mode: true` in `config.yaml` (or toggle via the Options menu) to enable feedback loops. The pipeline pauses after each major stage and displays its output. You can then:

- **Press Enter** to accept the output and continue to the next stage.
- **Type feedback** and press Enter to have the LLM revise the output. The revised version is displayed and you can provide additional feedback or press Enter to continue.

Interactive mode adds checkpoints after: question generation, question prioritization, analysis plans (legacy mode), analysis evaluations (both modes), visualizations, and the final report. In agent mode analysis, user feedback between iterations is saved to `analysis/background/user_feedback.md` and automatically incorporated into the next iteration's prompt.

### Prompt Testing Harness

The full pipeline takes ~2 hours, making prompt iteration slow. `test_prompt.py` lets you replay individual agent stages (`questions`, `viz`, `report`) against a completed project run, so you can test prompt changes in minutes.

```bash
# Preview the assembled prompt without running (free)
python test_prompt.py viz --project <completed_run> --dry-run

# Run the visualization stage against a completed project
python test_prompt.py viz --project <completed_run>

# A/B test with a modified prompt file
python test_prompt.py viz --project <completed_run> --prompt my_viz_v2.yaml --label "shorter-captions"

# Override provider/model
python test_prompt.py report --project <completed_run> --provider anthropic --model claude-opus-4-6

# Question generation with a specific topic
python test_prompt.py questions --project <completed_run> --task-description "How do attention heads specialize?"
```

The script creates lightweight test run directories under `test_runs/` using symlinks to the source project's input directories. Only the output directory is a real (empty) directory — no disk overhead. Provider/model defaults come from `.last_llm.json`. See `python test_prompt.py --help` for all options.

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
├── main.py                     # Main workflow orchestrator and CLI entrypoint
├── config.yaml                 # Default configuration
├── prompts/                    # Main prompt templates
├── persona_prompts/            # Optional persona overlays (`*_persona.yaml`)
├── src/
│   ├── core/                   # Shared runtime, UI, subprocess, and utility code
│   ├── questions/              # Question generation and prioritization
│   ├── analysis/               # Agent and legacy analysis pipeline
│   ├── visualization/          # Agent and legacy visualization pipeline
│   ├── reporting/              # Report generation and revision
│   ├── autocritique/           # AutoCritique and revision agents
│   ├── literature_review/      # Literature review agent
│   ├── repo/                   # Repo assembly agent
│   ├── notebook/               # Notebook generation agent
│   └── publish/                # Publishing integration
├── citation_graph/             # Citation graph tooling and literature-search support
├── misc/title.txt              # Startup banner
├── test_prompt.py              # Prompt testing harness against completed runs
└── requirements.txt            # Python dependencies
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

By default, visualizations are generated by a CLI agent subprocess (Claude CLI or Codex CLI) that reads all analysis outputs, produces publication-quality figures, and writes captions. The agent writes `Visualization_log.md` (working notes), `figure_{n}.py` (scripts), `figure_{n}.png` (figures), and `caption_{n}.txt` (captions) to the `visualizations/` directory. When agent mode is unavailable, the system falls back to a legacy multi-call pipeline:

- **Visualization Planner**: Plans appropriate visualizations for successful analysis results
- **Visualization Generator**: Creates Python visualization code using matplotlib, seaborn, and other libraries
- **Visualization Evaluator**: Uses multimodal LLMs to assess visualization quality and detect issues

Set `visualization.use_agent: false` in `config.yaml` to always use the legacy pipeline.

### Reporting

By default, the final report is generated by a CLI agent subprocess (Claude CLI or Codex CLI) that reads all analysis outputs and visualizations, then writes an academic-style research report autonomously. The agent writes `Reporter_log.md` (working notes) and `{title}.md` (the final report) to the `reports/` directory. When agent mode is unavailable, the system falls back to a legacy multi-call pipeline:

- **Report Generator**: Produces comprehensive reports with findings, visualizations, and insights in multiple formats

Set `reporting.use_agent: false` in `config.yaml` to always use the legacy pipeline.

### AutoCritique

When `autocritique.enabled: true` (default) and `autocritique.use_agent: true`, an automated peer review step runs after report generation. A CLI agent subprocess reads the report, analyses, and visualizations, then produces a formal review with a verdict (**Reject**, **Revise and Resubmit**, or **Accept**). The agent is instructed to include one of three exact sentinel strings (`Verdict: Reject`, `Verdict: Revise and Resubmit`, `Verdict: Accept`) which the pipeline parses programmatically.

If the verdict is **Revise and Resubmit**, the agent also writes individual `Recommendation_N.md` files — one per issue — that are designed to be fed directly into the analysis agent prompts during a revision cycle.

If the final verdict is **Reject**, the pipeline renames the project directory with a `REJECT_` prefix (e.g. `REJECT_my_study_2026-03-17T10-00-00`) before proceeding to repo assembly and notebook generation.

AutoCritique outputs are organized into round-based subdirectories (`autocritique/round_1/`, `autocritique/round_2/`, etc.) to support multiple review rounds:

```
autocritique/
  round_1/
    AutoCritique_log.md       # Working notes / scratchpad
    AutoCritique_review.md    # Formal review with verdict
    Recommendation_1.md       # (only on "Revise and Resubmit")
    Recommendation_2.md
    ...
  round_2/                    # Created if round 1 triggers revisions
    ...
```

There is no legacy fallback — if the agent can't run (CLI not found, unsupported provider), the step is skipped. Toggle via the Options menu (#13) or set `autocritique.enabled: false` in `config.yaml`.

### Pipeline UI & HTML Dashboard

During each run, AutoInterp writes a self-contained `dashboard.html` file to the project directory. The dashboard provides:

- **Per-step tabs** — Questions, Prioritize, Analysis, Visualization, Report, AutoCritique — each showing all LLM prompts and responses for that stage
- **Real-time progress** — All CLI agent subprocesses (question generation, analysis, visualization, report, autocritique) emit progress updates via filesystem polling every 3 seconds. Named milestone files (plans, scripts, figures, evaluations) get descriptive messages; any other new file the agent creates is also reported. Progress appears as `[~]` lines in the terminal and as a timestamped log in the dashboard
- **Heartbeat** — When no new files are detected for 2 minutes, a "Still running... Xm Ys elapsed" message is emitted so the user always sees activity during long-running agents. An "Agent finished (Xm Ys)" message is emitted when the subprocess exits
- **Agent-only timeout** — Timeout budgets count only agent thinking time, not time spent waiting on child processes (e.g., running a Python analysis script). This uses `psutil` to detect active child processes and pause the timeout clock, so long-running but legitimate script executions won't trigger a premature timeout
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
