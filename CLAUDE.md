# CLAUDE.md

## Project Overview

AutoInterp is an automated mechanistic interpretability research framework. It takes a research question as input and produces a research report with original analyses, visualizations, and interpretation. Each step of the research process is executed by an LLM agent.

## Quick Start

```bash
pip install -r requirements.txt
python main.py          # interactive provider selection, then full pipeline
python main.py run      # same as above
python main.py context-pack  # run context pack only (no full pipeline)
```

Environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN` (set whichever provider you use).

## Key Architecture

### Pipeline Flow (`main.py` → `streamlined_pipeline()`)

1. **Context Pack** (optional) — sample 3 papers from citation graph, download articles (PDF or HTML) to `literature/`, generate research questions
2. **Question Generation** — LLM writes candidate research questions (skipped if context pack produced questions)
3. **Question Prioritization** — LLM selects best question, extracts TITLE, renames project dir (always runs)
4. **Iterative Analysis** — agent mode (default): CLI agent subprocess plans, codes, executes, debugs, evaluates autonomously; legacy mode: plan → generate code → execute in sandbox → evaluate → repeat until confident
5. **Visualization** — plan and generate visualizations for analysis results
6. **Report Generation** — produce final markdown/HTML report

### Context Pack Question Generation

When `context_pack.enabled: true` in `config.yaml`, the system builds a 3-paper pack from `arxiv_interp_graph/output/graph_state.json` and generates questions. The strategy depends on the LLM provider:

- **Anthropic** → runs `claude` CLI agent (subprocess with `--dangerously-skip-permissions`)
- **OpenAI** → runs `codex` CLI agent (subprocess with `-s workspace-write`)
- **Other/fallback** → direct LLM API call via `_generate_question_llm()`

The agent reads articles from `literature/pdfs/` (PDFs and HTML files) and writes `Research_Questions.txt`. Output is copied to `questions/questions.txt` and the prioritizer always runs afterward.

### Article Download Pipeline

Each paper in the citation graph stores download metadata to avoid live API calls:

| Source | Coverage | Method |
|--------|----------|--------|
| `arxiv_id` | 937/1003 (93%) | Construct `https://arxiv.org/pdf/{id}.pdf` directly |
| `open_access_url` | 16/1003 | Stored URL from S2 `openAccessPdf` (Distill, journals) |
| S2 API fallback | 49/1003 | Live API call at download time |

HTML articles (Distill, Transformer Circuits Thread) are saved as `.html` files alongside PDFs. The agent prompt and LLM fallback both handle both formats.

To re-enrich the graph after adding new papers:
```bash
cd arxiv_interp_graph && python enrich_arxiv_ids.py
```

**Prerequisites for agent mode:** The `claude` CLI must be installed and authenticated:
```bash
curl -fsSL https://claude.ai/install.sh | bash   # install
claude                                             # first run: follow login prompts
```
For OpenAI, the `codex` CLI must be installed and authenticated separately.

Config fields:
- `context_pack.use_agent` (default `true`) — use CLI agent; `false` = always use LLM API
- `context_pack.agent_timeout` (default `600`) — subprocess timeout in seconds

Agent logic lives in `arxiv_interp_graph/context_pack/agent_questions.py`.

**Smoke-tested:** The agent subprocess path (Anthropic/`claude`) has been verified end-to-end — prompt substitution, subprocess invocation, PDF reading by the agent, `Research_Questions.txt` creation, and content retrieval all work. The full pipeline path requires `arxiv_interp_graph/output/graph_state.json` to exist (build it with `python arxiv_interp_graph/cli.py build`).

### Analysis Agent Mode (`analysis/agent_analysis.py`)

When `analysis.use_agent: true` (default) and the provider is `anthropic` or `openai`, each analysis iteration is handled by a single CLI agent subprocess instead of the 4-module legacy pipeline. The agent autonomously plans, writes code, executes it, debugs failures, writes an evaluation, and updates a confidence tracker.

Both agent and legacy pipelines write to the unified `analysis/` directory. The directory layouts are compatible but distinct:

**Directory layout (agent mode):**
```
analysis/
  background/
    Research_Question.md    # Copied from prioritized_question.txt
    confidence.json         # Running confidence tracker
  analysis_1/
    ANALYSIS_1_PLAN.md
    *.py                    # Analysis scripts (written + executed by agent)
    *.png                   # Generated figures
    ANALYSIS_1_EVALUATION.md
  analysis_2/
    ...
```

**Directory layout (legacy mode):**
```
analysis/
  a1_analysis_plan_*.txt   # Plans from AnalysisPlanner
  a2_analysis_plan_*.txt
  analysis_1/
    attempt_1/
      analysis_*.py         # Generated script
      analysis_generator_*.txt  # Prompt + response debug file
      stdout.txt            # Execution output
    attempt_2/
      ...
  analysis_2/
    ...
```

**Fallback rules:**
- `analysis.use_agent: false` → always use legacy pipeline
- Provider not `anthropic`/`openai` → legacy pipeline
- CLI binary not found → legacy pipeline with warning
- 2 consecutive agent failures → stop analysis early

Config fields:
```yaml
analysis:
  use_agent: true       # default true; false = legacy pipeline
  agent_timeout: 1800   # per-iteration subprocess timeout (seconds)
```

Agent logic lives in `analysis/agent_analysis.py`. Prompt template in `prompts/agent_analysis.yaml`.

### Pipeline UI (`core/pipeline_ui.py` + `core/dashboard_template.py`)

The pipeline produces two forms of output:

1. **CLI terminal** — the original colorful verbose ANSI output from `LLMInterface` (always printed)
2. **HTML dashboard** — a self-contained, auto-refreshing `dashboard.html` written to the project directory

`PipelineUI` is created in `async_main()` and attached to `LLMInterface.pipeline_ui`. Every `generate()` call records an `LLMInteraction` and rewrites the dashboard. Step lifecycle calls (`step_start`, `step_complete`, etc.) are made from `streamlined_pipeline()`.

The dashboard uses a dark "Civilization 2" theme with per-step color coding: green (questions), blue (prioritize), gold→burnt umber gradient (analysis), teal (visualization), coral (report). Tabs double as a progress bar. All prompt/response sections are collapsed by default. Auto-refresh uses `fetch()` + DOM diffing to preserve tab state, scroll position, and open/closed details.

Config (`config.yaml`):
```yaml
ui:
  rich_terminal: true      # Enable Rich library for terminal panels (unused currently — verbose output always prints)
  html_dashboard: true     # Write dashboard.html to project dir
  dashboard_refresh: 5     # Auto-refresh interval in seconds
  auto_open_browser: true  # Open dashboard in browser on pipeline start
```

## Important File Locations

| File | Purpose |
|------|---------|
| `main.py` | Main orchestrator, CLI entry point, `streamlined_pipeline()` |
| `config.yaml` | All configuration (providers, agents, execution, context pack, UI) |
| `core/llm_interface.py` | LLM API abstraction (Anthropic, OpenAI, OpenRouter) |
| `core/pipeline_ui.py` | `PipelineUI` class — step tracking, LLM interaction recording, HTML dashboard |
| `core/dashboard_template.py` | HTML template, CSS, JS, and render helpers for the dashboard |
| `core/utils.py` | `PathResolver` singleton, utilities |
| `questions/question_manager.py` | Question file I/O, prioritization |
| `arxiv_interp_graph/context_pack/` | Context pack: sampling, download, agent questions, run |
| `arxiv_interp_graph/context_pack/agent_questions.py` | CLI agent subprocess invocation |
| `arxiv_interp_graph/context_pack/run.py` | Context pack orchestration + LLM fallback |
| `arxiv_interp_graph/context_pack/download.py` | Article download (PDF + HTML) + manifest writing |
| `prompts/question_manager.yaml` | All question prompts (generator, prioritizer, agent question generator) |
| `analysis/agent_analysis.py` | CLI agent analysis: workspace setup, subprocess, output reading |
| `prompts/agent_analysis.yaml` | Prompt template for analysis agent iterations |
| `arxiv_interp_graph/enrich_arxiv_ids.py` | Batch-enrich graph with arxiv_id + open_access_url |
| `.last_llm.json` | Persisted provider/model selection from last run |
| `prompts/*.yaml` | Agent-specific prompt templates |

## Project Output Structure

Each run creates `projects/<project_id>/` with:
```
literature/           # Context pack outputs (when enabled)
  manifest.json       # Paper metadata
  pdfs/               # Downloaded articles (PDFs and HTML files)
  Research_Questions.txt  # Agent-generated questions
questions/            # Question generation + prioritization
  questions.txt       # Generated or context-pack questions
  prioritized_question.txt  # Selected question (from prioritizer)
analysis/             # All analysis output (agent and legacy)
  background/         # Agent mode: research question + confidence tracker
  analysis_1/         # Agent mode: plans, scripts, evaluations directly here
                      # Legacy mode: attempt_1/, attempt_2/ subdirectories
  analysis_2/         # ...
  a1_analysis_plan_*.txt  # Legacy mode: planner output
visualizations/       # Generated plots
reports/              # Final report
dashboard.html        # Auto-refreshing HTML dashboard (written during run)
```

## Conventions

- `PathResolver` is a singleton; use `path_resolver.ensure_path("component")` to get/create project subdirectories
- LLM config is read from `config.yaml` at startup and persisted to `.last_llm.json`
- The `arxiv_interp_graph` directory is added to `sys.path` at runtime so its subpackages can be imported directly (e.g., `from context_pack.run import ...`)
- Agent subprocess commands run with `cwd=literature_dir` so the agent sees PDFs relative to its working directory
