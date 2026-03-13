# CLAUDE.md

## Project Overview

AutoInterp is an automated mechanistic interpretability research framework. It takes a research question as input and produces a research report with original analyses, visualizations, and interpretation. Each step of the research process is executed by an LLM agent.

## Quick Start

```bash
pip install -r requirements.txt
python main.py          # interactive provider selection, then full pipeline
python main.py run      # same as above
python main.py literature-search  # run literature search only (no full pipeline)
```

Environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN` (set whichever provider you use).

## Key Architecture

### Pipeline Flow (`main.py` → `streamlined_pipeline()`)

1. **Literature Search** (optional) — sample 3 papers from citation graph, download articles (PDF or HTML) to `literature/`, generate research questions
2. **Question Generation** — LLM writes candidate research questions (skipped if literature search produced questions)
3. **Question Prioritization** — LLM selects best question, extracts TITLE, renames project dir (always runs)
4. **Iterative Analysis** — agent mode (default): CLI agent subprocess plans, codes, executes, debugs, evaluates autonomously; legacy mode: plan → generate code → execute in sandbox → evaluate → repeat until confident
5. **Visualization** — agent mode (default): CLI agent subprocess reads all analyses and generates publication-quality figures autonomously; legacy mode: per-analysis planner → generator → executor → evaluator pipeline
6. **Report Generation** — agent mode (default): CLI agent subprocess reads all analyses and visualizations, writes an academic-style report autonomously; legacy mode: LLM API calls generate each report section
7. **AutoCritique** (optional) — CLI agent subprocess performs automated peer review, producing a verdict (Accept / Revise and Resubmit / Reject) and recommendation files
8. **Revision** (conditional) — if AutoCritique verdict is "Revise and Resubmit", a CLI agent subprocess addresses each recommendation one at a time, performing new/revised analyses
9. **Report Revision** (conditional) — after all per-recommendation revisions complete, a CLI agent subprocess reads the original report, review, responses, and revised analyses, then produces a revised report incorporating all changes

### Literature Search Question Generation

When `literature_search.enabled: true` in `config.yaml`, the system builds a 3-paper pack from `arxiv_interp_graph/output/graph_state.json` and generates questions. The strategy depends on the LLM provider:

- **Anthropic** → runs `claude` CLI agent (subprocess with `--dangerously-skip-permissions`)
- **OpenAI** → runs `codex` CLI agent (subprocess with `-s workspace-write`)
- **Other/fallback** → direct LLM API call via `_generate_question_llm()`

The agent reads articles from `literature/pdfs/` (PDFs and HTML files) and writes `Research_Questions.txt`. Output is copied to `questions/questions.txt` and the prioritizer always runs afterward.

### Article Download Pipeline

Each paper in the citation graph stores download metadata to avoid live API calls:

| Source | Coverage | Method |
|--------|----------|--------|
| `arxiv_id` | 969/1003 (96.6%) | Construct `https://arxiv.org/pdf/{id}.pdf` directly |
| `open_access_url` | 29/1003 (2.9%) | Stored URL from S2, Distill, Springer OA, Frontiers, ACL, bioRxiv, etc. |
| No URL | 5/1003 (0.5%) | Paywalled (Elsevier/IEEE) — excluded from sampling |

**Sampling filter:** Papers without any download URL (`arxiv_id` or `open_access_url`) are automatically excluded from the literature search candidate pool. They remain in the graph for topology/statistics but will never be selected as seed, forward, or backward papers. See `_has_download_url()` in `sampling.py`.

**Download retry:** If a sampled paper's download fails at runtime (broken URL, timeout, etc.), the system automatically replaces it with another downloadable paper from the graph (up to 3 attempts per slot). See `_retry_failed_downloads()` in `run.py`.

HTML articles (Distill, Transformer Circuits Thread) are saved as `.html` files alongside PDFs. The agent prompt and LLM fallback both handle both formats.

To re-enrich the graph after adding new papers:
```bash
cd arxiv_interp_graph && python enrich_arxiv_ids.py
cd arxiv_interp_graph && python enrich_missing_urls.py  # S2 batch + arxiv title search for remaining gaps
```

**Prerequisites for agent mode:** The `claude` CLI must be installed and authenticated:
```bash
curl -fsSL https://claude.ai/install.sh | bash   # install
claude                                             # first run: follow login prompts
```
For OpenAI, the `codex` CLI must be installed and authenticated separately.

Config fields:
- `literature_search.use_agent` (default `true`) — use CLI agent; `false` = always use LLM API
- `literature_search.agent_timeout` (default `600`) — subprocess timeout in seconds

Agent logic lives in `arxiv_interp_graph/literature_search/agent_questions.py`.

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

### Visualization Agent Mode (`visualization/agent_visualization.py`)

When `visualization.use_agent: true` (default) and the provider is `anthropic` or `openai`, all visualizations are generated by a single CLI agent subprocess. The agent reads the `analysis/` directory (background materials and all analysis iterations), produces publication-quality figures, and writes captions. The agent handles all visualizations in one invocation (not per-analysis).

**Output directory:**
```
visualizations/
  Visualization_log.md    # Agent working notes / scratchpad
  figure_1.py             # Self-contained visualization script
  figure_1.png            # Generated figure
  caption_1.txt           # Brief caption for the figure
  figure_2.py
  figure_2.png
  caption_2.txt
  ...
```

**Fallback rules:**
- `visualization.use_agent: false` → always use legacy pipeline
- Provider not `anthropic`/`openai` → legacy pipeline
- CLI binary not found → legacy pipeline with warning
- Agent finishes but no `figure_*.png` found → legacy pipeline

Config fields:
```yaml
visualization:
  use_agent: true       # default true; false = legacy pipeline
  agent_timeout: 900    # subprocess timeout (seconds)
```

Agent logic lives in `visualization/agent_visualization.py`. Prompt template in `prompts/agent_visualization.yaml`.

### Report Agent Mode (`reporting/agent_report.py`)

When `reporting.use_agent: true` (default) and the provider is `anthropic` or `openai`, the final report is generated by a single CLI agent subprocess. The agent reads the `analysis/` and `visualizations/` directories autonomously, writes `Reporter_log.md` (notes), and then writes the full academic-style report as `{title}.md` in the `reports/` directory.

**Fallback rules:**
- `reporting.use_agent: false` → always use legacy pipeline
- Provider not `anthropic`/`openai` → legacy pipeline
- CLI binary not found → legacy pipeline with warning
- Agent finishes but no `.md` report found → legacy pipeline

Config fields:
```yaml
reporting:
  use_agent: true       # default true; false = legacy pipeline
  agent_timeout: 900    # subprocess timeout (seconds)
```

Agent logic lives in `reporting/agent_report.py`. Prompt template in `prompts/agent_report.yaml`.

### AutoCritique Agent Mode (`autocritique/agent_autocritique.py`)

When `autocritique.enabled: true` (default) and `autocritique.use_agent: true`, an automated peer review step runs after report generation. A CLI agent subprocess (claude/codex) reads the report, analyses, and visualizations, then produces a formal review with a verdict (Reject / Revise and Resubmit / Accept).

**Iterative review loop:** The autocritique → revision → report revision cycle can repeat up to `max_revision_rounds` times (default 2). When a review returns "Revise and Resubmit", the pipeline runs per-recommendation revision agents, then a report revision agent to produce a new report, then loops back for another autocritique review of the revised report. The loop terminates when:
- The verdict is **Accept** or **Reject**
- `max_revision_rounds` is reached (the final review still runs but no further revisions are triggered)
- The review has no recommendation files
- An agent fails or doesn't produce expected output

On rounds 2+, the autocritique prompt explicitly directs the reviewer to read the most recent revised report (`*_revision_{k}.md`) rather than the original.

**No legacy fallback:** Unlike report generation, AutoCritique simply skips if the agent can't run. There is no non-agent path.

**Skip rules:**
- `autocritique.enabled: false` → step marked as skipped
- `autocritique.use_agent: false` → step skipped
- Provider not `anthropic`/`openai` → step skipped with message
- CLI binary not found → step skipped with message

Config fields:
```yaml
autocritique:
  enabled: true              # default true; false = skip autocritique
  use_agent: true            # must be true (no legacy fallback)
  agent_timeout: 600         # subprocess timeout (seconds)
  revision_timeout: 1800     # per-recommendation timeout for revision agent
  report_revision_timeout: 900  # timeout for report revision agent
  max_revision_rounds: 2     # max autocritique→revision→report revision cycles (0 = review only)
```

**Output directory (round-based):**
```
autocritique/
  round_1/
    AutoCritique_log.md       # Agent working notes
    AutoCritique_review.md    # Formal review (Summary, Strengths, Weaknesses, Minor Issues, Questions, Verdict, Caveats)
    Recommendation_1.md       # (only if verdict is "Revise and Resubmit")
    Recommendation_2.md
    Reviewer_1_log.md         # Revision agent working notes (rec 1)
    Response_1.md             # Revision agent formal response (rec 1)
    Reviewer_2_log.md         # Revision agent working notes (rec 2)
    Response_2.md             # Revision agent formal response (rec 2)
    ...
  round_2/                    # (only if round 1 verdict was "Revise and Resubmit")
    ...
```

Agent logic lives in `autocritique/agent_autocritique.py`. Prompt template in `prompts/agent_autocritique.yaml`. Toggleable via Options menu (#12).

### Revision Agent Mode (`autocritique/agent_revision.py`)

When AutoCritique produces a "Revise and Resubmit" verdict with `Recommendation_N.md` files, the pipeline automatically runs a Revision Agent for each recommendation. Each recommendation spawns its own CLI agent subprocess (claude/codex). The agent reads the study materials, the review, and its assigned recommendation, then performs the revision work (new/modified analyses), documenting everything in the `autocritique/round_{k}/` directory.

**Trigger:** Runs only when AutoCritique verdict contains `**Revise and Resubmit**` and at least one `Recommendation_N.md` file exists.

**Per-recommendation output:**
- `Reviewer_{i}_log.md` — agent working notes / scratchpad
- `Response_{i}.md` — formal response documenting changes made, results, and assessment

New/revised analysis files go into `analysis/` per the agent's judgment (new `analysis_{n+1}/` dirs or modifications to existing dirs).

Agent logic lives in `autocritique/agent_revision.py`. Prompt template in `prompts/agent_revision.yaml`.

### Report Revision Agent Mode (`reporting/agent_report_revision.py`)

After all per-recommendation revision agents complete, a Report Revision Agent runs to holistically incorporate all revision work into the research report. It reads the original report, the review, all responses, and the revised analyses/visualizations, then produces a new report file (`*_revision_{k}.md`) in `reports/`. The agent does not mention the revision process in the revised report (like peer review, the revised manuscript does not discuss the review itself).

**Trigger:** Runs only after the per-recommendation revision agents complete successfully (verdict was "Revise and Resubmit").

**Output (in `reports/`):**
- `Report_revision_{k}.log` — agent working notes
- `{original_title}_revision_{k}.md` — the revised report

Agent logic lives in `reporting/agent_report_revision.py`. Prompt template in `prompts/agent_report_revision.yaml`.

### Agent Subprocess Progress Polling (`core/agent_subprocess.py`)

CLI agent subprocesses (question generation, analysis, report, autocritique) use `run_agent_with_polling()` instead of blocking `subprocess.run(capture_output=True)`. This provides real-time progress during long-running agent calls:

- Uses `subprocess.Popen` with a 3-second polling loop (`POLL_INTERVAL = 3.0`)
- Watches the agent's working directory for milestone files via `MilestoneSpec` / `MilestonePattern`
- Calls an `on_progress` callback for each new file detected (e.g. "Wrote analysis plan", "Wrote script: foo.py", "Generated figure: plot.png")
- **Broad file scanning**: Detects *any* new file in the watch directory, not just milestone-pattern matches. Files matching a milestone pattern get a descriptive message; other files are reported as "New file: {name}"
- **Heartbeat**: Emits "Still running... Xm Ys elapsed" every 30 seconds (`HEARTBEAT_INTERVAL = 30.0`) when no milestones or new files have been detected, so the user always sees activity
- **Completion signal**: Emits "Agent finished (Xm Ys)" when the subprocess exits
- Progress messages flow to both the CLI terminal (`[~]` prefix) and the HTML dashboard (timestamped log)
- Timeout enforcement: kills the process if wall-clock time exceeds the configured limit
- Backward compatible: `on_progress=None` and `milestone=None` are valid (no polling, just Popen + wait)

Milestone patterns per call site:
- **Question generation** — watches `literature/` for `Research_Questions.txt`
- **Analysis agent** — watches `analysis/analysis_{n}/` for `ANALYSIS_{n}_PLAN.md`, `*.py`, `*.png`/`*.jpg`/`*.svg`, `ANALYSIS_{n}_EVALUATION.md`
- **Visualization agent** — watches `visualizations/` for `Visualization_log.md`, `figure_*.py`, `figure_*.png`/`*.jpg`/`*.svg`, `caption_*.txt`
- **Report agent** — watches `reports/` for `Reporter_log.md`, `*.md`
- **AutoCritique agent** — watches `autocritique/round_{n}/` for `AutoCritique_log.md`, `AutoCritique_review.md`, `Recommendation_*.md`
- **Revision agent** — watches `autocritique/round_{k}/` for `Reviewer_{i}_log.md`, `Response_{i}.md`
- **Report revision agent** — watches `reports/` for `Report_revision_{k}.log`, `*_revision_{k}.md`

### Pipeline UI (`core/pipeline_ui.py` + `core/dashboard_template.py`)

The pipeline produces two forms of output:

1. **CLI terminal** — the original colorful verbose ANSI output from `LLMInterface` (always printed)
2. **HTML dashboard** — a self-contained, auto-refreshing `dashboard.html` written to the project directory

`PipelineUI` is created in `async_main()` and attached to `LLMInterface.pipeline_ui`. Every `generate()` call records an `LLMInteraction` and rewrites the dashboard. Step lifecycle calls (`step_start`, `step_complete`, `step_progress`, etc.) are made from `streamlined_pipeline()`.

`step_progress(step_id, message)` records a `ProgressMessage` on the step, prints `[~] {message}` to the terminal, and rewrites the dashboard. Progress messages are rendered as a timestamped log (`+42s Wrote script: foo.py`) above the LLM interaction cards in each step's tab.

The dashboard uses a dark "Civilization 2" theme with per-step color coding: green (questions), blue (prioritize), gold→burnt umber gradient (analysis), teal (visualization), coral (report), purple (critique). Tabs double as a progress bar. All prompt/response sections are collapsed by default. Auto-refresh uses `fetch()` + DOM diffing to preserve tab state, scroll position, and open/closed details.

Config (`config.yaml`):
```yaml
ui:
  rich_terminal: true      # Enable Rich library for terminal panels (unused currently — verbose output always prints)
  html_dashboard: true     # Write dashboard.html to project dir
  dashboard_refresh: 5     # Auto-refresh interval in seconds
  auto_open_browser: true  # Open dashboard in browser on pipeline start
```

### Interactive Mode (`core/interactive.py`)

When `interactive_mode: true` in `config.yaml` (or toggled via the Options menu), the pipeline pauses after each stage to display output and solicit user feedback. If the user types feedback, the system revises the output via an LLM call and re-displays it. Pressing Enter with no input continues to the next stage.

Checkpoints are inserted at:

| Stage | Mode | What happens on feedback |
|-------|------|--------------------------|
| Literature Search Questions | both | LLM revision call; rewrites `questions/questions.txt` |
| Question Generation | both | LLM revision call; rewrites `questions/questions.txt` |
| Question Prioritization | both | LLM revision call; rewrites `questions/prioritized_question.txt` (before title extraction) |
| Analysis Plan | legacy | LLM revision call; rewrites plan file; revised plan passed to code generator |
| Analysis Evaluation | legacy | Feedback stored in `config["_interactive_guidance"]` for next planner iteration |
| Analysis Evaluation | agent | Feedback appended to `analysis/background/user_feedback.md`; read by next iteration's prompt |
| Visualizations | both | Feedback noted; passed to report generation as additional context |
| Report | both | LLM revision call; overwrites the report file |

Config:
```yaml
interactive_mode: false   # top-level key in config.yaml
```

Revision prompts live in `prompts/interactive.yaml`. Core helpers in `core/interactive.py`: `is_interactive()`, `format_stage_output()`, `interactive_checkpoint()`, `make_revision_call()`.

### Options Menu (`main.py`)

At startup, "Options" appears as choice `[5]` in the provider/model selection menu. Selecting it opens an interactive menu to override common config settings without editing `config.yaml`. After closing the Options menu, the user is returned to the provider selection screen to choose a provider and model:

| # | Setting | Config Key | Type |
|---|---------|-----------|------|
| 1 | Max analysis iterations | `analysis.max_iterations` | int |
| 2 | Confidence threshold | `analysis.confidence_threshold` | float (displayed as %) |
| 3 | Use CLI agent for analysis | `analysis.use_agent` | bool |
| 4 | Use CLI agent for report | `reporting.use_agent` | bool |
| 5 | Use CLI agent for visualization | `visualization.use_agent` | bool |
| 6 | Literature search | `literature_search.enabled` | bool |
| 7 | Articles for question gen | `literature_search.n_papers` | int |
| 8 | Visualization format | `visualization.default_format` | str (png/svg/pdf) |
| 9 | Visualization DPI | `visualization.dpi` | int |
| 10 | HTML dashboard | `ui.html_dashboard` | bool |
| 11 | Auto-open browser | `ui.auto_open_browser` | bool |
| 12 | Interactive mode (feedback loops) | `interactive_mode` | bool |
| 13 | AutoCritique (peer review) | `autocritique.enabled` | bool |
| 14 | Max revision rounds | `autocritique.max_revision_rounds` | int |

After editing, the user chooses "Just this time" (in-memory only) or "Make default" (persisted to `.user_options.json`). Saved defaults are loaded automatically on future runs via `load_user_options()`, which runs right after `initialize_framework()`. Only keys present in `OPTIONS_SETTINGS` are applied; stale keys in the JSON file are ignored.

Key functions (all in `main.py`): `OPTIONS_SETTINGS`, `_get_config_value()`, `_set_config_value()`, `load_user_options()`, `save_user_options()`, `show_options_menu()`.

### Manual Configuration (`main.py`)

Choice `[4] Manual Configuration` in the provider selection menu opens an interactive per-stage model picker. It shows all 10 LLM-backed agents with their current provider/model and lets the user change each one independently (mix-and-match providers).

Agents listed (in pipeline order): Question Generator, Question Prioritizer, Analysis Planner, Analysis Generator, Analysis Evaluator, Visualization Planner, Visualization Generator, Visualization Evaluator, Report Generator, Title Generator.

The model picker shows all models from `MODEL_MAPPINGS` (Anthropic, OpenAI, OpenRouter) plus any custom (provider, model) pairs already present in the agent configs from `config.yaml`. To add a custom model to the picker, add it to the relevant agent's `llm` section in `config.yaml`.

After editing, the user chooses "Just this time" (in-memory only) or "Make default" (persisted to `.user_manual_models.json`). Saved defaults are loaded automatically on future runs via `load_manual_model_config()`. After closing the menu the user proceeds directly to the pipeline (no return to provider selection).

Key functions (all in `main.py`): `MODEL_MAPPINGS`, `MANUAL_CONFIG_AGENTS`, `_build_model_list()`, `show_manual_config_menu()`, `load_manual_model_config()`, `save_manual_model_config()`.

## Important File Locations

| File | Purpose |
|------|---------|
| `main.py` | Main orchestrator, CLI entry point, `streamlined_pipeline()` |
| `config.yaml` | All configuration (providers, agents, execution, literature search, UI) |
| `core/llm_interface.py` | LLM API abstraction (Anthropic, OpenAI, OpenRouter) |
| `core/agent_subprocess.py` | Shared `Popen` + filesystem-polling runner for CLI agent subprocesses |
| `core/pipeline_ui.py` | `PipelineUI` class — step tracking, LLM interaction recording, progress messages, HTML dashboard |
| `core/dashboard_template.py` | HTML template, CSS, JS, and render helpers for the dashboard |
| `core/utils.py` | `PathResolver` singleton, utilities |
| `core/interactive.py` | Interactive mode: feedback loops, revision calls |
| `prompts/interactive.yaml` | Per-stage revision prompt templates for interactive mode |
| `questions/question_manager.py` | Question file I/O, prioritization |
| `arxiv_interp_graph/literature_search/` | Literature search: sampling, download, agent questions, run |
| `arxiv_interp_graph/literature_search/agent_questions.py` | CLI agent subprocess invocation |
| `arxiv_interp_graph/literature_search/run.py` | Literature search orchestration + LLM fallback |
| `arxiv_interp_graph/literature_search/download.py` | Article download (PDF + HTML) + manifest writing |
| `prompts/question_manager.yaml` | All question prompts (generator, prioritizer, agent question generator) |
| `analysis/agent_analysis.py` | CLI agent analysis: workspace setup, subprocess, output reading |
| `prompts/agent_analysis.yaml` | Prompt template for analysis agent iterations |
| `visualization/agent_visualization.py` | CLI agent visualization: subprocess, output reading |
| `prompts/agent_visualization.yaml` | Prompt template for visualization agent |
| `reporting/agent_report.py` | CLI agent report generation: subprocess, output reading |
| `prompts/agent_report.yaml` | Prompt template for report agent |
| `autocritique/agent_autocritique.py` | CLI agent autocritique: subprocess, output reading |
| `prompts/agent_autocritique.yaml` | Prompt template for autocritique agent |
| `autocritique/agent_revision.py` | CLI agent revision: per-recommendation subprocess, output reading |
| `prompts/agent_revision.yaml` | Prompt template for revision agent |
| `reporting/agent_report_revision.py` | CLI agent report revision: incorporates revisions into report |
| `prompts/agent_report_revision.yaml` | Prompt template for report revision agent |
| `arxiv_interp_graph/enrich_arxiv_ids.py` | Batch-enrich graph with arxiv_id + open_access_url |
| `.last_llm.json` | Persisted provider/model selection from last run |
| `.user_options.json` | Persisted user option overrides (gitignored) |
| `.user_manual_models.json` | Persisted per-agent model overrides from Manual Configuration (gitignored) |
| `prompts/*.yaml` | Agent-specific prompt templates |

## Project Output Structure

Each run creates `projects/<project_id>/` with:
```
literature/           # Literature search outputs (when enabled)
  manifest.json       # Paper metadata
  pdfs/               # Downloaded articles (PDFs and HTML files)
  Research_Questions.txt  # Agent-generated questions
questions/            # Question generation + prioritization
  questions.txt       # Generated or literature-search questions
  prioritized_question.txt  # Selected question (from prioritizer)
analysis/             # All analysis output (agent and legacy)
  background/         # Agent mode: research question + confidence tracker + user feedback
    user_feedback.md  # Interactive mode: accumulated user feedback (agent mode only)
  analysis_1/         # Agent mode: plans, scripts, evaluations directly here
                      # Legacy mode: attempt_1/, attempt_2/ subdirectories
  analysis_2/         # ...
  a1_analysis_plan_*.txt  # Legacy mode: planner output
visualizations/       # Generated plots (agent or legacy)
  Visualization_log.md  # Agent mode: working notes
  figure_1.py           # Agent mode: visualization script
  figure_1.png          # Agent mode: generated figure
  caption_1.txt         # Agent mode: figure caption
reports/              # Final report (and revised reports)
  Report_revision_*.log       # Report revision agent working notes (one per round)
  *_revision_*.md             # Revised reports (one per round)
autocritique/         # AutoCritique outputs (when enabled)
  round_1/              # First review round (round_2/ etc. if revisions triggered)
    AutoCritique_log.md       # Agent working notes
    AutoCritique_review.md    # Formal review with verdict
    Recommendation_*.md       # Actionable revision items (only on "Revise and Resubmit")
    Reviewer_*_log.md         # Revision agent working notes (one per recommendation)
    Response_*.md             # Revision agent formal responses (one per recommendation)
dashboard.html        # Auto-refreshing HTML dashboard (written during run)
```

## Conventions

- `PathResolver` is a singleton; use `path_resolver.ensure_path("component")` to get/create project subdirectories
- LLM config is read from `config.yaml` at startup and persisted to `.last_llm.json`
- The `arxiv_interp_graph` directory is added to `sys.path` at runtime so its subpackages can be imported directly (e.g., `from literature_search.run import ...`)
- Agent subprocess commands run with `cwd=literature_dir` so the agent sees PDFs relative to its working directory
