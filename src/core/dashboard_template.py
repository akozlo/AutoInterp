"""
HTML dashboard template and rendering helpers for PipelineUI.

Provides the self-contained HTML skeleton (Civ 2 inspired dark theme, tabbed
interface with integrated progress indicators) and functions to render dynamic
content (tab buttons, LLM cards, analysis columns).
"""

import html as _html
import re
from collections import OrderedDict
from datetime import datetime
from typing import List, Dict, Any, Optional


def escape_html(text: str) -> str:
    """Escape HTML special characters in text."""
    return _html.escape(str(text)) if text else ""


def _format_duration(seconds: Optional[float]) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds is None:
        return ""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def _format_chars(text: Optional[str]) -> str:
    """Format character count with comma separators."""
    if text is None:
        return "0"
    return f"{len(text):,}"


# ---------------------------------------------------------------------------
# Step configuration
# ---------------------------------------------------------------------------

STEP_CONFIG = {
    "question_generation": {"label": "questions", "tab_id": "questions"},
    "question_prioritization": {"label": "prioritize", "tab_id": "prioritize"},
    "literature_review": {"label": "lit review", "tab_id": "litreview"},
    "iterative_analysis": {"label": "analysis", "tab_id": "analysis"},
    "visualization": {"label": "visualization", "tab_id": "visualization"},
    "report_generation": {"label": "report", "tab_id": "report"},
    "autocritique": {"label": "critique", "tab_id": "critique"},
    "revision": {"label": "revision", "tab_id": "revision"},
    "report_revision": {"label": "revised report", "tab_id": "report_revision"},
    "repo": {"label": "repo", "tab_id": "repo"},
    "notebook": {"label": "notebook", "tab_id": "notebook"},
}

# Status → CSS class and icon for the merged tab/progress bar
_STATUS_CSS = {
    "completed": ("step-done", "+"),
    "running": ("step-run", "~"),
    "failed": ("step-run", "!"),
    "skipped": ("step-pend", "-"),
    "pending": ("step-pend", "-"),
}

# Status text shown in overview table
_STATUS_TEXT = {
    "completed": ("done", "st-done"),
    "running": ("running", "st-run"),
    "failed": ("failed", "st-fail"),
    "skipped": ("skipped", "st-pend"),
    "pending": ("pending", "st-pend"),
}

# Analysis column gradient: gold → burnt orange → burnt umber (+ deeper stops)
_ANALYSIS_GRADIENT = [
    ("var(--c-analysis-1)", "var(--c-analysis-1-bright)", "var(--c-analysis-1-dim)"),
    ("var(--c-analysis-2)", "var(--c-analysis-2-bright)", "var(--c-analysis-2-dim)"),
    ("var(--c-analysis-3)", "var(--c-analysis-3-bright)", "var(--c-analysis-3-dim)"),
]


# ---------------------------------------------------------------------------
# Tab buttons (merged progress bar + tabs)
# ---------------------------------------------------------------------------

def render_tab_buttons(steps: List[Dict[str, Any]]) -> str:
    """Render tab buttons with integrated progress indicators."""
    parts = ['<button class="tab-btn active" data-tab="overview">overview</button>']
    for step in steps:
        cfg = STEP_CONFIG.get(step["step_id"], {"label": step["step_id"], "tab_id": step["step_id"]})
        tab_id = cfg["tab_id"]
        label = cfg["label"]
        css_cls, icon = _STATUS_CSS.get(step["status"], ("step-pend", "-"))

        parts.append(
            f'<button class="tab-btn {css_cls}" data-tab="{escape_html(tab_id)}">'
            f'<span class="status-icon">{icon}</span>{escape_html(label)}'
            f'</button>'
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM interaction cards
# ---------------------------------------------------------------------------

def render_llm_card(interaction: Dict[str, Any]) -> str:
    """Render a single LLM interaction as a card with collapsible sections.

    Section order: SYSTEM (collapsed), USER (collapsed), ASSISTANT (collapsed).
    """
    agent = escape_html(interaction.get("display_name", interaction.get("agent_name", "Unknown")))
    model = escape_html(interaction.get("model", ""))
    provider = escape_html(interaction.get("provider", ""))
    duration = _format_duration(interaction.get("duration_seconds"))
    prompt_chars = _format_chars(interaction.get("prompt"))
    response_chars = _format_chars(interaction.get("response"))

    prompt_html = escape_html(interaction.get("prompt", ""))
    sys_msg_html = escape_html(interaction.get("system_message", ""))
    response_html = escape_html(interaction.get("response", ""))

    sys_section = ""
    if sys_msg_html:
        sys_section = f'''
        <details class="card-section">
            <summary>SYSTEM</summary>
            <pre class="card-pre sys">{sys_msg_html}</pre>
        </details>'''

    return f'''
    <div class="card">
        <div class="card-head">
            <span class="name">{agent}</span>
            <span>{model} / {provider} / {duration} / {prompt_chars}-&gt;{response_chars} chars</span>
        </div>
        {sys_section}
        <details class="card-section">
            <summary>USER</summary>
            <pre class="card-pre user">{prompt_html}</pre>
        </details>
        <details class="card-section">
            <summary>ASSISTANT</summary>
            <pre class="card-pre asst">{response_html}</pre>
        </details>
    </div>'''


# ---------------------------------------------------------------------------
# Analysis column layout helpers
# ---------------------------------------------------------------------------

_ANALYSIS_ROLE_MAP = {
    "analysis_planner": "PLANNER",
    "analysis_generator": "GENERATOR",
    "evaluator": "EVALUATOR",
    "analysis_agent": "AGENT",
}


def _group_analysis_interactions(interactions: List[Dict[str, Any]]) -> OrderedDict:
    """Group analysis interactions into {analysis_num -> {attempt_num -> [interactions]}}.

    Within each analysis number, attempts are inferred by counting generator calls:
    each generator call starts a new attempt. The planner only fires on attempt 1.
    """
    by_analysis: OrderedDict = OrderedDict()
    for inter in interactions:
        a_num = inter.get("iteration_number") or 0
        by_analysis.setdefault(a_num, []).append(inter)

    result: OrderedDict = OrderedDict()
    for a_num, items in by_analysis.items():
        attempts: OrderedDict = OrderedDict()
        current_attempt = 1
        for item in items:
            role = _ANALYSIS_ROLE_MAP.get(item.get("agent_name"), "")
            if role == "GENERATOR" and current_attempt in attempts:
                has_gen = any(
                    _ANALYSIS_ROLE_MAP.get(x.get("agent_name")) == "GENERATOR"
                    for x in attempts[current_attempt]
                )
                if has_gen:
                    current_attempt += 1
            attempts.setdefault(current_attempt, []).append(item)
        result[a_num] = attempts
    return result


def _render_analysis_role_card(interaction: Dict[str, Any], role: str) -> str:
    """Render a compact card for a single role inside an analysis attempt."""
    model = escape_html(interaction.get("model", ""))
    duration = _format_duration(interaction.get("duration_seconds"))

    prompt_html = escape_html(interaction.get("prompt", ""))
    sys_msg_html = escape_html(interaction.get("system_message", ""))
    response_html = escape_html(interaction.get("response", ""))

    sys_section = ""
    if sys_msg_html:
        sys_section = f'''
            <details class="role-section">
                <summary>SYSTEM</summary>
                <pre class="card-pre sys">{sys_msg_html}</pre>
            </details>'''

    return f'''
        <div class="role">
            <div class="role-head">
                <span class="rn">{role}</span>
                <span>{model} / {duration}</span>
            </div>
            {sys_section}
            <details class="role-section">
                <summary>USER</summary>
                <pre class="card-pre user">{prompt_html}</pre>
            </details>
            <details class="role-section">
                <summary>ASSISTANT</summary>
                <pre class="card-pre asst">{response_html}</pre>
            </details>
        </div>'''


def render_analysis_columns(interactions: List[Dict[str, Any]]) -> str:
    """Render the iterative analysis tab as horizontal-scrolling columns."""
    if not interactions:
        return '<p class="no-data">No analysis interactions recorded.</p>'

    grouped = _group_analysis_interactions(interactions)

    columns = []
    for col_idx, (a_num, attempts) in enumerate(grouped.items()):
        # Pick gradient color for this column (clamp to last stop)
        grad_idx = min(col_idx, len(_ANALYSIS_GRADIENT) - 1)
        col_color, col_bright, col_dim = _ANALYSIS_GRADIENT[grad_idx]

        attempt_tabs = []
        attempt_panes = []
        for att_num, items in attempts.items():
            active_cls = "active" if att_num == 1 else ""
            attempt_tabs.append(
                f'<button class="att-tab {active_cls}" '
                f'data-att="a{a_num}-att{att_num}">attempt {att_num}</button>'
            )
            role_cards = []
            for item in items:
                role = _ANALYSIS_ROLE_MAP.get(item.get("agent_name"), item.get("agent_name", ""))
                role_cards.append(_render_analysis_role_card(item, role))

            attempt_panes.append(
                f'<div class="att-pane {active_cls}" id="a{a_num}-att{att_num}">'
                f'{"".join(role_cards)}'
                f'</div>'
            )

        tabs_html = "\n".join(attempt_tabs)
        panes_html = "\n".join(attempt_panes)

        columns.append(f'''
        <div class="analysis-col" style="--col-color: {col_color}; --col-bright: {col_bright}; --col-dim: {col_dim};">
            <div class="col-title">analysis #{a_num}</div>
            <div class="att-bar">{tabs_html}</div>
            {panes_html}
        </div>''')

    return f'<div class="analysis-row">{"".join(columns)}</div>'


# ---------------------------------------------------------------------------
# Progress log renderer
# ---------------------------------------------------------------------------

def _format_study_title(task_name: str, project_id: str) -> str:
    """Format a human-readable study title from task_name or project_id.

    Strips trailing _YYYY-MM-DDTHH-MM-SS timestamp, strips REJECT_ prefix,
    replaces underscores with spaces, and lowercases.
    """
    raw = project_id or task_name
    if not raw:
        return ""
    # Strip trailing timestamp like _2026-03-12T18-14-54
    raw = re.sub(r'_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$', '', raw)
    # Strip REJECT_ prefix
    raw = re.sub(r'^REJECT_', '', raw)
    # Replace underscores with spaces
    raw = raw.replace('_', ' ')
    return raw.lower()


def _render_markdown(text: str) -> str:
    """Render markdown text to HTML. Falls back to escaped <pre> if markdown library unavailable."""
    try:
        import markdown
        return markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc'])
    except ImportError:
        return f'<pre class="card-pre asst">{escape_html(text)}</pre>'


def _extract_round_number(filename: str) -> int:
    """Extract round number from filenames like 'AutoCritique_review.md (round 2)'. Returns 1 if not found."""
    m = re.search(r'\(round\s+(\d+)\)', filename)
    return int(m.group(1)) if m else 1


def render_round_columns(
    output_files: List[Dict[str, Any]],
    step_id: str,
    color_var: str,
    color_bright_var: str,
) -> str:
    """Group output files by round number and render as horizontal columns."""
    if not output_files:
        return ""

    # Group by round
    rounds: Dict[int, List[Dict[str, Any]]] = {}
    for of in output_files:
        rn = _extract_round_number(of.get("filename", ""))
        rounds.setdefault(rn, []).append(of)

    columns = []
    for rn in sorted(rounds.keys()):
        cards = "\n".join(
            render_output_card(of, step_id=step_id, collapsed=True) for of in rounds[rn]
        )
        columns.append(
            f'<div class="round-col" style="border-color: {color_var};">'
            f'<h3 style="color: {color_bright_var};">round {rn}</h3>'
            f'{cards}'
            f'</div>'
        )

    return f'<div class="round-row">{"".join(columns)}</div>'


_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.svg'}


def render_output_card(
    output_file: Dict[str, Any],
    step_id: str = "",
    collapsed: bool = False,
) -> str:
    """Render a single output file as a collapsible card.

    - Image files (.png, .jpg, .svg) in visualization step render as <img> tags.
    - Markdown files in report_generation/report_revision steps render as formatted HTML.
    - Other files render as <pre> text blocks.
    """
    filename = output_file.get("filename", "")
    content_raw = output_file.get("content", "")
    chars = _format_chars(content_raw)
    filename_escaped = escape_html(filename)

    open_attr = "" if collapsed else " open"

    # Determine file extension
    import os.path
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    # Image files: render as <img>
    if ext in _IMAGE_EXTENSIONS:
        # content is a relative path like "visualizations/figure_1.png"
        img_src = escape_html(content_raw) if content_raw else escape_html(filename)
        return f'''
    <div class="card viz-figure">
        <div class="card-head">
            <span class="name">{filename_escaped}</span>
        </div>
        <details class="card-section"{open_attr}>
            <summary>FIGURE</summary>
            <div style="padding: 10px 12px; background: #060606;">
                <img src="{img_src}" alt="{filename_escaped}">
            </div>
        </details>
    </div>'''

    # Markdown files in report steps: render as formatted HTML
    if ext == '.md' and step_id in ('report_generation', 'report_revision'):
        rendered = _render_markdown(content_raw)
        return f'''
    <div class="card">
        <div class="card-head">
            <span class="name">{filename_escaped}</span>
            <span>{chars} chars</span>
        </div>
        <details class="card-section"{open_attr}>
            <summary>REPORT</summary>
            <div class="rendered-md">{rendered}</div>
        </details>
    </div>'''

    # Default: plain text
    content = escape_html(content_raw)
    return f'''
    <div class="card">
        <div class="card-head">
            <span class="name">{filename_escaped}</span>
            <span>{chars} chars</span>
        </div>
        <details class="card-section"{open_attr}>
            <summary>CONTENT</summary>
            <pre class="card-pre asst">{content}</pre>
        </details>
    </div>'''


def render_output_cards(
    output_files: List[Dict[str, Any]],
    step_id: str = "",
    collapsed: bool = False,
) -> str:
    """Render all output file cards for a step."""
    if not output_files:
        return ""
    return "\n".join(
        render_output_card(of, step_id=step_id, collapsed=collapsed)
        for of in output_files
    )


def render_progress_log(
    progress_messages: List[Dict[str, Any]],
    step_start_time: Optional[datetime] = None,
) -> str:
    """Render a progress log div from a list of progress message dicts."""
    if not progress_messages:
        return ""

    entries = []
    for pm in progress_messages:
        msg_text = pm.get("message", "")
        if "Still running..." in msg_text:
            continue
        ts = pm.get("timestamp")
        msg = escape_html(msg_text)
        offset_str = ""
        if step_start_time and ts:
            delta = (ts - step_start_time).total_seconds()
            offset_str = f"+{int(delta)}s"
        entries.append(
            f'<div class="progress-entry">'
            f'<span class="progress-ts">{offset_str}</span>'
            f'<span class="progress-msg">{msg}</span>'
            f'</div>'
        )

    if not entries:
        return ""

    return (
        '<div class="progress-log">'
        + "\n".join(entries)
        + '</div>'
    )


# ---------------------------------------------------------------------------
# Tab content renderer
# ---------------------------------------------------------------------------

def render_tab_content(
    steps: List[Dict[str, Any]],
    task_name: str = "",
    pipeline_start_time: Optional[datetime] = None,
    project_id: str = "",
) -> str:
    """Render the content for all tabs."""
    parts = []

    # -- Overview tab --

    # Total run time (wall clock)
    if pipeline_start_time:
        # Use end_time of the last completed/failed step, or now
        last_end = None
        for s in steps:
            et = s.get("end_time")
            if et and (last_end is None or et > last_end):
                last_end = et
        if last_end is None:
            last_end = datetime.now()
        total_wall = (last_end - pipeline_start_time).total_seconds()
    else:
        total_wall = 0.0

    # Current step: find the first running step, or show COMPLETE
    current_step_html = '<span style="color: #fff;">COMPLETE</span>'
    for s in steps:
        if s["status"] == "running":
            cfg = STEP_CONFIG.get(s["step_id"], {"label": s["step_id"], "tab_id": s["step_id"]})
            tab_id = cfg["tab_id"]
            current_step_html = f'<span style="color: var(--c-{tab_id});">{escape_html(cfg["label"])}</span>'
            break
    else:
        # Check if any step is not pending/skipped (pipeline may not have completed fully)
        any_active = any(s["status"] in ("completed", "running", "failed") for s in steps)
        if not any_active:
            current_step_html = '<span style="color: #555;">not started</span>'

    # Study title
    study_title = _format_study_title(task_name, project_id)
    study_title_html = f'<div class="study-title">{escape_html(study_title)}</div>' if study_title else ""

    overview_rows = ""
    for step in steps:
        cfg = STEP_CONFIG.get(step["step_id"], {"label": step["step_id"], "tab_id": step["step_id"]})
        tab_id = cfg["tab_id"]
        label = cfg["label"]
        status = step["status"]
        status_text, status_cls = _STATUS_TEXT.get(status, ("pending", "st-pend"))

        # Step link gets its color when completed/running, muted when pending
        if status in ("completed", "running", "failed"):
            link_cls = f"step-link step-link-{tab_id}"
        else:
            link_cls = "step-link step-link-pend"

        elapsed = ""
        if step.get("start_time") and step.get("end_time"):
            elapsed = _format_duration((step["end_time"] - step["start_time"]).total_seconds())
        elif step.get("start_time") and step["status"] == "running":
            elapsed = _format_duration((datetime.now() - step["start_time"]).total_seconds())

        overview_rows += f'''
        <tr>
            <td><a class="{link_cls}" href="#" data-tab="{escape_html(tab_id)}">{escape_html(label)}</a></td>
            <td><span class="st {status_cls}">{status_text}</span></td>
            <td>{elapsed}</td>
        </tr>'''

    parts.append(f'''
    <div class="tab-content active" id="tab-overview">
        <h2>overview</h2>
        {study_title_html}
        <div class="stats">
            <div class="stat"><div class="stat-val">{_format_duration(total_wall)}</div><div class="stat-lbl">run time</div></div>
            <div class="stat"><div class="stat-val">{current_step_html}</div><div class="stat-lbl">current step</div></div>
        </div>
        <table>
            <thead><tr><th>step</th><th>status</th><th>run time</th></tr></thead>
            <tbody>{overview_rows}</tbody>
        </table>
    </div>''')

    # -- Per-step tabs --
    for step in steps:
        interactions = step.get("llm_interactions", [])
        progress_msgs = step.get("progress_messages", [])
        output_files = step.get("output_files", [])
        step_id = step["step_id"]
        cfg = STEP_CONFIG.get(step_id, {"label": step_id, "tab_id": step_id})
        tab_id = cfg["tab_id"]

        # Render progress log (shown above main content when present)
        progress_html = render_progress_log(progress_msgs, step.get("start_time"))

        # Critique and revision: group output files by round in horizontal columns
        if step_id == "autocritique" and output_files:
            output_html = render_round_columns(
                output_files, step_id,
                "var(--c-critique)", "var(--c-critique-bright)",
            )
        elif step_id == "revision" and output_files:
            output_html = render_round_columns(
                output_files, step_id,
                "var(--c-revision)", "var(--c-revision-bright)",
            )
        else:
            # Regular output cards
            output_html = render_output_cards(output_files, step_id=step_id)

        if step_id == "iterative_analysis":
            content = render_analysis_columns(interactions)
        elif interactions:
            content = "\n".join(render_llm_card(i) for i in interactions)
        elif output_files:
            # Step with output files but no LLM interactions (agent subprocess steps)
            content = ""
        else:
            status = step["status"]
            if status == "pending":
                content = '<p class="no-data">pending</p>'
            elif status == "skipped":
                content = f'<p class="no-data">skipped{" — " + escape_html(step.get("summary", "")) if step.get("summary") else ""}</p>'
            elif progress_msgs:
                # Running step with progress but no LLM interactions yet
                content = ""
            else:
                content = '<p class="no-data">No LLM interactions recorded for this step.</p>'

        # Prepend progress log + output files
        content = progress_html + output_html + content

        parts.append(f'''
    <div class="tab-content" id="tab-{escape_html(tab_id)}">
        <h2>{escape_html(step["display_name"])}</h2>
        {content}
    </div>''')

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Full HTML template — Civ 2 inspired dark theme
# ---------------------------------------------------------------------------

DASHBOARD_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoInterp Dashboard — {task_name}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'SF Mono', 'Cascadia Code', 'Consolas', 'DejaVu Sans Mono', monospace; background: #0a0a0a; color: #ccc; font-size: 14px; line-height: 1.6; }}
a {{ color: #ccc; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

/* Civ 2 palette — bright saturated colors on black */
:root {{
    --c-questions: #36d058;
    --c-questions-bright: #5aeb7c;
    --c-questions-dim: #28a044;
    --c-prioritize: #3d8bfd;
    --c-prioritize-bright: #6aabff;
    --c-prioritize-dim: #2d6abe;
    --c-litreview: #ff9800;
    --c-litreview-bright: #ffb74d;
    --c-litreview-dim: #c47600;
    --c-analysis: #ebc934;
    --c-analysis-bright: #f2d95c;
    --c-analysis-dim: #b89c28;
    --c-visualization: #2dd4bf;
    --c-visualization-bright: #5ef0db;
    --c-visualization-dim: #22a393;
    --c-report: #f06292;
    --c-report-bright: #f48aaf;
    --c-report-dim: #c04070;
    --c-critique: #ab47bc;
    --c-critique-bright: #ce93d8;
    --c-critique-dim: #7b1fa2;
    --c-revision: #e65100;
    --c-revision-bright: #ff8a50;
    --c-revision-dim: #ac3900;
    --c-report_revision: #e57373;
    --c-report_revision-bright: #ff8a80;
    --c-report_revision-dim: #c62828;
    --c-repo: #c6a700;
    --c-repo-bright: #e6c300;
    --c-repo-dim: #9e8600;
    --c-notebook: #5c6bc0;
    --c-notebook-bright: #7986cb;
    --c-notebook-dim: #3949ab;
    /* Analysis gradient: gold -> burnt orange -> burnt umber */
    --c-analysis-1: #ebc934;
    --c-analysis-1-bright: #f2d95c;
    --c-analysis-1-dim: #b89c28;
    --c-analysis-2: #d4842a;
    --c-analysis-2-bright: #e09a4a;
    --c-analysis-2-dim: #a06620;
    --c-analysis-3: #b85a1e;
    --c-analysis-3-bright: #d07038;
    --c-analysis-3-dim: #8a4416;
}}

.header {{ padding: 12px 20px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }}
.header-title {{ color: #999; font-size: 13px; }}
.header-time {{ color: #777; font-size: 13px; }}

/* Tab bar doubles as progress bar */
.tab-bar {{ display: flex; gap: 0; border-bottom: 1px solid #333; }}
.tab-btn {{ background: none; border: none; color: #777; padding: 8px 16px; cursor: pointer; font-family: inherit; font-size: 13px; border-bottom: 2px solid transparent; }}
.tab-btn:hover {{ color: #bbb; }}
.tab-btn.active {{ color: #ccc; border-bottom-color: #888; }}
/* Step tabs colored by status even when inactive */
.tab-btn.step-done[data-tab="questions"] {{ color: var(--c-questions); }}
.tab-btn.step-done[data-tab="prioritize"] {{ color: var(--c-prioritize); }}
.tab-btn.step-done[data-tab="litreview"] {{ color: var(--c-litreview); }}
.tab-btn.step-done[data-tab="analysis"] {{ color: var(--c-analysis); }}
.tab-btn.step-done[data-tab="visualization"] {{ color: var(--c-visualization); }}
.tab-btn.step-done[data-tab="report"] {{ color: var(--c-report); }}
.tab-btn.step-done[data-tab="critique"] {{ color: var(--c-critique); }}
.tab-btn.step-run[data-tab="questions"] {{ color: var(--c-questions); }}
.tab-btn.step-run[data-tab="prioritize"] {{ color: var(--c-prioritize); }}
.tab-btn.step-run[data-tab="litreview"] {{ color: var(--c-litreview); }}
.tab-btn.step-run[data-tab="analysis"] {{ color: var(--c-analysis); }}
.tab-btn.step-run[data-tab="visualization"] {{ color: var(--c-visualization); }}
.tab-btn.step-run[data-tab="report"] {{ color: var(--c-report); }}
.tab-btn.step-run[data-tab="critique"] {{ color: var(--c-critique); }}
.tab-btn.step-done[data-tab="revision"] {{ color: var(--c-revision); }}
.tab-btn.step-run[data-tab="revision"] {{ color: var(--c-revision); }}
.tab-btn.step-done[data-tab="report_revision"] {{ color: var(--c-report_revision); }}
.tab-btn.step-run[data-tab="report_revision"] {{ color: var(--c-report_revision); }}
.tab-btn.step-done[data-tab="repo"] {{ color: var(--c-repo); }}
.tab-btn.step-run[data-tab="repo"] {{ color: var(--c-repo); }}
.tab-btn.step-done[data-tab="notebook"] {{ color: var(--c-notebook); }}
.tab-btn.step-run[data-tab="notebook"] {{ color: var(--c-notebook); }}
.tab-btn.step-pend {{ color: #555; }}
/* Active tab: brightest + colored underline */
.tab-btn.step-done[data-tab="questions"].active {{ color: var(--c-questions-bright); border-bottom-color: var(--c-questions); }}
.tab-btn.step-done[data-tab="prioritize"].active {{ color: var(--c-prioritize-bright); border-bottom-color: var(--c-prioritize); }}
.tab-btn.step-done[data-tab="litreview"].active {{ color: var(--c-litreview-bright); border-bottom-color: var(--c-litreview); }}
.tab-btn.step-run[data-tab="litreview"].active {{ color: var(--c-litreview-bright); border-bottom-color: var(--c-litreview); }}
.tab-btn.step-done[data-tab="analysis"].active,
.tab-btn.step-run[data-tab="analysis"].active {{ color: var(--c-analysis-bright); border-bottom-color: var(--c-analysis); }}
.tab-btn.step-run[data-tab="questions"].active {{ color: var(--c-questions-bright); border-bottom-color: var(--c-questions); }}
.tab-btn.step-run[data-tab="prioritize"].active {{ color: var(--c-prioritize-bright); border-bottom-color: var(--c-prioritize); }}
.tab-btn.step-run[data-tab="visualization"].active {{ color: var(--c-visualization-bright); border-bottom-color: var(--c-visualization); }}
.tab-btn.step-run[data-tab="report"].active {{ color: var(--c-report-bright); border-bottom-color: var(--c-report); }}
.tab-btn.step-run[data-tab="critique"].active {{ color: var(--c-critique-bright); border-bottom-color: var(--c-critique); }}
.tab-btn.step-done[data-tab="visualization"].active {{ color: var(--c-visualization-bright); border-bottom-color: var(--c-visualization); }}
.tab-btn.step-done[data-tab="report"].active {{ color: var(--c-report-bright); border-bottom-color: var(--c-report); }}
.tab-btn.step-done[data-tab="critique"].active {{ color: var(--c-critique-bright); border-bottom-color: var(--c-critique); }}
.tab-btn.step-done[data-tab="revision"].active {{ color: var(--c-revision-bright); border-bottom-color: var(--c-revision); }}
.tab-btn.step-run[data-tab="revision"].active {{ color: var(--c-revision-bright); border-bottom-color: var(--c-revision); }}
.tab-btn.step-done[data-tab="report_revision"].active {{ color: var(--c-report_revision-bright); border-bottom-color: var(--c-report_revision); }}
.tab-btn.step-run[data-tab="report_revision"].active {{ color: var(--c-report_revision-bright); border-bottom-color: var(--c-report_revision); }}
.tab-btn.step-done[data-tab="repo"].active {{ color: var(--c-repo-bright); border-bottom-color: var(--c-repo); }}
.tab-btn.step-run[data-tab="repo"].active {{ color: var(--c-repo-bright); border-bottom-color: var(--c-repo); }}
.tab-btn.step-done[data-tab="notebook"].active {{ color: var(--c-notebook-bright); border-bottom-color: var(--c-notebook); }}
.tab-btn.step-run[data-tab="notebook"].active {{ color: var(--c-notebook-bright); border-bottom-color: var(--c-notebook); }}
.tab-btn .status-icon {{ margin-right: 4px; }}

.main {{ padding: 20px; max-width: 1400px; margin: 0 auto; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

h2 {{ font-size: 14px; font-weight: normal; color: #888; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 2px; }}

/* Overview */
.stats {{ display: flex; gap: 24px; margin-bottom: 20px; }}
.stat-val {{ font-size: 22px; color: #ddd; }}
.stat-lbl {{ font-size: 12px; color: #777; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{ text-align: left; padding: 6px 10px; font-weight: normal; font-size: 12px; color: #777; border-bottom: 1px solid #333; }}
td {{ padding: 6px 10px; border-bottom: 1px solid #1a1a1a; font-size: 13px; color: #bbb; }}
.st {{ font-size: 12px; }}
.st-done {{ color: #fff; }}
.st-run {{ color: #ebc934; }}
.st-fail {{ color: #f06292; }}
.st-pend {{ color: #555; }}

/* Step-colored links in overview */
.step-link-questions {{ color: var(--c-questions); }}
.step-link-prioritize {{ color: var(--c-prioritize); }}
.step-link-litreview {{ color: var(--c-litreview); }}
.step-link-analysis {{ color: var(--c-analysis); }}
.step-link-visualization {{ color: var(--c-visualization); }}
.step-link-report {{ color: var(--c-report); }}
.step-link-critique {{ color: var(--c-critique); }}
.step-link-revision {{ color: var(--c-revision); }}
.step-link-report_revision {{ color: var(--c-report_revision); }}
.step-link-repo {{ color: var(--c-repo); }}
.step-link-notebook {{ color: var(--c-notebook); }}
.step-link-pend {{ color: #555; }}
.step-link:hover {{ text-decoration: underline; }}

/* LLM cards */
.card {{ border: 1px solid #222; margin-bottom: 12px; }}
.card-head {{ padding: 8px 12px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #222; color: #999; font-size: 13px; }}
.card-head .name {{ color: #ddd; }}
.card-section {{ border-top: 1px solid #1a1a1a; }}
.card-section summary {{ padding: 6px 12px; cursor: pointer; font-size: 12px; color: #999; }}
.card-section summary:hover {{ color: #ccc; }}
.card-pre {{ padding: 10px 12px; white-space: pre-wrap; word-wrap: break-word; font-family: inherit; font-size: 12px; line-height: 1.5; background: #060606; }}

/* SYSTEM text always muted */
.card-pre.sys {{ color: #777; }}

/* Per-step card colors */
#tab-questions .card-head .name {{ color: var(--c-questions-bright); }}
#tab-questions .card-head {{ border-left: 3px solid var(--c-questions-dim); }}
#tab-questions h2 {{ color: var(--c-questions); }}
#tab-questions .card-pre.user {{ color: var(--c-questions-bright); }}
#tab-questions .card-pre.asst {{ color: var(--c-questions-dim); }}

#tab-prioritize .card-head .name {{ color: var(--c-prioritize-bright); }}
#tab-prioritize .card-head {{ border-left: 3px solid var(--c-prioritize-dim); }}
#tab-prioritize h2 {{ color: var(--c-prioritize); }}
#tab-prioritize .card-pre.user {{ color: var(--c-prioritize-bright); }}
#tab-prioritize .card-pre.asst {{ color: var(--c-prioritize-dim); }}

#tab-litreview .card-head .name {{ color: var(--c-litreview-bright); }}
#tab-litreview .card-head {{ border-left: 3px solid var(--c-litreview-dim); }}
#tab-litreview h2 {{ color: var(--c-litreview); }}
#tab-litreview .card-pre.user {{ color: var(--c-litreview-bright); }}
#tab-litreview .card-pre.asst {{ color: var(--c-litreview-dim); }}

#tab-analysis h2 {{ color: var(--c-analysis); }}
.analysis-col .col-title {{ color: var(--col-color); }}
.analysis-col .att-tab.active {{ color: var(--col-bright); border-bottom-color: var(--col-color); }}
.analysis-col .role-head .rn {{ color: var(--col-bright); }}
.analysis-col .role-head {{ border-left: 3px solid var(--col-dim); }}
.analysis-col .card-pre.user {{ color: var(--col-bright); }}
.analysis-col .card-pre.asst {{ color: var(--col-dim); }}

#tab-visualization h2 {{ color: var(--c-visualization); }}
#tab-visualization .card-head .name {{ color: var(--c-visualization-bright); }}
#tab-visualization .card-head {{ border-left: 3px solid var(--c-visualization-dim); }}
#tab-visualization .card-pre.user {{ color: var(--c-visualization-bright); }}
#tab-visualization .card-pre.asst {{ color: var(--c-visualization-dim); }}

#tab-report h2 {{ color: var(--c-report); }}
#tab-report .card-head .name {{ color: var(--c-report-bright); }}
#tab-report .card-head {{ border-left: 3px solid var(--c-report-dim); }}
#tab-report .card-pre.user {{ color: var(--c-report-bright); }}
#tab-report .card-pre.asst {{ color: var(--c-report-dim); }}

#tab-critique h2 {{ color: var(--c-critique); }}
#tab-critique .card-head .name {{ color: var(--c-critique-bright); }}
#tab-critique .card-head {{ border-left: 3px solid var(--c-critique-dim); }}
#tab-critique .card-pre.user {{ color: var(--c-critique-bright); }}
#tab-critique .card-pre.asst {{ color: var(--c-critique-dim); }}

#tab-revision h2 {{ color: var(--c-revision); }}
#tab-revision .card-head .name {{ color: var(--c-revision-bright); }}
#tab-revision .card-head {{ border-left: 3px solid var(--c-revision-dim); }}
#tab-revision .card-pre.user {{ color: var(--c-revision-bright); }}
#tab-revision .card-pre.asst {{ color: var(--c-revision-dim); }}

#tab-report_revision h2 {{ color: var(--c-report_revision); }}
#tab-report_revision .card-head .name {{ color: var(--c-report_revision-bright); }}
#tab-report_revision .card-head {{ border-left: 3px solid var(--c-report_revision-dim); }}
#tab-report_revision .card-pre.user {{ color: var(--c-report_revision-bright); }}
#tab-report_revision .card-pre.asst {{ color: var(--c-report_revision-dim); }}

#tab-repo h2 {{ color: var(--c-repo); }}
#tab-repo .card-head .name {{ color: var(--c-repo-bright); }}
#tab-repo .card-head {{ border-left: 3px solid var(--c-repo-dim); }}
#tab-repo .card-pre.user {{ color: var(--c-repo-bright); }}
#tab-repo .card-pre.asst {{ color: var(--c-repo-dim); }}

#tab-notebook h2 {{ color: var(--c-notebook); }}
#tab-notebook .card-head .name {{ color: var(--c-notebook-bright); }}
#tab-notebook .card-head {{ border-left: 3px solid var(--c-notebook-dim); }}
#tab-notebook .card-pre.user {{ color: var(--c-notebook-bright); }}
#tab-notebook .card-pre.asst {{ color: var(--c-notebook-dim); }}

/* Analysis columns */
.analysis-row {{ display: flex; gap: 12px; overflow-x: auto; padding-bottom: 8px; align-items: flex-start; }}
.analysis-col {{ min-width: 450px; max-width: 560px; flex-shrink: 0; border: 1px solid #222; }}
.col-title {{ padding: 8px 12px; font-size: 13px; color: #999; border-bottom: 1px solid #222; }}
.att-bar {{ display: flex; gap: 0; border-bottom: 1px solid #222; }}
.att-tab {{ background: none; border: none; color: #666; padding: 6px 14px; cursor: pointer; font-family: inherit; font-size: 12px; border-bottom: 2px solid transparent; }}
.att-tab:hover {{ color: #aaa; }}
.att-tab.active {{ color: #ccc; border-bottom-color: #888; }}
.att-pane {{ display: none; padding: 8px; }}
.att-pane.active {{ display: block; }}
.role {{ margin-bottom: 6px; border: 1px solid #1a1a1a; }}
.role-head {{ padding: 6px 10px; font-size: 12px; color: #999; border-bottom: 1px solid #1a1a1a; display: flex; justify-content: space-between; }}
.role-head .rn {{ color: #bbb; }}
.role-section {{ border-top: 1px solid #141414; }}
.role-section summary {{ padding: 5px 10px; cursor: pointer; font-size: 12px; color: #999; }}
.role-section summary:hover {{ color: #ccc; }}

.no-data {{ color: #555; padding: 20px 0; font-size: 13px; }}

/* Progress log */
.progress-log {{ margin-bottom: 16px; border: 1px solid #1a1a1a; padding: 8px 12px; background: #060606; }}
.progress-entry {{ display: flex; gap: 12px; padding: 2px 0; font-size: 12px; line-height: 1.5; }}
.progress-ts {{ color: #555; min-width: 50px; text-align: right; flex-shrink: 0; }}
.progress-msg {{ color: #888; }}

/* Study title */
.study-title {{ color: #ddd; font-size: 16px; margin-bottom: 16px; font-weight: normal; }}

/* Visualization figures */
.viz-figure img {{ max-width: 100%; border: 1px solid #222; }}

/* Rendered markdown */
.rendered-md {{ padding: 10px 12px; font-size: 13px; line-height: 1.7; color: #ccc; }}
.rendered-md h1 {{ font-size: 18px; color: #ddd; margin: 20px 0 10px 0; font-weight: bold; }}
.rendered-md h2 {{ font-size: 16px; color: #ddd; margin: 18px 0 8px 0; font-weight: bold; }}
.rendered-md h3 {{ font-size: 14px; color: #ddd; margin: 14px 0 6px 0; font-weight: bold; }}
.rendered-md h4, .rendered-md h5, .rendered-md h6 {{ font-size: 13px; color: #ccc; margin: 12px 0 4px 0; font-weight: bold; }}
.rendered-md p {{ margin: 8px 0; }}
.rendered-md ul, .rendered-md ol {{ margin: 8px 0; padding-left: 24px; }}
.rendered-md li {{ margin: 2px 0; }}
.rendered-md code {{ background: #1a1a1a; padding: 1px 4px; font-size: 12px; }}
.rendered-md pre {{ background: #0a0a0a; padding: 10px 12px; overflow-x: auto; margin: 8px 0; }}
.rendered-md pre code {{ background: none; padding: 0; }}
.rendered-md table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
.rendered-md th {{ text-align: left; padding: 6px 10px; font-weight: bold; font-size: 12px; color: #aaa; border-bottom: 1px solid #333; }}
.rendered-md td {{ padding: 6px 10px; border-bottom: 1px solid #1a1a1a; font-size: 12px; color: #bbb; }}
.rendered-md blockquote {{ border-left: 3px solid #333; padding-left: 12px; color: #999; margin: 8px 0; }}
.rendered-md img {{ max-width: 100%; }}

/* Round-based grouping (critique, revision) */
.round-row {{ display: flex; gap: 12px; overflow-x: auto; padding-bottom: 8px; align-items: flex-start; }}
.round-col {{ min-width: 400px; max-width: 600px; flex-shrink: 0; border: 1px solid #222; padding: 8px; }}
.round-col h3 {{ font-size: 13px; color: #999; margin-bottom: 8px; font-weight: normal; text-transform: uppercase; letter-spacing: 1px; }}

.footer {{ padding: 16px 20px; color: #555; font-size: 12px; border-top: 1px solid #1a1a1a; margin-top: 40px; }}
</style>
</head>
<body>
<div class="header">
    <span class="header-title">autointerp / pipeline</span>
    <span class="header-time" id="dash-timestamp">{timestamp}</span>
</div>
<div class="tab-bar" id="dash-tab-bar">{tab_buttons}</div>
<div class="main" id="dash-main">
{tab_content}
</div>
<div class="footer">autointerp</div>
<script>
/* ---- Tab switching ---- */
function switchTab(tabId) {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    const btn = document.querySelector('.tab-btn[data-tab="' + tabId + '"]');
    if (btn) btn.classList.add('active');
    const tab = document.getElementById('tab-' + tabId);
    if (tab) tab.classList.add('active');
}}
function bindEvents() {{
    document.querySelectorAll('.tab-btn').forEach(btn => {{
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    }});
    document.querySelectorAll('.step-link').forEach(link => {{
        link.addEventListener('click', (e) => {{
            e.preventDefault();
            switchTab(link.dataset.tab);
        }});
    }});
    document.querySelectorAll('.att-tab').forEach(btn => {{
        btn.addEventListener('click', () => {{
            const col = btn.closest('.analysis-col');
            col.querySelectorAll('.att-tab').forEach(b => b.classList.remove('active'));
            col.querySelectorAll('.att-pane').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            const pane = col.querySelector('#' + btn.dataset.att);
            if (pane) pane.classList.add('active');
        }});
    }});
}}
bindEvents();

/* ---- Smart auto-refresh (preserves tab state) ---- */
const REFRESH_INTERVAL = {refresh_interval_ms};
const IS_FINAL = {is_final};

function getActiveTabId() {{
    const active = document.querySelector('.tab-btn.active');
    return active ? active.dataset.tab : 'overview';
}}

function getOpenDetails() {{
    const state = {{}};
    document.querySelectorAll('details').forEach((el, idx) => {{
        state[idx] = el.open;
    }});
    return state;
}}

function restoreOpenDetails(state) {{
    document.querySelectorAll('details').forEach((el, idx) => {{
        if (idx in state) el.open = state[idx];
    }});
}}

function getActiveAttempts() {{
    const state = {{}};
    document.querySelectorAll('.analysis-col').forEach((col, idx) => {{
        const active = col.querySelector('.att-tab.active');
        if (active) state[idx] = active.dataset.att;
    }});
    return state;
}}

function restoreActiveAttempts(state) {{
    document.querySelectorAll('.analysis-col').forEach((col, idx) => {{
        if (state[idx]) {{
            col.querySelectorAll('.att-tab').forEach(b => b.classList.remove('active'));
            col.querySelectorAll('.att-pane').forEach(p => p.classList.remove('active'));
            const btn = col.querySelector('.att-tab[data-att="' + state[idx] + '"]');
            if (btn) btn.classList.add('active');
            const pane = col.querySelector('#' + state[idx]);
            if (pane) pane.classList.add('active');
        }}
    }});
}}

async function smartRefresh() {{
    try {{
        const resp = await fetch(window.location.href);
        if (!resp.ok) return;
        const html = await resp.text();
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');

        const activeTab = getActiveTabId();
        const scrollY = window.scrollY;
        const detailsState = getOpenDetails();
        const attState = getActiveAttempts();

        const regions = ['dash-tab-bar', 'dash-main', 'dash-timestamp'];
        regions.forEach(id => {{
            const oldEl = document.getElementById(id);
            const newEl = doc.getElementById(id);
            if (oldEl && newEl) oldEl.innerHTML = newEl.innerHTML;
        }});

        bindEvents();
        switchTab(activeTab);
        window.scrollTo(0, scrollY);
        restoreOpenDetails(detailsState);
        restoreActiveAttempts(attState);

        const newScript = doc.querySelector('script');
        if (newScript && newScript.textContent.includes('IS_FINAL = true')) {{
            clearInterval(window._refreshTimer);
        }}
    }} catch (e) {{ /* ignore */ }}
}}

if (!IS_FINAL && REFRESH_INTERVAL > 0) {{
    window._refreshTimer = setInterval(smartRefresh, REFRESH_INTERVAL);
}}
</script>
</body>
</html>'''
