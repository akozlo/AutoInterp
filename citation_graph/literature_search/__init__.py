"""
Literature search from citation graph: seed + forward (citing) + backward (cited).
Build a fixed 3-paper pack, download PDFs, write manifest, optionally generate research question(s).
"""

from .agent_questions import run_agent_question_generation
from .run import run_literature_search
from .sampling import build_literature_search

__all__ = ["run_literature_search", "build_literature_search", "run_agent_question_generation"]
