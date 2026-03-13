"""
Context pack from citation graph: seed + forward (citing) + backward (cited).
Build a fixed 3-paper pack, download PDFs, write manifest, optionally generate research question(s).
"""

from .agent_questions import run_agent_question_generation
from .run import run_context_pack
from .sampling import build_context_pack

__all__ = ["run_context_pack", "build_context_pack", "run_agent_question_generation"]
