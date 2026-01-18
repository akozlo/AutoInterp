"""AutoInterp agent framework package."""

from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point proxy so ``AutoInterp.main`` is importable without side effects."""
    from .main import main as _main

    _main(list(argv) if argv is not None else None)


__all__ = ["main"]
