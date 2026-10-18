"""Choose a Times-compatible font available in the current plotting runtime."""
from __future__ import annotations

import os

from matplotlib import font_manager


def paper_font() -> str:
    """Prefer the manuscript font, with Matplotlib's bundled STIX fallback."""
    requested = os.environ.get("ASCT_FIGURE_FONT")
    candidates = (requested,) if requested else ("Times New Roman", "STIXGeneral")
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        return name
    raise RuntimeError(f"No available figure font among {candidates}")
