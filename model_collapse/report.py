"""Stable offline reporting interface for every named workflow."""
from pathlib import Path

from .common import read_json
from .workflow.registry import handler


def report(root, out=None, *, for_backup=False, reference_comparison=None):
    c = read_json(Path(root) / "config.json", {})
    render = handler(c, "report")
    if c.get("workflow") == "completion_comparison":
        return render(root, out=out, for_backup=for_backup, reference=reference_comparison)
    if reference_comparison is not None:
        raise ValueError("A reference comparison is supported by the completion workflow")
    return render(root, out=out, for_backup=for_backup)


__all__ = ["report"]
