"""Compatibility entry point for offline analysis of saved measurements."""
from .analysis.reporting import report as _report


def report(root, out=None, *, for_backup=False):
    from pathlib import Path
    from .common import read_json
    if read_json(Path(root) / "config.json", {}).get("workflow") == "paired":
        from .workflow.paired import report_pair
        return report_pair(root, out=out, for_backup=for_backup)
    return _report(root, out=out, for_backup=for_backup)

__all__ = ["report"]
