"""Compatibility entry point for offline analysis of saved measurements."""
from .analysis.reporting import report as _report


def report(root, out=None, *, for_backup=False, reference_comparison=None):
    from pathlib import Path
    from .common import read_json
    if read_json(Path(root) / "config.json", {}).get("workflow") == "completion_comparison":
        from .analysis.completion import report as report_completion
        return report_completion(root, out=out, for_backup=for_backup, reference=reference_comparison)
    if reference_comparison is not None:
        raise ValueError("A reference comparison is supported by the completion workflow")
    if read_json(Path(root) / "config.json", {}).get("workflow") == "fresh_real":
        from .analysis.fresh_real import report as report_fresh
        return report_fresh(root, out=out, for_backup=for_backup)
    if read_json(Path(root) / "config.json", {}).get("workflow") == "paired":
        from .workflow.paired import report_pair
        return report_pair(root, out=out, for_backup=for_backup)
    if read_json(Path(root) / "config.json", {}).get("workflow") == "mixed_campaign":
        from .workflow.mixed_campaign import report_campaign
        return report_campaign(root, out=out, for_backup=for_backup)
    return _report(root, out=out, for_backup=for_backup)

__all__ = ["report"]
