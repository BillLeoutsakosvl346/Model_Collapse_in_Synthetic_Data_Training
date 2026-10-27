"""Compatibility entry point for the MANUAL PAID shutdown probe.

This creates cloud compute. It is not part of ordinary offline verification.
"""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "operations" / "check_shutdown_timeout.py"), run_name="__main__")
