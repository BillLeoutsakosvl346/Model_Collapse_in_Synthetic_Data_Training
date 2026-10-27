"""Explicit workspace locations; stdlib-only for the bootstrap watchdog."""
from __future__ import annotations

import os
from pathlib import Path


def project_root():
    override = os.environ.get("MODEL_COLLAPSE_PROJECT_ROOT")
    return Path(override).expanduser().resolve() if override else Path(__file__).resolve().parent.parent


PROJECT = project_root()
STATE = Path(os.environ.get("MODEL_COLLAPSE_STATE_DIR", str(PROJECT / ".state"))).expanduser().resolve()
