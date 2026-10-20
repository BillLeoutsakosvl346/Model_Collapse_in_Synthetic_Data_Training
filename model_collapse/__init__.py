"""Reproducible OPT-125M recursive synthetic-data experiments."""

import os

# Large private checkpoint transfers can take more than the Hub's 10-second read default.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
