"""Rebuild the smoke figures from the small committed provenance records."""
from pathlib import Path

from model_collapse.report import report


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    print(report(here / "records", here / "reproduced_figures"))
