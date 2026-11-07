"""Rebuild the FineWeb smoke figures and metrics from published small records."""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from publication import PROJECT, new_destination, verified_publication, write_provenance
from model_collapse.analysis.fresh_real import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "reports/20261106T143315Z-smoke-fresh-real")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    verified_publication(source)
    output = new_destination(source, args.out)
    records = source / "records"
    with tempfile.TemporaryDirectory(prefix="fineweb-offline-report-") as temporary:
        workspace = Path(temporary) / source.name
        shutil.copytree(records, workspace)
        report(workspace, out=output)
    write_provenance(source, output, Path(__file__), list(records.rglob("*.json")),
                     scope="Offline regeneration of published FineWeb pipeline-validation metrics and figures.")
    print(output)


if __name__ == "__main__":
    main()
