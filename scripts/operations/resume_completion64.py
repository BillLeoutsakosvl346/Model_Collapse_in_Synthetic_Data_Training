"""Resume the accepted completion workflow after private-storage admission."""
from __future__ import annotations

import argparse
import json
import os

from huggingface_hub import HfApi, get_token

from model_collapse.common import PROJECT, STATE, atomic_json, load_secrets, read_json
from model_collapse.remote import launch
from scripts.operations.hf_storage_admission import check_storage


def resume(run_id):
    load_secrets()
    c = read_json(PROJECT / "runs" / run_id / "config.json", {})
    destination = read_json(STATE / "completion64-new-backup-destination.json", {})
    repair = read_json(PROJECT / "docs/validation/2026-11-06_completion64-backup-capacity-repair.json", {})
    if c.get("workflow") != "completion_comparison" or c.get("kind") != "full":
        raise ValueError("This entry point requires the recorded full completion comparison")
    if repair.get("run_id") != run_id or not repair.get("temporary_cpu_removed"):
        raise RuntimeError("Finish verified backup repair and stop its temporary CPU before resume")
    repos = {"model": os.environ.get("HF_MODEL_REPO"), "dataset": os.environ.get("HF_DATASET_REPO")}
    if repos != destination.get("repos") or not destination.get("free_plan"):
        raise RuntimeError("Backup destinations differ from the verified dedicated free namespace")
    evidence = check_storage(repos, destination["organization"], 100_000_000_000,
                             40_000_000_000, 10_000_000_000,
                             HfApi(token=os.environ.get("HF_TOKEN") or get_token()))
    atomic_json(PROJECT / "runs" / run_id / "storage_admission_before_resume.json", evidence)
    return launch(resume_id=run_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(resume(args.run_id), indent=2))


if __name__ == "__main__":
    main()
