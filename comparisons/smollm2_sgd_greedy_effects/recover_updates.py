"""Selective pinned recovery of diagnostics, without weights or cloud compute."""
from __future__ import annotations

import os
from pathlib import Path

# HF's nested symlink creation needs extended Windows paths for attempt journals.
if os.name == "nt":
    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub")).resolve()
    os.environ["HF_HUB_CACHE"] = str(cache) if str(cache).startswith("\\\\?\\") else "\\\\?\\" + str(cache)

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from model_collapse.artifacts.gradient_sources import Snapshot
from model_collapse.common import atomic_json, digest, load_secrets, read_json, sha256
from model_collapse.completion_mixing import select_rows
from model_collapse.science.sgd_effects import verify_seal
from model_collapse.workflow.completion import sgd_update_probe
from model_collapse.workflow.selection import training_contract
from model_collapse.workflow.sgd_validation import audit_records

RUN = "20261114T001352Z-full-completion64-mixtures"


def recover(workspace):
    load_secrets()
    workspace = Path(workspace)
    receipt = read_json("docs/validation/2026-11-14_sgd-greedy-full-completion-status.json")
    parent = Snapshot(receipt["parent_backup_snapshot"], workspace / RUN)
    for name in ("config.json", "manifest.json", "environment.json", "completion.json",
                 "child_snapshots.json", "seed_42/gen_00/metrics.json"):
        parent.get(name)
    refs = parent.json("child_snapshots.json")
    sources = {arm: Snapshot(refs[arm], workspace / (RUN + "-" + arm))
               for arm in ("real", "mixed99", "synthetic")}
    jobs = [(src, name) for src in sources.values() for name in
            ("config.json", "completion.json", "seed_42/gen_00/metrics.json")]
    for src in sources.values():
        for cycle in range(1, 11):
            jobs.extend((src, f"seed_42/gen_{cycle:02d}/{name}") for name in
                        ("diagnostics_manifest.json", "training.json", "training_history.json",
                         "stage_completion.json", "metrics.json"))
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda pair: pair[0].get(pair[1]), jobs))
        journals = []
        for src in sources.values():
            for cycle in range(1, 11):
                prefix = f"seed_42/gen_{cycle:02d}/"
                seal = read_json(src.root / prefix / "diagnostics_manifest.json")
                journals.extend((src, prefix + name) for name in seal["files"])
        list(pool.map(lambda pair: pair[0].get(pair[1]), journals))
    audits = {}
    for arm, src in sources.items():
        c = read_json(src.root / "config.json")
        probe = sgd_update_probe("data/smollm2_completion64_v2", c)
        results = []
        for cycle in range(1, 11):
            stage = src.root / f"seed_42/gen_{cycle:02d}"
            training = read_json(stage / "training.json")
            lineage = training["lineage"]
            binding = {"training_contract_sha256": training_contract(c),
                       "input_tokens_sha256": training["input_tokens_sha256"],
                       "initial_weights_sha256": lineage["initializer_weights_sha256"],
                       "optimizer_parent_sha256": lineage["optimizer_parent_sha256"],
                       "lineage_sha256": digest(lineage),
                       "total_optimizer_steps": training["total_optimizer_steps"]}
            seal = verify_seal(stage, expected_weights=training["selected_weights_sha256"],
                expected_probe_identity=probe["identity"],
                expected_steps=range((cycle - 1) * 79 + 1, cycle * 79 + 1),
                expected_binding=binding)
            assert seal == training["update_diagnostics"]
            completed = read_json(stage / "stage_completion.json")["files"]
            for name in ("diagnostics_manifest.json", "training.json", "training_history.json", *seal["files"]):
                expected = completed[name]
                expected = expected["sha256"] if isinstance(expected, dict) else expected
                assert sha256(stage / name) == expected, (arm, cycle, name)
            records = [json.loads(line) for line in (stage / seal["records_path"]).read_text(encoding="utf-8").splitlines()]
            selection, _ = select_rows(10000, c["synthetic_fraction"], c["seed"], cycle)
            results.append({"cycle": cycle, **audit_records(records, selection["source_labels"], c, cycle)})
        audits[arm] = results
        print(arm, "verified", sum(row["updates"] for row in results), "updates", flush=True)
    result = {"run_id": RUN, "sources": {"parent": parent.receipt(),
              **{arm: src.receipt() for arm, src in sources.items()}}, "audits": audits,
              "scope": "All selected diagnostic seals/journals/probe arrays verified; no bulk model or optimizer restoration.",
              "new_training_or_inference": False}
    atomic_json(workspace / "update_recovery.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".state/sgd-update-analysis")
    args = parser.parse_args()
    recover(args.workspace)
