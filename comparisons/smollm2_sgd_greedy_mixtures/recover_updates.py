"""Read completed retained-volume diagnostics against sealed private-backup hashes.

No model weights, inference, training or cloud resource mutations. Operational
registry files are never recovery destinations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from model_collapse.artifacts.paths import safe_relative
from model_collapse.cloud.s3_artifacts import client
from model_collapse.common import atomic_json, digest, read_json, sha256
from model_collapse.completion_mixing import select_rows
from model_collapse.remote import local_resources
from model_collapse.science.sgd_effects import verify_seal
from model_collapse.workflow.completion import sgd_update_probe
from model_collapse.workflow.selection import training_contract
from model_collapse.workflow.sgd_validation import audit_records

RUN = "20261114T150912Z-full-completion64-mixtures"
ARMS = ("mixed50", "mixed80")


def recover(workspace):
    workspace = Path(workspace)
    resources = local_resources(RUN)
    assert resources["volume_id"] == "hmcfo68us7" and resources["pod_id"] == "o9i0oynem3qpe3"
    s3 = client({"backend": "runpod_s3", "bucket": resources["volume_id"],
                 "region": resources["data_center"], "prefix": "analysis-read"})

    def raw(run, relative):
        safe_relative(relative)
        return s3.get_object(Bucket=resources["volume_id"], Key=f"runs/{run}/{relative}")["Body"].read()

    parent_ref = json.loads(raw(RUN, "backup_snapshot.json"))
    indexes, receipts = {}, {}

    def index(ref):
        run = ref["run_id"]
        data = raw(run, "artifact_index.json")
        if hashlib.sha256(data).hexdigest() != ref["artifact_index_sha256"]:
            raise RuntimeError("Retained index differs from the sealed backup identity")
        obj = json.loads(data)
        if obj["run_id"] != run or obj["repos"] != ref["repos"]:
            raise RuntimeError("Backup owner differs")
        folder = workspace / run
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "artifact_index.json").write_bytes(data)
        indexes[run] = obj["files"]
        receipts[run] = {"snapshot": ref, "files": {}, "retrieval": "retained_volume_bytes_verified_against_sealed_private_backup_index"}

    def get(run, relative):
        safe_relative(relative)
        meta = indexes[run][relative]
        out = workspace / run / relative
        if not out.exists() or sha256(out) != meta["sha256"]:
            data = raw(run, relative)
            if len(data) != meta["bytes"] or hashlib.sha256(data).hexdigest() != meta["sha256"]:
                raise RuntimeError("Diagnostic source differs: " + relative)
            out.parent.mkdir(parents=True, exist_ok=True)
            temporary = out.with_suffix(out.suffix + ".partial")
            temporary.write_bytes(data); temporary.replace(out)
        receipts[run]["files"][relative] = meta
        return out

    index(parent_ref)
    for name in ("config.json", "manifest.json", "environment.json", "completion.json", "child_snapshots.json",
                 "seed_42/gen_00/metrics.json", "seed_42/gen_00/probe_examples.json",
                 "seed_42/gen_00/probe/synthetic_for_gen_01/tokens.npy"):
        get(RUN, name)
    assert read_json(workspace / RUN / "completion.json")["complete"]
    refs = read_json(workspace / RUN / "child_snapshots.json")
    assert tuple(refs) == ARMS
    for arm in ARMS:
        assert refs[arm]["run_id"] == RUN + "-" + arm
        index(refs[arm])
    jobs = []
    for arm in ARMS:
        run = refs[arm]["run_id"]
        jobs.extend((run, name) for name in ("config.json", "completion.json", "seed_42/gen_00/metrics.json"))
        for cycle in range(1, 11):
            prefix = f"seed_42/gen_{cycle:02d}/"
            jobs.extend((run, prefix + name) for name in ("diagnostics_manifest.json", "training.json",
                "training_history.json", "stage_completion.json", "metrics.json", "probe_examples.json",
                "training_input/manifest.json", "mixture/manifest.json", "mixture/selection.json",
                "mixture/source_labels.npy", f"probe/synthetic_for_gen_{cycle + 1:02d}/tokens.npy"))
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda job: get(*job), jobs))
        journals = []
        for arm in ARMS:
            for cycle in range(1, 11):
                run, prefix = refs[arm]["run_id"], f"seed_42/gen_{cycle:02d}/"
                seal = read_json(workspace / run / prefix / "diagnostics_manifest.json")
                journals.extend((run, prefix + name) for name in seal["files"])
        list(pool.map(lambda job: get(*job), sorted(set(journals))))
    audits = {}
    for arm in ARMS:
        run, results = refs[arm]["run_id"], []
        root = workspace / run; c = read_json(root / "config.json")
        probe = sgd_update_probe("data/smollm2_completion64_v2", c)
        previous = read_json(root / "seed_42/gen_00/metrics.json")["weights_sha256"]
        optimizer_parent = None
        for cycle in range(1, 11):
            stage = root / f"seed_42/gen_{cycle:02d}"
            training = read_json(stage / "training.json"); lineage = training["lineage"]
            assert training["initial_weights_sha256"] == lineage["initializer_weights_sha256"] == previous
            assert lineage["generator_weights_sha256"] == previous
            assert training["optimizer_parent_sha256"] == optimizer_parent
            assert lineage["optimizer_parent_sha256"] == optimizer_parent
            binding = {"training_contract_sha256": training_contract(c),
                       "input_tokens_sha256": training["input_tokens_sha256"],
                       "initial_weights_sha256": previous, "optimizer_parent_sha256": optimizer_parent,
                       "lineage_sha256": digest(lineage), "total_optimizer_steps": training["total_optimizer_steps"]}
            seal = verify_seal(stage, expected_weights=training["selected_weights_sha256"],
                expected_probe_identity=probe["identity"], expected_steps=range((cycle - 1) * 79 + 1, cycle * 79 + 1),
                expected_binding=binding)
            assert seal == training["update_diagnostics"]
            completed = read_json(stage / "stage_completion.json")["files"]
            for name in ("diagnostics_manifest.json", "training.json", "training_history.json", *seal["files"]):
                expected = completed[name]; expected = expected["sha256"] if isinstance(expected, dict) else expected
                assert sha256(stage / name) == expected
            records = [json.loads(line) for line in (stage / seal["records_path"]).read_text(encoding="utf-8").splitlines()]
            labels = np.load(stage / "mixture/source_labels.npy", allow_pickle=False)
            expected, _ = select_rows(10000, c["synthetic_fraction"], c["seed"], cycle)
            assert np.array_equal(labels, expected["source_labels"])
            mixture = read_json(stage / "mixture/manifest.json")
            assert mixture["synthetic_fraction"] == c["synthetic_fraction"]
            assert mixture["generator_weights_sha256"] == previous
            assert read_json(stage / "training_input/manifest.json")["lineage"] == lineage
            results.append({"cycle": cycle, **audit_records(records, labels, c, cycle)})
            previous = training["selected_weights_sha256"]
            optimizer_parent = indexes[run][f"seed_42/gen_{cycle:02d}/optimizer_state/cycle.pt"]["sha256"]
        audits[arm] = results
        print(arm, "verified", sum(x["updates"] for x in results), "updates", flush=True)
    result = {"run_id": RUN, "sources": {"parent": receipts[RUN], **{a: receipts[refs[a]["run_id"]] for a in ARMS}},
              "audits": audits, "scope": "Selected diagnostic journals, scores, seals, metrics, source labels and fixed output probes only; no bulk checkpoint/optimizer restoration.",
              "new_training_or_inference": False}
    for source in result["sources"].values():
        source["verified_bytes"] = sum(v["bytes"] for v in source["files"].values())
    atomic_json(workspace / "update_recovery.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".state/sgd-mixture-update-analysis")
    recover(parser.parse_args().workspace)
