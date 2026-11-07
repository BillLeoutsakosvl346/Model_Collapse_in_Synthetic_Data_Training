"""Offline export guards protect private/bulk artifacts and scientific evidence."""
from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

from model_collapse.common import PROJECT, config, digest, sha256
from model_collapse.workflow.profiles import CAMPAIGN_CHECKS, mixture_checks, protocol_fingerprint
from scripts.analysis.export_campaign_acceptance import _accepted_source, _selected, _validate, _verified_source, export


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_export_allowlist_excludes_bulk_private_and_unknown_stages(tmp_path):
    write(tmp_path / "config.json", {"seed": 42, "generations": 2})
    for rel in ("resources.json", "tokenizer/tokenizer.json", "seed_42/gen_01/checkpoint/config.json",
                "seed_42/gen_03/metrics.json", "seed_other/gen_01/metrics.json", "secrets.json",
                "seed_42/gen_01/metrics.json", "mixed_for_gen_01/selection.json",
                "shared_mixture_verification.json"):
        write(tmp_path / rel, {})
    (tmp_path / "data").mkdir()
    (tmp_path / "data/train.npy").write_bytes(b"private bulk")
    selected = {p.relative_to(tmp_path).as_posix() for p in _selected(tmp_path)}
    assert selected == {"config.json", "seed_42/gen_01/metrics.json", "mixed_for_gen_01/selection.json",
                        "shared_mixture_verification.json"}


def test_export_requires_indexed_scientific_bytes(tmp_path):
    path = write(tmp_path / "config.json", {"kind": "smoke"})
    with pytest.raises(RuntimeError, match="absent from the pinned"):
        _verified_source(tmp_path, path)
    write(tmp_path / "artifact_index.json", {"files": {"config.json": {"sha256": sha256(path)}}})
    assert _verified_source(tmp_path, path) == (path, True)
    write(path, {"kind": "changed"})
    with pytest.raises(RuntimeError, match="Missing or corrupt"):
        _verified_source(tmp_path, path)


def test_shared_mixture_source_proof_is_selected_and_index_verified(tmp_path):
    write(tmp_path / "config.json", {"seed": 42, "generations": 3})
    proof = write(tmp_path / "shared_mixture_verification.json", {"full_source_verified": True})
    write(tmp_path / "artifact_index.json", {"files": {"shared_mixture_verification.json": {"sha256": sha256(proof)}}})
    assert proof in _selected(tmp_path)
    assert _verified_source(tmp_path, proof) == (proof, True)


def test_export_rejects_full_condition_before_any_output(tmp_path):
    write(tmp_path / "config.json", {"workflow": "mixed_campaign", "kind": "full"})
    with pytest.raises(ValueError, match="smoke directory"):
        _validate(tmp_path)


def test_export_refuses_nonempty_destination_without_reading_source(tmp_path):
    source, out = tmp_path / "run", tmp_path / "already-published"
    source.mkdir()
    write(out / "report.json", {"preserve": True})
    before = (out / "report.json").read_bytes()
    with pytest.raises(ValueError, match="new or empty"):
        export(source, out)
    assert (out / "report.json").read_bytes() == before


def test_parent_export_allowlist_restricts_job_seals(tmp_path):
    write(tmp_path / "job_seals/reset50.json", {})
    write(tmp_path / "job_seals/credential.json", {"private": True})
    assert _selected(tmp_path, parent=True) == [tmp_path / "job_seals/reset50.json"]


def accepted_fixture(source):
    c = config(PROJECT / "comparisons/mixed_fraction_final5/smoke.yaml")
    runtime = "a" * 64
    certificate = {
        "workflow": "mixed_campaign", "validated_run_id": source.name,
        "source_code_sha256": runtime, "protocol_fingerprint": protocol_fingerprint(c),
        "provider_pod_state": "EXITED", "unique_training_stages": 6, "executed_epochs": 30,
        "checks": {key: True for key in CAMPAIGN_CHECKS},
        "child_certificates": {name: {
            "source_code_sha256": runtime, "protocol_fingerprint": protocol_fingerprint(job),
            "checks": {key: True for key in mixture_checks(job, include_operational=False)}}
            for name, job in c["jobs"].items()},
        "shared20_mixture": {f"{arm}_{kind}_sha256": "b" * 64
                             for arm in ("continued", "reset") for kind in ("tokens", "selection")},
    }
    write(source / "config.json", c)
    write(source / "manifest.json", {"code_sha256": runtime, "config_hash": digest(c)})
    write(source / "validation_results.json", certificate)
    return c, certificate


def test_export_acceptance_binds_valid_run_runtime_and_all_job_protocols(tmp_path):
    c, receipt = accepted_fixture(tmp_path)
    assert _accepted_source(tmp_path, c) == receipt


@pytest.mark.parametrize("change,match", [
    ("run", "different campaign run"), ("runtime", "source version"),
    ("protocol", "scientific protocol"), ("missing_check", "complete mixed-data campaign"),
    ("child_protocol", "mixed-data child protocol"), ("missing_child", "mixed-data child protocol"),
    ("failed_check", "complete mixed-data campaign"), ("manifest", "manifest does not bind"),
])
def test_export_rejects_stale_or_incomplete_acceptance(tmp_path, change, match):
    c, valid = accepted_fixture(tmp_path)
    receipt = copy.deepcopy(valid)
    if change == "run":
        receipt["validated_run_id"] = "another-campaign"
    elif change == "runtime":
        receipt["source_code_sha256"] = "c" * 64
    elif change == "protocol":
        receipt["protocol_fingerprint"] = "d" * 64
    elif change == "missing_check":
        receipt["checks"].pop("single_owner_and_deadline")
    elif change == "failed_check":
        receipt["checks"]["three_private_restores"] = False
    elif change == "child_protocol":
        receipt["child_certificates"]["reset20"]["protocol_fingerprint"] = "d" * 64
    elif change == "missing_child":
        receipt["child_certificates"].pop("reset50")
    elif change == "manifest":
        write(tmp_path / "manifest.json", {"code_sha256": "a" * 64, "config_hash": "d" * 64})
    write(tmp_path / "validation_results.json", receipt)
    with pytest.raises(RuntimeError, match=match):
        _accepted_source(tmp_path, c)
