"""Publication tooling must stay offline and preserve recorded scientific evidence."""
import csv
import hashlib
import json
import re
import shutil
import socket
from pathlib import Path

import pytest

from model_collapse.common import PROJECT
from scripts.analysis.build_mixed_acceptance import DEFAULT_SOURCE, build
from scripts.analysis.build_week3_report import ROOT, build as build_week3
from scripts.analysis.publication import checksum, new_destination
from scripts.maintenance.check_repository import markdown_errors, publication_errors


def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Offline report construction attempted network access")
    monkeypatch.setattr(socket.socket, "connect", denied)


def test_report_destination_protects_evidence_and_existing_output(tmp_path):
    for target in (DEFAULT_SOURCE, DEFAULT_SOURCE / "analysis", DEFAULT_SOURCE.parent):
        with pytest.raises(ValueError):
            new_destination(DEFAULT_SOURCE, target)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="new or empty"):
        new_destination(DEFAULT_SOURCE, occupied)
    assert (occupied / "keep.txt").read_text() == "keep"


def test_mixed_report_rebuild_is_offline_and_preserves_measurements(tmp_path, monkeypatch):
    deny_network(monkeypatch)
    before = {p: checksum(p) for p in DEFAULT_SOURCE.rglob("*") if p.is_file() and "generated" not in p.parts}
    result = build(DEFAULT_SOURCE, tmp_path / "mixed")
    out = Path(result["report"]).parent
    rows = list(csv.DictReader((out / "analysis/metrics.csv").open(encoding="utf-8")))
    for row in rows:
        metric = json.loads((DEFAULT_SOURCE / f"records/seed_42/gen_{int(row['generation']):02d}/metrics.json").read_text())
        assert float(row["test_perplexity"]) == metric["test"]["perplexity"]
        if row["generation"] != "0":
            assert float(row["real_blocks"]) == float(row["synthetic_blocks"]) == 128
            assert row["selected_epoch"] == "5"
    assert result["mixtures_checked"] == 2
    assert "not current cloud state" in (out / "report.md").read_text()
    provenance = json.loads((out / "publication_reproduction.json").read_text())
    assert provenance["analysis"]["script_sha256"]
    assert provenance["inputs"] and provenance["outputs"]
    assert all(checksum(out / name) == meta["sha256"] for name, meta in provenance["outputs"].items())
    assert all(checksum(p) == h for p, h in before.items())


def test_corrupted_publication_fails_before_creating_output(tmp_path, monkeypatch):
    deny_network(monkeypatch)
    source = tmp_path / "source"
    shutil.copytree(DEFAULT_SOURCE, source, ignore=shutil.ignore_patterns("generated"))
    (source / "dataset_preview.json").write_text("[]")
    with pytest.raises(RuntimeError, match="corrupt publication"):
        build(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_self_consistent_manifest_cannot_hide_wrong_mixture_counts(tmp_path, monkeypatch):
    deny_network(monkeypatch)
    source = tmp_path / "source"
    shutil.copytree(DEFAULT_SOURCE, source, ignore=shutil.ignore_patterns("generated"))
    relative = "records/mixed_for_gen_02/manifest.json"
    target = source / relative
    d = json.loads(target.read_text())
    d.update(real_blocks=127, synthetic_blocks=129)
    target.write_text(json.dumps(d))
    manifest_path = source / "publication_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][relative].update(sha256=checksum(target), bytes=target.stat().st_size)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="counts or shuffle"):
        build(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_week3_builder_requires_safe_output_and_keeps_links_valid(tmp_path, monkeypatch):
    deny_network(monkeypatch)
    before = checksum(ROOT / "report.md")
    with pytest.raises(ValueError):
        build_week3(ROOT)
    out = tmp_path / "week3"
    build_week3(out)
    assert checksum(ROOT / "report.md") == before
    assert (out / "draft.json").is_file() and (out / "publication_reproduction.json").is_file()
    links = re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", (out / "report.md").read_text())
    assert links
    for link in links:
        if not re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", link):
            assert (out / link).resolve().exists(), link
    with pytest.raises(ValueError, match="new or empty"):
        build_week3(out)


def test_markdown_checker_detects_missing_target_but_ignores_code(tmp_path):
    doc = tmp_path / "README.md"
    doc.write_text('[missing](absent.md)\n```text\n[example](not-a-real-file.md)\n```\n[external](https://example.invalid/)\n')
    errors = markdown_errors(tmp_path, [doc])
    assert len(errors) == 1 and "absent.md" in errors[0]


def test_publication_checker_detects_tampering_and_path_escape(tmp_path):
    root = tmp_path / "reports/run"
    root.mkdir(parents=True)
    payload = root / "record.json"
    payload.write_bytes(b'{}')
    manifest = root / "publication_manifest.json"
    manifest.write_text(json.dumps({"files": {"record.json": {"bytes": 2, "sha256": hashlib.sha256(b'{}').hexdigest()}}}))
    assert publication_errors(tmp_path, [manifest]) == []
    payload.write_bytes(b'{"changed":true}')
    assert "checksum/size differs" in publication_errors(tmp_path, [manifest])[0]
    manifest.write_text(json.dumps({"files": {"../../outside": {"bytes": 0, "sha256": "0" * 64}}}))
    assert "unsafe evidence path" in publication_errors(tmp_path, [manifest])[0]
