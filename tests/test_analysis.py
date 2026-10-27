"""Offline analysis must preserve evidence and distinguish inherited work."""
import subprocess
import sys
from pathlib import Path

import pytest

from model_collapse.analysis.forecasts import forecast
from model_collapse.analysis.records import read_metrics_csv
from model_collapse.common import PROJECT, atomic_json, read_json, sha256
from model_collapse.report import report


@pytest.mark.parametrize("run_id", ["20261020T211121Z-full", "20261026T192740Z-full-continued"])
def test_historical_reports_regenerate_offline_without_changing_evidence(tmp_path, run_id):
    original = PROJECT / "reports" / run_id
    hashes = {p: sha256(p) for p in original.rglob("*") if p.is_file() and "generated" not in p.parts}
    result = report(original, tmp_path / run_id)
    rows = read_metrics_csv(Path(result["report"]).parent / "metrics.csv")
    historical = read_metrics_csv(original / "metrics.csv")
    assert len(rows) == len(historical) == result["completed_generations"] == 11
    for actual, expected in zip(rows, historical):
        for field, value in expected.items():
            if field == "cycle":
                continue
            assert field in actual
            if isinstance(value, float):
                assert actual[field] == pytest.approx(value, abs=1e-12)
            else:
                assert actual[field] == value
    assert {p: sha256(p) for p in hashes} == hashes
    manifest = read_json(result["analysis_manifest"])
    assert manifest["analysis"]["source_sha256"]
    assert manifest["scientific_config_sha256"]
    assert manifest["scope"].startswith("Offline")
    for name, details in manifest["inputs"].items():
        assert sha256(Path(manifest["source_root"]) / name) == details["sha256"]
    for name, details in manifest["outputs"].items():
        assert sha256(Path(result["report"]).parent / name) == details["sha256"]
    text = Path(result["report"]).read_text(encoding="utf-8")
    assert "not predictive uncertainty" in text
    assert "no independent model re-evaluation" in manifest["scope"]


def test_report_refuses_published_or_existing_outputs(tmp_path):
    root = PROJECT / "reports/20261026T192740Z-full-continued"
    for out in (root, root / "records", root / "reproduced_figures"):
        with pytest.raises(ValueError, match="published evidence"):
            report(root, out)
    existing = tmp_path / "curated"
    existing.mkdir()
    (existing / "report.md").write_text("Authored interpretation", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        report(root, existing)
    assert (existing / "report.md").read_text() == "Authored interpretation"


def test_default_output_is_unique_generated_analysis(monkeypatch, tmp_path):
    import model_collapse.analysis.reporting as analysis
    root = PROJECT / "reports/20261026T192740Z-full-continued"
    # Replace only plotting/identity; default output policy itself is exercised.
    monkeypatch.setattr(analysis, "PROJECT", tmp_path)
    monkeypatch.setattr(analysis, "_plots", lambda *args: None)
    monkeypatch.setattr(analysis, "_analysis_identity", lambda: {"source_sha256": "test"})
    a = report(root)
    b = report(root)
    assert Path(a["report"]).parent != Path(b["report"]).parent
    assert Path(a["report"]).is_relative_to(tmp_path / "reports" / root.name / "generated")


def test_imported_shared_stages_keep_measurements_but_exclude_old_runtime(tmp_path):
    source = PROJECT / "reports/20261026T192740Z-full-continued/records"
    run = tmp_path / "reset-smoke"
    c = read_json(source / "config.json")
    c.update(initialization="reset_s0", kind="smoke")
    atomic_json(run / "config.json", c)
    for g in (0, 1, 2):
        stage = run / f"seed_42/gen_{g:02d}"
        m = read_json(source / f"seed_42/gen_{g:02d}/metrics.json")
        atomic_json(stage / "metrics.json", m)
        if g < 2:
            atomic_json(stage / "import.json", {"source_run_id": source.parent.name,
                        "source_stage_index": g, "measurement_scope": "original_source_splits",
                        "execution": "imported_shared_stage", "new_training_seconds": 0,
                        "new_generation_seconds": 0})
    result = report(run, tmp_path / "derived")
    rows = read_metrics_csv(Path(result["report"]).parent / "metrics.csv")
    assert rows[1]["train_seconds"] > 0
    for row in rows[:2]:
        assert row["stage_origin"] == "imported_shared_stage"
        assert row["new_train_seconds"] == row["new_generation_seconds"] == 0
        assert row["source_run_id"] == source.parent.name
    assert rows[2]["new_train_seconds"] == rows[2]["train_seconds"] > 0
    assert rows[2]["new_generation_seconds"] == rows[2]["generation_seconds"] > 0
    text = Path(result["report"]).read_text(encoding="utf-8")
    assert "reset to fixed real-data-trained S0" in text
    assert "First new training is cycle 2" in text
    assert "active splits may differ" in text
    assert not (Path(result["report"]).parent / "forecast.json").exists()
    assert len(read_json(result["analysis_manifest"])["imported_stage_receipts"]) == 2


def test_csv_reader_normalizes_stage_without_rewriting(tmp_path):
    path = tmp_path / "legacy.csv"
    path.write_bytes(b"cycle,test_perplexity,selected_epoch,initial_weights_sha256\r\n2,42.97,,abc\r\n")
    original = path.read_bytes()
    row, = read_metrics_csv(path)
    assert row == {"cycle": 2, "generation": 2, "test_perplexity": 42.97,
                   "selected_epoch": None, "initial_weights_sha256": "abc"}
    assert path.read_bytes() == original
    path.write_text("generation,cycle\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting"):
        read_metrics_csv(path)


def test_worker_analysis_is_backup_eligible_without_replacing_source(tmp_path):
    import shutil
    from model_collapse.hub import artifact_files
    original = PROJECT / "reports/20261026T192740Z-full-continued/records"
    run = tmp_path / "remote-run"
    run.mkdir()
    shutil.copyfile(original / "config.json", run / "config.json")
    stage = run / "seed_42/gen_01"
    stage.mkdir(parents=True)
    shutil.copyfile(original / "seed_42/gen_01/metrics.json", stage / "metrics.json")
    before = artifact_files(run)
    result = report(run, for_backup=True)
    dest = Path(result["report"]).parent
    assert dest.parent == run / "generated"
    files = artifact_files(run)
    for relative, meta in before.items():
        assert files[relative] == meta
    for filename in ("report.md", "perplexity.png", "runtime.png", "diversity.png", "analysis_manifest.json"):
        assert (dest / filename).relative_to(run).as_posix() in files
    with pytest.raises(ValueError, match="do not pass out"):
        report(run, run / "seed_42", for_backup=True)
    second = report(run, for_backup=True)
    assert Path(second["report"]).parent != dest


def test_forecast_counts_only_new_reset_work():
    bench = {"full_split_blocks": {"train": 37946, "validation": 3922, "test": 4494},
             "training_target_tokens_per_second": 10000, "generation_new_tokens_per_second": 1200}
    reset = forecast(bench, 10, .57, initialization="reset_s0", first_new_cycle=2)
    continued = forecast(bench, 10, .57, continued=True)
    fresh = forecast(bench, 10, .57)
    assert reset["new_training_stages"] == 9
    assert reset["new_full_generation_stages"] == 8
    assert reset["estimated_hours"] < continued["estimated_hours"] < fresh["estimated_hours"]


def test_reporting_import_does_not_import_training_or_cloud():
    # A fresh process makes this meaningful even when the full test suite has
    # already imported torch to test the training engine.
    result = subprocess.run([sys.executable, "-c",
        "import sys; from model_collapse.report import report; "
        "assert not any(x in sys.modules for x in ('torch', 'model_collapse.pipeline', 'model_collapse.cloud'))"],
        cwd=PROJECT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
