"""Derived forecasts use the recorded campaign volume, without changing evidence."""
import shutil

import pytest

from model_collapse.common import PROJECT, atomic_json, read_json


def test_child_report_uses_recorded_sixty_gb_storage(tmp_path, monkeypatch):
    from model_collapse.analysis import reporting

    source = PROJECT / "reports/20261026T192740Z-full-continued/records"
    root = tmp_path / "child"
    root.mkdir()
    shutil.copyfile(source / "config.json", root / "config.json")
    stage = root / "seed_42/gen_01"
    stage.mkdir(parents=True)
    for filename in ("metrics.json", "training_history.json"):
        shutil.copyfile(source / "seed_42/gen_01" / filename, stage / filename)
    hourly = 60 * .07 / 730
    atomic_json(root / "resources.json", {
        "quoted_gpu_hourly_usd": .57, "network_volume_gb": 60,
        "volume_hourly_usd": hourly})
    atomic_json(root / "benchmark.json", {
        "gpu": "test", "full_split_blocks": {"train": 37946, "validation": 3922, "test": 4494},
        "training_target_tokens_per_second": 34047,
        "generation_new_tokens_per_second": 1306})
    monkeypatch.setattr(reporting, "_plots", lambda *args: None)
    output = tmp_path / "analysis"
    reporting.report(root, output)
    forecast = read_json(output / "forecast.json")
    assert forecast["network_volume_gb"] == 60
    assert forecast["network_volume_hourly_usd"] == hourly
    assert forecast["hourly_usd"] == pytest.approx(.57 + hourly + 20 * .1 / 730)
    assert "60 GB network" in forecast["storage_assumption"]
