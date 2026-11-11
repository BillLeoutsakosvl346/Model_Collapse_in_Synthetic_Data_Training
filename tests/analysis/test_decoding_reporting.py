"""Equal synthetic fractions must remain distinct decoder curves."""
import copy
from pathlib import Path

import matplotlib.axes
import yaml

from model_collapse.common import atomic_json, read_json
from model_collapse.analysis import completion


def test_decoder_report_has_separate_greedy_sampling_and_historical_beam_lines(tmp_path, monkeypatch):
    reference = (Path(__file__).resolve().parents[2] /
                 "reports/20261110T234038Z-full-completion64-decoding/records/20261106T193918Z-full-completion64")
    rows, inputs, examples = completion._collect(reference)
    root = tmp_path / "new"
    config = yaml.safe_load(Path("comparisons/fineweb_completion64_decoding/full.yaml").read_text())
    atomic_json(root / "config.json", config)
    atomic_json(root / "seed_42/gen_00/metrics.json", read_json(reference / "seed_42/gen_00/metrics.json"))
    synthetic = [r for r in rows if r["arm"] == "synthetic"]
    new_rows = []
    for name in ("greedy", "sampling"):
        for old in synthetic:
            new_rows.append({**copy.deepcopy(old), "arm": name, "training_decoder": name, "comparison_id": root.name})
    monkeypatch.setattr(completion, "_collect", lambda source: (new_rows, {}, {}) if source == root else (rows, inputs, examples))
    labels = []
    original = matplotlib.axes.Axes.plot
    def plot(self, *args, **kwargs):
        labels.append(kwargs.get("label"))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", plot)
    result = completion.report(root, out=tmp_path / "analysis", reference=reference)
    assert result["rows"] == 44
    assert set(labels) == {"0% synthetic", "100% synthetic / beam", "100% synthetic / greedy", "100% synthetic / sampling"}
    assert {r["training_decoder"] for r in read_json(tmp_path / "analysis/metrics.json")} == {
        "real_targets", "beam", "greedy", "sampling"}
    assert read_json(tmp_path / "analysis/analysis_manifest.json")["reference_execution_difference"] == config["reference_execution_difference"]
    wrong = copy.deepcopy(config)
    wrong["learning_rate"] = 6e-6
    atomic_json(root / "config.json", wrong)
    import pytest
    with pytest.raises(RuntimeError, match="scientific settings differ"):
        completion.report(root, out=tmp_path / "wrong", reference=reference)
