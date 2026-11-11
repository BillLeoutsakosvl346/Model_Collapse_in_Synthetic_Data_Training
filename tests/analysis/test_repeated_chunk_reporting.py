"""Keep same-fraction fresh/repeated conditions distinct in four-curve reports."""
import json
import math
from pathlib import Path

import matplotlib.image as image
import pytest

from model_collapse.common import atomic_json, read_json, sha256
from scripts.analysis import build_repeated_chunk_comparison as analysis


def test_four_condition_report_preserves_sources_and_square_figures(tmp_path, monkeypatch):
    root, reference, destination = tmp_path / "new", tmp_path / "old", tmp_path / "analysis"
    for source in (root, reference):
        atomic_json(source / "completion.json", {"complete": True})
        atomic_json(source / "children.json", {})
        config = {"training_chunk_policy": "repeat_fixed" if source == root else "fresh_per_cycle", "secondary_corpora": ["fineweb_edu"]}
        if source == root:
            config["fixed_chunk"] = 1
        atomic_json(source / "config.json", config)
    def collect(source):
        rows = []
        for arm in ("real", "synthetic"):
            for cycle in range(11):
                loss = 3 + cycle * (.01 if arm == "real" else .1)
                rows.append({"arm": arm, "cycle": cycle, "global_step": cycle * 79, "test_targets": 457664,
                    "synthetic_fraction":0.0 if arm == "real" else 1.0, "weights_sha256":"start" if cycle == 0 else str(cycle),
                    "test_perplexity":math.exp(loss), "validation_perplexity":math.exp(loss), "test_loss":loss,
                    "test_nll_p95":loss + 2,
                    "token_entropy_nats":6., "mean_within_entropy_nats":3., "distinct_2":.4, "distinct_3":.5,
                    "mean_within_repeated_4gram_rate":.4, "fixed_real_anchor_loss":2.9,
                    "real_anchor_generalization_gap":loss - 2.9, "learning_rate_end":5e-6 - cycle*4e-7,
                    "secondary_fineweb_edu_perplexity":math.exp(loss)})
        return rows, {}, {}
    monkeypatch.setattr(analysis, "PROJECT", tmp_path)
    monkeypatch.setattr(analysis, "_collect", collect)
    monkeypatch.setattr(analysis, "comparison_contract", lambda root: {"bank":"same", "settings":"same", "baseline_weights_sha256":"start"})
    before = {p:sha256(p) for source in (root,reference) for p in source.rglob("*.json")}
    result = analysis.build(root,reference,destination)
    rows = read_json(destination / "metrics.json")
    assert len(rows) == 44
    assert {r["condition"] for r in rows} == set(analysis.LABELS)
    assert len(read_json(destination / "figure_manifest.json")) == 14
    dimensions = image.imread(destination / "01_test_perplexity.png").shape
    assert dimensions[0] == dimensions[1]
    assert result["inputs"]
    assert before == {p:sha256(p) for p in before}
    with pytest.raises(ValueError, match="new or empty"):
        analysis.build(root,reference,destination)
    old = read_json(reference / "config.json")
    atomic_json(reference / "config.json", {**old, "training_chunk_policy":"repeat_fixed", "fixed_chunk":1})
    with pytest.raises(RuntimeError, match="fresh-per-cycle"):
        analysis.build(root,reference,tmp_path / "wrong_reference")
