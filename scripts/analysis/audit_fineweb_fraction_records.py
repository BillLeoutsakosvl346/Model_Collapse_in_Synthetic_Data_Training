"""Independent read-only audit of preserved FineWeb fraction measurements.

Only a new explicitly selected output directory is written. No cloud calls, training,
inference, credentials or edits to published evidence.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model_collapse.science.metrics import diversity, probe_diagnostics
from scripts.analysis.publication import new_destination

read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
relative = lambda p: Path(p).resolve().relative_to(ROOT).as_posix()

publication = ROOT / "reports/20261107T214406Z-full-completion64-mixed99p9"
published = publication / "analysis/seven-fractions"
raw = ROOT / "reports/20261111_fineweb_fraction_audit/records"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True, help="New/empty generated-analysis directory")
args = parser.parse_args()
out = new_destination(raw, args.out)
rows = read(published / "metrics.json")
provenance = read(published / "input_provenance.json")
aggregate = ROOT / "reports/20261107T113638Z-full-completion64-mixed80/analysis/five-fractions/metrics.json"
aggregate_hash = sha(aggregate)
evidence = []
probe_checks = []
example_checks = []

for row in rows:
    fraction, cycle = row["synthetic_fraction"], row["cycle"]
    ref = provenance["references"][str(fraction)]
    child = raw / ref["run_id"]
    path = child / f"seed_42/gen_{cycle:02d}/metrics.json"
    metric = read(path)
    index = read(child / "artifact_index.json")
    rel = path.relative_to(child).as_posix()
    assert sha(path) == index["files"][rel]["sha256"]
    for split, target_count in (("test", 457664), ("validation", 468608)):
        value = metric[split]
        assert value["scored_tokens"] == target_count
        assert math.isclose(value["loss"], value["nll_sum"] / target_count, rel_tol=1e-13)
        assert math.isclose(value["perplexity"], math.exp(value["loss"]), rel_tol=1e-13)
        assert row[split + "_perplexity"] == value["perplexity"]
    for key in ("distinct_1", "distinct_2", "distinct_3", "repeated_4gram_rate",
                "duplicate_continuation_rate", "token_entropy_nats", "mean_within_entropy_nats",
                "mean_within_repeated_4gram_rate", "newline_token_50118_share"):
        assert row[key] == metric["probe"][key]
    actual = sha(path)
    inherited = row["source_metrics_sha256"]
    if inherited == actual:
        scope = "stage_metrics_file"
    else:
        assert inherited == aggregate_hash
        scope = "inherited_aggregate_table"
    evidence.append({"synthetic_fraction": fraction, "cycle": cycle,
                     "actual_source_metrics_path": relative(path), "actual_source_metrics_sha256": actual,
                     "index_sha256": sha(child / "artifact_index.json"),
                     "reported_source_metrics_sha256": inherited, "reported_checksum_scope": scope,
                     "reported_aggregate_path": relative(aggregate) if scope == "inherited_aggregate_table" else None,
                     "numerical_source_match": True})
    if cycle:
        history = read(path.parent / "training_history.json")
        assert len(history) == 1 and history[0]["global_step"] == 79 * cycle
        assert math.isclose(history[0]["learning_rate_end"], 1e-6 + 2e-6*(1+math.cos(math.pi*cycle/10)), rel_tol=1e-10)
        probe = path.parent / f"probe/synthetic_for_gen_{cycle+1:02d}/tokens.npy"
        assert sha(probe) == index["files"][probe.relative_to(child).as_posix()]["sha256"]
        tokens = np.load(probe, allow_pickle=False)
        assert tokens.shape == (256, 64)
        actual_metrics = {**diversity(tokens), **probe_diagnostics(tokens)}
        errors = {key: abs(value - metric["probe"][key]) for key, value in actual_metrics.items()}
        assert max(errors.values()) < 1e-12
        probe_checks.append({"fraction": fraction, "cycle": cycle, "array_path": relative(probe),
                             "sha256": sha(probe), "metrics": len(actual_metrics), "max_abs_error": max(errors.values())})

for fraction in (0., .5, 1.):
    rid = provenance["references"][str(fraction)]["run_id"]
    for path in (raw / rid).glob("seed_42/gen_*/test_sequences.json"):
        seq, metric = read(path), read(path.parent / "metrics.json")
        values = np.asarray(seq["sequence_mean_nll"])
        counts = np.asarray(seq["scored_tokens_per_sequence"])
        assert len(values) == 7151 and np.all(counts == 64)
        loss = float(np.average(values, weights=counts))
        error = abs(loss - metric["test"]["loss"])
        assert error < 1e-12
        assert math.isclose(float(np.exp(loss)), metric["test"]["perplexity"], rel_tol=1e-12)
        assert math.isclose(float(np.percentile(values, 95)), metric["test"]["p95_example_nll"], rel_tol=1e-12)
        example_checks.append({"run_id": rid, "cycle": metric["generation"], "source_path": relative(path),
                               "source_sha256": sha(path), "targets": int(counts.sum()), "absolute_nll_error": error})

ten_source = raw / "aggregate_metrics_110.json"
ten = read(ten_source)
assert len(ten) == 110
trajectories = {}
for fraction in sorted({r["synthetic_fraction"] for r in ten}):
    selected = sorted([r for r in ten if r["synthetic_fraction"] == fraction], key=lambda r:r["cycle"])
    assert [r["cycle"] for r in selected] == list(range(11))
    for r in selected:
        assert r["test_targets"] == 457664 and r["validation_targets"] == 468608
        assert math.isclose(math.exp(r["test_loss"]), r["test_perplexity"], rel_tol=1e-12)
    vals = [r["test_perplexity"] for r in selected]
    trajectories[str(fraction)] = {"test_ppl": vals, "increments": np.diff(vals).tolist(),
                                 "final_over_initial": vals[-1]/vals[0],
                                 "gpu": selected[-1]["gpu"],
                                 "scope": "110-row preserved aggregate; seven selected conditions additionally checked against raw pinned records above."}

half = sorted([r for r in rows if r["synthetic_fraction"] == .5], key=lambda r:r["cycle"])
report = {
    "scope": "Offline arithmetic, source-record/index and probe-token checks; no new model inference or bulk checkpoint restoration.",
    "inputs": {relative(published / "metrics.json"): sha(published / "metrics.json"),
               relative(published / "input_provenance.json"): sha(published / "input_provenance.json"),
               relative(ten_source): sha(ten_source), relative(aggregate): aggregate_hash},
    "checks": {"raw_stage_rows_verified": len(evidence), "trained_probe_arrays_recomputed": len(probe_checks),
               "available_example_loss_arrays_checked": len(example_checks), "aggregate_conditions": len(trajectories),
               "ppl_arithmetic_relative_tolerance": 1e-13, "probe_metric_absolute_tolerance": 1e-12},
    "provenance_scope_issue": {"field": "source_metrics_sha256", "affected_rows": sum(r["reported_checksum_scope"] == "inherited_aggregate_table" for r in evidence),
        "description": "One column mixes aggregate source-table hashes for inherited 0/50/80/100% rows with stage-file hashes for newer conditions. Numerical values agree with raw pinned stage records; this is ambiguous metadata, not a numerical plot fault.",
        "preserve_originals": True},
    "per_row_sources": evidence, "probe_checks": probe_checks, "direct_example_loss_checks": example_checks,
    "trajectories": trajectories,
    "half_synthetic_secondary_final_over_baseline": {key:half[-1][key]/half[0][key] for key in half[-1] if key.startswith("secondary_")},
    "interpretation": [
        "99% peaks at cycle 5 then remains about 82.7–83.3 during cycles 6–10. It is finite-horizon flattening under LR decay, not proof of permanent stability.",
        "50% final primary PPL is close to M0, but within-output repetition is about 47.7% versus baseline39% and real-only40.6%; it is not unchanged in every metric.",
        "50% secondary PPL rises about9.3% on PG19 and5.5% on AGNews despite near-baseline primary FineWeb PPL.",
        "Hardware groups differ: 85/90/95/99/99.9 use4500 and0/20/50/80/100 use4000. All seeds42, but independently measured baseline beam outputs differ slightly.",
        "100% synthetic still uses real source prompts. Only continuation labels are synthetic; this cannot establish behavior of fully synthetic context distributions.",
    ],
}
out.mkdir(parents=True, exist_ok=True)
destination = out / "metrics-review.json"
report["audit_script_sha256"] = sha(__file__)
with (out / "corrected_provenance.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(evidence[0]))
    writer.writeheader()
    writer.writerows(evidence)
destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"out":relative(destination), "checks":report["checks"], "provenance_affected_rows":report["provenance_scope_issue"]["affected_rows"]}))
