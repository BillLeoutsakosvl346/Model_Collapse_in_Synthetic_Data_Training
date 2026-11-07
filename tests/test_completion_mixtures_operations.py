import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_collapse.common import atomic_json, config, read_json, sha256
from model_collapse.workflow import completion
from model_collapse.workflow.profiles import protocol_fingerprint, require_acceptance

CONFIG = Path("comparisons/fineweb_completion64_mixtures")


def test_mixture_profiles_bind_both_ratios_and_preserve_historical_protocol():
    full, smoke = (config(CONFIG / (kind + ".yaml")) for kind in ("full", "smoke"))
    assert protocol_fingerprint(full) == protocol_fingerprint(smoke)
    assert completion.allocation_id(full) == "fineweb_completion64_mixtures"
    assert completion.acceptance_campaign(full) != completion.ACCEPTANCE
    old = config("comparisons/fineweb_real_vs_synthetic_completion64/full.yaml")
    old["prepared_bank"] = read_json("docs/validation/2026-11-06_completion64-data.json")["bank"]
    assert protocol_fingerprint(old) == "b22e6d66d82ea7393ee265b757a0d11f9e1d16f5519059faf798e0e285ab45ac"
    assert completion.allocation_id(old) == completion.ALLOCATION
    changed = copy.deepcopy(full)
    changed["completion_arms"][1]["synthetic_fraction"] = .3
    assert protocol_fingerprint(changed) != protocol_fingerprint(full)
    certificate = {"source_code_sha256": "runtime", "protocol_fingerprint": protocol_fingerprint(full),
                   "workflow": "completion_comparison", "checks": {k: True for k in completion.CHECKS}}
    with pytest.raises(RuntimeError, match="both completion arms"):
        require_acceptance(full, certificate, "runtime")
    certificate["checks"].update({k: True for k in completion.acceptance_checks(full)})
    require_acceptance(full, certificate, "runtime")
    with pytest.raises(RuntimeError, match="scientific protocol"):
        require_acceptance(changed, certificate, "runtime")


@pytest.mark.parametrize("definitions", [[], [{"id": "../bad", "synthetic_fraction": .5}, {"id": "ok", "synthetic_fraction": .2}],
                                       [{"id": "same", "synthetic_fraction": .5}, {"id": "same", "synthetic_fraction": .2}],
                                       [{"id": "a", "synthetic_fraction": True}, {"id": "b", "synthetic_fraction": .2}]])
def test_arm_definitions_reject_invalid_or_unsafe_conditions(definitions):
    c = config(CONFIG / "full.yaml")
    c["completion_arms"] = definitions
    with pytest.raises(ValueError):
        completion.validate_config(c)


def test_mixture_forecast_counts_only_selected_training_generations():
    c = config(CONFIG / "full.yaml")
    b = {"full_primary_targets": {"test": 1, "validation": 1}, "full_secondary_targets": 1,
         "training_target_tokens_per_second": 100, "generation_new_tokens_per_second": 100,
         "primary_target_tokens_per_second": 100, "secondary_target_tokens_per_second": 100}
    forecast = completion.forecast(b, c, .57)
    assert forecast["generated_training_continuations"] == 70000
    assert forecast["generated_training_tokens"] == 4480000
    assert forecast["training_target_exposures"] == 12800000
    assert forecast["unique_training_stages"] == forecast["generation_stages"] == 20


def test_parallel_validation_is_limited_to_one_registered_protected_comparison(monkeypatch):
    from model_collapse import remote
    c = config(CONFIG / "smoke.yaml")
    run = c["parallel_validation_with"]
    state = {"allocations": {completion.MIXTURE_ALLOCATION: {"parallel_validation_with": run}}}
    owner = {"kind": "full", "workflow": "completion_comparison", "pod_id": "protected", "volume_id": "preserve"}
    monkeypatch.setattr(remote, "local_resources", lambda name: owner if name == run else None)
    pod = {"id": "protected", "name": "iib-" + run, "status": "RUNNING",
           "mounts": {"network": [{"volumeId": "preserve"}]}}
    remote.check_active_pods(c, [pod], state)
    for changed, pods, ledger in (({**c, "kind": "full"}, [pod], state),
                                  (c, [pod, {**pod, "id": "second"}], state),
                                  (c, [{**pod, "id": "unknown"}], state),
                                  (c, [{**pod, "mounts": {}}], state),
                                  (c, [pod], {})):
        with pytest.raises(RuntimeError):
            remote.check_active_pods(changed, pods, ledger)


def test_capacity_failure_prevents_any_new_provider_resource(tmp_path, monkeypatch):
    from model_collapse import remote
    c = config(CONFIG / "smoke.yaml")
    repos = {"model": "project/models", "dataset": "project/data"}
    atomic_json(tmp_path / "completion64-new-backup-destination.json",
                {"free_plan": True, "organization": "project", "repos": repos})
    calls = []
    class Provider:
        def request(self, method, path):
            calls.append((method, path))
            assert (method, path) == ("GET", "/pods")
            return {"pods": []}
    class Hub:
        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(private=True, sha="pinned", siblings=[])
        def list_lfs_files(self, *args, **kwargs):
            return [SimpleNamespace(file_oid="large", size=45000000000)]
    monkeypatch.setattr(remote, "config", lambda path: copy.deepcopy(c))
    monkeypatch.setattr(remote, "STATE", tmp_path)
    monkeypatch.setattr(remote, "load_secrets", lambda: None)
    monkeypatch.setattr(remote, "Runpod", Provider)
    monkeypatch.setattr(remote, "preflight", lambda **kwargs: {"hugging_face": {"repos": repos}})
    monkeypatch.setattr(remote, "HfApi", lambda **kwargs: Hub())
    monkeypatch.setattr(completion, "resolve_inputs", lambda value: value)
    with pytest.raises(RuntimeError, match="capacity is insufficient"):
        remote._launch(smoke=True, config_path="unused")
    assert calls == [("GET", "/pods")]


def test_private_index_contains_sealed_nested_probe_journals(tmp_path):
    from model_collapse.hub import artifact_files
    atomic_json(tmp_path / "status.json", {"state": "running"})
    (tmp_path / "events.jsonl").write_text("root volatile\n")
    journal = tmp_path / "seed_42/gen_01/probe/events.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text("sealed probe journal\n")
    files = artifact_files(tmp_path)
    assert "status.json" not in files and "events.jsonl" not in files
    assert files["seed_42/gen_01/probe/events.jsonl"]["sha256"] == sha256(journal)


def test_reference_report_rejects_changed_bank_or_scientific_settings_before_writing(tmp_path, monkeypatch):
    from model_collapse.analysis import completion as analysis
    a, b = tmp_path / "a", tmp_path / "b"
    c = config(CONFIG / "full.yaml")
    c["prepared_bank"] = {"manifest_sha256": "same-bank"}
    baseline = {"weights_sha256": "pretrained", "test": {"scored_tokens": 457664}, "validation": {"scored_tokens": 468608}}
    for root in (a, b):
        atomic_json(root / "config.json", c)
        atomic_json(root / "seed_42/gen_00/metrics.json", baseline)
        atomic_json(root / "completion.json", {"complete": True})
    monkeypatch.setattr(analysis, "_collect", lambda root: ([], {}, {}))
    changed = copy.deepcopy(c)
    changed["prepared_bank"]["manifest_sha256"] = "different-bank"
    atomic_json(b / "config.json", changed)
    destination = tmp_path / "analysis"
    with pytest.raises(RuntimeError, match="scientific settings differ"):
        analysis.report(a, out=destination, reference=b)
    assert not destination.exists()


def test_matched_reference_report_combines_four_fractions_without_changing_evidence(tmp_path):
    import math
    from matplotlib import image
    from model_collapse.analysis.completion import report
    main, reference = tmp_path / "mixtures", tmp_path / "endpoints"
    probe = {"distinct_2": .3, "token_entropy_nats": 3., "mean_within_entropy_nats": 2.,
             "mean_within_repeated_4gram_rate": .1, "repeated_4gram_rate": .2, "duplicate_continuation_rate": 0.}
    metric = {"weights_sha256": "original", "validation": {"loss": 1., "perplexity": math.e, "scored_tokens": 128},
              "test": {"loss": 1., "perplexity": math.e, "scored_tokens": 128, "p95_example_nll": 1., "median_example_nll": 1.},
              "secondary": {"fineweb_edu": {"perplexity": math.e}}, "probe": probe}
    for root, mixtures in ((main, True), (reference, False)):
        c = config(CONFIG / "full.yaml")
        c.update(generations=1, train_pairs_per_cycle=10, evaluation_pairs=2, probe_count=2, secondary_corpora=["fineweb_edu"])
        if not mixtures:
            c.pop("completion_arms")
        atomic_json(root / "config.json", c)
        atomic_json(root / "completion.json", {"complete": True})
        atomic_json(root / "seed_42/gen_00/metrics.json", metric)
        for child in completion.child_roots(root, c).values():
            for cycle in (0, 1):
                atomic_json(child / f"seed_42/gen_{cycle:02d}/metrics.json", metric)
    before = {p: sha256(p) for root in (main, reference) for p in root.rglob("*.json")}
    destination = tmp_path / "comparison"
    result = report(main, out=destination, reference=reference)
    rows = read_json(destination / "metrics.json")
    assert result["rows"] == 8 and {row["synthetic_fraction"] for row in rows} == {0, .2, .5, 1}
    assert read_json(destination / "analysis_manifest.json")["reference_contract"] is not None
    dimensions = image.imread(destination / "test_perplexity.png").shape
    assert dimensions[0] == dimensions[1]
    assert before == {p: sha256(p) for p in before}
