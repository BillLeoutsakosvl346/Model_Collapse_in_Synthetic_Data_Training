"""Offline checks of acceptance accounting and source-effect interpretation."""
import copy
import math

import numpy as np
import pytest

from model_collapse.analysis.sgd_effects import effect_summary
from model_collapse.workflow.sgd_validation import audit_greedy_generation, audit_records, audit_replay, audit_state


def _records(labels, cycle=1):
    c = {"effective_batch": 128, "microbatch": 32, "learning_rate": .001, "seed": 42}
    order = np.random.default_rng(42 + cycle).permutation(len(labels))
    total_steps = math.ceil(len(labels) / 128)
    result = []
    for offset in range(0, len(labels), 128):
        indices = order[offset:offset + 128]
        real = int(np.sum(labels[indices] == 0)); synthetic = len(indices) - real
        start = 3.
        real_loss, synthetic_loss = start - real * 1e-4, start + synthetic * 1e-5
        mixed = real_loss + synthetic_loss - start
        if not real: real_loss = start; mixed = synthetic_loss
        if not synthetic: synthetic_loss = start; mixed = real_loss
        losses = {"no_update": start, "real": real_loss, "synthetic": synthetic_loss, "mixed": mixed}
        effects = {"real_only_delta_nll": real_loss - start, "synthetic_only_delta_nll": synthetic_loss - start,
                   "mixed_delta_nll": mixed - start, "real_conditional_delta_nll": mixed - synthetic_loss,
                   "synthetic_conditional_delta_nll": mixed - real_loss,
                   "interaction_nll": mixed - real_loss - synthetic_loss + start}
        gradient = {"real_norm": real / len(indices), "synthetic_norm": synthetic / len(indices),
                    "total_norm": 1., "real_synthetic_cosine": 0. if real and synthetic else None,
                    "reconstruction_relative_error": 1e-7}
        gradient["layers"] = {"layers.0": dict(gradient)}
        result.append({"step": (cycle - 1) * total_steps + offset // 128 + 1,
                       "cycle": cycle, "learning_rate": .001, "batch_indices": indices.tolist(),
                       "real_pairs": real, "synthetic_pairs": synthetic, "real_scored_targets": real * 64,
                       "synthetic_scored_targets": synthetic * 64, "total_scored_targets": len(indices) * 64,
                       "real_source_mean_nll": 4. if real else None,
                       "synthetic_source_mean_nll": .2 if synthetic else None,
                       "gradient": gradient, "mixed_shadow_exact": True, "mode_preserved": True,
                       "rng_preserved": True, "update_l2_norm": .02, "effects": effects,
                       "states": {name: {"loss": value, "perplexity": math.exp(value)} for name, value in losses.items()}})
    return result, c


def _state():
    return {"optimizer": {"state": {}, "param_groups": [{"lr": .001, "momentum": 0., "dampening": 0.,
                "weight_decay": 0., "nesterov": False, "maximize": False, "foreach": False, "fused": False}]},
            "scheduler": None, "global_step": 4, "model_weights_sha256": "a" * 64,
            "rng": {"torch": [1]}}


def test_plain_sgd_state_has_only_resume_bookkeeping():
    state = _state()
    result = audit_state(state, {"selected_weights_sha256": "a" * 64}, {"learning_rate": .001}, 4)
    assert result["optimizer_history_entries"] == 0
    assert result["scheduler"] is None


def test_training_and_output_probe_require_same_explicit_greedy_protocol():
    manifest = {"provenance": {"parent_weights_sha256": "a" * 64, "prompt_count": 32,
        "new_tokens": 64, "num_beams": 1, "decoder": "greedy", "repetition_penalty": 1.,
        "decoding": {"do_sample": False, "min_new_tokens": 64, "max_new_tokens": 64}}}
    audit_greedy_generation(manifest, "a" * 64, 32)
    manifest["provenance"]["num_beams"] = 5
    with pytest.raises(RuntimeError, match="greedy"):
        audit_greedy_generation(manifest, "a" * 64, 32)


@pytest.mark.parametrize("change", ["adam", "scheduler", "rate", "weights", "rng", "step"])
def test_incompatible_optimizer_or_state_rejected(change):
    state = _state()
    if change == "adam": state["optimizer"]["state"][0] = {"exp_avg": [1]}
    if change == "scheduler": state["scheduler"] = {}
    if change == "rate": state["optimizer"]["param_groups"][0]["lr"] = .002
    if change == "weights": state["model_weights_sha256"] = "b" * 64
    if change == "rng": state["rng"] = None
    if change == "step": state["global_step"] = 3
    with pytest.raises(RuntimeError):
        audit_state(state, {"selected_weights_sha256": "a" * 64}, {"learning_rate": .001}, 4)


@pytest.mark.parametrize("fraction", [0., .99, 1.])
def test_actual_counts_endpoints_and_sixteen_pair_tail(fraction):
    from model_collapse.completion_mixing import select_rows
    selection, _ = select_rows(400, fraction, 42, 1)
    records, c = _records(selection["source_labels"])
    result = audit_records(records, selection["source_labels"], c, 1)
    assert result["updates"] == 4
    assert result["final_batch_pairs"] == 16
    assert result["real_pairs"] == {0.: 400, .99: 4, 1.: 0}[fraction]
    if fraction == .99:
        assert result["mixed_microbatches"] > 0
        assert result["real_present_updates"] < 4


def test_second_cycle_step_numbers_and_fresh_seed():
    labels = np.ones(400, dtype=np.int32); labels[:4] = 0
    records, c = _records(labels, cycle=2)
    assert [r["step"] for r in records] == [5, 6, 7, 8]
    audit_records(records, labels, c, 2)


@pytest.mark.parametrize("change", ["duplicate", "order", "counts", "step", "rate", "reconstruction", "shadow", "rng", "effect", "perplexity", "layers"])
def test_bad_update_evidence_rejected(change):
    labels = np.ones(400, dtype=np.int32); labels[:4] = 0
    records, c = _records(labels)
    if change == "duplicate": records[0]["batch_indices"][0] = records[0]["batch_indices"][1]
    if change == "order": records[0]["batch_indices"] = records[0]["batch_indices"][::-1]
    if change == "counts": records[0]["real_scored_targets"] += 64
    if change == "step": records[1]["step"] = records[0]["step"]
    if change == "rate": records[0]["learning_rate"] = .1
    if change == "reconstruction": records[0]["gradient"]["reconstruction_relative_error"] = .01
    if change == "shadow": records[0]["mixed_shadow_exact"] = False
    if change == "rng": records[0]["rng_preserved"] = False
    if change == "effect": records[0]["effects"]["real_conditional_delta_nll"] += .01
    if change == "perplexity": records[0]["states"]["mixed"]["perplexity"] *= 2
    if change == "layers": records[0]["gradient"]["layers"] = {}
    with pytest.raises(RuntimeError): audit_records(records, labels, c, 1)


def test_absent_source_is_exact_noop_and_has_no_mean():
    labels = np.ones(400, dtype=np.int32)
    records, c = _records(labels)
    audit_records(records, labels, c, 1)
    records[0]["real_source_mean_nll"] = 0.
    with pytest.raises(RuntimeError, match="absent"):
        audit_records(records, labels, c, 1)


def test_epoch_effect_uncertainty_resamples_documents_not_updates():
    scores = {"no_update": np.full((2, 4), 192.), "real": np.full((2, 4), 190.),
              "synthetic": np.full((2, 4), 193.), "mixed": np.full((2, 4), 191.),
              "counts": np.full(4, 64), "documents": np.arange(4)}
    records = [{"real_pairs": 1, "synthetic_pairs": 127}, {"real_pairs": 0, "synthetic_pairs": 16}]
    result = effect_summary(scores, records)
    assert result["real_conditional_mean_delta_nll"] == pytest.approx(-2 / 64)
    assert result["synthetic_conditional_mean_delta_nll"] == pytest.approx(1 / 64)
    assert result["actual_mixed_mean_delta_nll"] == pytest.approx(-1 / 64)
    assert result["real_conditional_document_bootstrap_95"] == pytest.approx([-2 / 64] * 2)
    assert result["probe_documents"] == 4
    assert result["updates"] == 2
    assert result["real_present_updates"] == 1


def test_epoch_effect_sums_do_not_fake_more_real_rows():
    scores = {"no_update": np.full((2, 4), 192.), "real": np.full((2, 4), 192.),
              "synthetic": np.full((2, 4), 193.), "mixed": np.full((2, 4), 193.),
              "counts": np.full(4, 64), "documents": np.array([0, 0, 1, 1])}
    records = [{"real_pairs": 0, "synthetic_pairs": 128}, {"real_pairs": 0, "synthetic_pairs": 16}]
    result = effect_summary(scores, records)
    assert result["real_conditional_mean_when_present"] is None
    assert result["real_conditional_mean_delta_nll"] == 0
    assert result["probe_documents"] == 2


def test_single_replay_preserves_first_update_and_excludes_old_attempt(tmp_path):
    from model_collapse.common import atomic_json
    import json
    records, _ = _records(np.ones(400, dtype=np.int32), cycle=2)
    atomic_json(tmp_path / "diagnostics_manifest.json", {"attempt_id": "accepted"})
    folder = tmp_path / "update_diagnostics/attempts/old"
    folder.mkdir(parents=True)
    (folder / "updates.jsonl").write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    result = audit_replay(tmp_path, records)
    assert result["abandoned_recorded_updates"] == 1
    assert result["abandoned_attempts_excluded"] is True
    altered = copy.deepcopy(records)
    altered[0]["states"]["mixed"]["loss"] += .001
    with pytest.raises(RuntimeError, match="Replayed"):
        audit_replay(tmp_path, altered)


def test_per_layer_summary_cannot_disagree_with_actual_global_norm():
    labels = np.ones(400, dtype=np.int32)
    records, c = _records(labels)
    records[0]["gradient"]["layers"]["layers.0"]["synthetic_norm"] *= 2
    with pytest.raises(RuntimeError, match="Per-layer"):
        audit_records(records, labels, c, 1)


@pytest.mark.parametrize("fault", [None, "binding", "probe", "training_seal"])
def test_completed_stage_checks_current_binding_and_probe(tmp_path, fault):
    import torch
    from types import SimpleNamespace
    from model_collapse.common import atomic_json, digest, read_json
    from model_collapse.science.engine import shifted_nll, weights_hash
    from model_collapse.science.sgd_effects import SourceAccumulator, UpdateObserver
    from model_collapse.workflow.completion import verify_stage
    from model_collapse.workflow.selection import training_contract

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(7, 3)
            self.head = torch.nn.Linear(3, 7)

        def forward(self, input_ids, attention_mask, use_cache=False):
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    c = dict(workflow="completion_arm", optimizer="sgd", learning_rate_policy="constant", momentum=0,
             learning_rate=.001, min_learning_rate=.001, weight_decay=0, precision="fp32", epochs=1,
             seed=42, microbatch=2, effective_batch=2, checkpoint_selection="final_epoch",
             initialization="continued", prepared_bank={"manifest_sha256": "b" * 64}, data_kind="mixed",
             synthetic_fraction=.99, generation_decoder="greedy", num_beams=1, do_sample=False,
             block_size=64, new_tokens=64, repetition_penalty=1., update_diagnostics={"enabled": True},
             train_pairs_per_cycle=2)
    model = TinyLM(); initial = weights_hash(model)
    lineage = {"initializer_weights_sha256": initial, "optimizer_parent_sha256": None}
    binding = {"training_contract_sha256": training_contract(c), "input_tokens_sha256": "i" * 64,
               "initial_weights_sha256": initial, "optimizer_parent_sha256": None,
               "lineage_sha256": digest(lineage), "total_optimizer_steps": 10}
    ids = np.ones((2, 3), dtype=np.int64); attention = np.ones_like(ids)
    scoring = np.zeros_like(ids); scoring[:, 2] = 1
    observer = UpdateObserver(model, (ids, attention, scoring), root=tmp_path, cycle=1,
                              probe_identity="current-probe", eval_batch=2, binding=binding)
    accumulator = SourceAccumulator(model, 2)
    losses = shifted_nll(model(torch.tensor(ids), torch.tensor(attention)).logits,
                         torch.tensor(ids), torch.tensor(scoring)).sum(1)
    accumulator.backward(losses, torch.ones(2, dtype=torch.int64), [0, 1])
    observer.before_step(accumulator, .001, step=1, batch_indices=np.arange(2))
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, foreach=False, fused=False)
    optimizer.step(); observer.after_step()
    seal = observer.seal(weights_hash(model))
    training = {**binding, "initial_weights_sha256": initial, "selected_weights_sha256": weights_hash(model),
                "terminal_weights_sha256": weights_hash(model), "lineage": lineage,
                "checkpoint_selection": "final_epoch", "selected_epoch": 1,
                "update_diagnostics": seal}
    atomic_json(tmp_path / "training.json", training)
    atomic_json(tmp_path / "training_history.json", [{"epoch": 1}])
    atomic_json(tmp_path / "stage_completion.json", {"files": {}})
    atomic_json(tmp_path / "metrics.json", {"fixture": "unchanged"})
    expected_probe = "current-probe"
    if fault == "binding":
        altered = read_json(tmp_path / "diagnostics_manifest.json")
        altered["binding"]["input_tokens_sha256"] = "changed"
        atomic_json(tmp_path / "diagnostics_manifest.json", altered)
    elif fault == "probe":
        expected_probe = "other-probe"
    elif fault == "training_seal":
        training["update_diagnostics"]["unique_probe_evaluations"] += 1
        atomic_json(tmp_path / "training.json", training)
    if fault:
        with pytest.raises(RuntimeError, match={"binding": "binding", "probe": "probe identity",
                                               "training_seal": "training record"}[fault]):
            verify_stage(tmp_path, c, lineage, update_probe_identity=expected_probe)
    else:
        assert verify_stage(tmp_path, c, lineage, update_probe_identity=expected_probe) == {"fixture": "unchanged"}


def test_report_reads_real_observer_seals_without_changing_evidence(tmp_path):
    import torch
    from types import SimpleNamespace
    from model_collapse.analysis.sgd_effects import report
    from model_collapse.common import atomic_json, sha256
    from model_collapse.science.engine import shifted_nll, weights_hash
    from model_collapse.science.sgd_effects import SourceAccumulator, UpdateObserver
    from model_collapse.workflow.sgd_profile import ARMS, PROFILE

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(7, 3)
            self.head = torch.nn.Linear(3, 7)

        def forward(self, input_ids, attention_mask, use_cache=False):
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    root = tmp_path / "comparison"
    c = {"workflow": "completion_comparison", "completion_profile": PROFILE, "completion_arms": ARMS,
         "model": "tiny-fixture", "generations": 1, "train_pairs_per_cycle": 4,
         "learning_rate": .001, "min_learning_rate": .001, "weight_decay": 0.,
         "secondary_corpora": [], "prepared_bank": {"manifest_sha256": "b" * 64},
         "generation_decoder": "greedy", "seed": 42}
    atomic_json(root / "config.json", c)
    ids = np.ones((4, 129), dtype=np.int64); attention = np.ones_like(ids)
    scoring = np.zeros_like(ids); scoring[:, 65:] = 1
    probe = {"distinct_2": .5, "token_entropy_nats": 1., "mean_within_entropy_nats": .5,
             "mean_within_repeated_4gram_rate": .1, "repeated_4gram_rate": .2,
             "duplicate_continuation_rate": 0.}
    def metrics(weights, training=None):
        return {"weights_sha256": weights, "validation": {"loss": 2., "perplexity": math.exp(2.), "scored_tokens": 256},
                "test": {"loss": 2., "perplexity": math.exp(2.), "scored_tokens": 256,
                         "p95_example_nll": 2.1, "median_example_nll": 2.}, "probe": probe,
                "secondary": {}, "training": training or {}}
    for arm in ("real", "mixed99", "synthetic"):
        torch.manual_seed(42); model = TinyLM(); model.train()
        child = root.with_name(root.name + "-" + arm)
        base = metrics(weights_hash(model))
        atomic_json(child / "seed_42/gen_00/metrics.json", base)
        atomic_json(root / "seed_42/gen_00/metrics.json", base)
        stage = child / "seed_42/gen_01"
        observer = UpdateObserver(model, (ids, attention, scoring), root=stage, cycle=1,
                                  probe_identity="p" * 64, documents=np.arange(4), eval_batch=4)
        labels = np.zeros(4, dtype=np.int64) if arm == "real" else np.ones(4, dtype=np.int64)
        if arm == "mixed99": labels[0] = 0
        accumulator = SourceAccumulator(model, 256)
        losses = shifted_nll(model(torch.tensor(ids), torch.tensor(attention)).logits, torch.tensor(ids), torch.tensor(scoring)).sum(1)
        accumulator.backward(losses, torch.full((4,), 64), labels)
        observer.before_step(accumulator, .001, step=1, batch_indices=np.arange(4))
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=0, weight_decay=0, foreach=False, fused=False)
        optimizer.step(); observer.after_step(); observer.seal(weights_hash(model))
        training = {"selected_weights_sha256": weights_hash(model), "global_step": 1,
                    "selected_epoch": 1, "seconds": .01}
        atomic_json(stage / "training.json", training)
        atomic_json(stage / "metrics.json", metrics(weights_hash(model), training))
        atomic_json(stage / "stage_completion.json", {"files": {}})
    evidence = {str(p): sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    out = tmp_path / "rendered"
    result = report(root, out)
    assert result["updates"] == 3
    assert (out / "update_effects.csv").is_file()
    assert (out / "sgd_conditional_source_effects.png").is_file()
    assert "not independent seed replicates" in (out / "report.md").read_text(encoding="utf-8")
    assert all(sha256(path) == expected for path, expected in evidence.items())
