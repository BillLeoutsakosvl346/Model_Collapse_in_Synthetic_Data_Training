"""Actual-batch source gradients and isolated, observational SGD interventions.

Source components are token-weighted contributions to the ordinary batch gradient,
not separately renormalized real-only/synthetic-only objectives. No probe gradient
is computed or used to train. Temporary gradient tensors never leave memory.
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import random
import re
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from ..common import atomic_json, digest, read_json, sha256, verify_files
from .gradients import sequence_losses

POLICY = "actual_batch_weighted_sgd_effects_v1"
ACTIONS = ("no_update", "real", "synthetic", "mixed")


@contextlib.contextmanager
def preserve_rng():
    """Diagnostics do not consume any training random stream."""
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def _named(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _layer(name):
    match = re.search(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.", name)
    return "layers." + match[1] if match else name.rsplit(".", 1)[0]


def _norm_summary(real2, synthetic2, total2, dot, residual2):
    rn, sn, tn = math.sqrt(real2), math.sqrt(synthetic2), math.sqrt(total2)
    return {"real_norm": rn, "synthetic_norm": sn, "total_norm": tn,
            "real_synthetic_cosine": None if min(rn, sn) <= 1e-12 else max(-1., min(1., dot / (rn * sn))),
            "real_synthetic_dot": dot,
            "reconstruction_absolute_error": math.sqrt(residual2),
            "reconstruction_relative_error": math.sqrt(residual2) / max(tn, rn + sn, 1e-12)}


def _require_finite(gradient, message):
    values = [torch.isfinite(value).all() for value in gradient if value is not None]
    if values and not bool(torch.stack(values).all()):
        raise RuntimeError(message)


class SourceAccumulator:
    """Backward one effective batch, sharing the current training forward graph.

    ``backward`` replaces the usual scalar ``loss.backward`` call. It first
    calculates the ordinary total gradient, accumulating that into ``p.grad``.
    In a mixed microbatch two additional backward calls measure components
    directly. A one-source microbatch reuses its ordinary total gradient.
    """
    def __init__(self, model, total_targets):
        if type(total_targets) is not int or total_targets <= 0:
            raise ValueError("Effective batch needs a positive target count")
        self.model, self.named, self.total_targets = model, _named(model), total_targets
        if not self.named:
            raise ValueError("No trainable parameters")
        if any(p.grad is not None for _, p in self.named):
            raise ValueError("Clear parameter gradients before beginning an effective batch")
        self.real = {n: torch.zeros_like(p) for n, p in self.named}
        self.synthetic = {n: torch.zeros_like(p) for n, p in self.named}
        self.counts = {"real": 0, "synthetic": 0}
        self.pairs = {"real": 0, "synthetic": 0}
        self.loss_sums = {"real": 0., "synthetic": 0.}

    def backward(self, sequence_nll_sums, sequence_target_counts, source_labels):
        losses = sequence_nll_sums
        if losses.ndim != 1 or not len(losses) or not torch.isfinite(losses).all():
            raise ValueError("Per-sequence losses must be finite nonempty sums")
        counts = torch.as_tensor(sequence_target_counts, device=losses.device)
        labels = torch.as_tensor(source_labels, device=losses.device)
        if (counts.shape != losses.shape or labels.shape != losses.shape
                or torch.any(counts <= 0) or torch.any(counts != counts.long())
                or torch.any((labels != 0) & (labels != 1))):
            raise ValueError("Invalid source labels or target counts")
        unique = labels.unique().tolist()
        params = [p for _, p in self.named]
        total = torch.autograd.grad(losses.sum() / self.total_targets, params,
                                    retain_graph=len(unique) == 2, allow_unused=True)
        _require_finite(total, "Non-finite ordinary training gradient")
        for (name, parameter), value in zip(self.named, total):
            if value is not None:
                if parameter.grad is None:
                    parameter.grad = value.detach().clone()
                else:
                    parameter.grad.add_(value.detach())
        for position, label in enumerate(unique):
            source = "synthetic" if label else "real"
            rows = labels == label
            if len(unique) == 1:
                gradient = total
            else:
                gradient = torch.autograd.grad(losses[rows].sum() / self.total_targets, params,
                    retain_graph=position < len(unique) - 1, allow_unused=True)
                _require_finite(gradient, "Non-finite source gradient")
            destination = getattr(self, source)
            for (name, _), value in zip(self.named, gradient):
                if value is not None:
                    destination[name].add_(value.detach())
            self.counts[source] += int(counts[rows].sum())
            self.pairs[source] += int(rows.sum())
            self.loss_sums[source] += float(losses[rows].detach().double().sum())
        if sum(self.counts.values()) > self.total_targets:
            raise ValueError("Microbatches exceed the effective batch target count")

    def metrics(self):
        if sum(self.counts.values()) != self.total_targets:
            raise ValueError("Incomplete effective batch")
        tensors = {}
        for name, p in self.named:
            real, synthetic = self.real[name].double(), self.synthetic[name].double()
            total = torch.zeros_like(real) if p.grad is None else p.grad.detach().double()
            values = [real.square().sum(), synthetic.square().sum(), total.square().sum(),
                      (real * synthetic).sum(), (total - real - synthetic).square().sum()]
            tensors.setdefault(_layer(name), []).append(torch.stack(values))
        groups = {name: torch.stack(values).sum(0).cpu().tolist() for name, values in tensors.items()}
        combined = [math.fsum(v[i] for v in groups.values()) for i in range(5)]
        gradient = _norm_summary(*combined)
        gradient["layers"] = {name: _norm_summary(*values) for name, values in groups.items()}
        gradient["reconstruction_normalization"] = "max(total_norm, real_norm + synthetic_norm, 1e-12)"
        if not all(math.isfinite(v) for v in combined) or gradient["reconstruction_relative_error"] > 1e-5:
            raise RuntimeError("Direct real/synthetic gradients do not reconstruct the ordinary gradient")
        result = {"total_scored_targets": self.total_targets, "gradient": gradient}
        for source in ("real", "synthetic"):
            result.update({source + "_pairs": self.pairs[source],
                           source + "_scored_targets": self.counts[source],
                           source + "_source_mean_nll": self.loss_sums[source] / self.counts[source] if self.counts[source] else None,
                           source + "_weighted_training_nll": self.loss_sums[source] / self.total_targets})
        return result


def _versions(model):
    return tuple((n, p._version) for n, p in model.named_parameters()) + tuple(
        (n, p._version) for n, p in model.named_buffers())


def _verify_attempt_bindings(root, cycle, probe_identity, binding):
    """An unfinished attempt cannot be replayed under different scientific inputs."""
    attempts = Path(root) / "update_diagnostics" / "attempts"
    for attempt in attempts.iterdir() if attempts.exists() else ():
        if not attempt.is_dir():
            continue
        metadata = read_json(attempt / "attempt.json")
        if metadata is None:
            journal = attempt / "updates.jsonl"
            if journal.exists() and journal.stat().st_size:
                raise RuntimeError("Unfinished diagnostic attempt lost its scientific binding")
            continue  # Constructor interrupted before any update was recorded.
        expected = {"policy": POLICY, "cycle": cycle, "probe_identity": probe_identity,
                    "binding": binding, "attempt_id": attempt.name}
        if metadata != expected:
            raise RuntimeError("Unfinished diagnostic attempt binding changed")


def prepare_replay(root, *, cycle, probe_identity, binding, expected_steps):
    """Archive a verified seal whose training transaction never committed.

    This is called only by the observed-SGD engine after reloading the verified
    cycle initializer. A standalone observer still refuses to overwrite a seal.
    The old evidence stays in its original attempt; only its uncommitted root
    pointer moves. Completed stages and incompatible/corrupt attempts fail closed.
    """
    root = Path(root)
    if (root / "training.json").exists() or (root / "stage_completion.json").exists():
        raise RuntimeError("Completed training diagnostics cannot be replayed")
    _verify_attempt_bindings(root, cycle, probe_identity, binding)
    source = root / "diagnostics_manifest.json"
    if not source.exists():
        return None
    manifest = verify_seal(root, expected_probe_identity=probe_identity,
                           expected_steps=expected_steps, expected_binding=binding)
    if manifest["cycle"] != cycle:
        raise RuntimeError("Uncommitted diagnostic seal belongs to a different cycle")
    checksum = sha256(source)
    archived = root / "update_diagnostics" / "attempts" / manifest["attempt_id"] / "uncommitted_seal.json"
    if archived.exists():
        if sha256(archived) != checksum:
            raise RuntimeError("An archived uncommitted seal already differs")
        source.unlink()  # An exact verified archived copy already exists.
    else:
        os.replace(source, archived)
    return {"attempt_id": manifest["attempt_id"], "sha256": checksum,
            "archived_path": archived.relative_to(root).as_posix(), "updates": manifest["updates"]}


class UpdateObserver:
    """One scratch model, four logical interventions, and one cycle's journal."""
    def __init__(self, model, probe, *, root, cycle, probe_identity, documents=None,
                 eval_batch=32, verify_shadow=True, binding=None, numerical_floor=1e-6):
        self.root, self.model = Path(root), model
        if (self.root / "diagnostics_manifest.json").exists():
            raise RuntimeError("Completed diagnostics cannot be overwritten")
        self.probe = tuple(np.array(v, copy=True) for v in probe)
        if len(self.probe) != 3 or any(v.shape != self.probe[0].shape for v in self.probe):
            raise ValueError("Probe tokens and masks differ")
        self.documents = np.arange(len(self.probe[0])) if documents is None else np.asarray(documents)
        if self.documents.shape != (len(self.probe[0]),) or not probe_identity or eval_batch <= 0:
            raise ValueError("Invalid probe identity, documents or batch size")
        self.cycle, self.probe_identity, self.eval_batch = cycle, probe_identity, eval_batch
        self.verify_shadow, self.binding, self.numerical_floor = verify_shadow, binding or {}, numerical_floor
        _verify_attempt_bindings(self.root, self.cycle, self.probe_identity, self.binding)
        self.attempt_id = uuid.uuid4().hex
        self.attempt = self.root / "update_diagnostics" / "attempts" / self.attempt_id
        self.attempt.mkdir(parents=True)
        self.journal = self.attempt / "updates.jsonl"
        self.journal.touch()
        self.records, self.scores, self._pending, self._cache = [], [], None, None
        with preserve_rng():
            self.scratch = copy.deepcopy(model).eval()
            self.scratch.zero_grad(set_to_none=True)
        atomic_json(self.attempt / "attempt.json", {"policy": POLICY, "cycle": cycle,
                    "probe_identity": probe_identity, "binding": self.binding, "attempt_id": self.attempt_id})

    def _evaluate(self):
        return sequence_losses(self.scratch, *self.probe, microbatch=self.eval_batch)

    @torch.no_grad()
    def _intervene(self, gradient, learning_rate):
        self.scratch.load_state_dict(self.model.state_dict())
        if gradient is not None:
            for name, p in _named(self.scratch):
                value = gradient[name]
                if value is not None:
                    p.add_(value, alpha=-learning_rate)

    def before_step(self, accumulator, learning_rate, step, batch_indices):
        if self._pending is not None or accumulator.model is not self.model:
            raise RuntimeError("Observer has an unfinished or foreign update")
        if not math.isfinite(learning_rate) or learning_rate <= 0 or type(step) is not int or step < 1:
            raise ValueError("Invalid SGD update identity")
        if self.records and step != self.records[-1]["step"] + 1:
            raise RuntimeError("Update journal is not contiguous")
        start = time.perf_counter()
        mode = self.model.training
        metrics = accumulator.metrics()
        batch_indices = np.asarray(batch_indices)
        if batch_indices.ndim != 1 or len(batch_indices) != sum(accumulator.pairs.values()):
            raise ValueError("Batch indices do not match source rows")
        total = {n: p.grad for n, p in accumulator.named}
        scores, measurement_ids, evaluated = {}, {}, 0
        versions = _versions(self.model)
        with preserve_rng():
            self._intervene(None, learning_rate)
            if self._cache is not None and self._cache["step"] == step - 1 and self._cache["versions"] == versions:
                scores["no_update"], counts = self._cache["values"], self._cache["counts"]
                measurement_ids["no_update"] = self._cache["measurement_id"]
            else:
                scores["no_update"], counts = self._evaluate()
                measurement_ids["no_update"] = f"{self.attempt_id}:{step}:no_update"
                evaluated += 1
            for action in ("real", "synthetic"):
                if not accumulator.counts[action]:
                    scores[action] = scores["no_update"]
                    measurement_ids[action] = measurement_ids["no_update"]
                elif not accumulator.counts["synthetic" if action == "real" else "real"]:
                    # The active endpoint component is exactly the ordinary total gradient.
                    continue
                else:
                    self._intervene(getattr(accumulator, action), learning_rate)
                    scores[action], actual_counts = self._evaluate()
                    if not np.array_equal(counts, actual_counts):
                        raise RuntimeError("Shadow target counts changed")
                    measurement_ids[action] = f"{self.attempt_id}:{step}:{action}"
                    evaluated += 1
            self._intervene(total, learning_rate)
            scores["mixed"], actual_counts = self._evaluate()
            if not np.array_equal(counts, actual_counts):
                raise RuntimeError("Actual-update probe target counts changed")
            measurement_ids["mixed"] = f"{self.attempt_id}:{step}:mixed"
            evaluated += 1
            for action in ("real", "synthetic"):
                if action not in scores:
                    scores[action] = scores["mixed"]
                    measurement_ids[action] = measurement_ids["mixed"]
            actual_parameters = dict(self.model.named_parameters())
            displacement2 = float(torch.stack([(p.detach().double() - actual_parameters[n].detach().double()).square().sum()
                                               for n, p in _named(self.scratch)]).sum())
        if self.model.training != mode or _versions(self.model) != versions:
            raise RuntimeError("Observational diagnostics changed the training model")
        means = {a: float(scores[a].sum() / counts.sum()) for a in ACTIONS}
        effects = {"real_only_delta_nll": means["real"] - means["no_update"],
                   "synthetic_only_delta_nll": means["synthetic"] - means["no_update"],
                   "mixed_delta_nll": means["mixed"] - means["no_update"],
                   "real_conditional_delta_nll": means["mixed"] - means["synthetic"],
                   "synthetic_conditional_delta_nll": means["mixed"] - means["real"],
                   "interaction_nll": means["mixed"] - means["real"] - means["synthetic"] + means["no_update"]}
        exact = {"real_only_delta_nll": measurement_ids["real"] == measurement_ids["no_update"],
                 "synthetic_only_delta_nll": measurement_ids["synthetic"] == measurement_ids["no_update"],
                 "mixed_delta_nll": False,
                 "real_conditional_delta_nll": measurement_ids["mixed"] == measurement_ids["synthetic"],
                 "synthetic_conditional_delta_nll": measurement_ids["mixed"] == measurement_ids["real"],
                 "interaction_nll": not all(accumulator.counts.values())}
        for key in effects:
            if exact[key]:
                effects[key] = 0.
        record = {"policy": POLICY, "attempt_id": self.attempt_id, "cycle": self.cycle,
                  "step": step, "learning_rate": learning_rate, "batch_indices": batch_indices.tolist(),
                  "batch_indices_sha256": digest(batch_indices.tolist()), **metrics,
                  "states": {a: {"loss": means[a], "perplexity": math.exp(means[a]),
                                 "measurement_id": measurement_ids[a]} for a in ACTIONS},
                  "effects": effects, "unique_probe_evaluations": evaluated,
                  "numerical_floor_nll": self.numerical_floor,
                  "numerically_unresolved": {k: not exact[k] and abs(v) <= self.numerical_floor for k, v in effects.items()},
                  "effect_status": {k: "exact_shared_measurement" if exact[k] else
                                    "numerically_unresolved" if abs(v) <= self.numerical_floor else "resolved"
                                    for k, v in effects.items()},
                  "mode_preserved": True, "rng_preserved": True,
                  "update_l2_norm": math.sqrt(displacement2),
                  "diagnostic_seconds": time.perf_counter() - start}
        self._pending = {"record": record, "scores": scores, "counts": counts,
                         "mode": mode, "before_versions": versions}
        return record

    def after_step(self, step_seconds=None):
        finish_start = time.perf_counter()
        if self._pending is None:
            raise RuntimeError("No pending SGD intervention")
        if step_seconds is not None and (not math.isfinite(step_seconds) or step_seconds < 0):
            raise ValueError("Invalid effective-update duration")
        pending, record = self._pending, self._pending["record"]
        if self.model.training != pending["mode"]:
            raise RuntimeError("SGD update changed the model mode")
        if self.verify_shadow:
            actual = dict(self.model.named_parameters())
            differences = [(p.detach() - actual[n].detach()).abs().max() for n, p in _named(self.scratch)]
            error = float(torch.stack(differences).max())
            if error != 0.:
                raise RuntimeError("Mixed shadow does not equal the actual ordinary SGD update")
            record.update(mixed_shadow_exact=True, mixed_shadow_max_absolute_error=error)
        else:
            record.update(mixed_shadow_exact=None, mixed_shadow_max_absolute_error=None)
        record["after_step_seconds"] = time.perf_counter() - finish_start
        record["step_seconds"] = None if step_seconds is None else step_seconds + record["after_step_seconds"]
        self.records.append(record)
        self.scores.append({a: np.array(pending["scores"][a], copy=True) for a in ACTIONS})
        self._counts = np.array(pending["counts"], copy=True)
        self._cache = {"step": record["step"], "versions": _versions(self.model),
                       "values": self.scores[-1]["mixed"], "counts": self._counts,
                       "measurement_id": record["states"]["mixed"]["measurement_id"]}
        with self.journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        self._pending = None
        return record

    def seal(self, terminal_weights_sha256):
        if self._pending is not None or not self.records or len(terminal_weights_sha256) != 64:
            raise RuntimeError("Cannot seal incomplete update diagnostics")
        destination = self.root / "diagnostics_manifest.json"
        if destination.exists():
            raise RuntimeError("Completed diagnostics cannot be replaced")
        arrays = self.attempt / "probe_scores.npz"
        with arrays.with_suffix(".partial").open("wb") as handle:
            np.savez_compressed(handle, **{a: np.stack([r[a] for r in self.scores]) for a in ACTIONS},
                counts=self._counts, documents=self.documents,
                steps=np.asarray([r["step"] for r in self.records], dtype=np.int64))
        os.replace(arrays.with_suffix(".partial"), arrays)
        manifest = {"policy": POLICY, "attempt_id": self.attempt_id, "cycle": self.cycle,
                    "probe_identity": self.probe_identity, "probe_pairs": len(self.documents),
                    "binding": self.binding, "terminal_weights_sha256": terminal_weights_sha256,
                    "updates": len(self.records), "first_step": self.records[0]["step"],
                    "last_step": self.records[-1]["step"], "score_kind": "per_example_nll_sums",
                    "records_path": self.journal.relative_to(self.root).as_posix(),
                    "scores_path": arrays.relative_to(self.root).as_posix(),
                    "unique_probe_evaluations": sum(r["unique_probe_evaluations"] for r in self.records),
                    "files": {p.relative_to(self.root).as_posix(): sha256(p)
                              for p in (self.journal, arrays, self.attempt / "attempt.json")}}
        atomic_json(destination, manifest)
        verify_seal(self.root, expected_weights=terminal_weights_sha256,
                    expected_probe_identity=self.probe_identity, expected_steps=len(self.records))
        return manifest


def verify_seal(root, *, expected_weights=None, expected_probe_identity=None, expected_steps=None, expected_binding=None):
    root = Path(root)
    manifest = read_json(root / "diagnostics_manifest.json")
    if not manifest or manifest.get("policy") != POLICY:
        raise RuntimeError("Completed update diagnostic seal missing or incompatible")
    if expected_weights is not None and manifest["terminal_weights_sha256"] != expected_weights:
        raise RuntimeError("Diagnostics belong to different terminal weights")
    if expected_probe_identity is not None and manifest["probe_identity"] != expected_probe_identity:
        raise RuntimeError("Diagnostic probe identity changed")
    if expected_binding is not None and manifest.get("binding") != expected_binding:
        raise RuntimeError("Diagnostic scientific binding changed")
    if isinstance(expected_steps, int) and manifest["updates"] != expected_steps:
        raise RuntimeError("Diagnostic update count changed")
    prefix = "update_diagnostics/attempts/" + manifest["attempt_id"] + "/"
    expected_files = {prefix + n for n in ("updates.jsonl", "probe_scores.npz", "attempt.json")}
    if set(manifest["files"]) != expected_files or any(".." in Path(p).parts for p in manifest["files"]):
        raise RuntimeError("Diagnostic seal references unexpected files")
    verify_files(root, manifest["files"])
    attempt = root / prefix
    metadata = read_json(attempt / "attempt.json")
    if metadata != {"policy": POLICY, "cycle": manifest["cycle"], "probe_identity": manifest["probe_identity"],
                    "binding": manifest["binding"], "attempt_id": manifest["attempt_id"]}:
        raise RuntimeError("Diagnostic attempt metadata differs from its seal")
    records = [json.loads(line) for line in (attempt / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
    steps = [r["step"] for r in records]
    if (len(records) != manifest["updates"] or not steps or
            steps != list(range(manifest["first_step"], manifest["last_step"] + 1)) or
            any(r["attempt_id"] != manifest["attempt_id"] or r["cycle"] != manifest["cycle"] for r in records)):
        raise RuntimeError("Completed diagnostic journal has missing/foreign updates")
    if expected_steps is not None and not isinstance(expected_steps, int) and steps != list(expected_steps):
        raise RuntimeError("Diagnostic global steps changed")
    if manifest.get("records_path") != prefix + "updates.jsonl" or manifest.get("scores_path") != prefix + "probe_scores.npz":
        raise RuntimeError("Diagnostic record/score paths differ from sealed attempt")
    with np.load(attempt / "probe_scores.npz", allow_pickle=False) as arrays:
        shape = (manifest["updates"], manifest["probe_pairs"])
        if (any(arrays[a].shape != shape or not np.isfinite(arrays[a]).all() for a in ACTIONS)
                or arrays["counts"].shape != (shape[1],) or np.any(arrays["counts"] <= 0)
                or arrays["documents"].shape != (shape[1],) or arrays["steps"].tolist() != steps):
            raise RuntimeError("Completed probe-score arrays differ from the journal")
        for i, row in enumerate(records):
            for action in ACTIONS:
                mean = float(arrays[action][i].sum() / arrays["counts"].sum())
                if abs(mean - row["states"][action]["loss"]) > 1e-12:
                    raise RuntimeError("Recorded probe loss differs from per-example arrays")
    return manifest
