"""Export small, verified campaign acceptance evidence without cloud access.

Run after metadata recovery and local GPU/provider validation. The source is a
recovered campaign directory with its three sibling job directories. Model
weights, token arrays, tokenizers, secrets and operational resource registries
are never copied. The export records saved evidence, not a new GPU validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

from model_collapse.common import digest, read_json, sha256
from model_collapse.hub import verify_recovered_files
from model_collapse.mixing import mixture_counts
from model_collapse.workflow.mixed_campaign import JOB_ORDER, job_roots, report_campaign
from model_collapse.workflow.profiles import protocol_fingerprint, require_acceptance
from model_collapse.workflow.selection import verify_selection

try:
    from .publication import new_destination, write_provenance
except ImportError:  # Direct script invocation.
    from publication import new_destination, write_provenance


PARENT_FILES = {
    "config.json", "manifest.json", "completion.json", "remote_validation.json",
    "validation_results.json", "children.json", "child_snapshots.json",
    "continued20_seal.json", "shared20_mixture.json", "artifact_index.json",
    "recovery_verified.json", "detached_verification.json", "watchdog_exit.json",
}
JOB_FILES = {
    "config.json", "manifest.json", "completion.json", "environment.json",
    "benchmark.json", "shared_prefix.json", "shared_mixture_verification.json", "remote_validation.json",
    "artifact_index.json", "recovery_verified.json", "validation_g1_snapshot.json",
}
STAGE_FILES = {
    "metrics.json", "training.json", "training_history.json", "stage_completion.json",
    "test_sequences.json", "reference_sequences.json", "samples.json", "import.json",
}
LOCAL_RECEIPTS = {
    "artifact_index.json", "validation_results.json", "recovery_verified.json",
    "detached_verification.json", "watchdog_exit.json",
}


def _selected(root, parent=False):
    """An allowlist of scientific JSON, never recursive bulk/cache discovery."""
    root = Path(root)
    paths = [root / name for name in (PARENT_FILES if parent else JOB_FILES)]
    if parent:
        paths += [root / "job_seals" / f"{name}.json" for name in JOB_ORDER]
    else:
        c = read_json(root / "config.json")
        for g in range(c["generations"] + 1):
            stage = root / f"seed_{c['seed']}/gen_{g:02d}"
            paths += [stage / name for name in STAGE_FILES]
        for pattern in ("mixed_for_gen_*/manifest.json", "mixed_for_gen_*/selection.json",
                        "synthetic_for_gen_*/manifest.json"):
            paths += list(root.glob(pattern))
        paths += [root / "data/manifest.json"]
        for pattern in ("shared_source/data/manifest.json", "shared_source/config.json",
                        "shared_source/mixed_for_gen_*/manifest.json",
                        "shared_source/mixed_for_gen_*/selection.json",
                        "shared_source/synthetic_for_gen_*/manifest.json"):
            paths += list(root.glob(pattern))
    return sorted({p for p in paths if p.is_file()})


def _verified_source(root, path):
    """Honor metadata fetch's historical-control redirects, without replacing them."""
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root) or path.suffix != ".json":
        raise ValueError("Selected evidence must be JSON inside its run directory")
    rel = path.relative_to(root).as_posix()
    meta = read_json(root / "artifact_index.json", {}).get("files", {}).get(rel)
    if meta is None:
        # Local validation/recovery and volatile provider receipts are not in the
        # immutable remote index. Their current bytes are recorded as inputs.
        if rel not in LOCAL_RECEIPTS:
            raise RuntimeError(f"Scientific evidence is absent from the pinned artifact index: {rel}")
        return path, False
    verify_recovered_files(root, {rel: meta})
    if sha256(path) == meta["sha256"]:
        return path, True
    redirect = read_json(root / "recovery_snapshot.json", {}).get("preserved_control_files", {}).get(rel, {})
    target = (root / redirect.get("recovered_path", "")).resolve()
    if not target.is_relative_to(root) or not target.is_file() or sha256(target) != meta["sha256"]:
        raise RuntimeError(f"Verified historical evidence path is unavailable: {rel}")
    return target, True


def _accepted_source(source, c):
    """Bind the saved local certificate to this run, runtime and all job protocols."""
    manifest = read_json(source / "manifest.json", {})
    runtime = manifest.get("code_sha256")
    if (not isinstance(runtime, str) or len(runtime) != 64
            or manifest.get("config_hash") != digest(c)):
        raise RuntimeError("Campaign manifest does not bind its configuration and runtime")
    acceptance = read_json(source / "validation_results.json", {})
    if acceptance.get("validated_run_id") != source.name:
        raise RuntimeError("Local acceptance belongs to a different campaign run")
    if acceptance.get("provider_pod_state") != "EXITED":
        raise RuntimeError("Provider-confirmed shutdown is required")
    require_acceptance(c, acceptance, runtime)
    if any(value is not True for value in acceptance["checks"].values()):
        raise RuntimeError("A recorded acceptance check failed")
    return acceptance


def _validate(source):
    """Check recorded acceptance and six actual stages, without loading tensors."""
    c = read_json(source / "config.json", {})
    completion = read_json(source / "completion.json", {})
    if c.get("workflow") != "mixed_campaign" or c.get("kind") != "smoke":
        raise ValueError("Expected a recovered mixed_campaign smoke directory")
    if not completion.get("complete") or completion.get("unique_training_stages") != 6 or completion.get("executed_epochs") != 30:
        raise RuntimeError("Campaign must have completed six new stages / thirty epochs")
    acceptance = _accepted_source(source, c)
    roots, stages, mixtures = job_roots(source), [], []
    for name in JOB_ORDER:
        child = roots[name]
        job = read_json(child / "config.json", {})
        if not read_json(child / "completion.json", {}).get("complete"):
            raise RuntimeError(f"Campaign job is incomplete: {name}")
        if job.get("epochs") != 5 or job.get("checkpoint_selection") != "final_epoch":
            raise RuntimeError("Every new stage must execute and retain epoch 5")
        child_audit_path = child / "remote_validation.json"
        child_audit = read_json(child_audit_path, {})
        accepted_audit = acceptance["child_certificates"][name]
        child_manifest = read_json(child / "manifest.json", {})
        if (child_audit != accepted_audit
                or acceptance.get("child_audits", {}).get(name) != sha256(child_audit_path)
                or child_audit.get("source_code_sha256") != acceptance["source_code_sha256"]
                or child_audit.get("protocol_fingerprint") != protocol_fingerprint(job)
                or child_manifest.get("code_sha256") != acceptance["source_code_sha256"]
                or child_manifest.get("config_hash") != digest(job)
                or child_audit.get("artifact_recovery", {}).get("run_id") != child.name):
            raise RuntimeError("Child proof differs from the accepted run, runtime or scientific protocol")
        start = 1 if name == "continued20" else 2
        expected_counts = mixture_counts(256, job["synthetic_fraction"])
        for g in range(start, job["generations"] + 1):
            gen = child / f"seed_{job['seed']}/gen_{g:02d}"
            metric = read_json(gen / "metrics.json")
            training = read_json(gen / "training.json")
            history = read_json(gen / "training_history.json")
            if not metric or metric["training"] != training:
                raise RuntimeError("Stage training/metric evidence differs")
            verify_selection(training, history, job)
            if training.get("executed_epochs") != 5 or training.get("selected_epoch") != 5:
                raise RuntimeError("Terminal epoch-5 training evidence is missing")
            for split in ("validation", "test"):
                score = metric[split]
                if (score["scored_tokens"] != 64 * 63 or not math.isfinite(score["loss"])
                        or not math.isclose(score["loss"], score["nll_sum"] / score["scored_tokens"], abs_tol=1e-10)
                        or not math.isclose(score["perplexity"], math.exp(score["loss"]), rel_tol=1e-10)):
                    raise RuntimeError("Saved held-out NLL/PPL arithmetic or scope differs")
            mixed = read_json(child / f"mixed_for_gen_{g:02d}/manifest.json")
            selection = read_json(child / f"mixed_for_gen_{g:02d}/selection.json")
            if not mixed or not selection or (mixed["real_blocks"], mixed["synthetic_blocks"]) != expected_counts:
                raise RuntimeError("Saved mixture counts differ from the configured fraction")
            if mixed["training_tokens_sha256"] != training["input_tokens_sha256"]:
                raise RuntimeError("Mixture and actual training-token identities differ")
            real_count, synthetic_count = expected_counts
            if (len(selection["real_indices"]) != real_count or len(set(selection["real_indices"])) != real_count
                    or len(selection["synthetic_indices"]) != synthetic_count
                    or len(set(selection["synthetic_indices"])) != synthetic_count
                    or sorted(selection["permutation"]) != list(range(256))
                    or selection["source_labels"] != [int(i >= real_count) for i in selection["permutation"]]):
                raise RuntimeError("Saved source selections, labels or shuffle are inconsistent")
            stages.append({"job": name, "generation": g, "selected_epoch": 5,
                           "validation_loss": metric["validation"]["loss"],
                           "validation_perplexity": metric["validation"]["perplexity"],
                           "test_loss": metric["test"]["loss"], "test_perplexity": metric["test"]["perplexity"],
                           "initial_weights_sha256": training["initial_weights_sha256"],
                           "selected_weights_sha256": training["selected_weights_sha256"]})
            mixtures.append({"job": name, "generation": g, **{key: mixed[key] for key in
                              ("real_blocks", "synthetic_blocks", "real_tokens", "synthetic_tokens", "training_tokens_sha256")}})
    if len(stages) != 6:
        raise RuntimeError("Acceptance export requires exactly six newly trained stages")
    for rel in ("seed_42/gen_00/metrics.json", "seed_42/gen_01/metrics.json"):
        if sha256(roots["continued20"] / rel) != sha256(roots["reset20"] / rel):
            raise RuntimeError("The 20/80 shared measurements differ")
    paths = [roots[name] / "mixed_for_gen_02/selection.json" for name in ("continued20", "reset20")]
    if sha256(paths[0]) != sha256(paths[1]):
        raise RuntimeError("The 20/80 shared cycle-2 selections differ")
    hashes = [read_json(roots[name] / "mixed_for_gen_02/manifest.json")["training_tokens_sha256"]
              for name in ("continued20", "reset20")]
    if len(set(hashes)) != 1:
        raise RuntimeError("The 20/80 shared cycle-2 mixed token hashes differ")
    return acceptance, stages, mixtures


def _table(rows, columns):
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row[key]
            cells.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def export(source, out):
    source = Path(source).resolve()
    out = new_destination(source, out)
    acceptance, stages, mixtures = _validate(source)
    roots = {"campaign": source, **job_roots(source)}
    selected = []
    for name, root in roots.items():
        for path in _selected(root, parent=name == "campaign"):
            actual, indexed = _verified_source(root, path)
            selected.append((root, path.relative_to(root), actual, indexed))
    out.mkdir(parents=True, exist_ok=True)
    copied = {}
    for root, relative, actual, indexed in selected:
        dest = out / "records" / root.name / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual, dest)
        copied[dest.relative_to(out).as_posix()] = {"sha256": sha256(dest), "bytes": dest.stat().st_size,
                                                   "verified_against_remote_index": indexed}
    copied_source = out / "records" / source.name
    report_campaign(copied_source, out / "analysis")
    for filename, rows in (("stages.csv", stages), ("mixture_counts.csv", mixtures)):
        with (out / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Three-experiment mixed-data campaign: GPU acceptance", "",
             f"Campaign: `{source.name}`.", "",
             "**Pipeline validation only. This report summarizes saved smoke evidence, not a full experiment, current cloud state or a new GPU validation.**", "",
             "## Verified execution", "",
             "- One sequential campaign executed six new training stages / thirty actual epochs, retaining epoch 5 throughout.",
             "- Every job's recorded runtime matches the accepted campaign. The saved acceptance reports successful private restore/re-evaluation, interruption recovery, detached progress and provider-confirmed shutdown.",
             "- Reset50 imports historical full-data S0/S1/D2. Its active 256-block mixture uses a reduced population and is not historical full M2; full-source reconstruction is a separate recorded worker check.",
             "- Continued20 and reset20 share exact S0/S1 measurements, D2 and cycle-2 mixed token/selection identities. Their new stage initializers follow continued versus fixed-S0 policies.", "",
             "## New-stage measurements", "",
             _table(stages, [("job", "Job"), ("generation", "Cycle"), ("selected_epoch", "Selected epoch"),
                             ("validation_perplexity", "Validation PPL"), ("test_perplexity", "Test PPL")]), "",
             "These rows describe only newly executed stages on the frozen 64-block smoke evaluation sets. Imported full-source measurements retain their original scope in the per-job records and plots.", "",
             "## Training mixtures", "",
             _table(mixtures, [("job", "Job"), ("generation", "Cycle"), ("real_blocks", "Real blocks"),
                               ("synthetic_blocks", "Synthetic blocks"), ("real_tokens", "Real tokens"),
                               ("synthetic_tokens", "Synthetic tokens")]), "",
             "## Figure families", ""]
    for title, filename in (("Fixed real-text evaluation", "perplexity.png"),
                            ("Pooled diversity and repetition", "diversity.png"),
                            ("Within-output and formatting diagnostics", "probe_diagnostics.png")):
        lines.extend([f"### {title}", ""])
        for name in JOB_ORDER:
            if (out / "analysis" / name / filename).is_file():
                lines.extend([f"![{name}: {title}](analysis/{name}/{filename})", ""])
    lines.extend(["## Evidence and limitations", "",
                  "The export verifies indexed selected JSON, finite NLL/PPL arithmetic, epoch selection and saved mixture/selection consistency. It copies no model weights, token arrays, tokenizers or resource registries. Remote bulk restoration/re-evaluation is recorded evidence; this offline export does not repeat it or reconstruct absent full arrays.", "",
                  f"Recorded runtime: `{acceptance['source_code_sha256']}`. Protocol: `{acceptance['protocol_fingerprint']}`.", "",
                  "Original publications and recovered source files are unchanged. Per-job saved metric tables/figures are in `analysis/`; selected records and their hashes are preserved alongside this report. Acceptance spending and full-run affordability require the separate dated operational receipt."])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {"run_id": source.name, "runtime_sha256": acceptance["source_code_sha256"],
               "protocol_fingerprint": acceptance["protocol_fingerprint"], "new_training_stages": 6,
               "executed_epochs": 30, "provider_pod_state_recorded": acceptance["provider_pod_state"],
               "checks": acceptance["checks"], "selected_records": copied,
               "scope": "Selected metadata only; no new training, cloud reads, bulk recovery or model evaluation."}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_provenance(source.parent, out, __file__, [actual for _, _, actual, _ in selected],
                     scope="Offline metadata-only campaign acceptance export; historical publications and source records unchanged.")
    files = {p.relative_to(out).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
             for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {"schema_version": 1, "run_id": source.name, "runtime_sha256": acceptance["source_code_sha256"],
                "protocol_fingerprint": acceptance["protocol_fingerprint"], "files": files,
                "scope": "Selected JSON and generated analysis; original remote artifact indexes also list uncopied bulk files."}
    (out / "publication_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"report": str(out / "report.md"), "new_training_stages": len(stages), "executed_epochs": 30,
            "selected_json_files": len(copied), "published_files": len(files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Recovered campaign directory, with its three sibling jobs")
    parser.add_argument("--out", type=Path, required=True, help="New empty export destination; published evidence is protected")
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.out), indent=2))
