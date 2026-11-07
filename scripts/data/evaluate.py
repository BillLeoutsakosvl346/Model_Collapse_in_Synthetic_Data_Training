"""Evaluation-only cross-corpus worker. Reuses two existing FineWeb checkpoints."""
from __future__ import annotations

import argparse
import gc
import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from model_collapse.common import atomic_json, event, now, read_json, sha256, verify_files


def load_inputs(root, spec):
    result = {}
    verify_files(root, spec["input_files"])
    for name, info in spec["collections"].items():
        tokens = np.load(root / f"inputs/{name}/tokens.npy", allow_pickle=False)
        masks = np.load(root / f"inputs/{name}/masks.npy", allow_pickle=False)
        if (tokens.shape != masks.shape or tokens.ndim != 2 or tokens.shape[1] != 512
                or np.any((masks != 0) & (masks != 1)) or np.any(masks[:, 1:] > masks[:, :-1])
                or int(masks[:, 1:].sum()) != info["scored_tokens"]):
            raise RuntimeError("Cross-corpus input masks/counts changed")
        result[name] = tokens, masks
    return result


def comparison_rows(evaluations):
    original = evaluations["pretrained"]
    adapted = evaluations["fineweb_epoch1"]
    rows = []
    for corpus in original:
        before, after = original[corpus], adapted[corpus]
        if before["scored_tokens"] != after["scored_tokens"]:
            raise RuntimeError("Models were evaluated on different target counts")
        rows.append({"corpus": corpus, "scored_tokens": before["scored_tokens"],
                     "pretrained_perplexity": before["perplexity"], "after_one_epoch_perplexity": after["perplexity"],
                     "perplexity_change_percent": 100 * (after["perplexity"] / before["perplexity"] - 1),
                     "nll_change": after["loss"] - before["loss"]})
    return rows


def report(root, output=None):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    destination = Path(output) if output else root / "analysis"
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("Choose a new report directory")
    destination.mkdir(parents=True, exist_ok=True)
    evaluations = read_json(root / "evaluations.json")
    rows = comparison_rows(evaluations)
    atomic_json(destination / "comparison.json", rows)
    with (destination / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    names = [r["corpus"].replace("_", " ") for r in rows]
    values = [r["perplexity_change_percent"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_box_aspect(1)
    bars = ax.bar(names, values, color=["#b44d45" if x > 0 else "#287f9b" for x in values])
    ax.axhline(0, color="#333", linewidth=1)
    ax.bar_label(bars, labels=[f"{x:+.2f}%" for x in values], padding=5)
    ax.margins(y=.2)
    ax.set_ylabel("Held-out perplexity change (%)")
    ax.set_title("After one FineWeb-Edu epoch\nNegative = improved, positive = worsened")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(destination / "perplexity_change.png", dpi=180)
    plt.close(fig)
    inputs = {"evaluations.json": sha256(root / "evaluations.json"), "experiment.json": sha256(root / "experiment.json")}
    atomic_json(destination / "analysis_manifest.json", {"inputs": inputs, "script_sha256": sha256(__file__),
                "outputs": {p.name: sha256(p) for p in destination.iterdir() if p.is_file()},
                "matplotlib": matplotlib.__version__})
    return rows


def backup_analysis(root, repos):
    """Explicit dataset-only allowlist, with no model/cache or credential upload."""
    import os
    from huggingface_hub import HfApi, get_token
    from model_collapse.hub import artifact_files, retry_transfer
    files = artifact_files(root)
    if any(meta["repo_type"] != "dataset" for meta in files.values()):
        raise RuntimeError("Evaluation roots must not contain downloaded model checkpoints")
    atomic_json(root / "artifact_index.json", {"run_id": root.name, "repos": repos, "files": files})
    allow = [*files, "artifact_index.json", "status.json", "events.jsonl"]
    retry_transfer(HfApi(token=os.environ.get("HF_TOKEN") or get_token()).upload_folder,
                   repo_id=repos["dataset"], repo_type="dataset", folder_path=root, path_in_repo=root.name,
                   allow_patterns=allow, commit_message=f"Save cross-corpus evaluation {root.name}")


def run(root):
    import torch
    from model_collapse import engine
    from model_collapse.hub import download_snapshot_file
    root = Path(root)
    spec = read_json(root / "experiment.json")
    r = read_json(root / "resources.json")
    if sha256(root / "evaluator.py") != spec["evaluator_sha256"]:
        raise RuntimeError("Evaluation script identity differs")
    c = spec["model_config"]
    active = threading.Event()
    def heartbeat():
        import os
        while not active.is_set():
            atomic_json(root / "supervisor.json", {"pid": os.getpid(), "time": now(), "stage": "evaluation"})
            active.wait(3)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        inputs = load_inputs(root, spec)
        if sha256(root / "checkpoint_index.json") != spec["checkpoint_index_sha256"]:
            raise RuntimeError("Pinned checkpoint index differs")
        index = read_json(root / "checkpoint_index.json")
        evaluations, checks = {}, {}
        for label, generation in (("pretrained", 0), ("fineweb_epoch1", 1)):
            atomic_json(root / "status.json", {"state": "running", "checkpoint": label, "time": now()})
            relative = f"seed_42/gen_{generation:02d}"
            cache = root.parent / "evaluation_models" / root.name / label
            cache.mkdir(parents=True, exist_ok=True)
            for name, meta in index["files"].items():
                if not name.startswith(relative + "/checkpoint/"):
                    continue
                src = download_snapshot_file(spec["source_run_id"], spec["source_repos"], name, meta, spec["source_revisions"])
                destination = cache / Path(name).name
                import shutil
                shutil.copyfile(src, destination)
            model = engine.load_model(c, cache)
            weights = engine.weights_hash(model)
            if weights != spec["expected_weights"][label]:
                raise RuntimeError("Restored checkpoint tensor hash differs")
            evaluations[label] = {}
            for corpus, (tokens, masks) in inputs.items():
                value = engine.evaluate(model, tokens, c, masks=masks)
                if corpus == "fineweb_edu":
                    expected = spec["original_fineweb_metrics"][label]["test"]["loss"]
                    delta = abs(value["loss"] - expected)
                    same_gpu = torch.cuda.get_device_name() == spec["source_gpu"]
                    checks[label] = {"tensor_hash": weights, "fineweb_test_loss_error": delta,
                                     "historical_fineweb_loss": expected, "current_fineweb_loss": value["loss"],
                                     "same_gpu_as_historical_run": same_gpu,
                                     "historical_loss_within_tolerance": delta <= spec["fineweb_reproduction_nll_tolerance"]}
                    atomic_json(root / "checks.json", checks)
                    if same_gpu and delta > spec["fineweb_reproduction_nll_tolerance"]:
                        raise RuntimeError("Original FineWeb test loss failed reproduction")
                evaluations[label][corpus] = value
                event(root, "corpus_evaluated", checkpoint=label, corpus=corpus, **value)
                atomic_json(root / "evaluations.json", evaluations)
            ids, attention = engine.tensor(tokens[:2]), engine.tensor(masks[:2])
            with torch.inference_mode(), engine.autocast(c):
                logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
            direct = -torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1) * attention[:, 1:]
            error = float((direct - engine.shifted_nll(logits, ids, attention)).abs().max())
            if error > 2e-6:
                raise RuntimeError("Masked next-token NLL calculation differs")
            checks[label]["direct_nll_max_error"] = error
            del model, logits, direct
            gc.collect()
            torch.cuda.empty_cache()
        rows = report(root)
        atomic_json(root / "checks.json", checks)
        atomic_json(root / "completion.json", {"complete": True, "training_updates_executed": 0,
                    "models_evaluated": 2, "test_corpora": len(rows), "time": now()})
        atomic_json(root / "status.json", {"state": "complete", "time": now()})
        backup_analysis(root, r["repos"])
        atomic_json(root / "stop_request.json", {"reason": "complete", "time": now()})
        while True:
            time.sleep(3)
    except BaseException as error:
        import os
        import traceback
        details = traceback.format_exc()
        for secret in (os.environ.get("HF_TOKEN"), os.environ.get("RUNPOD_API_KEY")):
            if secret:
                details = details.replace(secret, "[REDACTED]")
        atomic_json(root / "failure.json", {"traceback": details, "time": now()})
        atomic_json(root / "completion.json", {"complete": False, "error": str(error), "time": now()})
        atomic_json(root / "status.json", {"state": "failed", "error": str(error), "time": now()})
        try:
            backup_analysis(root, r["repos"])
        except Exception:
            pass
        atomic_json(root / "stop_request.json", {"reason": "failure", "time": now()})
        raise
    finally:
        active.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.report_only:
        print(json.dumps(report(args.root, args.out), indent=2))
    else:
        run(args.root)


if __name__ == "__main__":
    main()
