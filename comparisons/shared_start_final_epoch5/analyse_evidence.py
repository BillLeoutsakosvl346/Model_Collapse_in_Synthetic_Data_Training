"""Selectively recover and audit Week 3 without model loading or cloud mutation.

Run from the repository root with its environment: python
comparisons/shared_start_final_epoch5/analyse_evidence.py.  Immutable Hub commits
and indexes are taken from the already verified completion snapshot.  Only
scientific JSON records and twenty unique 256x64 int32 probe arrays are read;
full token datasets and model weights remain in private storage.
"""
from __future__ import annotations

import csv
import argparse
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from huggingface_hub import hf_hub_download, get_token
from tokenizers import Tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from model_collapse.common import PROJECT, atomic_json, load_secrets, read_json, sha256
from model_collapse.hub import download_snapshot_file, retry_transfer
from model_collapse.science.metrics import diversity

PAIR = "20261030T180755Z-full-paired-final5"
LOCAL = PROJECT / "runs" / PAIR
ROOT = PROJECT / "reports" / PAIR
RECORDS = ROOT / "records"
OUT = ROOT / "analysis" / "audit"


def copy_verified(src, target, meta):
    assert src.stat().st_size == meta["bytes"] and sha256(src) == meta["sha256"], str(src)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha256(target) == meta["sha256"], f"Refuse to replace changed evidence: {target}"
    else:
        shutil.copyfile(src, target)


def selected(rel):
    if rel in {"config.json", "manifest.json", "completion.json", "environment.json", "data/manifest.json",
               "initial_dataset_import.json", "shared_prefix.json", "benchmark.json", "requirements-tested.txt"}:
        return True
    if rel.startswith("shared_source/") and rel.endswith(".json"):
        return True
    if rel.startswith("synthetic_for_gen_") and rel.endswith("/manifest.json"):
        return True
    return rel.startswith("seed_42/") and Path(rel).name in {
        "metrics.json", "training.json", "training_history.json", "stage_completion.json", "import.json",
        "test_sequences.json", "reference_sequences.json", "samples.json", "config.json"} and "/checkpoint/" not in rel


def recover():
    load_secrets()
    pins = read_json(LOCAL / "completion_snapshot_pins.json")
    parent_dir = LOCAL / "completion_evidence" / "parent"
    child_snapshots = read_json(parent_dir / "child_snapshots.json")
    inputs, tasks, indexes = {}, [], {}
    parents = pins["index"]
    parent_index = Path(retry_transfer(hf_hub_download, parents["repos"]["dataset"],
                        f"{PAIR}/artifact_index.json", repo_type="dataset", revision=pins["revisions"]["dataset"], token=get_token()))
    assert sha256(parent_index) == pins["artifact_index_sha256"]
    assert read_json(parent_index) == parents
    copy_verified(parent_index, RECORDS/"parent"/"artifact_index.json", {"sha256":sha256(parent_index),"bytes":parent_index.stat().st_size})
    inputs["parent/artifact_index.json"]={"sha256":sha256(parent_index),"bytes":parent_index.stat().st_size}
    assert sha256(parent_dir / "child_snapshots.json") == parents["files"]["child_snapshots.json"]["sha256"]
    for arm in ("continued", "reset"):
        identity = child_snapshots[arm]
        evidence = LOCAL / "completion_evidence" / arm
        indexpath = evidence / "artifact_index.json"
        assert sha256(indexpath) == identity["artifact_index_sha256"]
        index = read_json(indexpath)
        assert index["run_id"] == identity["run_id"]
        indexes[arm] = index
        target = RECORDS / arm / "artifact_index.json"
        copy_verified(indexpath, target, {"bytes": indexpath.stat().st_size, "sha256": identity["artifact_index_sha256"]})
        inputs[f"{arm}/artifact_index.json"] = {"bytes": target.stat().st_size, "sha256": sha256(target), "immutable_identity": identity}
        for rel, meta in index["files"].items():
            if selected(rel):
                tasks.append((arm, identity, evidence, rel, meta))
    # Parent control snapshots are archived records, never current resource state.
    for rel in ("config.json", "manifest.json", "completion.json", "children.json", "child_snapshots.json", "continued_seal.json"):
        meta = parents["files"][rel]
        src = parent_dir / rel
        if not src.exists():
            src = download_snapshot_file(PAIR, parents["repos"], rel, meta, pins["revisions"])
        target = RECORDS / "parent" / rel
        copy_verified(src, target, meta)
        inputs[f"parent/{rel}"] = meta

    def transfer(task):
        arm, identity, evidence, rel, meta = task
        dst = RECORDS / arm / rel
        candidates = (dst, evidence / rel, PROJECT / "runs" / identity["run_id"] / rel)
        src = next((p for p in candidates if p.is_file() and p.stat().st_size == meta["bytes"] and sha256(p) == meta["sha256"]), None)
        if src is None:
            src = download_snapshot_file(identity["run_id"], identity["repos"], rel, meta, identity["revisions"])
        copy_verified(src, dst, meta)
        return f"{arm}/{rel}", meta
    with ThreadPoolExecutor(max_workers=8) as executor:
        for key, meta in executor.map(transfer, tasks):
            inputs[key] = meta
    # One small tokenizer document verifies the textual meaning of probe IDs.
    rel="seed_42/gen_00/tokenizer/tokenizer.json"
    key,meta=transfer(("continued",child_snapshots["continued"],LOCAL/"completion_evidence/continued",rel,indexes["continued"]["files"][rel]))
    inputs[key]=meta
    # G0 uses the exact historical D1; no duplicate full training array is fetched.
    initial = read_json(RECORDS / "continued" / "config.json")["initial_dataset"]
    oldidxfile = PROJECT / "reports" / initial["run_id"] / "artifact_index.json"
    if not oldidxfile.exists() or sha256(oldidxfile) != initial["artifact_index_sha256"]:
        oldidxfile = Path(retry_transfer(hf_hub_download, initial["repos"]["dataset"],
                         f"{initial['run_id']}/artifact_index.json", repo_type="dataset",
                         revision=initial["revisions"]["dataset"], token=get_token()))
    assert sha256(oldidxfile) == initial["artifact_index_sha256"]
    oldindex = read_json(oldidxfile)
    oldtarget = RECORDS / "source_d1" / "artifact_index.json"
    copy_verified(oldidxfile, oldtarget, {"sha256": initial["artifact_index_sha256"], "bytes": oldidxfile.stat().st_size})
    inputs["source_d1/artifact_index.json"] = {"sha256": sha256(oldtarget), "bytes": oldtarget.stat().st_size, "immutable_identity": initial}
    for rel in ("config.json", "data/manifest.json", "synthetic_for_gen_01/manifest.json", "seed_42/gen_00/training.json", "seed_42/gen_00/samples.json",
                "seed_42/gen_01/training_history.json", "seed_42/gen_01/training.json", "seed_42/gen_01/metrics.json"):
        meta=oldindex["files"][rel]
        dst=RECORDS/"source_d1"/rel
        candidates=(dst, PROJECT/"reports"/initial["run_id"]/"records"/rel, PROJECT/"runs"/initial["run_id"]/rel)
        src=next((p for p in candidates if p.is_file() and sha256(p)==meta["sha256"]),None)
        if src is None:
            src=download_snapshot_file(initial["run_id"],initial["repos"],rel,meta,initial["revisions"])
        copy_verified(src,dst,meta)
        inputs[f"source_d1/{rel}"]={**meta,"source_run_id":initial["run_id"],"revisions":initial["revisions"]}
    for arm in ("continued", "reset"):
        for g in range(11):
            if arm == "reset" and g < 2:
                continue
            identity = initial if g == 0 else child_snapshots[arm]
            index = oldindex if g == 0 else indexes[arm]
            rel = f"synthetic_for_gen_{g+1:02d}/shard_0000000.npy"
            meta = index["files"][rel]
            target = RECORDS / "probes" / arm / f"gen_{g:02d}.npy"
            candidates = [target, PROJECT / "runs" / identity["run_id"] / "probe_tokens" / f"gen_{g:02d}.npy"]
            src = next((p for p in candidates if p.is_file() and sha256(p) == meta["sha256"]), None)
            if src is None:
                src = download_snapshot_file(identity["run_id"], identity["repos"], rel, meta, identity["revisions"])
            copy_verified(src, target, meta)
            key = target.relative_to(RECORDS).as_posix()
            inputs[key] = {**meta, "source_run_id": identity["run_id"], "source_relative_path": rel,
                           "revisions": identity["revisions"]}
    assert len(list((RECORDS / "probes").rglob("*.npy"))) == 20
    receipt = {"pair_id": PAIR, "scope": "Selective scientific metadata plus twenty 256x64 probes; no weights, logits or full token arrays",
               "files": inputs, "verified_files": len(inputs), "verified_bytes": sum(v["bytes"] for v in inputs.values()),
               "probe_arrays": 20, "probe_bytes": sum(v["bytes"] for k, v in inputs.items() if k.startswith("probes/"))}
    atomic_json(RECORDS / "recovery_manifest.json", receipt)
    return receipt, indexes


def local_diversity(values):
    """Separate corpus vocabulary variety from loops inside individual outputs."""
    row_entropies, row_repeat4, row_distinct = [], [], {n: [] for n in (1, 2, 3, 4)}
    for row in values.tolist():
        counts = np.asarray(list(Counter(row).values()), dtype=float)
        p = counts / counts.sum()
        row_entropies.append(float(-(p * np.log(p)).sum()))
        for n in row_distinct:
            grams = [tuple(row[i:i+n]) for i in range(len(row)-n+1)]
            row_distinct[n].append(len(set(grams))/len(grams))
        row_repeat4.append(1-row_distinct[4][-1])
    return {"mean_within_entropy_nats": float(np.mean(row_entropies)),
            "mean_within_repeated_4gram_rate": float(np.mean(row_repeat4)),
            "fraction_outputs_with_repeated_4gram": float(np.mean(np.asarray(row_repeat4)>0)),
            "corpus_distinct_4": 1-diversity(values)["repeated_4gram_rate"],
            **{f"mean_within_distinct_{n}": float(np.mean(v)) for n, v in row_distinct.items()}}


def newline_sensitivity(values, reference_nll):
    """Post-hoc diagnostic only: exclude newlines without joining broken n-grams."""
    newline=(values==50118)
    selected=values[~newline]
    counts=np.asarray(list(Counter(selected.tolist()).values()),dtype=float)
    p=counts/counts.sum()
    grams=Counter(tuple(row[i:i+4]) for row in values.tolist() for i in range(61) if 50118 not in row[i:i+4])
    positions=sum(grams.values())
    return {"newline_token_id":50118,"newline_tokens":int(newline.sum()),"total_tokens":int(values.size),
            "newline_share":float(newline.mean()),"outputs_with_at_least_32_newlines":int((newline.sum(axis=1)>=32).sum()),
            "all_newline_outputs":int((newline.sum(axis=1)==64).sum()),
            "pooled_entropy_excluding_newline_nats":float(-(p*np.log(p)).sum()),
            "original_4gram_positions_without_newline":positions,
            "pooled_repeated_4gram_rate_without_newline":(positions-len(grams))/positions,
            "frozen_s0_score_subgroups_posthoc":{label:{"outputs":int(mask.sum()),
                "mean_nll":float(reference_nll[mask].mean()) if mask.any() else None}
                for label,mask in (("all_outputs",np.ones(len(values),dtype=bool)),
                                   ("all_newline",newline.sum(axis=1)==64),
                                   ("at_least_32_newlines",newline.sum(axis=1)>=32),
                                   ("fewer_than_32_newlines",newline.sum(axis=1)<32))}}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def audit(receipt, indexes):
    rows, histories, stage_checks, raw_metrics = [], [], [], {}
    tokenizer=Tokenizer.from_file(str(RECORDS/"continued/seed_42/gen_00/tokenizer/tokenizer.json"))
    assert tokenizer.decode([50118],skip_special_tokens=False)=="\n"
    prompt_texts = None
    config0, env0, data0 = None, None, None
    for arm in ("continued", "reset"):
        base = RECORDS / arm
        get = lambda rel: read_json(base / rel)
        c, env, data = get("config.json"), get("environment.json"), get("data/manifest.json")
        assert get("completion.json")["complete"]
        assert c["epochs"] == 5 and c["checkpoint_selection"] == "final_epoch"
        assert c["kind"] == "full" and c["generations"] == 10
        raw_metrics[arm] = []
        if arm == "continued":
            config0, env0, data0 = c, env, data
        else:
            for k in ("seed", "model_revision", "dataset_revision", "block_size", "new_tokens", "num_beams", "repetition_penalty", "epochs", "checkpoint_selection",
                      "learning_rate", "min_learning_rate", "weight_decay", "microbatch", "effective_batch", "generation_batch", "eval_batch", "precision", "probe_count", "start_checkpoint"):
                assert c[k] == config0[k], k
            assert data == data0 and env == env0
        for split, count in (("train", 37946), ("validation", 3922), ("test", 4494)):
            assert data["stats"][split]["used_blocks"] == count
            assert indexes[arm]["files"][f"data/{split}.npy"] == indexes["continued"]["files"][f"data/{split}.npy"]
            assert data["files"][f"{split}.npy"] == indexes[arm]["files"][f"data/{split}.npy"]["sha256"]
        for g in range(11):
            rel = f"seed_42/gen_{g:02d}"
            m, t = get(rel+"/metrics.json"), get(rel+"/training.json")
            seq, reference = get(rel+"/test_sequences.json"), get(rel+"/reference_sequences.json")
            samples = get(rel+"/samples.json")
            raw_metrics[arm].append(m)
            assert t == m["training"]
            prompts = [s["prompt"] for s in samples]
            if prompt_texts is None: prompt_texts = prompts
            assert prompts == prompt_texts and len(samples) == 256
            if arm == "reset" and g < 2:
                assert m == raw_metrics["continued"][g]
                assert get(rel+"/import.json")
            probe_arm = "continued" if g < 2 else arm
            values = np.load(RECORDS/"probes"/probe_arm/f"gen_{g:02d}.npy", allow_pickle=False)
            assert values.shape == (256, 64) and values.dtype == np.int32
            assert all(PreTrainedTokenizerBase.clean_up_tokenization(tokenizer.decode(row.tolist(),skip_special_tokens=False))==s["continuation"] for row,s in zip(values,samples)), "Saved decoded examples do not match probe IDs"
            measured = diversity(values)
            assert all(math.isclose(measured[k], v, abs_tol=1e-12) for k, v in m["diversity"].items())
            block_nll = np.asarray(seq["sequence_mean_nll"])
            reference_nll = np.asarray(reference["sequence_mean_nll"])
            assert len(block_nll) == 4494 and len(reference_nll) == 256
            assert np.isfinite(block_nll).all() and np.isfinite(reference_nll).all()
            assert math.isclose(float(block_nll.mean()), m["test"]["loss"], abs_tol=1e-12)
            assert math.isclose(float(reference_nll.mean()), m["frozen_g0_reference"]["loss"], abs_tol=1e-12)
            assert math.isclose(float(np.exp(block_nll).mean()), m["test"]["mean_sequence_perplexity"], abs_tol=1e-10)
            assert math.isclose(float(np.exp(block_nll).std()), m["test"]["std_sequence_perplexity"], abs_tol=1e-10)
            assert seq["scored_tokens_per_sequence"]==reference["scored_tokens_per_sequence"]==63
            for split, count in (("validation", 3922), ("test", 4494)):
                score = m[split]
                assert score["scored_tokens"] == count*63
                assert math.isclose(score["loss"], score["nll_sum"]/score["scored_tokens"], abs_tol=1e-12)
                assert math.isclose(score["perplexity"], math.exp(score["loss"]), rel_tol=1e-12)
            new_stage = g >= (1 if arm == "continued" else 2)
            if new_stage:
                parent = raw_metrics[arm][g-1]
                expected = parent["training"]["selected_weights_sha256"] if arm == "continued" else c["start_checkpoint"]["weights_sha256"]
                h = get(rel+"/training_history.json")
                assert len(h) == 5 and [v["epoch"] for v in h] == [1,2,3,4,5]
                assert all(math.isfinite(v["train_loss"]) and math.isfinite(v["validation_loss"]) for v in h)
                assert t["initial_weights_sha256"] == expected and t["selected_weights_sha256"] != expected
                assert t["selected_epoch"] == t["executed_epochs"] == 5
                assert t["checkpoint_selection"] == "final_epoch" and t["selected_weights_sha256"] == t["terminal_weights_sha256"]
                assert t["selected_validation_loss"] == h[-1]["validation_loss"]
                assert t["best_validation_loss"] == min(v["validation_loss"] for v in h)
                assert t["input_tokens_sha256"] == parent["generation_timing"]["continuation_tokens_sha256"]
                assert t["training_blocks"] == 37946 and t["optimizer_initialization"] == "fresh_each_cycle"
                lineage = t["lineage"]
                assert lineage["initializer_weights_sha256"] == expected
                assert lineage["generator_weights_sha256"] == parent["training"]["selected_weights_sha256"]
                assert lineage["training_tokens_sha256"] == t["input_tokens_sha256"]
                dataset_meta=indexes[arm]["files"][f"synthetic_for_gen_{g:02d}/tokens.npy"]
                assert lineage["training_dataset_file_sha256"] == dataset_meta["sha256"]
                stage = get(rel+"/stage_completion.json")
                assert stage["selected_weights_sha256"] == t["selected_weights_sha256"]
                for f, digest in stage["files"].items():
                    assert indexes[arm]["files"][rel+"/"+f]["sha256"] == digest
                for f, digest in t["checkpoint_files"].items():
                    assert indexes[arm]["files"][rel+"/checkpoint/"+f]["sha256"] == digest
                histories.extend({"arm": arm,"cycle": g, **v} for v in h)
                stage_checks.append({"arm": arm,"cycle":g,"initial_weights_sha256":expected,"selected_weights_sha256":t["selected_weights_sha256"],
                                     "dataset_tokens_sha256":t["input_tokens_sha256"],"selected_epoch":5,"best_validation_epoch":t["best_validation_epoch"]})
            generated = m["generation_timing"]
            assert generated["provenance"]["parent_weights_sha256"] == t["selected_weights_sha256"]
            assert generated["provenance"]["prompt_count"] == (37946 if g < 10 else 256)
            assert generated["provenance"]["num_beams"] == 5 and generated["provenance"]["new_tokens"] == 64
            assert generated["provenance"]["repetition_penalty"] == 1
            if g < 10:
                assert generated["provenance"]["prompt_tokens_sha256"] == raw_metrics["continued"][0]["generation_timing"]["provenance"]["prompt_tokens_sha256"]
            if not (arm == "reset" and g < 2):
                assert get(f"synthetic_for_gen_{g+1:02d}/manifest.json") == generated
                for f, digest in generated["files"].items():
                    assert indexes[arm]["files"][f"synthetic_for_gen_{g+1:02d}/{f}"]["sha256"] == digest
            rows.append({"arm":arm,"cycle":g,"stage_origin":"shared" if g<2 else "executed",
                         "test_loss":m["test"]["loss"],"test_perplexity":m["test"]["perplexity"],
                         "validation_loss":m["validation"]["loss"],"validation_perplexity":m["validation"]["perplexity"],
                         "test_block_nll_p50":float(np.percentile(block_nll,50)),"test_block_nll_p95":float(np.percentile(block_nll,95)),
                         "reference_nll_mean":float(reference_nll.mean()),"reference_nll_p50":float(np.percentile(reference_nll,50)),"reference_nll_p95":float(np.percentile(reference_nll,95)),
                         "selected_epoch":t["selected_epoch"],"training_seconds":t["seconds"] if new_stage else 0,
                         "generation_seconds":generated["seconds"] if new_stage else 0,
                         "duplicate_outputs":round(measured["duplicate_continuation_rate"]*256),
                         **measured, **local_diversity(values)})
    assert len(stage_checks)==19 and len(histories)==95
    shared = read_json(RECORDS/"reset"/"shared_prefix.json")
    assert shared["d2_full_tokens_sha256"] == shared["d2_active_tokens_sha256"] == raw_metrics["continued"][1]["generation_timing"]["continuation_tokens_sha256"]
    assert raw_metrics["continued"][2]["training"]["input_tokens_sha256"] == raw_metrics["reset"][2]["training"]["input_tokens_sha256"]
    old_h=read_json(RECORDS/"source_d1/seed_42/gen_01/training_history.json")
    new_h=read_json(RECORDS/"continued/seed_42/gen_01/training_history.json")
    old_t=read_json(RECORDS/"source_d1/seed_42/gen_01/training.json")
    new_t=read_json(RECORDS/"continued/seed_42/gen_01/training.json")
    trajectory_keys=("epoch","learning_rate","train_loss","validation_loss")
    assert len(old_h)==len(new_h)==5
    assert all(all(old[k]==new[k] for k in trajectory_keys) for old,new in zip(old_h,new_h))
    assert old_t["selected_epoch"]==1 and new_t["selected_epoch"]==5
    assert old_t["initial_weights_sha256"]==new_t["initial_weights_sha256"]
    assert old_t["input_tokens_sha256"]==new_t["input_tokens_sha256"]
    assert old_t["selected_weights_sha256"]!=new_t["selected_weights_sha256"]
    old_d1=read_json(RECORDS/"source_d1/synthetic_for_gen_01/manifest.json")
    new_d1=raw_metrics["continued"][0]["generation_timing"]
    assert old_d1["continuation_tokens_sha256"]==new_d1["continuation_tokens_sha256"]==new_t["input_tokens_sha256"]
    assert old_d1["provenance"]==new_d1["provenance"]
    assert read_json(RECORDS/"source_d1/data/manifest.json")==data0
    imported_d1=read_json(RECORDS/"continued/initial_dataset_import.json")
    assert imported_d1["full_tokens_sha256"]==imported_d1["active_tokens_sha256"]==new_t["input_tokens_sha256"]
    crossweek={"same_s0_hash":new_t["initial_weights_sha256"],"same_d1_tokens_hash":new_t["input_tokens_sha256"],
               "epoch_trajectory_exact_fields":list(trajectory_keys),"five_epoch_training_validation_trajectory_exact":True,
               "old_selected_epoch":1,"new_selected_epoch":5,"old_selected_weights_sha256":old_t["selected_weights_sha256"],
               "new_selected_weights_sha256":new_t["selected_weights_sha256"],
               "scope":"Saved scalar losses and LR match exactly; no claim of comparing unsaved intermediate tensors."}
    ca=np.asarray(read_json(RECORDS/"continued"/"seed_42/gen_10/test_sequences.json")["sequence_mean_nll"])
    ra=np.asarray(read_json(RECORDS/"reset"/"seed_42/gen_10/test_sequences.json")["sequence_mean_nll"])
    result={"pair_id":PAIR,"verification_scope":receipt["scope"],"checks_passed":True,"unique_stages":19,"new_executed_epochs":95,
            "shared_prefix_identical":True,"fixed_evaluation_hashes":{s:indexes["continued"]["files"][f"data/{s}.npy"]["sha256"] for s in ("train","validation","test")},
            "probe_size":256,"tokens_per_continuation":64,"training_tokens_per_stage":37946*64,"shifted_training_targets_per_epoch":37946*63,
            "validation_scored_tokens":3922*63,"test_scored_tokens":4494*63,
            "test_blocks_with_lower_nll_reset":int((ra<ca).sum()),"test_blocks":4494,"mean_test_nll_continued_minus_reset":float((ca-ra).mean()),
            "best_validation_epoch_counts":dict(Counter(str(s["best_validation_epoch"]) for s in stage_checks)),
            "week2_vs_week3_cycle1":crossweek,
            "newline_sensitivity_posthoc":{arm:newline_sensitivity(np.load(RECORDS/"probes"/arm/"gen_10.npy",allow_pickle=False),
                np.asarray(read_json(RECORDS/arm/"seed_42/gen_10/reference_sequences.json")["sequence_mean_nll"])) for arm in ("continued","reset")},
            "limitations":["Bulk model tensors and full token datasets are checked by indexed/recorded hashes only, not downloaded or re-evaluated here.",
                           "NLL and perplexity arithmetic verified from saved sequence scores and aggregate NLL, without a fresh forward pass.",
                           "Diversity exactly recomputed on the fixed first 256 training prompts; probes are not independent random samples.",
                           "One seed, deterministic beam decoding and one dataset/model; block comparisons are descriptive, not independent-run significance.",
                           "Corpus token entropy measures pooled token frequencies, not model predictive entropy or sequence order; within-output metrics are reported separately."]}
    OUT.mkdir(parents=True,exist_ok=True)
    write_csv(OUT/"metrics.csv",rows); write_csv(OUT/"epoch_history.csv",histories); write_csv(OUT/"stage_lineage.csv",stage_checks)
    atomic_json(OUT/"audit.json",result)
    manifest={"created_at":datetime.now(timezone.utc).isoformat(),"git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT,text=True).strip(),
              "analysis_script_sha256":sha256(Path(__file__)),"environment":{"python":platform.python_version(),"numpy":np.__version__,"huggingface_hub":importlib.metadata.version("huggingface_hub")},
              "input_files":receipt["files"],"outputs":{p.name:{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in OUT.iterdir() if p.is_file() and p.name!="analysis_manifest.json"}}
    atomic_json(OUT/"analysis_manifest.json",manifest)
    print(json.dumps({"recovery":{k:v for k,v in receipt.items() if k!="files"},"audit":result},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records",type=Path,default=RECORDS)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--recover",action="store_true",help="Read pinned private backups for missing records; no compute")
    args=parser.parse_args()
    RECORDS=args.records.resolve()
    OUT=(args.out or ROOT/"generated"/("audit-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))).resolve()
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Choose a new empty audit output directory; published evidence must not be overwritten")
    if OUT==RECORDS or OUT.is_relative_to(RECORDS):
        raise RuntimeError("Derived audit output must be separate from immutable records")
    if args.recover:
        receipt,indexes=recover()
    else:
        receipt=read_json(RECORDS/"recovery_manifest.json")
        if not receipt or receipt.get("pair_id")!=PAIR:
            raise RuntimeError("Verified records missing; supply --recover with the saved private run provenance")
        for rel,meta in receipt["files"].items():
            path=(RECORDS/rel).resolve()
            assert path.is_relative_to(RECORDS) and path.is_file(),rel
            assert path.stat().st_size==meta["bytes"] and sha256(path)==meta["sha256"],rel
        indexes={arm:read_json(RECORDS/arm/"artifact_index.json") for arm in ("continued","reset")}
    print("Selected records recovered and checksum verified",flush=True)
    audit(receipt,indexes)
