"""Recover small evidence and audit the continued-weight mixture comparison.

Default: offline saved-record/probe analysis. --recover uses pinned private
backups, with no paid compute, model loading or full training-array downloads.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from model_collapse.common import PROJECT, STATE, atomic_json, digest, read_json, sha256

sys.path.insert(0, str(PROJECT))
from scripts.analysis.publication import new_destination, write_provenance

RUN = "20261031T163428Z-full-continued"
PAIR = "20261030T180755Z-full-paired-final5"
PUBLISHED = PROJECT / "reports" / RUN
RECORDS = PUBLISHED / "records"
OLD = PROJECT / "reports" / PAIR / "records"
LABELS = {"synthetic_100": "100% synthetic", "mixed_50": "50% real / 50% synthetic"}
COLORS = {"synthetic_100": "#d66a32", "mixed_50": "#178b87", "s0": "#6e7780"}


def immutable_copy(source, target, expected):
    source, target = Path(source), Path(target)
    if sha256(source) != expected:
        raise RuntimeError(f"Source checksum differs: {source}")
    if target.exists():
        if sha256(target) != expected:
            raise RuntimeError(f"Refusing to replace preserved evidence: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def recover():
    from model_collapse import remote
    from model_collapse.hub import download_snapshot_file, fetch
    from model_collapse.common import load_secrets
    load_secrets()
    local = PROJECT / "runs" / RUN
    completion = read_json(PROJECT / "docs/validation/2026-11-01_mixed-50-full-completion.json")
    snapshot = completion["backup"]
    proof = read_json(local / "recovery_verified.json")
    if not proof or proof["revisions"] != snapshot["revisions"]:
        proof = fetch(RUN, snapshot["repos"], local, metadata_only=True, revisions=snapshot["revisions"])
        atomic_json(local / "recovery_verified.json", proof)
    assert proof["artifact_index_sha256"] == snapshot["artifact_index_sha256"]
    index = read_json(local / "artifact_index.json")
    names = ["config.json", "manifest.json", "environment.json", "data/manifest.json", "benchmark.json",
             "initial_dataset_import.json", "completion.json", "resources.json"]
    for g in range(11):
        names += [f"seed_42/gen_{g:02d}/{n}" for n in ("metrics.json", "training.json", "samples.json",
                    "test_sequences.json", "reference_sequences.json", "stage_completion.json")]
        if g:
            names.append(f"seed_42/gen_{g:02d}/training_history.json")
            names.append(f"mixed_for_gen_{g:02d}/manifest.json")
        names.append(f"synthetic_for_gen_{g+1:02d}/manifest.json")
    copied = {}
    for name in names:
        meta = index["files"][name]
        source = local / name
        if not source.exists() or sha256(source) != meta["sha256"]:
            redirected = proof.get("preserved_control_files", {}).get(name)
            source = local / redirected["recovered_path"] if redirected else download_snapshot_file(
                RUN, snapshot["repos"], name, meta, snapshot["revisions"])
        immutable_copy(source, RECORDS / name, meta["sha256"])
        copied[name] = meta
    immutable_copy(local / "artifact_index.json", RECORDS / "artifact_index.json", snapshot["artifact_index_sha256"])
    copied["artifact_index.json"] = {"sha256": snapshot["artifact_index_sha256"], "bytes": (local / "artifact_index.json").stat().st_size}
    def probe(g):
        rel = f"synthetic_for_gen_{g+1:02d}/shard_0000000.npy"
        meta = index["files"][rel]
        cached = local / "probe_tokens" / f"gen_{g:02d}.npy"
        source = cached if cached.exists() and sha256(cached) == meta["sha256"] else download_snapshot_file(
            RUN, snapshot["repos"], rel, meta, snapshot["revisions"])
        key = f"probes/gen_{g:02d}.npy"
        immutable_copy(source, RECORDS / key, meta["sha256"])
        return key, {**meta, "source_run": RUN, "source_relative_path": rel}
    with ThreadPoolExecutor(max_workers=4) as pool:
        copied.update(dict(pool.map(probe, range(1, 11))))
    baseline = OLD / "probes/continued/gen_00.npy"
    initial = read_json(RECORDS / "config.json")["initial_dataset"]
    original = read_json(OLD / "continued/config.json")["initial_dataset"]
    assert initial["tokens_sha256"] == original["tokens_sha256"]
    baseline_hash = read_json(OLD / "source_d1/synthetic_for_gen_01/manifest.json")["files"]["shard_0000000.npy"]
    immutable_copy(baseline, RECORDS / "probes/gen_00.npy", baseline_hash)
    copied["probes/gen_00.npy"] = {"sha256": baseline_hash, "bytes": baseline.stat().st_size,
                                      "source_run": initial["run_id"], "source_relative_path": "synthetic_for_gen_01/shard_0000000.npy"}
    cores, selection_functions = {}, {}
    for code in ("962243987067fd6e4aaaa7ac77acee05ae0775779d3904d7cdc6f6712e6583ca",
                 "266afa984a2a51b63f850390232a12b3ec98a93975ad27f9401983d1afc060c5"):
        archive = STATE / f"code-{code}.tar.gz"
        with tarfile.open(archive) as bundle:
            cores[code] = {name: hashlib.sha256(bundle.extractfile(name).read().replace(b"\r\n", b"\n")).hexdigest()
                           for name in ("model_collapse/engine.py", "model_collapse/data.py", "model_collapse/workflow/selection.py")}
            tree = ast.parse(bundle.extractfile("model_collapse/workflow/selection.py").read().decode())
            selection_functions[code] = {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                                        if isinstance(node, ast.FunctionDef) and node.name in ("selection_policy", "verify_selection")}
    assert all(len({item[name] for item in cores.values()}) == 1 for name in ("model_collapse/engine.py", "model_collapse/data.py"))
    assert len({digest(x) for x in selection_functions.values()}) == 1
    core_record = RECORDS / "runtime_core_identity.json"
    if not core_record.exists():
        atomic_json(core_record, {"source_bundles": cores, "shared_training_data_core_identical": True,
            "selection_policy_and_verifier_ast_identical": True,
            "selection_module_difference": "training_contract adds synthetic_fraction to its resume-integrity fingerprint; checkpoint policy/verifier and training/data engine are unchanged."})
    copied["runtime_core_identity.json"] = {"sha256": sha256(core_record), "bytes": core_record.stat().st_size}
    receipt = {"run_id": RUN, "revisions": snapshot["revisions"], "metadata_recovery": proof,
               "selected_files": copied, "selected_files_count": len(copied),
               "selected_files_bytes": sum(v["bytes"] for v in copied.values()),
               "new_probe_arrays_downloaded": 10, "new_probe_bytes": 10 * 65664,
               "shared_s0_probe_reused": True,
               "scope": "All small metadata fetched privately; selected scientific records/probes published. No weights/full training arrays fetched. Full source-selection lists remain in private backups; offline analysis reconstructs their exact hashes."}
    target = RECORDS / "recovery_manifest.json"
    if target.exists():
        assert read_json(target) == receipt, "Existing recovery receipt differs"
    else:
        atomic_json(target, receipt)
    return receipt


def entropy(values):
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def grams(values, n):
    return [tuple(row[i:i+n]) for row in values for i in range(len(row)-n+1)]


def repetition(row):
    g = grams([row], 4)
    return 1 - len(set(g)) / len(g)


def probe_statistics(probe, samples):
    values = probe.tolist()
    within = np.asarray([repetition(r) for r in values])
    newline = (probe == 50118).sum(axis=1)
    spaces = (probe == 1437).sum(axis=1)
    kept4 = [g for g in grams(values, 4) if 50118 not in g]
    result = {f"distinct_{n}": len(set(grams(values, n))) / (len(probe) * (65-n)) for n in (1, 2, 3, 4)}
    result.update(repeated_4gram_rate=1-result["distinct_4"], repeated_4gram_denominator=256*61,
        duplicate_continuation_rate=1-len(set(map(tuple, values)))/len(probe),
        duplicate_outputs=len(probe)-len(set(map(tuple, values))), token_entropy_nats=entropy(probe.flatten().tolist()),
        probe_count=len(probe), mean_within_entropy_nats=float(np.mean([entropy(r) for r in values])),
        mean_within_repeated_4gram_rate=float(within.mean()), fraction_outputs_with_repeated_4gram=float(np.mean(within>0)),
        newline_token_50118_count=int(newline.sum()), newline_token_50118_share=float(newline.sum()/probe.size),
        newline_heavy_outputs_ge32=int(np.sum(newline>=32)), all_newline_outputs=int(np.sum(newline==64)),
        standalone_space_share=float(spaces.sum()/probe.size), standalone_space_heavy_outputs_ge32=int(np.sum(spaces>=32)),
        whitespace_only_decoded_outputs=sum(not s["continuation"].strip() for s in samples),
        outputs_without_alphabetic_characters=sum(not any(c.isalpha() for c in s["continuation"]) for s in samples),
        tokens_retained_excluding_newline=int(np.count_nonzero(probe!=50118)),
        pooled_entropy_excluding_newline_nats=entropy(probe[probe!=50118].tolist()),
        nonnewline_original_4gram_windows=len(kept4), repeated4_excluding_newline_windows=1-len(set(kept4))/len(kept4))
    for n in (1, 2, 3, 4):
        result[f"mean_within_distinct_{n}"] = float(np.mean([len(set(grams([r], n)))/(65-n) for r in values]))
    return result, within


def analyse(out):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerBase
    out = new_destination(PUBLISHED, out)
    recovery = read_json(RECORDS / "recovery_manifest.json")
    if not recovery:
        raise RuntimeError("Verified selected records missing; use --recover first")
    inputs = set()
    old_recovery = read_json(OLD / "recovery_manifest.json")
    def track(p):
        p = Path(p)
        if p.is_relative_to(OLD):
            relative = p.relative_to(OLD).as_posix()
            expected = old_recovery["files"].get(relative)
            if expected:
                assert sha256(p) == expected["sha256"], relative
        inputs.add(p)
        return p
    def read(p):
        return read_json(track(p))
    for rel, meta in recovery["selected_files"].items():
        assert sha256(track(RECORDS / rel)) == meta["sha256"], rel
    configs = {"mixed_50": read(RECORDS / "config.json"), "synthetic_100": read(OLD / "continued/config.json")}
    a, b = configs["mixed_50"], configs["synthetic_100"]
    matching_keys = ("seed", "model", "model_revision", "dataset", "dataset_revision", "dataset_config", "block_size", "new_tokens",
        "num_beams", "repetition_penalty", "epochs", "checkpoint_selection", "learning_rate", "min_learning_rate", "weight_decay",
        "microbatch", "effective_batch", "generation_batch", "eval_batch", "precision", "probe_count", "start_checkpoint")
    assert all(a[k]==b[k] for k in matching_keys)
    assert a["initial_dataset"]["tokens_sha256"]==b["initial_dataset"]["tokens_sha256"]
    assert a["initialization"]==b["initialization"]=="continued" and a["synthetic_fraction"]==.5
    env = read(RECORDS / "environment.json")
    assert env == read(OLD / "continued/environment.json")
    assert read(RECORDS / "data/manifest.json")["files"] == read(OLD / "continued/data/manifest.json")["files"]
    core = read(RECORDS / "runtime_core_identity.json")
    assert core["shared_training_data_core_identical"] and core["selection_policy_and_verifier_ast_identical"]
    tokenizer = Tokenizer.from_file(str(track(OLD / "continued/seed_42/gen_00/tokenizer/tokenizer.json")))
    assert tokenizer.decode([50118], skip_special_tokens=False)=="\n" and tokenizer.decode([1437],skip_special_tokens=False)==" "
    roots = {"mixed_50": RECORDS, "synthetic_100": OLD / "continued"}
    by, seqs, refs, probes, samples, within, histories, selections = {}, {}, {}, {}, {}, {}, [], []
    real_union, real_draws = set(), Counter()
    for arm, root in roots.items():
        by[arm], seqs[arm], refs[arm], probes[arm], samples[arm], within[arm] = [], {}, {}, {}, {}, {}
        for g in range(11):
            stage = root / f"seed_42/gen_{g:02d}"
            m, t = read(stage / "metrics.json"), read(stage / "training.json")
            assert t==m["training"]
            for key, count in (("test",283122),("validation",247086),("frozen_g0_reference",16128)):
                record=m[key]
                assert record["scored_tokens"]==count and record["context_tokens"]==64
                assert math.isclose(record["nll_sum"]/count,record["loss"],abs_tol=1e-10)
                assert math.isclose(math.exp(record["loss"]),record["perplexity"],rel_tol=1e-10)
            gp=m["generation_timing"]["provenance"]
            assert gp["prompt_count"]==(37946 if g<10 else 256)
            assert gp["new_tokens"]==64 and gp["num_beams"]==5 and gp["repetition_penalty"]==1.0
            block = np.asarray(read(stage / "test_sequences.json")["sequence_mean_nll"])
            reference = np.asarray(read(stage / "reference_sequences.json")["sequence_mean_nll"])
            assert len(block)==4494 and len(reference)==256
            for values, record in ((block,m["test"]),(reference,m["frozen_g0_reference"])):
                assert np.isfinite(values).all() and math.isclose(float(values.mean()),record["loss"],abs_tol=1e-10)
                assert math.isclose(math.exp(float(values.mean())),record["perplexity"],rel_tol=1e-10)
            assert math.isclose(float(np.mean(np.exp(block))),m["test"]["mean_sequence_perplexity"],rel_tol=1e-10)
            assert math.isclose(float(np.std(np.exp(block))),m["test"]["std_sequence_perplexity"],rel_tol=1e-10)
            probe_path = RECORDS / "probes" / f"gen_{g:02d}.npy" if arm=="mixed_50" else OLD / "probes/continued" / f"gen_{g:02d}.npy"
            probe = np.load(track(probe_path), allow_pickle=False)
            sample = read(stage / "samples.json")
            assert probe.shape==(256,64) and len(sample)==256
            for row, saved in zip(probe, sample):
                assert PreTrainedTokenizerBase.clean_up_tokenization(tokenizer.decode(row.tolist(),skip_special_tokens=False))==saved["continuation"]
            measured, inside = probe_statistics(probe,sample)
            for key, value in m["diversity"].items():
                assert math.isclose(measured[key],value,abs_tol=1e-12),(arm,g,key)
            for key, value in m.get("probe_diagnostics",{}).items():
                assert math.isclose(measured[key],value,abs_tol=1e-12),(arm,g,key)
            row = {"arm":arm,"cycle":g,"test_nll":float(block.mean()),"test_perplexity":m["test"]["perplexity"],
                "validation_nll":m["validation"]["loss"],"validation_perplexity":m["validation"]["perplexity"],
                "test_block_nll_p50":float(np.percentile(block,50)),"test_block_nll_p95":float(np.percentile(block,95)),
                "mean_passage_perplexity":m["test"]["mean_sequence_perplexity"],"passage_perplexity_sd":m["test"]["std_sequence_perplexity"],
                "frozen_s0_nll_mean":float(reference.mean()),"frozen_s0_nll_p95":float(np.percentile(reference,95)),
                "selected_epoch":t["selected_epoch"],"executed_epochs":t.get("executed_epochs",0),
                "best_validation_epoch":t.get("best_validation_epoch"),"selected_validation_loss":t.get("selected_validation_loss"),
                "minimum_validation_loss":t.get("best_validation_loss"),"train_seconds":t["seconds"] if g else 0,
                "generation_seconds":m["generation_timing"]["seconds"] if g else 0,
                "selected_weights_sha256":t["selected_weights_sha256"],
                "peak_allocated_gpu_bytes":t.get("peak_allocated_gpu_bytes",0),
                "new_training_stage":g>0,"training_blocks":37946 if g else 0,
                "real_blocks":(18973 if arm=="mixed_50" else 0) if g else 0,
                "synthetic_blocks":(18973 if arm=="mixed_50" else 37946) if g else 0,**measured}
            by[arm].append(row)
            seqs[arm][g],refs[arm][g],probes[arm][g],samples[arm][g],within[arm][g]=block,reference,probe,sample,inside
            if g:
                assert t["selected_epoch"]==t["executed_epochs"]==5 and t["checkpoint_selection"]=="final_epoch"
                assert t["selected_weights_sha256"]==t["terminal_weights_sha256"]
                assert t["selected_weights_sha256"]!=t["initial_weights_sha256"]
                assert t["optimizer_initialization"]=="fresh_each_cycle" and t["initialization"]=="continued"
                prior=read(root/f"seed_42/gen_{g-1:02d}/metrics.json")
                assert t["initial_weights_sha256"]==prior["training"]["selected_weights_sha256"]==t["lineage"]["generator_weights_sha256"]
                generation=prior["generation_timing"]
                if arm=="mixed_50":
                    mix=read(RECORDS/f"mixed_for_gen_{g:02d}/manifest.json")
                    assert mix["real_blocks"]==mix["synthetic_blocks"]==18973 and mix["training_blocks"]==37946
                    assert mix["training_tokens_sha256"]==t["input_tokens_sha256"]
                    assert mix["sources"]["synthetic_tokens_sha256"]==generation["continuation_tokens_sha256"]==t["lineage"]["synthetic_source_tokens_sha256"]
                    assert mix["sources"]["generator_weights_sha256"]==t["initial_weights_sha256"]
                    streams=[np.random.default_rng(np.random.SeedSequence([42,g,s])) for s in (0,1,2)]
                    selection={"real_indices":streams[0].choice(37946,18973,replace=False).tolist(),
                               "synthetic_indices":streams[1].choice(37946,18973,replace=False).tolist(),
                               "permutation":streams[2].permutation(37946).tolist()}
                    selection["source_labels"]=[int(i>=18973) for i in selection["permutation"]]
                    actual=hashlib.sha256((json.dumps(selection,indent=2,allow_nan=False)+"\n").encode()).hexdigest()
                    assert actual==mix["files"]["selection.json"],(g,"selection hash")
                    real_union.update(selection["real_indices"]);real_draws.update(selection["real_indices"])
                    selections.append({"cycle":g,"real_blocks":18973,"synthetic_blocks":18973,"real_tokens":1214272,
                        "synthetic_tokens":1214272,"training_tokens_sha256":t["input_tokens_sha256"],"selection_sha256":actual})
                else:
                    assert t["input_tokens_sha256"]==generation["continuation_tokens_sha256"]
                history=read(stage/"training_history.json")
                assert [r["epoch"] for r in history]==[1,2,3,4,5]
                assert math.isclose(history[-1]["validation_loss"],m["validation"]["loss"],abs_tol=1e-10)
                assert all(math.isfinite(v) for r in history for v in r.values() if isinstance(v,(int,float)))
                histories += [{"arm":arm,"cycle":g,**r} for r in history]
    assert np.array_equal(probes["mixed_50"][0],probes["synthetic_100"][0])
    assert np.array_equal(seqs["mixed_50"][0],seqs["synthetic_100"][0])
    prompts=[s["prompt"] for s in samples["synthetic_100"][0]]
    assert all([s["prompt"] for s in ss]==prompts for aa in samples.values() for ss in aa.values())
    assert len(histories)==100
    for g in range(1,11):
        learning_rates={arm:[r["learning_rate"] for r in histories if r["arm"]==arm and r["cycle"]==g] for arm in roots}
        assert learning_rates["mixed_50"]==learning_rates["synthetic_100"]
    gap=seqs["synthetic_100"][10]-seqs["mixed_50"][10]
    rgap=within["synthetic_100"][10]-within["mixed_50"][10]
    examples=[]
    anchors=[(0,"Fixed historical anchor; not selected for typicality"),
             (int(np.argmin(np.abs(rgap-np.median(rgap)))),"First output nearest median paired within-repetition gap"),
             (next(i for i,s in enumerate(samples["synthetic_100"][10]) if not s["continuation"].strip()),"First whitespace-only synthetic-only final output")]
    for i,rule in anchors:
        if any(x["prompt_index"]==i for x in examples):continue
        examples.append({"prompt_index":i,"selection_rule":rule,"prompt":prompts[i],
            "s0":samples["synthetic_100"][0][i]["continuation"],
            "outputs":{arm:{"text":samples[arm][10][i]["continuation"],"within_repeated4":float(within[arm][10][i])} for arm in roots}})
    summary={"checks_passed":True,"mixed_run":RUN,"synthetic_only_run":PAIR+"-continued","shared_s0":True,"same_initial_D1_pool":True,
        "verified_invariants":["Pinned mixed and historical input checksums", "Matched scientific settings and recorded environment",
            "Identical real split hashes, S0 weights and D1 pool", "Unchanged shared training/data engine and selection policy",
            "Every recorded NLL/PPL and passage distribution arithmetic", "All 22 probe metric sets independently recalculated",
            "All 5,632 saved continuations decoded from exact token IDs", "All 20 initializer/generator/data handoffs",
            "One hundred finite epoch records, terminal epoch 5 and identical learning-rate schedules",
            "Ten deterministic mixed selections reconstructed to their recorded SHA-256", "Fixed prompt order and evaluation context"],
        "matched_config_keys":list(matching_keys),"identical_recorded_environment":env,"unique_probes":21,
        "probe_scope":"The same first 256 real training prompts, 64 continuation tokens each; six adjacent source articles, not a random full-corpus sample.",
        "endpoint_metrics":{"s0":by["synthetic_100"][0],"synthetic_100":by["synthetic_100"][10],"mixed_50":by["mixed_50"][10]},
        "first_update_metrics":{arm:by[arm][1] for arm in roots},
        "final_ppl_reduction_percent":100*(1-by["mixed_50"][10]["test_perplexity"]/by["synthetic_100"][10]["test_perplexity"]),
        "mixed_final_ppl_above_s0_percent":100*(by["mixed_50"][10]["test_perplexity"]/by["mixed_50"][0]["test_perplexity"]-1),
        "paired_test_blocks":{"total":4494,"mixed_lower":int(np.sum(gap>0)),"synthetic_lower":int(np.sum(gap<0)),"ties":int(np.sum(gap==0)),"mean_nll_synthetic_minus_mixed":float(gap.mean()),"quantiles":{str(q):float(np.percentile(gap,q)) for q in [5,50,95]}},
        "test_blocks_worse_than_s0":{arm:int(np.sum(seqs[arm][10]>seqs[arm][0])) for arm in roots},
        "paired_within_repetition":{"synthetic_greater":int(np.sum(rgap>0)),"mixed_greater":int(np.sum(rgap<0)),"ties":int(np.sum(rgap==0))},
        "real_sampling":{"unique_real_blocks_selected":len(real_union),"real_pool_blocks":37946,"never_selected":37946-len(real_union),"selected_block_events":sum(real_draws.values()),"draw_count_range":[min(real_draws.values()),max(real_draws.values())]},
        "per_arm_training":{"new_stages":10,"executed_epochs":50,"input_tokens_per_dataset":2428544,"targets_per_epoch":2390598,"input_token_exposures":50*2428544,"target_exposures":50*2390598,"new_full_synthetic_pools":9},
        "new_recovery":{"all_metadata_files":recovery['metadata_recovery']['verified_files'],"all_metadata_bytes":recovery['metadata_recovery']['verified_bytes'],
                        "selected_publication_files":recovery['selected_files_count'],"selected_publication_bytes":recovery['selected_files_bytes'],
                        "new_probe_arrays":10,"new_probe_bytes":656640,"full_arrays_or_weights_downloaded":False},
        "recorded_work_seconds":{arm:{"training":sum(r["train_seconds"] for r in by[arm]),"generation":sum(r["generation_seconds"] for r in by[arm])} for arm in roots},
        "best_validation_epochs":{arm:[r["best_validation_epoch"] for r in by[arm][1:]] for arm in roots},
        "scope":"Independent arithmetic/diversity/selection/lineage audit of saved records. No new model forward pass or full token-array reconstruction; one seed, no across-run significance."}
    out.mkdir(parents=True,exist_ok=True)
    def csv_write(path, rows):
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    csv_write(out/"metrics.csv",by["synthetic_100"]+by["mixed_50"])
    csv_write(out/"training_history.csv",histories);csv_write(out/"mixture_counts.csv",selections)
    csv_write(out/"paired_test_blocks.csv",[{"block":i,"s0_nll":float(seqs['mixed_50'][0][i]),"synthetic_100_nll":float(seqs['synthetic_100'][10][i]),"mixed_50_nll":float(seqs['mixed_50'][10][i]),"synthetic_minus_mixed_nll":float(v)} for i,v in enumerate(gap)])
    atomic_json(out/"summary.json",summary);atomic_json(out/"examples.json",examples)
    plot(out,by,histories,seqs,summary,within)
    write_provenance(PROJECT,out,__file__,inputs|{RECORDS/'recovery_manifest.json',OLD/'recovery_manifest.json'},
                     scope=summary["scope"])
    return summary


def plot(out,by,histories,seqs,summary,within):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures=out/'figures';figures.mkdir()
    plt.rcParams.update({'font.size':9.2,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':7.5,
        'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans','savefig.facecolor':'white'})
    def curves(ax,key,scale=1,style='-',suffix=''):
        for arm in ['synthetic_100','mixed_50']:
            ax.plot(range(11),[r[key]*scale for r in by[arm]],style,marker='o' if style=='-' else None,ms=3,lw=1.5,color=COLORS[arm],label=LABELS[arm]+suffix)
        ax.scatter([0],[by['mixed_50'][0][key]*scale],marker='s',s=26,color=COLORS['s0'],zorder=5)
        ax.set(xlabel='Synthetic-training cycle',xticks=range(11));ax.grid(alpha=.2)
    fig,axes=plt.subplots(3,2,figsize=(7.4,7.2),layout='constrained')
    specs=[('test_perplexity',1,'A  Fixed real-test perplexity','Corpus perplexity'),('token_entropy_nats',1,'B  Pooled token entropy','Nats'),
           ('distinct_2',100,'C  Distinct-2 (solid) / -3 (dashed)','Unique / total windows (%)'),('repeated_4gram_rate',100,'D  Pooled repeated four-grams','Excess occurrences (%)'),
           ('duplicate_outputs',1,'E  Exact duplicate continuations','Additional copies / 256'),('frozen_s0_nll_mean',1,'F  Frozen S0 scores, no prompt','Nats / target')]
    for ax,(key,scale,title,ylabel) in zip(axes.flat,specs):
        curves(ax,key,scale)
        if key=='distinct_2':curves(ax,'distinct_3',100,'--')
        if key=='frozen_s0_nll_mean':curves(ax,'frozen_s0_nll_p95',1,'--')
        ax.set(title=title,ylabel=ylabel)
    axes[0,0].legend(loc='upper left');axes[2,1].text(.03,.96,'solid: mean; dashed: P95',transform=axes[2,1].transAxes,va='top',fontsize=7.5)
    fig.savefig(figures/'recursive_metrics.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.4,5.25),layout='constrained')
    values=[seqs['mixed_50'][0],seqs['synthetic_100'][10],seqs['mixed_50'][10]]
    bins=np.linspace(min(v.min() for v in values),max(v.max() for v in values),45)
    for label,values,color in [('S0',values[0],COLORS['s0']),('100% synthetic S10',values[1],COLORS['synthetic_100']),('50/50 S10',values[2],COLORS['mixed_50'])]:
        axes[0,0].hist(values,bins=bins,histtype='step',density=True,lw=1.5,label=label,color=color)
    axes[0,0].set(title='A  Fixed real-test blocks',xlabel='Mean block NLL',ylabel='Density');axes[0,0].legend()
    gap=seqs['synthetic_100'][10]-seqs['mixed_50'][10]
    axes[0,1].hist(gap,bins=45,color=COLORS['mixed_50'],alpha=.8);axes[0,1].axvline(0,color='black',lw=1)
    axes[0,1].set(title='B  Paired final block-loss gap',xlabel='100% NLL minus 50/50 NLL',ylabel='Test blocks')
    axes[0,1].text(.97,.95,f"50/50 lower on {int(np.sum(gap>0)):,} / 4,494",ha='right',va='top',transform=axes[0,1].transAxes,fontsize=8)
    for arm in ['synthetic_100','mixed_50']:
        for cycle,style in [(1,'--'),(10,'-')]:
            h=[r for r in histories if r['arm']==arm and r['cycle']==cycle]
            for ax,key in [(axes[1,0],'validation_loss'),(axes[1,1],'train_loss')]:
                ax.plot(range(1,6),[r[key] for r in h],style,color=COLORS[arm],label=LABELS[arm]+f', cycle {cycle}')
                ax.scatter([5],[h[-1][key]],marker='*',s=70,color=COLORS[arm],zorder=4)
    axes[1,0].set(title='C  Validation through five epochs',xlabel='Epoch (star = retained)',ylabel='Frozen real validation NLL',xticks=range(1,6))
    axes[1,1].set(title='D  Training fit: datasets differ',xlabel='Epoch (star = retained)',ylabel='Current training-dataset NLL',xticks=range(1,6))
    handles, labels = axes[1,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=7)
    for ax in axes.flat:ax.grid(alpha=.18)
    fig.savefig(figures/'real_text_and_training.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(7.4,7.2),layout='constrained')
    specs=[('newline_token_50118_share',100,'A  Formatting token concentration','Share of all probe tokens (%)'),
           ('token_entropy_nats',1,'B  Raw / newline-excluded entropy','Nats'),
           ('mean_within_repeated_4gram_rate',100,'C  Repetition inside each output','Mean repeated four-grams (%)'),
           ('repeated_4gram_rate',100,'D  Raw / newline-free windows','Pooled repeated four-grams (%)'),
           ('newline_heavy_outputs_ge32',1,'E  Formatting-heavy outputs (>=32/64)','Outputs / 256')]
    extra={'newline_token_50118_share':('standalone_space_share',100),'token_entropy_nats':('pooled_entropy_excluding_newline_nats',1),
           'repeated_4gram_rate':('repeated4_excluding_newline_windows',100),'newline_heavy_outputs_ge32':('standalone_space_heavy_outputs_ge32',1)}
    for ax,(key,scale,title,ylabel) in zip(list(axes.flat)[:5],specs):
        curves(ax,key,scale)
        if key in extra:curves(ax,*extra[key],'--')
        ax.set(title=title,ylabel=ylabel)
    axes[0,0].text(.02,.96,'solid: newline; dashed: standalone space',transform=axes[0,0].transAxes,va='top',fontsize=7)
    axes[2,0].text(.02,.96,'solid: newline; dashed: standalone space',transform=axes[2,0].transAxes,va='top',fontsize=7)
    x=within['mixed_50'][10]*100;y=within['synthetic_100'][10]*100
    axes[2,1].scatter(x,y,s=13,alpha=.45,color='#516b7a');axes[2,1].plot([0,100],[0,100],color='black',ls='--',lw=.8)
    axes[2,1].scatter([x[0]],[y[0]],color='#b32936',s=40,label='Anchor prompt 0');axes[2,1].legend()
    axes[2,1].set(title='F  Same prompts, different loops',xlabel='50/50 within-output repetition (%)',ylabel='100% within-output repetition (%)');axes[2,1].grid(alpha=.2)
    fig.savefig(figures/'formatting_diagnostics.png',dpi=220);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--recover',action='store_true')
    args=parser.parse_args()
    if args.recover:
        result=recover();print(json.dumps({'selected_recovery_files':result['selected_files_count'],'selected_bytes':result['selected_files_bytes']}),flush=True)
    result=analyse(args.out)
    print(json.dumps({k:v for k,v in result.items() if k in ['endpoint_metrics','paired_test_blocks','real_sampling','paired_within_repetition','final_ppl_reduction_percent','mixed_final_ppl_above_s0_percent','best_validation_epochs']},indent=2))
