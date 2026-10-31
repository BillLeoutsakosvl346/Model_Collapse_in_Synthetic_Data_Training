"""Reproduce Week 3 analysis from recovered immutable records and tiny probes.

No cloud writes, model loading, or paid compute. Recover evidence with
analyse_evidence.py first. Choose a new --out directory to preserve publications.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from model_collapse.common import sha256
from model_collapse.science.metrics import diversity

PAIR = "20261030T180755Z-full-paired-final5"
COLORS = {"reset": "#755493", "continued": "#cc6b27", "shared": "#666666"}
LABELS = {"reset": "Reset to S0 (A)", "continued": "Continued weights (B)"}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def entropy(tokens):
    counts = np.array(list(Counter(tokens).values()), dtype=float)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def row_repetition(tokens):
    grams = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    return 1 - len(set(grams)) / len(grams)


def derive(records, out):
    rows, histories, inputs, seqs, probes, samples = [], [], {}, {}, {}, {}

    def track(path):
        rel = str(path.resolve().relative_to(PROJECT)) if path.resolve().is_relative_to(PROJECT) else str(path.resolve())
        inputs[rel.replace("\\", "/")] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        return path

    def read(path):
        return json.loads(track(path).read_text(encoding="utf-8-sig"))

    for arm in ("continued", "reset"):
        seqs[arm], probes[arm], samples[arm] = {}, {}, {}
        for cycle in range(11):
            root = records / arm / f"seed_42/gen_{cycle:02d}"
            m = read(root / "metrics.json")
            t = read(root / "training.json")
            seq = np.asarray(read(root / "test_sequences.json")["sequence_mean_nll"], dtype=float)
            ref = np.asarray(read(root / "reference_sequences.json")["sequence_mean_nll"], dtype=float)
            sample = read(root / "samples.json")
            probe_arm = "continued" if cycle < 2 else arm
            p = track(records / "probes" / probe_arm / f"gen_{cycle:02d}.npy")
            probe = np.load(p, allow_pickle=False)
            assert probe.shape == (256, 64) and len(sample) == 256 and len(seq) == 4494 and len(ref) == 256
            assert np.isclose(seq.mean(), m["test"]["loss"], atol=1e-12, rtol=0)
            assert np.isclose(ref.mean(), m["frozen_g0_reference"]["loss"], atol=1e-12, rtol=0)
            for split, targets in (("validation", 247086), ("test", 283122)):
                s = m[split]
                assert s["scored_tokens"] == targets
                assert math.isclose(math.exp(s["nll_sum"] / targets), s["perplexity"], rel_tol=1e-12)
            measured = diversity(probe)
            assert all(math.isclose(measured[k], value, abs_tol=1e-12) for k, value in m["diversity"].items())
            seqs[arm][cycle], probes[arm][cycle], samples[arm][cycle] = seq, probe, sample
            internal_h = [entropy(row.tolist()) for row in probe]
            internal_r = [row_repetition(row.tolist()) for row in probe]
            heading_count = int(np.count_nonzero(probe == 5457))
            newline_counts = np.sum(probe == 50118, axis=1)
            # Keep original contiguous windows: removing tokens before forming
            # n-grams would invent phrases that the model never generated.
            nonnewline_windows = [tuple(row[i:i+4]) for row in probe.tolist() for i in range(61)
                                  if 50118 not in row[i:i+4]]
            all_newline_rows = np.flatnonzero(newline_counts == 64)
            for idx in all_newline_rows:
                assert sample[int(idx)]["continuation"] == "\n" * 64
            within_distinct = {}
            for n in (1, 2, 3, 4):
                within_distinct[f"mean_within_distinct_{n}"] = float(np.mean([
                    len({tuple(row[i:i+n]) for i in range(65-n)}) / (65-n) for row in probe.tolist()]))
            unique_stage = cycle >= 1 if arm == "continued" else cycle >= 2
            row = {
                "arm": arm, "cycle": cycle,
                "measurement_identity": "shared" if cycle < 2 else arm,
                "newly_executed_stage": unique_stage,
                "test_nll": m["test"]["loss"], "test_perplexity": m["test"]["perplexity"],
                "validation_nll": m["validation"]["loss"], "validation_perplexity": m["validation"]["perplexity"],
                "test_block_nll_p50": float(np.percentile(seq, 50)), "test_block_nll_p95": float(np.percentile(seq, 95)),
                "frozen_s0_nll_mean": float(ref.mean()), "frozen_s0_nll_p50": float(np.percentile(ref, 50)),
                "frozen_s0_nll_p95": float(np.percentile(ref, 95)), **measured,
                "distinct_4": 1-measured["repeated_4gram_rate"],
                "mean_within_entropy_nats": float(np.mean(internal_h)),
                "mean_within_repeated_4gram_rate": float(np.mean(internal_r)),
                "fraction_outputs_with_repeated_4gram": float(np.mean(np.array(internal_r) > 0)),
                **within_distinct,
                "heading_token_5457_count": heading_count, "heading_token_5457_share": heading_count / probe.size,
                "heading_token_5457_heavy_outputs": int(np.sum(np.sum(probe == 5457, axis=1) >= 32)),
                "newline_token_50118_count": int(newline_counts.sum()),
                "newline_token_50118_share": float(newline_counts.sum()/probe.size),
                "newline_heavy_outputs_ge32": int(np.sum(newline_counts >= 32)),
                "all_newline_outputs": int(np.sum(newline_counts == 64)),
                "tokens_retained_excluding_newline": int(np.count_nonzero(probe != 50118)),
                "pooled_entropy_excluding_newline_nats": entropy(probe[probe != 50118].tolist()),
                "nonnewline_original_4gram_windows": len(nonnewline_windows),
                "repeated4_excluding_newline_windows": 1-len(set(nonnewline_windows))/len(nonnewline_windows),
                "selected_epoch": t["selected_epoch"], "executed_epochs": t.get("executed_epochs", 0),
                "checkpoint_selection": t.get("checkpoint_selection", "historical_reference"),
                "selected_validation_loss": t.get("selected_validation_loss"),
                "best_validation_loss": t.get("best_validation_loss"), "best_validation_epoch": t.get("best_validation_epoch"),
                "recorded_training_seconds": t["seconds"], "recorded_generation_seconds": m["generation_timing"]["seconds"],
                "new_training_seconds": t["seconds"] if unique_stage else 0,
                "new_generation_seconds": m["generation_timing"]["seconds"] if unique_stage else 0,
                "selected_weights_sha256": t["selected_weights_sha256"],
            }
            rows.append(row)
            if unique_stage:
                hist = read(root / "training_history.json")
                assert [h["epoch"] for h in hist] == list(range(1, 6))
                assert t["selected_epoch"] == 5 and t["checkpoint_selection"] == "final_epoch"
                assert t["terminal_weights_sha256"] == t["selected_weights_sha256"]
                for h in hist:
                    histories.append({"arm": arm, "cycle": cycle, **h})
    for g in (0, 1):
        assert samples["continued"][g] == samples["reset"][g]
        assert np.array_equal(seqs["continued"][g], seqs["reset"][g])
        assert np.array_equal(probes["continued"][g], probes["reset"][g])
    prompts = [x["prompt"] for x in samples["continued"][0]]
    assert all([x["prompt"] for x in s] == prompts for arm in samples.values() for s in arm.values())
    assert len(histories) == 95
    by = {arm: [r for r in rows if r["arm"] == arm] for arm in ("continued", "reset")}
    a, b = by["reset"], by["continued"]
    gain = seqs["continued"][10] - seqs["reset"][10]
    baseline = seqs["continued"][0]
    all_examples = {arm: {str(k): values for k, values in data.items()} for arm, data in samples.items()}
    gaps = np.array([row_repetition(row.tolist()) for row in probes["continued"][10]]) - np.array([
        row_repetition(row.tolist()) for row in probes["reset"][10]])
    typical = int(np.argmin(np.abs(gaps-np.median(gaps))))
    negative = np.flatnonzero(gaps < 0)
    counter = int(negative[np.argmin(np.abs(gaps[negative]-np.median(gaps[negative])))]) if len(negative) else None
    selected = []
    all_blank = np.flatnonzero(np.all(probes["continued"][10] == 50118, axis=1))
    first_blank = int(all_blank[0]) if len(all_blank) else None
    for idx, rule in ((0, "Historical anchor prompt 0; not selected for typicality"),
                      (typical, "First prompt nearest the median continued-minus-reset within-output repeated-four-gram gap"),
                      (counter, "First prompt nearest the median negative gap among outputs where reset has greater internal repetition"),
                      (first_blank, "First prompt whose continued output consists of exactly 64 newline tokens")):
        if idx is None or any(e["prompt_index"] == idx for e in selected):
            continue
        selected.append({"prompt_index": idx, "selection_rule": rule, "prompt": prompts[idx],
                         "outputs": {arm: {"sample": samples[arm][10][idx],
                                             "within_output_entropy_nats": entropy(probes[arm][10][idx].tolist()),
                                             "within_output_repeated4_rate": row_repetition(probes[arm][10][idx].tolist())}
                                     for arm in ("continued", "reset")}})
    summary = {
        "pair_id": PAIR, "shared_measurements": [0, 1], "first_distinct_model_cycle": 2,
        "training_stages_new": 19, "epochs_executed_new": 95, "selected_epoch_each_stage": 5,
        "all_19_new_stages_best_validation_epoch": [r["best_validation_epoch"] for r in rows if r["newly_executed_stage"]],
        "train_blocks": 37946, "tokens_per_dataset": 37946*64, "targets_per_training_pass": 37946*63,
        "training_input_token_exposures": 95*37946*64, "training_target_exposures": 95*37946*63,
        "new_full_synthetic_datasets": 17, "new_full_synthetic_tokens": 17*37946*64,
        "probe_scope": "The same first 256 real training prompts; 64 tokens per continuation. Not a random full-corpus sample.",
        "common_s0_test_ppl": b[0]["test_perplexity"], "common_s1_test_ppl": b[1]["test_perplexity"],
        "common_first_jump_percent": 100*(b[1]["test_perplexity"]/b[0]["test_perplexity"]-1),
        "final_reset_test_ppl": a[10]["test_perplexity"], "final_continued_test_ppl": b[10]["test_perplexity"],
        "final_reset_vs_continued_ppl_percent": 100*(a[10]["test_perplexity"]/b[10]["test_perplexity"]-1),
        "final_test_nll_continued_minus_reset": float(gain.mean()),
        "final_test_blocks_lower_with_reset": int(np.sum(gain > 0)), "test_blocks": len(gain),
        "final_test_blocks_lower_with_reset_percent": float(np.mean(gain > 0)*100),
        "final_test_blocks_higher_than_s0": {arm: int(np.sum(seqs[arm][10] > baseline)) for arm in seqs},
        "final_test_difference_quantiles": {str(q): float(np.percentile(gain, q)) for q in (5, 50, 95)},
        "endpoint_metrics": {"s0": b[0], "s1": b[1], "reset_a10": a[10], "continued_b10": b[10]},
        "ppl_postfirst_minimum": {arm: min(by[arm][2:], key=lambda r:r["test_perplexity"]) for arm in by},
        "new_training_seconds": sum(r["new_training_seconds"] for r in rows),
        "new_generation_seconds": sum(r["new_generation_seconds"] for r in rows),
        "within_repetition_gap_summary": {"continued_greater": int(np.sum(gaps>0)), "reset_greater": int(np.sum(gaps<0)),
                                           "equal": int(np.sum(gaps==0)), "median_gap": float(np.median(gaps))},
        "selection_diagnostic": {"continued_cycle10_epoch5_minus_min_nll": b[10]["selected_validation_loss"]-b[10]["best_validation_loss"],
                                 "reset_cycle10_epoch5_minus_min_nll": a[10]["selected_validation_loss"]-a[10]["best_validation_loss"]},
        "formatting_diagnostic": {"token_50118_decode": "\\n", "all_newline_saved_decodes_checked": True,
            "scope": "Post-hoc token exclusion renormalizes unequal remaining token counts. Four-gram exclusion retains original contiguous windows only; no tokens are joined. Diagnostics do not replace the primary raw-token metrics or establish semantic quality.",
            "continued_final": {k:v for k,v in b[10].items() if "newline" in k},
            "reset_final": {k:v for k,v in a[10].items() if "newline" in k}},
        "inference_limit": "One seed. Block/probe distributions are descriptive, not across-run significance. No model re-evaluation or full synthetic-array verification in this script.",
    }
    historical = PROJECT / "reports/20261027T211852Z-full-reset-s0/analysis/metrics.csv"
    if historical.exists():
        with track(historical).open(encoding="utf-8-sig", newline="") as f:
            old = list(csv.DictReader(f))
        summary["historical_selection_context"] = {}
        for newarm, oldarm in (("continued", "continued"), ("reset", "reset_s0")):
            end = next(r for r in old if r["arm"] == oldarm and int(r["cycle"]) == 10)
            summary["historical_selection_context"][newarm] = {
                "historical_best_validation_selected_epoch": 1, "historical_final_test_ppl": float(end["test_perplexity"]),
                "current_final_epoch5_test_ppl": by[newarm][10]["test_perplexity"],
                "scope": "Historical seeded policy comparison; later synthetic datasets and common S1/D2 differ by construction. Not an independent replicated causal test.",
            }
    csv_write(out/"metrics.csv", rows)
    csv_write(out/"training_history.csv", histories)
    csv_write(out/"cycle_comparison.csv", [{"cycle": k,"shared": k<2,
        "reset_test_ppl":a[k]["test_perplexity"],"continued_test_ppl":b[k]["test_perplexity"],
        "continued_minus_reset_test_nll":b[k]["test_nll"]-a[k]["test_nll"],
        "reset_minus_continued_entropy_nats":a[k]["token_entropy_nats"]-b[k]["token_entropy_nats"],
        "reset_minus_continued_repeated4_pp":100*(a[k]["repeated_4gram_rate"]-b[k]["repeated_4gram_rate"])} for k in range(11)])
    save_json(out/"examples.json", {"selected_examples":selected,"all_saved_outputs":all_examples,
        "scope":"Complete 64-token continuations; generated factual claims unverified. Selection rules are descriptive and do not estimate prevalence."})
    save_json(out/"summary.json", summary)
    return by, histories, summary, seqs, inputs


def curves(ax, by, key, scale=1, linestyle="-", label_suffix=""):
    for arm in ("continued", "reset"):
        ax.plot(range(1,11), [r[key]*scale for r in by[arm][1:]], marker="o", ms=3.2,
                lw=1.6, ls=linestyle, color=COLORS[arm], label=LABELS[arm]+label_suffix)
    ax.plot([0,1], [by["continued"][k][key]*scale for k in (0,1)], "s"+linestyle,
            color=COLORS["shared"], lw=1.8, ms=4, zorder=5)


def plot(out, by, histories, summary, seqs):
    plt.rcParams.update({"font.size":10, "axes.titlesize":10.7,"axes.labelsize":9.5,
        "xtick.labelsize":9,"ytick.labelsize":9,"axes.spines.top":False,"axes.spines.right":False,
        "savefig.facecolor":"white","legend.fontsize":8.7,"font.family":"DejaVu Sans"})
    fig, axes = plt.subplots(3,2,figsize=(7.4,7.25),layout="constrained")
    specs = [("test_perplexity",1,"A  Fixed real-test perplexity","Corpus perplexity"),
             ("token_entropy_nats",1,"B  Pooled token entropy","Nats"),
             ("distinct_2",100,"C  Distinct-2 (solid) / -3 (dashed)","Unique / total n-grams (%)"),
             ("repeated_4gram_rate",100,"D  Pooled repeated four-grams","Repeated occurrences (%)"),
             ("duplicate_continuation_rate",100,"E  Exact duplicate outputs","Additional copies / 256 (%)"),
             ("frozen_s0_nll_mean",1,"F  Frozen S0 scores (no prompt)","Continuation NLL (nats/target)")]
    for ax,(key,scale,title,ylabel) in zip(axes.flat,specs):
        curves(ax,by,key,scale)
        if key == "distinct_2":
            curves(ax,by,"distinct_3",100,"--")
        if key == "frozen_s0_nll_mean":
            curves(ax,by,"frozen_s0_nll_p95",1,"--")
            ax.text(.3,.13,"Solid: mean; dashed: P95",transform=ax.transAxes,va="top",fontsize=8.4)
        ax.axvline(1,color="#999999",ls=":",lw=.75)
        ax.set(title=title,ylabel=ylabel,xlabel="Synthetic-training cycle",xticks=[0,2,4,6,8,10])
        ax.grid(alpha=.2)
    handles = [Line2D([0],[0],color=COLORS[a],marker="o",label=LABELS[a]) for a in ("continued","reset")]
    handles.append(Line2D([0],[0],color=COLORS["shared"],marker="s",label="Shared S0/S1"))
    fig.legend(handles=handles,loc="outside lower center",ncol=3,frameon=False)
    fig.suptitle("Week 3: retain final epoch 5 at every cycle",fontsize=12)
    fig.savefig(out/"figures/recursive_metrics.png",dpi=240)
    fig.savefig(out/"figures/recursive_metrics.pdf",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(7.4,5.2),layout="constrained")
    for arm,k,label,color,style in (("continued",0,"S0",COLORS["shared"],":"),
            ("continued",1,"Shared S1",COLORS["shared"],"--"),
            ("continued",10,"Continued B10",COLORS["continued"],"-"),("reset",10,"Reset A10",COLORS["reset"],"-")):
        arr=np.sort(seqs[arm][k]); axes[0,0].plot(arr,np.arange(1,len(arr)+1)/len(arr)*100,label=label,color=color,ls=style,lw=1.5)
    all_cdf_values = np.concatenate([seqs["continued"][0], seqs["continued"][1],
                                     seqs["continued"][10], seqs["reset"][10]])
    cdf_xlim = (min(0.8, math.floor(float(all_cdf_values.min()) * 5) / 5),
                math.ceil(float(all_cdf_values.max()) * 5) / 5)
    axes[0,0].set(title="A  Fixed real-test loss distribution",xlabel="Mean block NLL (nats/target)",ylabel="Cumulative blocks (%)",xlim=cdf_xlim)
    axes[0,0].legend(loc="lower right",fontsize=8)
    diff=seqs["continued"][10]-seqs["reset"][10]
    axes[0,1].hist(diff,bins=35,color="#477d99",alpha=.8)
    axes[0,1].axvline(0,color="#666666",ls="--",lw=1)
    axes[0,1].set(title="B  Matched final-block differences",xlabel="B10 NLL minus A10 NLL",ylabel="Number of blocks")
    axes[0,1].text(.96,.95,f"Reset lower on\n{summary['final_test_blocks_lower_with_reset']:,} / 4,494 blocks\nMean gap {diff.mean():.3f} nats",ha="right",va="top",transform=axes[0,1].transAxes,fontsize=8.8)
    for ax,cyc,title in ((axes[1,0],1,"C  Shared cycle 1: validation by epoch"),(axes[1,1],10,"D  Cycle 10: validation by epoch")):
        arms=("continued",) if cyc==1 else ("continued","reset")
        for arm in arms:
            h=[r for r in histories if r["arm"]==arm and r["cycle"]==cyc]
            color=COLORS["shared"] if cyc==1 else COLORS[arm]
            ax.plot([x["epoch"] for x in h],[x["validation_loss"] for x in h],"o-",color=color,label="Shared S1" if cyc==1 else LABELS[arm],ms=4)
            ax.scatter([5],[h[-1]["validation_loss"]],s=110,marker="*",color=color,zorder=4)
            best=min(h,key=lambda x:x["validation_loss"])
            ax.scatter([best["epoch"]],[best["validation_loss"]],s=70,facecolors="none",edgecolors=color,lw=1.5,zorder=4)
        ax.set(title=title,xlabel="Training epoch",ylabel="Fixed real-validation NLL",xticks=range(1,6))
        ax.legend(loc="upper left" if cyc==1 else "center left",fontsize=8)
    axes[1,0].text(.38,.13,"Star: retained epoch 5\nRing: lowest validation loss",transform=axes[1,0].transAxes,fontsize=8.6)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.savefig(out/"figures/generalization_and_selection.png",dpi=240)
    fig.savefig(out/"figures/generalization_and_selection.pdf",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)
    fig, axes=plt.subplots(2,2,figsize=(7.4,5.2),layout="constrained")
    for ax,key,scale,title,ylabel in (
        (axes[0,0],"newline_token_50118_share",100,"A  Newline concentration","Share of all tokens (%)"),
        (axes[0,1],"token_entropy_nats",1,"B  Pooled token entropy","Nats"),
        (axes[1,0],"mean_within_entropy_nats",1,"C  Mean per-output entropy","Nats"),
        (axes[1,1],"repeated_4gram_rate",100,"D  Pooled repeated four-grams","Repeated occurrences (%)")):
        curves(ax,by,key,scale)
        if key == "token_entropy_nats":
            curves(ax,by,"pooled_entropy_excluding_newline_nats",1,"--")
        if key == "repeated_4gram_rate":
            curves(ax,by,"repeated4_excluding_newline_windows",100,"--")
        ax.set(title=title,xlabel="Cycle",ylabel=ylabel,xticks=[0,2,4,6,8,10]);ax.grid(alpha=.2)
    fig.legend(handles=handles,loc="outside lower center",ncol=3,frameon=False,fontsize=8.2)
    fig.suptitle("Panels B/D: solid = raw; dashed = newline excluded",fontsize=10.5)
    fig.savefig(out/"figures/formatting_diagnostics.png",dpi=240)
    fig.savefig(out/"figures/formatting_diagnostics.pdf",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records",type=Path,default=PROJECT/"reports"/PAIR/"records")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    out=(args.out or PROJECT/"reports"/PAIR/"generated"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Choose a new empty output directory; published outputs must not be overwritten")
    (out/"figures").mkdir(parents=True,exist_ok=True)
    by,histories,summary,seqs,inputs=derive(args.records.resolve(),out)
    plot(out,by,histories,summary,seqs)
    files={str(p.relative_to(out)).replace("\\","/"):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in sorted(out.rglob("*")) if p.is_file()}
    save_json(out/"analysis_manifest.json",{"created_utc":datetime.now(timezone.utc).isoformat(),
        "analysis_script":str(Path(__file__).relative_to(PROJECT)).replace("\\","/"),"analysis_script_sha256":sha256(__file__),
        "analysis_git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT,text=True).strip(),
        "environment":{"python":platform.python_version(),**{p:importlib.metadata.version(p) for p in ("numpy","matplotlib")}},
        "input_files":inputs,"output_files":files,"hash_scope":"Excludes this self-referential manifest.",
        "verification_scope":"Recomputed saved scores, probe statistics and selection histories. Evidence audit independently checks source-index hashes and lineage. No new model inference."})
    print(json.dumps({"output":str(out),"rows":22,"unique_training_epochs":len(histories),"reset_lower_test_blocks":summary["final_test_blocks_lower_with_reset"]},indent=2))


if __name__=="__main__":
    main()
