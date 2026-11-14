"""Compare five matched SGD conditions using sealed diagnostics, without inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import tarfile
from pathlib import Path

import numpy as np

from comparisons.smollm2_sgd_greedy_effects.analyse_updates import interval, sign_counts, write_csv
from model_collapse.analysis.completion import comparison_contract
from model_collapse.analysis.sgd_effects import _collect_updates, source_summary
from model_collapse.common import atomic_json, read_json, sha256
from model_collapse.science.metrics import diversity
from model_collapse.workflow.smollm import probe_diagnostics

NEW = "20261114T150912Z-full-completion64-mixtures"
OLD = "20261114T001352Z-full-completion64-mixtures"
ARMS = ("real", "mixed50", "mixed80", "mixed99", "synthetic")
FRACTIONS = dict(zip(ARMS, (0., .5, .8, .99, 1.)))
COLORS = dict(zip(ARMS, ("#087F8C", "#2563A8", "#DA7920", "#7651AF", "#AD355C")))


def label(arm):
    return f"{100 * FRACTIONS[arm]:g}% synthetic"


def compatibility(old, new):
    if comparison_contract(old) != comparison_contract(new):
        raise RuntimeError("Historical controls differ in bank, evaluations, M0 or scientific settings")
    archive_names = ("02e45a8352f611b1004fa1bcf490e08384326eec141fa56576f9b28c70e839a0",
                     "790a2322864a98a5cd7efa66d776817511f2bfa035c613b50d9dec9c55e2971d")
    files = ("science/engine.py", "science/sgd_effects.py", "workflow/optimization.py",
             "workflow/sgd_calibration.py", "science/decoding.py", "science/metrics.py", "completion_mixing.py")
    identities = []
    for archive in archive_names:
        with tarfile.open(Path(".state") / ("code-" + archive + ".tar.gz")) as source:
            identities.append({name: hashlib.sha256(source.extractfile("model_collapse/" + name).read().replace(b"\r\n", b"\n")).hexdigest() for name in files})
    if identities[0] != identities[1]:
        raise RuntimeError("Training, diagnostic or decoding implementation changed across controls")
    return {"shared_contract": comparison_contract(new), "critical_runtime_files": identities[0],
            "historical_runtime": archive_names[0], "new_runtime": archive_names[1]}


def analyse(workspace, historical, out):
    workspace, historical, out = Path(workspace), Path(historical), Path(out)
    if out.exists():
        raise FileExistsError("Choose a new output destination")
    matched = compatibility(historical / OLD, workspace / NEW)
    sources, inputs, rows, layer_rows = {}, {}, [], []
    roots = {}
    for base, run in ((workspace, NEW), (historical, OLD)):
        recovery = read_json(base / "update_recovery.json")
        for key, source in recovery["sources"].items():
            directory = base / (run if key == "parent" else run + "-" + key)
            for name, meta in source["files"].items():
                if sha256(directory / name) != meta["sha256"]:
                    raise RuntimeError("Recovered input changed")
                inputs[run + ":" + key + ":" + name] = meta["sha256"]
        sources[run] = recovery
        c = read_json(base / run / "config.json")
        children = {a["id"]: base / (run + "-" + a["id"]) for a in c["completion_arms"]}
        roots.update(children)
        rr, ll, _, _ = _collect_updates(base / run, children)
        rows.extend(rr); layer_rows.extend(ll)
    if len(rows) != 3950 or len({r["probe_identity"] for r in rows}) != 1:
        raise RuntimeError("Missing updates or mismatched 128-pair diagnostic probe")
    cycles, tails, checks, endpoints, output_stats = [], [], {}, [], []
    score_cache = {}
    for arm in ARMS:
        values = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["step"])
        assert [r["step"] for r in values] == list(range(1, 791))
        continuity = max(abs(a["mixed_probe_nll"] - b["no_update_probe_nll"]) for a, b in zip(values, values[1:]))
        assert continuity < 1e-12
        vector_errors = []
        for r in values:
            r["real_symmetric_allocation"] = .5 * (r["real_only_delta_nll"] + r["real_conditional_delta_nll"])
            r["synthetic_symmetric_allocation"] = .5 * (r["synthetic_only_delta_nll"] + r["synthetic_conditional_delta_nll"])
            assert abs(r["real_symmetric_allocation"] + r["synthetic_symmetric_allocation"] - r["mixed_delta_nll"]) < 1e-12
            for source in ("real", "synthetic"):
                count = r[source + "_scored_targets"]
                expected = r[source + "_source_mean_nll"] * count / r["total_scored_targets"] if count else 0.
                assert abs(expected - r[source + "_weighted_training_nll"]) < 1e-12
            a, b, t, cosine = (r[k] for k in ("real_gradient_norm", "synthetic_gradient_norm", "total_gradient_norm", "source_cosine"))
            dot = a * b * cosine if cosine is not None else 0.
            expected = a*a + b*b + 2*dot
            error = abs(t*t - expected) / max(a*a + b*b + 2*abs(dot), 1e-30)
            assert error < 1e-5
            vector_errors.append(error)
        checks[arm] = {"updates": 790, "probe_continuity_max_error": continuity,
            "gradient_reconstruction_max_error": max(r["gradient_reconstruction_relative_error"] for r in values),
            "vector_identity_max_error": max(vector_errors),
            "real_effect_sign_counts": sign_counts([r["real_conditional_delta_nll"] for r in values]),
            "synthetic_effect_sign_counts": sign_counts([r["synthetic_conditional_delta_nll"] for r in values])}
        for cycle in range(11):
            m = read_json(roots[arm] / f"seed_42/gen_{cycle:02d}/metrics.json")
            assert abs(math.log(m["test"]["perplexity"]) - m["test"]["loss"]) < 1e-12
            endpoints.append({"arm": arm, "cycle": cycle, "validation_nll": m["validation"]["loss"],
                "validation_perplexity": m["validation"]["perplexity"], "test_nll": m["test"]["loss"],
                "test_perplexity": m["test"]["perplexity"], "test_nll_median": m["test"].get("median_example_nll"),
                "test_nll_p95": m["test"].get("p95_example_nll")})
            output_stats.append({"arm": arm, "cycle": cycle, **m["probe"]})
            if arm in ("mixed50", "mixed80"):
                path = (workspace / NEW / "seed_42/gen_00/probe/synthetic_for_gen_01/tokens.npy" if cycle == 0 else
                        roots[arm] / f"seed_42/gen_{cycle:02d}/probe/synthetic_for_gen_{cycle + 1:02d}/tokens.npy")
                tokens = np.load(path, allow_pickle=False)
                c = read_json(roots[arm] / "config.json")
                actual = {**diversity(tokens), **probe_diagnostics(tokens, c["newline_token_ids"])}
                assert tokens.shape == (256, 64)
                assert all(not isinstance(v, (int, float)) or abs(v - m["probe"][k]) < 1e-12 for k, v in actual.items())
            if not cycle:
                continue
            vv = [r for r in values if r["cycle"] == cycle]
            stage = roots[arm] / f"seed_42/gen_{cycle:02d}"
            seal = read_json(stage / "diagnostics_manifest.json")
            with np.load(stage / seal["scores_path"], allow_pickle=False) as arrays:
                arrays = {k: np.array(arrays[k], copy=True) for k in arrays.files}
            assert arrays["counts"].shape == (128,) and np.all(arrays["counts"] == 64)
            assert len(np.unique(arrays["documents"])) == 128
            score_cache[(arm, cycle)] = arrays
            row = {"arm": arm, "cycle": cycle, "updates": 79,
                "probe_ppl_before": vv[0]["no_update_probe_perplexity"],
                "probe_ppl_after": vv[-1]["mixed_probe_perplexity"]}
            effects = {"real_conditional": arrays["mixed"] - arrays["synthetic"],
                       "synthetic_conditional": arrays["mixed"] - arrays["real"],
                       "actual": arrays["mixed"] - arrays["no_update"],
                       "real_symmetric": .5 * (arrays["real"] - arrays["no_update"] + arrays["mixed"] - arrays["synthetic"]),
                       "synthetic_symmetric": .5 * (arrays["synthetic"] - arrays["no_update"] + arrays["mixed"] - arrays["real"])}
            for name, change in effects.items():
                mean, ci = interval(change.mean(0), arrays["counts"], arrays["documents"])
                row[name + "_mean_delta_nll"] = mean
                row[name + "_ci_low"], row[name + "_ci_high"] = ci
                row[name + "_sum_delta_nll"] = 79 * mean
            for source in ("real", "synthetic"):
                token_count = sum(r[source + "_scored_targets"] for r in vv)
                row[source + "_mean_training_nll"] = (math.fsum(r[source + "_source_mean_nll"] * r[source + "_scored_targets"] for r in vv if r[source + "_scored_targets"]) / token_count if token_count else None)
            assert abs(row["actual_sum_delta_nll"] - math.log(row["probe_ppl_after"] / row["probe_ppl_before"])) < 1e-10
            cycles.append(row)
            tail = vv[-1]
            tails.append({"arm": arm, "cycle": cycle, **{k: tail[k] for k in
                ("real_pairs", "synthetic_pairs", "mixed_delta_nll", "real_conditional_delta_nll", "synthetic_conditional_delta_nll")}})
    summaries = source_summary(rows)
    for r in summaries:
        same = [x for x in rows if x["arm"] == r["arm"] and x["cycle"] == r["cycle"] and
                (r["population"] == "all_updates" or (x["real_pairs"] + x["synthetic_pairs"] == 128) == (r["population"] == "full_batches"))]
        cosines = [x["source_cosine"] for x in same if x["source_cosine"] is not None]
        r["median_source_cosine"] = float(np.median(cosines)) if cosines else None
    phase_rows = []
    for phase, selected in (("early_cycles_1_to_3", range(1, 4)), ("late_cycles_6_to_10", range(6, 11))):
        for item in source_summary([{**r, "cycle": phase} for r in rows if r["cycle"] in selected]):
            item["phase"] = item.pop("cycle")
            phase_rows.append(item)
    out.mkdir(parents=True)
    for name, data in (("per_update.csv", rows), ("cycle_source_summary.csv", summaries),
        ("cycle_effects.csv", cycles), ("phase_summary.csv", phase_rows), ("tail_batches.csv", tails),
        ("test_trajectory.csv", endpoints), ("output_metrics.csv", output_stats)):
        write_csv(out / name, data)
    atomic_json(out / "verification.json", checks)
    atomic_json(out / "compatibility.json", matched)
    atomic_json(out / "summary.json", {"cycles": cycles, "source_summaries": summaries, "phases": phase_rows,
        "endpoints": [r for r in endpoints if r["cycle"] == 10], "new_fixed_probe_recomputations": 21,
        "controls": "Historical saved output statistics; all update journals/scores rechecked.", "checks": checks})
    draw(out, summaries, cycles, endpoints, output_stats, rows)
    atomic_json(out / "analysis_manifest.json", {"script_sha256": sha256(__file__),
        "recovery_script_sha256": sha256(Path(__file__).with_name("recover_updates.py")),
        "historical_math_helper_sha256": sha256("comparisons/smollm2_sgd_greedy_effects/analyse_updates.py"),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(), "numpy": np.__version__, "inputs": inputs,
        "recovery": sources, "new_training_or_inference": False,
        "scope": "3,950 sealed updates: 1,580 new and 2,370 historical; no bulk model/optimizer recovery.",
        "uncertainty": "2,000 paired document-bootstrap draws on 128 fixed documents; not training-seed uncertainty.",
        "outputs": {p.name: sha256(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({"out": str(out), "updates": len(rows), "endpoints": [r for r in endpoints if r["cycle"] == 10],
                      "phases": [r for r in phase_rows if r["population"] == "all_updates" and r["arm"] in ("mixed50", "mixed80")]}, indent=2))


def draw(out, sources, effects, endpoints, outputs, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12})
    def figure(title, ylabel):
        fig, ax = plt.subplots(figsize=(8, 8)); ax.set_box_aspect(1)
        ax.set(title=title, xlabel="Training cycle", ylabel=ylabel, xticks=range(1, 11)); ax.grid(alpha=.2)
        return fig, ax
    def save(fig, name, note):
        fig.text(.5, .015, note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .035, 1, 1)); fig.savefig(out / (name + ".png"), dpi=180)
        fig.savefig(out / (name + ".pdf"), metadata={"CreationDate": None, "ModDate": None}); plt.close(fig)
    for name, arms, log in (("01_test_perplexity_all", ARMS, True), ("02_test_perplexity_zoom", ARMS[:3], False)):
        fig, ax = figure("Fixed real-test continuation perplexity", "Corpus perplexity")
        for arm in arms:
            v = [r for r in endpoints if r["arm"] == arm]
            ax.plot([r["cycle"] for r in v], [r["test_perplexity"] for r in v], "o-", color=COLORS[arm], label=label(arm))
        ax.set_xticks(range(11)); ax.set_xlim(0, 10)
        if log: ax.set_yscale("log")
        ax.legend(); save(fig, name, "Same real-test pairs and pretrained M0. One seed; measured endpoints joined by lines.")
    all_rows = [r for r in sources if r["population"] == "all_updates"]
    fig, ax = figure("How much training loss comes from real targets?", "Real share of mean batch training loss (%)")
    for arm in ("mixed50", "mixed80", "mixed99"):
        v = [r for r in all_rows if r["arm"] == arm]
        ax.plot([r["cycle"] for r in v], [100*r["real_share_of_mean_batch_loss"] for r in v], "o-", label=label(arm), color=COLORS[arm])
    ax.set_ylim(0, 100); ax.legend(); save(fig, "03_real_loss_shares", "Source loss terms already include batch proportions. Loss share is not gradient share.")
    for arm in ("mixed50", "mixed80"):
        v = [r for r in all_rows if r["arm"] == arm];x=[r["cycle"] for r in v]
        fig, ax = figure(label(arm) + ": weighted gradients and their sum", "Mean gradient L2 norm per update")
        for key, title, color, style in (("mean_real_gradient_norm", "Weighted real", "#087F8C", "o-"),
            ("mean_synthetic_gradient_norm", "Weighted synthetic", "#BD365E", "o-"),
            ("mean_combined_gradient_norm", "Actual combined", "#433381", "s-"),
            ("mean_orthogonal_reference_norm", "Perpendicular reference", "#82909E", "--")):
            ax.plot(x, [r[key] for r in v], style, color=color, label=title)
        ax.set_ylim(bottom=0); ax.legend(fontsize=9)
        save(fig, "04_"+arm+"_gradients", "Means of 79 per-update norms; proportions already included. This is not an epoch-mean vector.")
        v = [r for r in effects if r["arm"] == arm]
        fig, ax = figure(label(arm) + ": immediate real-text effects", "Mean conditional real-probe ΔNLL per update")
        for source, title, color in (("real", "Including real", "#087F8C"), ("synthetic", "Including synthetic", "#BD365E")):
            prefix=source+"_conditional"
            ax.plot(x, [r[prefix+"_mean_delta_nll"] for r in v], "o-", color=color, label=title)
            ax.fill_between(x, [r[prefix+"_ci_low"] for r in v], [r[prefix+"_ci_high"] for r in v], color=color, alpha=.13)
        ax.axhline(0,color="#334155",lw=1);ax.legend();ax.ticklabel_format(axis="y",style="sci",scilimits=(-3,3))
        save(fig,"05_"+arm+"_help_hurt","Below zero improves the fixed 128-pair real probe. Bands resample documents, not training seeds.")
    for field,name,title,ylabel,percent in (("median_source_cosine","06_gradient_alignment","Source-gradient alignment","Median real/synthetic cosine",False),
        ("cancellation_vs_orthogonal_reference","07_gradient_cancellation","How strongly do the gradients oppose?","Norm reduction versus perpendicular reference (%)",True)):
        fig,ax=figure(title,ylabel)
        for arm in ("mixed50","mixed80","mixed99"):
            v=[r for r in all_rows if r["arm"]==arm];ax.plot([r["cycle"] for r in v],[r[field]*(100 if percent else 1) for r in v],"o-",label=label(arm),color=COLORS[arm])
        ax.axhline(0,color="#334155",lw=1);ax.legend()
        if not percent:ax.set_ylim(-1,1)
        save(fig,name,"Geometry describes opposition, not prediction quality. Undefined source alignments are excluded.")
    for key,name,title,ylabel in (("mean_within_entropy_nats","08_output_entropy","Token variety within generated outputs","Mean within-output token entropy (nats)"),
        ("mean_within_repeated_4gram_rate","09_output_repetition","Repeated phrases within generated outputs","Mean within-output repeated 4-grams (%)")):
        fig,ax=figure(title,ylabel)
        for arm in ARMS:
            v=[r for r in outputs if r["arm"]==arm];ax.plot([r["cycle"] for r in v],[r[key]*(100 if "repeated" in key else 1) for r in v],"o-",color=COLORS[arm],label=label(arm))
        ax.set_xticks(range(11));ax.legend();save(fig,name,"256 fixed prompts; greedy 64-token outputs. Empirical text statistics, not predictive entropy.")
    fig,ax=figure("Actual update-by-update real-probe perplexity", "Perplexity on 128 fixed real-validation pairs")
    for arm in ("mixed50","mixed80"):
        v=sorted((r for r in rows if r['arm']==arm),key=lambda r:r['step'])
        ax.plot([0]+[r['step']/79 for r in v],[v[0]['no_update_probe_perplexity']]+[r['mixed_probe_perplexity'] for r in v],
                color=COLORS[arm],lw=.9,label=label(arm))
        tail=[r for r in v if r['step']%79==0]
        ax.scatter([r['cycle'] for r in tail],[r['mixed_probe_perplexity'] for r in tail],s=20,color='#334155',
                   label='After 16-row final batch' if arm=='mixed50' else None,zorder=4)
    last=[r for r in rows if r['arm']=='mixed80' and r['step']==790][0]
    ax.annotate(f"80% final update: {last['real_pairs']} real / 16\n{last['no_update_probe_perplexity']:.2f} → {last['mixed_probe_perplexity']:.2f}",
                xy=(10,last['mixed_probe_perplexity']),xytext=(6.1,18.0),fontsize=9,
                arrowprops={'arrowstyle':'->','color':'#334155'})
    ax.set_xticks(range(11));ax.set_xlim(0,10.1);ax.legend(loc='upper left',fontsize=9)
    save(fig,'10_update_path_and_tails','Fractional cycle = progress within an epoch. Probe values are not the complete-test perplexity.')


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",default=".state/sgd-mixture-update-analysis")
    p.add_argument("--historical",default=".state/sgd-update-analysis")
    p.add_argument("--out",required=True)
    a=p.parse_args();analyse(a.workspace,a.historical,a.out)
