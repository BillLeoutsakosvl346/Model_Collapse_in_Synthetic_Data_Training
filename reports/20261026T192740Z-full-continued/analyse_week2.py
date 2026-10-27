"""Audit retrieved statistics and reproduce the Week 2 research figures, without a GPU."""
from pathlib import Path
import csv
import hashlib
import math
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_collapse.common import PROJECT, atomic_json, read_json, verify_files
from model_collapse.engine import diversity


def analyse():
    out = Path(__file__).resolve().parent
    rid = out.name
    root = PROJECT / "runs" / rid
    c = read_json(root / "config.json")
    index = read_json(root / "artifact_index.json")
    old = PROJECT / "reports/20261020T211121Z-full"
    old_index = read_json(old / "artifact_index.json")
    stats = read_json(root / "data/manifest.json")["stats"]
    assert c["kind"] == "full" and c["initialization"] == "continued" and c["generations"] == 10
    for split, count in (("train", 37946), ("validation", 3922), ("test", 4494)):
        assert stats[split]["used_blocks"] == count
        assert index["files"][f"data/{split}.npy"]["sha256"] == old_index["files"][f"data/{split}.npy"]["sha256"]
    required = ["config.json", "resources.json", "environment.json", "data/manifest.json", "benchmark.json", "completion.json",
                "seed_42/pretrained/metrics.json"]
    rows, metrics, epochs, handoffs = [], [], [], []
    initial = c["start_checkpoint"]["weights_sha256"]
    for g in range(11):
        prefix = f"seed_42/gen_{g:02d}"
        names = [f"{prefix}/{n}.json" for n in ("metrics", "training", "test_sequences", "reference_sequences", "samples")]
        if g:
            names += [f"{prefix}/training_history.json"]
        names += [f"synthetic_for_gen_{g+1:02d}/manifest.json"]
        required += names
        m = read_json(root / names[0])
        t = m["training"]
        assert t["initial_weights_sha256"] == initial
        if g:
            parent = read_json(root / f"synthetic_for_gen_{g:02d}/manifest.json")
            assert parent["continuation_tokens_sha256"] == t["input_tokens_sha256"]
            assert parent["provenance"]["parent_weights_sha256"] == initial
            assert t["selected_weights_sha256"] != initial
            assert t["training_blocks"] == 37946 and t["optimizer_initialization"] == "fresh_each_cycle"
            history = read_json(root / f"{prefix}/training_history.json")
            assert len(history) == 5
            assert t["selected_epoch"] == min(history, key=lambda h: h["validation_loss"])["epoch"]
            assert t["best_validation_loss"] == m["validation"]["loss"]
            epochs.append(t["selected_epoch"])
            handoffs.append({"cycle": g, "initial_weights_sha256": initial,
                             "selected_weights_sha256": t["selected_weights_sha256"], "input_tokens_sha256": t["input_tokens_sha256"]})
        else:
            assert t["training_blocks"] == 0 and t["selected_weights_sha256"] == initial
            prior_start = read_json(old / "records/seed_42/gen_00/metrics.json")["training"]["selected_weights_sha256"]
            assert initial == prior_start
        generated = read_json(root / f"synthetic_for_gen_{g+1:02d}/manifest.json")
        assert generated["provenance"]["parent_weights_sha256"] == t["selected_weights_sha256"]
        assert generated["provenance"]["prompt_count"] == (37946 if g < 10 else 256)
        probe = np.load(root / "probe_tokens" / f"gen_{g:02d}.npy", allow_pickle=False)
        assert probe.shape == (256, 64)
        probe_rel = f"synthetic_for_gen_{g+1:02d}/shard_0000000.npy"
        verify_files(root / "probe_tokens", {f"gen_{g:02d}.npy": index["files"][probe_rel]["sha256"]})
        recalculated = diversity(probe)
        assert all(math.isclose(recalculated[k], v, abs_tol=1e-12) for k, v in m["diversity"].items())
        seq = np.asarray(read_json(root / f"{prefix}/test_sequences.json")["sequence_mean_nll"])
        assert len(seq) == 4494
        for split, count in (("validation", 3922), ("test", 4494)):
            a = m[split]
            assert a["scored_tokens"] == count * 63
            assert math.isclose(math.exp(a["nll_sum"] / a["scored_tokens"]), a["perplexity"], rel_tol=1e-12)
        assert math.isclose(float(seq.mean()), m["test"]["loss"], abs_tol=1e-12)
        assert math.isclose(float(np.exp(seq).std()), m["test"]["std_sequence_perplexity"], rel_tol=1e-12)
        references = np.asarray(read_json(root / f"{prefix}/reference_sequences.json")["sequence_mean_nll"])
        assert len(references) == 256 and math.isclose(references.mean(), m["frozen_g0_reference"]["loss"], abs_tol=1e-12)
        row = {"cycle": g, "validation_loss": m["validation"]["loss"], "validation_perplexity": m["validation"]["perplexity"],
               "test_loss": m["test"]["loss"], "test_perplexity": m["test"]["perplexity"],
               "test_block_nll_p50": float(np.percentile(seq, 50)), "test_block_nll_p95": float(np.percentile(seq, 95)),
               "test_sequence_perplexity_mean": m["test"]["mean_sequence_perplexity"],
               "test_sequence_perplexity_sd": m["test"]["std_sequence_perplexity"],
               "frozen_s0_nll": m["frozen_g0_reference"]["loss"], "selected_epoch": t["selected_epoch"],
               "train_seconds": t["seconds"], "generation_seconds": generated["seconds"],
               "peak_allocated_gpu_bytes": t.get("peak_allocated_gpu_bytes", 0), **recalculated}
        rows.append(row)
        metrics.append(m)
        initial = t["selected_weights_sha256"]
    verify_files(root, {p: index["files"][p] for p in required})
    for rel in required:
        p = out / "records" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, p)
    shutil.copyfile(root / "artifact_index.json", out / "artifact_index.json")
    shutil.copyfile(root / "report/metrics.csv", out / "metrics_cloud_original.csv")
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    changes = {"test_ppl_s0_to_s10_percent": 100 * (rows[10]["test_perplexity"] / rows[0]["test_perplexity"] - 1),
               "test_ppl_s3_to_s10_percent": 100 * (rows[10]["test_perplexity"] / rows[3]["test_perplexity"] - 1),
               "repetition_s0_to_s10_percentage_points": 100 * (rows[10]["repeated_4gram_rate"] - rows[0]["repeated_4gram_rate"]),
               "distinct2_s2_to_s10_percent": 100 * (rows[10]["distinct_2"] / rows[2]["distinct_2"] - 1)}
    audit = {"run_id": rid, "scope": "Checksummed statistics, saved lineage, evaluation exports and directly recomputed 256-continuation probe metrics. No bulk-model re-evaluation or full synthetic-array download.",
             "checks": {"all_statistic_files_match_index": True, "same_starting_g0_tensor_hash_as_week1": True,
                        "all_ten_weight_and_data_handoffs": True, "fixed_real_splits_byte_identical_to_week1": True,
                        "all_11_probe_diversity_metrics_recomputed": True, "all_corpus_ppl_and_passage_statistics_recomputed": True,
                        "all_50_epochs_and_checkpoint_selection_verified": True},
             "statistic_files_verified": len(required), "selected_epochs": epochs, "handoffs": handoffs,
             "changes": changes, "synthetic_input_tokens_per_cycle": 2428544, "training_input_token_exposures_executed": 121427200,
             "selected_parent_update_epochs": sum(epochs), "probe_npy_bytes_downloaded": 11 * 65664}
    atomic_json(out / "consistency_audit.json", audit)
    plot(out, rows, metrics, old)
    return {"audit": audit, "selected_rows": [rows[g] for g in (0, 1, 5, 10)]}


def plot(out, rows, metrics, old):
    with (old / "metrics.csv").open() as f:
        prior = list(csv.DictReader(f))
    blue, orange = "#24708b", "#cc6b27"
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.45), layout="constrained")
    x = np.arange(11)
    def oldvals(key): return [float(r[key]) for r in prior]
    def newvals(key): return [r[key] for r in rows]
    a=axes[0,0]
    a.plot(x, oldvals("test_perplexity"), "o-", color=blue, label="Week 1: fresh base", ms=3)
    a.plot(x, newvals("test_perplexity"), "o-", color=orange, label="Week 2: carried weights", ms=3)
    a.set(title="A  Fixed real-test perplexity", ylabel="Corpus perplexity");a.legend(fontsize=7)
    a=axes[0,1]
    for n,col in ((1,"#437f97"),(2,"#ce742c"),(3,"#4a8c63")):
        a.plot(x, np.array(newvals(f"distinct_{n}"))*100,"o-",color=col,label=f"Distinct-{n}",ms=3)
    a.set(title="B  Week 2 output diversity",ylabel="Unique / total n-grams (%)");a.legend(fontsize=7)
    a=axes[1,0]
    a.plot(x,np.array(oldvals("repeated_4gram_rate"))*100,"o-",color=blue,label="Week 1: repeated 4-grams",ms=3)
    a.plot(x,np.array(newvals("repeated_4gram_rate"))*100,"o-",color=orange,label="Week 2: repeated 4-grams",ms=3)
    a.plot(x,np.array(newvals("duplicate_continuation_rate"))*100,"s--",color="#7b5c91",label="Week 2: exact duplicates",ms=3)
    a.set(title="C  Repetition and duplication",ylabel="Probe fraction (%)");a.legend(fontsize=6.7)
    a=axes[1,1]
    a.plot(x,oldvals("token_entropy_nats"),"o-",color=blue,label="Week 1",ms=3)
    a.plot(x,newvals("token_entropy_nats"),"o-",color=orange,label="Week 2",ms=3)
    a.set(title="D  Empirical token entropy",ylabel="Nats");a.legend(fontsize=7)
    a=axes[2,0]
    old_metrics=[read_json(old/f"records/seed_42/gen_{g:02d}/metrics.json") for g in range(11)]
    a.plot(x,[m["frozen_g0_reference"]["loss"] for m in old_metrics],"o-",color=blue,label="Week 1",ms=3)
    a.plot(x,newvals("frozen_s0_nll"),"o-",color=orange,label="Week 2",ms=3)
    a.set(title="E  Output scores under frozen G0/S0",ylabel="Mean continuation NLL");a.legend(fontsize=7)
    a=axes[2,1]
    for g,col in ((1,"#437f97"),(5,"#ce742c"),(10,"#7b5c91")):
        h=read_json(out/f"records/seed_42/gen_{g:02d}/training_history.json")
        a.plot(range(1,6),[r["validation_loss"] for r in h],"o-",color=col,label=f"S{g}",ms=3)
        a.scatter([1],[h[0]["validation_loss"]],marker="*",s=70,color=col,zorder=4)
        a.text(5.08,h[-1]["validation_loss"],f"S{g}",color=col,va="center",fontsize=7.5)
    a.set(title="F  Within-cycle validation loss",ylabel="Real-validation NLL",xlabel="Epoch",xticks=range(1,6))
    a.set_xlim(.8,5.6)
    a.text(.04,.96,"Epoch 1 selected in all ten cycles",transform=a.transAxes,va="top",fontsize=7)
    for i,a in enumerate(axes.flat):
        legend=a.get_legend()
        if legend:
            for text in legend.get_texts(): text.set_fontsize(7.5)
        a.grid(alpha=.2)
        if i<5:a.set(xlabel="Synthetic-training cycle",xticks=[0,2,4,6,8,10])
    fig.savefig(out/"week2_metrics_panels.png",dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    import json,sys
    if "--figures-only" in sys.argv:
        here=Path(__file__).resolve().parent
        with (here/"metrics.csv").open() as f:
            rows=[{k:float(v) if v else None for k,v in row.items()} for row in csv.DictReader(f)]
        plot(here,rows,[],PROJECT/"reports/20261020T211121Z-full")
        print("Rebuilt the six-panel figure from committed records; no network or GPU used.")
    else:
        print(json.dumps(analyse(), indent=2))
