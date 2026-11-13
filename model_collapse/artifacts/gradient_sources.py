"""Selective, immutable restoration for the gradient diagnostic."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from ..common import atomic_json, read_json, sha256, verify_files
from .hub import download_snapshot_file


class Snapshot:
    def __init__(self, ref, root):
        self.ref, self.root = ref, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        cached = self.root / "artifact_index.json"
        path = cached if cached.is_file() and sha256(cached) == ref["artifact_index_sha256"] else Path(hf_hub_download(
            ref["repos"]["dataset"], ref["run_id"] + "/artifact_index.json", repo_type="dataset", revision=ref["revisions"]["dataset"]))
        if sha256(path) != ref["artifact_index_sha256"]:
            raise RuntimeError("Pinned gradient source index differs")
        self.index = read_json(path)["files"]
        self.used = {}
        if Path(path).resolve() != cached.resolve():
            shutil.copyfile(path, cached)

    def get(self, relative):
        if relative not in self.index:
            raise RuntimeError("Required diagnostic source absent: " + relative)
        meta = self.index[relative]
        destination = self.root / relative
        if not destination.is_file() or sha256(destination) != meta["sha256"]:
            source = download_snapshot_file(self.ref["run_id"], self.ref["repos"], relative, meta, self.ref["revisions"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        verify_files(self.root, {relative: meta})
        self.used[relative] = meta
        return destination

    def json(self, relative):
        return read_json(self.get(relative))

    def array(self, relative):
        return np.load(self.get(relative), allow_pickle=False)

    def checkpoint(self, prefix, *, restore_weights=True):
        files = [n for n in self.index if n.startswith(prefix + "/checkpoint/")]
        if not any(n.endswith(".safetensors") for n in files):
            raise RuntimeError("Diagnostic checkpoint weights are absent")
        for name in files:
            if restore_weights or not name.endswith(".safetensors"):
                self.get(name)
        return self.root / prefix / "checkpoint"

    def receipt(self):
        return {"snapshot": self.ref, "files": self.used,
                "verified_bytes": sum(m["bytes"] for m in self.used.values())}


def check_input(tokens, attention, scoring, prompts, real, generated, selection, manifest, weights, bank_hash, chunk_hash):
    """Validate saved input bytes against independently restored real/generated sources."""
    from ..completion_data import assemble
    from ..science.identity import token_hash
    lineage = manifest["lineage"]
    if any(lineage.get(k) != weights for k in ("generator_weights_sha256", "initializer_weights_sha256")):
        raise RuntimeError("Diagnostic generator or initializer differs")
    if lineage["bank_manifest_sha256"] != bank_hash or lineage["chunk_manifest_sha256"] != chunk_hash:
        raise RuntimeError("Diagnostic bank/chunk identity differs")
    target = np.array(real, copy=True)
    target[selection] = generated
    expected = assemble(prompts, target, int(tokens[0, 0]))
    if any(not np.array_equal(a, b) for a, b in zip((tokens, attention, scoring), expected)):
        raise RuntimeError("Diagnostic input does not reconstruct from its sources")
    if (manifest["prompt_content_sha256"] != token_hash(prompts)
            or manifest["target_content_sha256"] != token_hash(target)
            or manifest["scored_targets"] != len(tokens) * 64):
        raise RuntimeError("Diagnostic target identity/count differs")
    if not np.all(scoring[:, :65] == 0) or not np.all(scoring[:, 65:] == 1):
        raise RuntimeError("Diagnostic continuation masks differ")
    return target


def restore_sources(root, c, *, include_weights=True):
    """No generation or training; only indexed source recovery."""
    from ..completion_mixing import select_rows
    from ..science.identity import token_hash
    root = Path(root)
    source_root = root / ".sources"
    parent = Snapshot(c["source_parent"], source_root / "parent")
    scientific = parent.json("config.json")
    if sha256(parent.root / "config.json") != c["source_config_sha256"]:
        raise RuntimeError("Source scientific configuration differs")
    snapshots = parent.json("child_snapshots.json")
    bank = Snapshot(scientific["prepared_bank"], source_root / "bank")
    bank_manifest = bank.json("manifest.json")
    bank_hash = sha256(bank.root / "manifest.json")
    if bank_hash != scientific["prepared_bank"]["manifest_sha256"]:
        raise RuntimeError("Source bank manifest differs")
    # The coordinator's saved config predates arm-specific special-token binding.
    # Read BOS from the pinned native bank, as the original workflow does.
    scientific = {**scientific, "bos_token_id": bank_manifest["bos_token_id"]}
    for name in ("prompts.npy", "targets.npy", "windows.npy", "documents.jsonl", "manifest.json"):
        bank.get("validation/" + name)
    baseline = parent.json("seed_42/gen_00/metrics.json")
    base_checkpoint = parent.checkpoint("seed_42/gen_00", restore_weights=include_weights)
    cases, children = [], {}
    for arm in c["arms"]:
        child = children[arm] = Snapshot(snapshots[arm], source_root / arm)
        child_config = child.json("config.json")
        for cycle in c["cycles"]:
            prefix = f"seed_42/gen_{cycle:02d}"
            previous = f"seed_42/gen_{cycle - 1:02d}"
            stage_training = child.json(prefix + "/training.json")
            if cycle == 1:
                checkpoint, state, weights = base_checkpoint, None, baseline["weights_sha256"]
            else:
                prior = child.json(previous + "/training.json")
                seal = child.json(previous + "/stage_completion.json")
                checkpoint = child.checkpoint(previous, restore_weights=include_weights)
                state_relative = previous + "/optimizer_state/cycle.pt"
                state = child.get(state_relative) if include_weights else child.root / state_relative
                weights = prior["selected_weights_sha256"]
                declared = {**{"checkpoint/" + k: v for k, v in prior["checkpoint_files"].items()}, **prior["state_files"]}
                if include_weights:
                    verify_files(child.root / previous, declared)
                elif any(child.index[previous + "/" + k]["sha256"] != value for k, value in declared.items()):
                    raise RuntimeError("Indexed source weights/optimizer differ from their training seal")
                for relative in ["checkpoint/model.safetensors", "optimizer_state/cycle.pt"]:
                    if include_weights:
                        verify_files(child.root / previous, {relative: seal["files"][relative]})
                    elif child.index[previous + "/" + relative]["sha256"] != seal["files"][relative]:
                        raise RuntimeError("Indexed source checkpoint differs from completed stage")
            if stage_training["initial_weights_sha256"] != weights:
                raise RuntimeError("Restored checkpoint is not the cycle initializer")
            state_hash = (sha256(state) if include_weights else child.index[previous + "/optimizer_state/cycle.pt"]["sha256"]) if state else None
            if stage_training["optimizer_parent_sha256"] != state_hash:
                raise RuntimeError("Restored optimizer is not the cycle parent")
            collection = f"cycle_{cycle:02d}"
            for name in ("prompts.npy", "targets.npy", "manifest.json"):
                bank.get(collection + "/" + name)
            prompts, real = (bank.array(collection + "/" + n) for n in ("prompts.npy", "targets.npy"))
            tokens, attention, scoring = (child.array(prefix + "/training_input/" + n)
                                         for n in ("tokens.npy", "attention.npy", "loss_mask.npy"))
            if not np.all(tokens[:, 0] == scientific["bos_token_id"]):
                raise RuntimeError("Historical BOS differs from the pinned native bank")
            manifest = child.json(prefix + "/training_input/manifest.json")
            verify_files(child.root / prefix / "training_input", manifest["files"])
            generation = child.json(f"synthetic_for_gen_{cycle:02d}/manifest.json")
            generated = child.array(f"synthetic_for_gen_{cycle:02d}/tokens.npy")
            verify_files(child.root / f"synthetic_for_gen_{cycle:02d}", {"tokens.npy": generation["files"]["tokens.npy"]})
            if arm == "mixed99":
                selection, real_count = select_rows(len(prompts), .99, scientific["seed"], cycle)
                mixture = child.json(prefix + "/mixture/manifest.json")
                for name in ("selection.json", "permutation.npy", "synthetic_indices.npy", "source_labels.npy", "row_indices.npy", "targets.npy"):
                    child.get(prefix + "/mixture/" + name)
                verify_files(child.root / prefix / "mixture", mixture["files"])
                for name, expected in selection.items():
                    if not np.array_equal(child.array(prefix + "/mixture/" + name + ".npy"), expected):
                        raise RuntimeError("Historical mixture selection differs")
                indices = selection["synthetic_indices"]
                if real_count != 100 or len(indices) != 9900 or mixture["generator_weights_sha256"] != weights:
                    raise RuntimeError("Historical 99% source counts/generator differ")
                if sha256(child.root / prefix / "mixture/manifest.json") != manifest["lineage"]["mixture_manifest_sha256"]:
                    raise RuntimeError("Historical mixture seal differs")
            else:
                indices = np.arange(len(prompts))
            provenance = generation["provenance"]
            generation_prompts = np.concatenate([np.full((len(indices), 1), scientific["bos_token_id"], dtype=np.int32), prompts[indices]], axis=1)
            if (provenance["parent_weights_sha256"] != weights or provenance["prompt_tokens_sha256"] != token_hash(generation_prompts)
                    or generation["continuation_tokens_sha256"] != token_hash(generated)
                    or (provenance["num_beams"], provenance["new_tokens"], provenance["repetition_penalty"]) != (5, 64, 1.0)):
                raise RuntimeError("Stored generation provenance differs")
            if sha256(child.root / f"synthetic_for_gen_{cycle:02d}/manifest.json") != manifest["lineage"]["generation_manifest_sha256"]:
                raise RuntimeError("Historical generation seal differs")
            check_input(tokens, attention, scoring, prompts, real, generated, indices, manifest, weights,
                        bank_hash, sha256(bank.root / collection / "manifest.json"))
            cases.append({"arm": arm, "cycle": cycle, "checkpoint": str(checkpoint), "optimizer": str(state) if state else None,
                          "weights": weights, "child": child, "prefix": prefix, "tokens": tokens,
                          "attention": attention, "scoring": scoring, "prompts": prompts, "real": real,
                          "synthetic_rows": np.sort(indices), "config": child_config})
    receipt = {"parent": parent.receipt(), "bank": bank.receipt(), "checkpoint_bytes_restored": include_weights,
               "children": {k: v.receipt() for k, v in children.items()}, "scope": "selected_checkpoints_optimization_states_and_cycle_inputs"}
    atomic_json(root / "source_recovery.json", receipt)
    return cases, bank, scientific


def prepare_input_cache(c, destination):
    """Reuse checksum-verified local metadata; bulk model/state bytes remain remote.

    This cache is an optimization only: clean checkouts still restore pinned files.
    Every reused byte is checked against the same immutable index.
    """
    from ..common import PROJECT, STATE
    destination = Path(destination)
    reference_root = STATE / "smollm-analysis-recovery"
    source = c["source_parent"]
    candidates = [("parent", source)]
    parent = Snapshot(source, destination / ".sources/parent")
    scientific = parent.json("config.json")
    snapshots = parent.json("child_snapshots.json")
    candidates += [(name, snapshots[name]) for name in c["arms"]]
    candidates.append(("bank", scientific["prepared_bank"]))
    for name, ref in candidates:
        target = Snapshot(ref, destination / ".sources" / name)
        original = PROJECT / "data/smollm2_completion64_v2" if name == "bank" else reference_root / ref["run_id"]
        for relative, meta in target.index.items():
            path = original / relative
            # Training weights and states are deliberately not copied to the laptop/cache bundle.
            allowed = relative.endswith(".json") or (name == "bank" and relative.startswith(("validation/", "cycle_01/", "cycle_05/", "cycle_10/")))
            if not allowed or not path.is_file() or sha256(path) != meta["sha256"]:
                continue
            output = target.root / relative
            if not output.is_file() or sha256(output) != meta["sha256"]:
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, output)
    return restore_sources(destination, c, include_weights=False)
