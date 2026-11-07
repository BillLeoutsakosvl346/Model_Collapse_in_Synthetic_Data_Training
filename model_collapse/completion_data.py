"""Immutable FineWeb prompt/continuation pairs shared by both comparison arms."""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from filelock import FileLock

from .common import PROJECT, STATE, atomic_json, digest, read_json, sha256, verify_files
from .data import save_array
from .fresh_data import document_keys, passage_hashes, shuffled_documents


def bank_contract(c):
    keys = ("model", "model_revision", "dataset", "dataset_revision", "dataset_config", "dataset_files",
            "seed", "shuffle_buffer", "full_cycles", "full_train_pairs", "corpus_bank")
    return digest({"settings": {k: c[k] for k in keys}, "prompt_source_tokens": 64,
                   "continuation_tokens": 64, "bos": True, "implementation": sha256(__file__)})


def source_bank(c):
    """Read only the pinned source files needed for selection and secondary tests."""
    from huggingface_hub import hf_hub_download
    ref = c["corpus_bank"]
    local = PROJECT / "data/corpus_bank/cross_corpus_v1"
    if not (local / "manifest.json").exists():
        local = STATE / "completion64-source-bank"
        local.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(ref["repo"], ref["path"] + "/manifest.json", repo_type="dataset", revision=ref["revision"])
        shutil.copyfile(path, local / "manifest.json")
    if sha256(local / "manifest.json") != ref["manifest_sha256"]:
        raise RuntimeError("Source corpus bank identity differs")
    manifest = read_json(local / "manifest.json")
    required = [n for n in manifest["files"] if n.endswith("documents.jsonl") and "/train/" not in n]
    required += [f"{name}/test/{file}" for name in c["secondary_corpora"] for file in ("tokens.npy", "masks.npy")]
    for name in required:
        path = local / name
        if not path.exists():
            downloaded = hf_hub_download(ref["repo"], ref["path"] + "/" + name, repo_type="dataset", revision=ref["revision"])
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(downloaded, path)
        verify_files(local, {name: manifest["files"][name]})
    return local, manifest


def records(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _write_collection(destination, docs, windows, prompt, target, permutation):
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {"prompts.npy": np.asarray(prompt, dtype=np.int32)[permutation],
              "targets.npy": np.asarray(target, dtype=np.int32)[permutation],
              "windows.npy": np.asarray(windows, dtype=np.int64)[permutation],
              "permutation.npy": np.asarray(permutation, dtype=np.int64)}
    for name, values in arrays.items():
        save_array(destination / name, values)
    temporary = destination / "documents.jsonl.partial"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in docs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(destination / "documents.jsonl")
    result = {"pairs": len(prompt), "documents": len(docs), "source_tokens": len(prompt) * 128,
              "target_tokens": len(prompt) * 64,
              "files": {n: sha256(destination / n) for n in (*arrays, "documents.jsonl")}}
    atomic_json(destination / "manifest.json", result)
    return result


def _pairs(docs, limit=None):
    selected, windows, prompt, target = [], [], [], []
    for row, ids in docs:
        count = len(ids) // 128
        if not count:
            continue
        doc_index = len(selected)
        selected.append(row)
        for offset in range(0, count * 128, 128):
            windows.append([doc_index, offset])
            prompt.append(ids[offset:offset + 64])
            target.append(ids[offset + 64:offset + 128])
            if limit and len(prompt) == limit:
                return selected, windows, prompt, target
    return selected, windows, prompt, target


def assemble(prompts, targets, bos):
    """BOS + 64 visible prompt tokens + 64 scored continuation tokens."""
    prompts, targets = np.asarray(prompts), np.asarray(targets)
    if prompts.shape != targets.shape or prompts.ndim != 2 or prompts.shape[1] != 64 or not len(prompts):
        raise ValueError("Completion training requires equally sized nonempty 64-token pairs")
    tokens = np.concatenate([np.full((len(prompts), 1), bos, dtype=np.int32), prompts, targets], axis=1).astype(np.int32)
    attention = np.ones_like(tokens)
    loss_mask = np.zeros_like(tokens)
    loss_mask[:, 65:] = 1
    return tokens, attention, loss_mask


def load_pairs(bank, collection, limit=None):
    folder = Path(bank) / collection
    manifest = read_json(folder / "manifest.json")
    verify_files(folder, manifest["files"])
    prompt = np.load(folder / "prompts.npy", mmap_mode="r", allow_pickle=False)
    target = np.load(folder / "targets.npy", mmap_mode="r", allow_pickle=False)
    if prompt.shape != target.shape or prompt.shape != (manifest["pairs"], 64):
        raise RuntimeError("Pair array shape differs from its manifest")
    return prompt[:limit], target[:limit], manifest


def prepare_bank(root, c, tokenizer, *, source=None, iterator=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    contract = bank_contract(c)
    with FileLock(str(root / "prepare.lock"), timeout=0):
        prior = read_json(root / "preparation.json")
        if prior and prior["contract"] != contract:
            raise RuntimeError("Completion bank configuration/source changed, choose a new output directory")
        complete = read_json(root / "manifest.json")
        if complete:
            verify_files(root, complete["files"])
            return complete
        atomic_json(root / "preparation.json", {"contract": contract})
        source = Path(source) if source is not None else source_bank(c)[0]
        seen, spans, collections = set(), set(), {}
        for path in sorted(source.glob("*/*/documents.jsonl")):
            if path.parent.name not in ("validation", "test"):
                continue
            for row in records(path):
                seen.update(document_keys(row))
                ids = tokenizer.encode(row["text"], add_special_tokens=False)
                start = row.get("window_start", 0)
                spans.update(passage_hashes(ids[start:start + row["used_source_tokens"]]))
        for split in ("validation", "test"):
            destination = root / split
            old = read_json(destination / "manifest.json")
            if old:
                verify_files(destination, old["files"])
                collections[split] = old
                continue
            values = ((row, tokenizer.encode(row["text"], add_special_tokens=False)[:row["used_source_tokens"]])
                      for row in records(source / "fineweb_edu" / split / "documents.jsonl"))
            docs, windows, prompt, target = _pairs(values)
            collections[split] = _write_collection(destination, docs, windows, prompt, target, np.arange(len(prompt)))
        coverage, rejected = {}, {"identity": 0, "passage": 0, "short": 0}
        stream = iter(iterator) if iterator is not None else shuffled_documents(c, coverage)
        try:
            for cycle in range(1, c["full_cycles"] + 1):
                name = f"cycle_{cycle:02d}"
                destination = root / name
                old = read_json(destination / "manifest.json")
                if old:
                    verify_files(destination, old["files"])
                    for row in records(destination / "documents.jsonl"):
                        seen.update(document_keys(row))
                        spans.update(passage_hashes(tokenizer.encode(row["text"], add_special_tokens=False)))
                    collections[name] = old
                    continue
                def eligible():
                    for row in stream:
                        keys = document_keys(row)
                        if not keys or seen.intersection(keys):
                            rejected["identity"] += 1
                            continue
                        seen.update(keys)
                        ids = tokenizer.encode(row["text"], add_special_tokens=False)
                        fingerprints = passage_hashes(ids)
                        if spans.intersection(fingerprints):
                            rejected["passage"] += 1
                            continue
                        if len(ids) < 128:
                            rejected["short"] += 1
                            continue
                        spans.update(fingerprints)
                        yield {k: row.get(k) for k in ("id", "url", "text", "source_file", "source_row", "dump", "file_path")}, ids
                docs, windows, prompt, target = _pairs(eligible(), c["full_train_pairs"])
                if len(prompt) != c["full_train_pairs"]:
                    raise RuntimeError("FineWeb stream exhausted before all completion pairs were selected")
                order = np.random.default_rng(c["seed"] + cycle).permutation(len(prompt))
                collections[name] = _write_collection(destination, docs, windows, prompt, target, order)
                print(f"Prepared {name}: {len(prompt)} pairs from {len(docs)} documents", flush=True)
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
        for name in c["secondary_corpora"]:
            for file in ("tokens.npy", "masks.npy"):
                destination = root / "secondary" / name / file
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name / "test" / file, destination)
        val_windows = np.load(root / "validation/windows.npy")
        by_doc = {}
        for index, (doc, offset) in enumerate(val_windows):
            by_doc.setdefault(int(doc), []).append(index)
        rng = random.Random(c["seed"])
        documents = rng.sample(sorted(by_doc), min(256, len(by_doc)))
        probe_indices = np.asarray([rng.choice(by_doc[doc]) for doc in documents], dtype=np.int64)
        save_array(root / "probe_indices.npy", probe_indices)
        examples = []
        for name, meta in collections.items():
            prompt = np.load(root / name / "prompts.npy", mmap_mode="r")
            target = np.load(root / name / "targets.npy", mmap_mode="r")
            selected_indices = random.Random(c["seed"] + len(examples)).sample(range(meta["pairs"]), min(2, meta["pairs"]))
            for index in selected_indices:
                examples.append({"collection": name, "index": index,
                                 "prompt": tokenizer.decode(prompt[index]), "real_continuation": tokenizer.decode(target[index])})
        atomic_json(root / "examples.json", examples)
        files = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*"))
                 if p.is_file() and p.name not in ("manifest.json", "prepare.lock") and not p.name.endswith(".partial")}
        files.update({f"{name}/manifest.json": sha256(root / name / "manifest.json") for name in collections})
        result = {"contract": contract, "collections": collections, "files": files,
                  "bos_token_id": tokenizer.bos_token_id, "coverage": coverage, "rejected": rejected,
                  "verification_scope": "Document IDs/normalized URLs/text and exact sliding 64-token source passages separated across collections. Human authorship and pretrained exposure unaudited."}
        atomic_json(root / "manifest.json", result)
        return result


def publish_bank(root, c, repos):
    from huggingface_hub import HfApi, get_token
    from .hub import resolve_snapshot, retry_transfer
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    verify_files(root, manifest["files"])
    name = "completion64-bank-" + manifest["contract"][:16]
    files = {p: {"sha256": h, "bytes": (root / p).stat().st_size, "repo_type": "dataset"}
             for p, h in {**manifest["files"], "manifest.json": sha256(root / "manifest.json")}.items()}
    atomic_json(root / "artifact_index.json", {"run_id": name, "repos": repos, "files": files})
    api = HfApi(token=os.environ.get("HF_TOKEN") or get_token())
    if not api.repo_info(repos["dataset"], repo_type="dataset").private:
        raise RuntimeError("Completion bank repository must be private")
    retry_transfer(api.upload_folder, repo_id=repos["dataset"], repo_type="dataset", folder_path=root,
                   path_in_repo=name, allow_patterns=[*files, "artifact_index.json"], commit_message="Freeze matched FineWeb completion bank")
    ref = {"run_id": name, "repos": repos, "revisions": resolve_snapshot(repos),
           "artifact_index_sha256": sha256(root / "artifact_index.json"), "manifest_sha256": sha256(root / "manifest.json"),
           "contract": bank_contract(c)}
    atomic_json(STATE / "completion64_bank.json", ref)
    return ref


def restore_bank(ref, destination):
    from .hub import fetch
    destination = Path(destination)
    if not (destination / "manifest.json").exists():
        receipt = fetch(ref["run_id"], ref["repos"], destination, revisions=ref["revisions"])
        if receipt["artifact_index_sha256"] != ref["artifact_index_sha256"]:
            raise RuntimeError("Completion bank backup index differs")
    if sha256(destination / "manifest.json") != ref["manifest_sha256"]:
        raise RuntimeError("Completion bank content identity differs")
    manifest = read_json(destination / "manifest.json")
    verify_files(destination, manifest["files"])
    return manifest


def audit_bank(root, tokenizer):
    """Reconstruct every selected token pair independently from saved text/offsets."""
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    verify_files(root, manifest["files"])
    seen, spans, counts = set(), set(), {}
    for name, meta in manifest["collections"].items():
        docs = list(records(root / name / "documents.jsonl"))
        encoded, current = [], set()
        for row in docs:
            keys = document_keys(row)
            if not keys or seen.intersection(keys):
                raise RuntimeError("Document identity crosses completion collections")
            seen.update(keys)
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            fingerprints = passage_hashes(ids)
            if spans.intersection(fingerprints):
                raise RuntimeError("Exact source passage crosses completion collections")
            current.update(fingerprints)
            encoded.append(ids)
        spans.update(current)
        prompt, target, _ = load_pairs(root, name)
        windows = np.load(root / name / "windows.npy", allow_pickle=False)
        order = np.load(root / name / "permutation.npy", allow_pickle=False)
        if not np.array_equal(np.sort(order), np.arange(meta["pairs"])):
            raise RuntimeError("Saved pair permutation is not bijective")
        for i, (doc, offset) in enumerate(windows):
            original = encoded[int(doc)][int(offset):int(offset) + 128]
            if (offset % 128 or len(original) != 128 or
                    not np.array_equal(np.concatenate([prompt[i], target[i]]), original)):
                raise RuntimeError("Pair reconstruction differs from saved text")
        counts[name] = {k: meta[k] for k in ("pairs", "target_tokens", "documents")}
    indices = np.load(root / "probe_indices.npy", allow_pickle=False)
    windows = np.load(root / "validation/windows.npy", allow_pickle=False)
    if len(set(windows[indices, 0])) != len(indices):
        raise RuntimeError("Probe must sample distinct validation documents")
    return {"all_file_hashes_verified": True, "all_pairs_reconstructed": True,
            "disjoint_documents_and_passages": True, "probe_document_stratification": True, "counts": counts}
