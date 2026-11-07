"""Pinned, shuffled FineWeb documents and disjoint token-budgeted collections."""
from __future__ import annotations

import hashlib
import json
import random
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from .common import atomic_json, digest, read_json, sha256, verify_files
from .data import save_array


def data_contract(c):
    keys = ("dataset", "dataset_revision", "dataset_config", "dataset_files", "model", "model_revision",
            "seed", "shuffle_buffer", "max_sequence_length", "source_tokens_per_cycle", "generations",
            "validation_documents", "test_documents")
    return digest({key: c[key] for key in keys})


def shuffled_documents(c, coverage):
    from datasets import load_dataset
    files = list(c["dataset_files"])
    random.Random(c["seed"]).shuffle(files)
    coverage.update(shard_order=files, read_rows_by_file={})

    def source():
        for name in files:
            url = f"https://huggingface.co/datasets/{c['dataset']}/resolve/{c['dataset_revision']}/{name}"
            dataset = load_dataset("parquet", data_files=url, split="train", streaming=True)
            for row, document in enumerate(dataset):
                coverage["read_rows_by_file"][name] = row + 1
                yield {**document, "source_file": name, "source_row": row}

    rng = random.Random(c["seed"] + 1)
    iterator = iter(source())
    buffer = []
    for document in iterator:
        if len(buffer) < c["shuffle_buffer"]:
            buffer.append(document)
        else:
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = document
    rng.shuffle(buffer)
    yield from buffer


def document_keys(document):
    text = document.get("text", "")
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    if not normalized:
        return set()
    keys = {"text:" + hashlib.sha256(normalized.encode()).hexdigest()}
    if document.get("id"):
        keys.add("id:" + str(document["id"]))
    if document.get("url"):
        try:
            url = urlsplit(document["url"])
            normalized_url = urlunsplit((url.scheme.lower(), url.netloc.lower(), url.path.rstrip("/"), url.query, ""))
        except ValueError:
            normalized_url = document["url"].split("#", 1)[0]
        keys.add("url:" + normalized_url)
    return keys


def passage_hashes(tokens):
    values = np.asarray(tokens, dtype="<i4")
    return {hashlib.blake2b(values[i:i + 64].tobytes(), digest_size=16).digest()
            for i in range(max(0, len(values) - 63))}


def encode_sequences(tokens, tokenizer, maximum, *, genuine_end):
    # Reserving two slots keeps every chunk policy independent of document length.
    payload = maximum - 2
    sequences = []
    for offset in range(0, len(tokens), payload):
        body = list(tokens[offset:offset + payload])
        ids = [tokenizer.bos_token_id] + body
        if genuine_end and offset + payload >= len(tokens):
            ids.append(tokenizer.eos_token_id)
        sequences.append(ids)
    return sequences


def load_collection(root, name):
    folder = Path(root) / "data" / name
    manifest = read_json(folder / "manifest.json")
    if not manifest:
        raise RuntimeError(f"Incomplete FineWeb collection: {name}")
    verify_files(folder, manifest["files"])
    tokens = np.load(folder / "tokens.npy", allow_pickle=False, mmap_mode="r")
    masks = np.load(folder / "masks.npy", allow_pickle=False, mmap_mode="r")
    if (tokens.shape != masks.shape or tokens.ndim != 2 or
            np.any((masks != 0) & (masks != 1)) or np.any(masks[:, 1:] > masks[:, :-1]) or
            np.any(masks.sum(1) < 2) or int(masks[:, 1:].sum()) != manifest["scored_tokens"]):
        raise RuntimeError("Invalid sequence masks or token accounting")
    return tokens, masks, manifest


def prepare_fresh(root, c, tokenizer, documents=None):
    root = Path(root)
    folder = root / "data"
    contract = data_contract(c)
    previous = read_json(folder / "preparation.json")
    if previous and previous["contract"] != contract:
        raise RuntimeError("FineWeb preparation configuration changed")
    old = read_json(folder / "manifest.json")
    if old:
        if old["contract"] != contract:
            raise RuntimeError("FineWeb data contract changed")
        verify_files(folder, old["files"])
        return old
    atomic_json(folder / "preparation.json", {"contract": contract})
    started = time.perf_counter()
    coverage = {"sampling": "Seeded shard permutation and bounded document shuffle; reproducible shuffled-stream sample."}
    iterator = iter(documents if documents is not None else shuffled_documents(c, coverage))
    seen, spans = set(), set()
    rejected = {"empty": 0, "duplicate_identity": 0, "shared_64_token_passage": 0}
    names = [("validation", c["validation_documents"], None), ("test", c["test_documents"], None)]
    names += [(f"cycle_{cycle:02d}", None, c["source_tokens_per_cycle"]) for cycle in range(1, c["generations"] + 1)]
    collections = {}
    for name, document_limit, token_limit in names:
        sequences, records, current_spans = [], [], set()
        source_count = 0
        while (len(records) < document_limit if document_limit is not None else source_count < token_limit):
            try:
                document = next(iterator)
            except StopIteration:
                raise RuntimeError("FineWeb stream exhausted before the disjoint collections were complete") from None
            keys = document_keys(document)
            if not keys:
                rejected["empty"] += 1
                continue
            if seen.intersection(keys):
                rejected["duplicate_identity"] += 1
                continue
            seen.update(keys)  # A rejected/trimmed identity cannot later re-enter the experiment.
            tokens = tokenizer.encode(document["text"], add_special_tokens=False)
            if not tokens:
                rejected["empty"] += 1
                continue
            original_length = len(tokens)
            if token_limit is not None:
                tokens = tokens[:token_limit - source_count]
            fingerprints = passage_hashes(tokens)
            if spans.intersection(fingerprints):
                rejected["shared_64_token_passage"] += 1
                continue
            current_spans.update(fingerprints)
            sequences.extend(encode_sequences(tokens, tokenizer, c["max_sequence_length"],
                                              genuine_end=len(tokens) == original_length))
            record = {key: document.get(key) for key in ("id", "url", "dump", "file_path", "source_file", "source_row", "text")}
            record.update(identity_keys=sorted(keys), source_token_count=original_length,
                          used_source_tokens=len(tokens), trimmed=len(tokens) < original_length,
                          used_tokens_sha256=hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest())
            records.append(record)
            source_count += len(tokens)
        spans.update(current_spans)
        tokens = np.full((len(sequences), c["max_sequence_length"]), tokenizer.pad_token_id, dtype=np.int32)
        masks = np.zeros_like(tokens)
        for index, ids in enumerate(sequences):
            tokens[index, :len(ids)] = ids
            masks[index, :len(ids)] = 1
        destination = folder / name
        save_array(destination / "tokens.npy", tokens)
        save_array(destination / "masks.npy", masks)
        temporary = destination / "documents.jsonl.partial"
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        temporary.replace(destination / "documents.jsonl")
        manifest = {"collection": name, "documents": len(records), "sequences": len(sequences),
                    "source_text_tokens": source_count, "non_padding_tokens": int(masks.sum()),
                    "special_tokens": int(masks.sum()) - source_count, "scored_tokens": int(masks[:, 1:].sum()),
                    "files": {file: sha256(destination / file) for file in ("tokens.npy", "masks.npy", "documents.jsonl")}}
        atomic_json(destination / "manifest.json", manifest)
        collections[name] = manifest
    files = {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and not p.name.endswith(".partial")}
    for name in collections:
        files[f"{name}/manifest.json"] = sha256(folder / name / "manifest.json")
    result = {"contract": contract, "collections": collections, "files": files, "coverage": coverage,
              "rejected_documents": rejected, "seconds": time.perf_counter() - started,
              "verification": "Disjoint accepted document identities and exact sliding 64-source-token spans across collections; finite 128-bit passage fingerprints.",
              "token_policy": "Source quotas exclude special/padding tokens; BOS per sequence, EOS at genuine document end, no EOS at trimmed end; right padding."}
    atomic_json(folder / "manifest.json", result)
    return result
