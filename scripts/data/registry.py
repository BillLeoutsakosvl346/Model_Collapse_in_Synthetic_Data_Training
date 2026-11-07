"""Frozen, document-disjoint source collections for evaluation and later mixing."""
from __future__ import annotations

import hashlib
import html
import json
import random
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
from filelock import FileLock

from model_collapse.common import atomic_json, digest, read_json, sha256, verify_files
from model_collapse.data import save_array
from model_collapse.fresh_data import document_keys, encode_sequences, passage_hashes
from scripts.data.readers import documents, stable_seed


def identity_keys(record):
    keys = document_keys(record)
    url = urlsplit(record.get("url") or "")
    if url.hostname in ("en.wikipedia.org", "www.en.wikipedia.org") and url.path.startswith("/wiki/"):
        title = unicodedata.normalize("NFKC", unquote(url.path[6:]).replace("_", " ")).casefold()
        keys.add("wikipedia-title:" + title)
    return keys


def token_window(record, tokenizer, c, name, role):
    full = tokenizer.encode(record["text"], add_special_tokens=False)
    limit = record.get("source_token_limit", len(full))
    if limit > len(full):
        raise RuntimeError("Recorded token quota exceeds the document")
    tokens = full[:limit]
    start = 0
    if name != "fineweb_edu" and len(tokens) > c["maximum_document_tokens"]:
        rng = random.Random(stable_seed(c["seed"], name, role, record["id"]))
        start = rng.randrange(len(tokens) - c["maximum_document_tokens"] + 1)
        tokens = tokens[start:start + c["maximum_document_tokens"]]
    genuine_end = bool(record.get("source_genuine_end", True) and start + len(tokens) == len(full))
    return tokens, start, len(full), genuine_end


def protected_fineweb(c, tokenizer, project):
    keys, spans = set(), set()
    coverage = {}
    for row in documents("fineweb_edu", "train", c, coverage, project):
        tokens = tokenizer.encode(row["text"], add_special_tokens=False)[:row["source_token_limit"]]
        keys.update(identity_keys(row))
        spans.update(passage_hashes(tokens))
    return keys, spans


def collection(root, name, role, c, tokenizer, seen, spans, protected, project, reader=documents):
    destination = root / name / role
    complete = read_json(destination / "manifest.json")
    if complete:
        verify_files(destination, complete["files"])
        current_spans = set()
        for line in (destination / "documents.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            keys = identity_keys(row)
            ids = tokenizer.encode(row["text"], add_special_tokens=False)[row["window_start"]:row["window_start"] + row["used_source_tokens"]]
            fingerprints = passage_hashes(ids)
            if seen.intersection(keys) or spans.intersection(fingerprints):
                raise RuntimeError("Completed corpus collection conflicts with earlier selections")
            if role != "train" and (protected[0].intersection(keys) or protected[1].intersection(fingerprints)):
                raise RuntimeError("Evaluation collection overlaps historical FineWeb training")
            seen.update(keys)
            current_spans.update(fingerprints)
        spans.update(current_spans)
        return complete
    target = None if name == "fineweb_edu" and role != "train" else c[f"{role}_source_tokens"]
    coverage = {"source": c["sources"][name], "role": role,
                "asset_metadata_cache": str((root / "source_objects" / name / role).resolve())}
    rejected = {"duplicate_identity": 0, "cross_collection_passage": 0, "historical_training_overlap": 0, "empty": 0}
    selected, sequences, current_spans, source_count = [], [], set(), 0
    iterator = reader(name, role, c, coverage, project)
    try:
        for row in iterator:
            keys = identity_keys(row)
            if not keys:
                rejected["empty"] += 1
                continue
            if keys.intersection(seen):
                rejected["duplicate_identity"] += 1
                continue
            ids, offset, full_length, genuine_end = token_window(row, tokenizer, c, name, role)
            if target is not None and len(ids) > target - source_count:
                ids = ids[:target - source_count]
                genuine_end = False
            if not ids:
                rejected["empty"] += 1
                continue
            fingerprints = passage_hashes(ids)
            if role != "train" and (keys.intersection(protected[0]) or fingerprints.intersection(protected[1])):
                rejected["historical_training_overlap"] += 1
                continue
            if fingerprints.intersection(spans):
                rejected["cross_collection_passage"] += 1
                continue
            seen.update(keys)
            current_spans.update(fingerprints)
            sequences.extend(encode_sequences(ids, tokenizer, c["max_sequence_length"], genuine_end=genuine_end))
            original_hash = row.get("used_tokens_sha256") if name == "fineweb_edu" else None
            record = {**row, "corpus": name, "role": role, "window_start": offset,
                      "full_source_tokens": full_length, "used_source_tokens": len(ids), "genuine_document_end": genuine_end,
                      "historical_used_tokens_sha256": original_hash,
                      "selected_tokens_sha256": hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()}
            selected.append(record)
            source_count += len(ids)
            if target is not None and source_count == target:
                break
    finally:
        iterator.close()
    if not selected or target is not None and source_count != target:
        raise RuntimeError(f"Insufficient eligible {name}/{role} text: {source_count} of {target} source tokens")
    spans.update(current_spans)
    tokens = np.full((len(sequences), c["max_sequence_length"]), tokenizer.pad_token_id, dtype=np.int32)
    masks = np.zeros_like(tokens)
    for i, ids in enumerate(sequences):
        tokens[i, :len(ids)] = ids
        masks[i, :len(ids)] = 1
    save_array(destination / "tokens.npy", tokens)
    save_array(destination / "masks.npy", masks)
    path = destination / "documents.jsonl"
    temporary = path.with_suffix(".jsonl.partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    complete = {"corpus": name, "role": role, "documents": len(selected), "sequences": len(sequences),
                "source_text_tokens": source_count, "special_tokens": int(masks.sum()) - source_count,
                "non_padding_tokens": int(masks.sum()), "scored_tokens": int(masks[:, 1:].sum()),
                "source_coverage": coverage, "rejected_documents": rejected,
                "files": {n: sha256(destination / n) for n in ("tokens.npy", "masks.npy", "documents.jsonl")}}
    atomic_json(destination / "manifest.json", complete)
    print(f"Prepared {name}/{role}: {len(selected)} documents, {source_count:,} source tokens", flush=True)
    return complete


def prepare(root, c, tokenizer, project, *, reader=documents, protected=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    contract = digest({"configuration": c, "implementation": {n: sha256(Path(__file__).with_name(n))
                       for n in ("registry.py", "readers.py")},
                       "sequence_helpers": sha256(Path(__file__).parents[2] / "model_collapse/fresh_data.py")})
    with FileLock(str(root / "prepare.lock"), timeout=0):
        previous = read_json(root / "preparation.json")
        if previous and previous["contract"] != contract:
            raise RuntimeError("Corpus bank configuration or preparation implementation changed; choose a new output")
        complete = read_json(root / "manifest.json")
        if complete:
            verify_files(root, complete["files"])
            return complete
        atomic_json(root / "preparation.json", {"contract": contract, "configuration": c})
        if protected is None:
            print("Indexing original FineWeb training identities and exact token passages", flush=True)
            protected = protected_fineweb(c, tokenizer, project)
        seen, spans, collections = set(), set(), {}
        for role in ("validation", "test", "train"):
            for name in c["sources"]:
                collections[f"{name}/{role}"] = collection(root, name, role, c, tokenizer, seen, spans, protected, project, reader)
        files = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*"))
                 if p.is_file() and p.name not in ("prepare.lock", "manifest.json") and ".partial" not in p.name}
        for key in collections:
            files[f"{key}/manifest.json"] = sha256(root / key / "manifest.json")
        complete = {"schema_version": 1, "contract": contract, "configuration": c, "collections": collections,
                    "files": files, "overlap_scope": "Document ID, canonical Wikipedia title, normalized URL/text and exact sliding 64-token passages across all bank collections. Evaluation also excludes all recorded FineWeb training identities/passages.",
                    "sampling": "Pinned source streams, seeded file/document shuffles and stable native-split partitions. Long source documents contribute a deterministic contiguous window up to the document token cap. Finite read coverage is recorded.",
                    "token_policy": "Original pretrained OPT tokenizer; BOS per sequence; up to 510 source tokens; EOS at genuine document endings only; right padding excluded from attention/targets."}
        atomic_json(root / "manifest.json", complete)
        return complete


def preview(root, tokenizer):
    root = Path(root)
    m = read_json(root / "manifest.json")
    rows = []
    for key, info in m["collections"].items():
        records = [json.loads(line) for line in (root / key / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
        rng = random.Random(stable_seed(m["configuration"]["seed"], "preview", key))
        for record in rng.sample(records, min(3, len(records))):
            ids = tokenizer.encode(record["text"], add_special_tokens=False)
            start = record["window_start"]
            # Small excerpts keep the review shareable without copying entire source pages.
            excerpt = " ".join(tokenizer.decode(ids[start:start + min(100, record["used_source_tokens"]) ]).split()[:22])
            rows.append({"collection": key, "id": record["id"], "url": record.get("url"),
                         "source_tokens": record["used_source_tokens"], "excerpt": excerpt})
    atomic_json(root / "examples.json", rows)
    body = ["<!doctype html><html><head><meta charset='utf-8'><title>Corpus bank preview</title>",
            "<style>body{font:17px/1.6 system-ui;max-width:1000px;margin:45px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}article{background:#f6f7fa;padding:18px;margin:18px 0;border-radius:10px}small{color:#555}</style></head><body>",
            "<h1>Cross-corpus dataset review</h1><p>Fixed seeded examples from verified selected documents. Original texts and arrays remain in the private corpus bank.</p>",
            "<table><tr><th>Collection</th><th>Documents</th><th>Source tokens</th><th>Scored targets</th></tr>"]
    for key, info in m["collections"].items():
        body.append(f"<tr><td>{html.escape(key)}</td><td>{info['documents']:,}</td><td>{info['source_text_tokens']:,}</td><td>{info['scored_tokens']:,}</td></tr>")
    body.append("</table><h2>Examples</h2>")
    for row in rows:
        body.append(f"<article><strong>{html.escape(row['collection'])}</strong><p>{html.escape(row['excerpt'])}</p><small>{html.escape(row['id'])} · {row['source_tokens']:,} selected source tokens</small></article>")
    body.append("</body></html>")
    (root / "preview.html").write_text("\n".join(body), encoding="utf-8")
    return rows
