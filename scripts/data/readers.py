"""Pinned source adapters. Original documents remain private local artifacts."""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import random
import re
from pathlib import Path
from urllib.parse import quote

import requests
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from model_collapse.common import atomic_json, digest, read_json, sha256


def stable_seed(seed, *parts):
    return int(digest([seed, *parts])[:16], 16)


def shuffled(iterator, size, seed):
    rng = random.Random(seed)
    buffer = []
    try:
        for document in iterator:
            if len(buffer) < size:
                buffer.append(document)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = document
        rng.shuffle(buffer)
        yield from buffer
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()


def hf_url(source, name):
    return f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{name}"


def parquet_rows(source, names, coverage):
    for name in names:
        coverage.setdefault("files", {})[name] = {"rows_read": 0}
        dataset = load_dataset("parquet", data_files=hf_url(source, name), split="train", streaming=True)
        for i, row in enumerate(dataset):
            coverage["files"][name]["rows_read"] = i + 1
            yield {**row, "source_file": name, "source_row": i}


ARTICLE = re.compile(r"^\s*=\s+([^=].*?)\s+=\s*$")


def wiki_documents(rows):
    title, parts, start, end = None, [], None, None
    for row in rows:
        match = ARTICLE.match(row["text"])
        if match:
            if title is not None:
                yield {"id": "wikipedia:" + title, "title": title, "text": "".join(parts),
                       "url": "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
                       "source_start": start, "source_end": end}
            title, parts = match.group(1).strip(), []
            start = {key: row[key] for key in ("source_file", "source_row")}
        if title is not None:
            parts.append(row["text"])
            end = {key: row[key] for key in ("source_file", "source_row")}
    if title is not None:
        yield {"id": "wikipedia:" + title, "title": title, "text": "".join(parts),
               "url": "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
               "source_start": start, "source_end": end}


def partition(value, denominator):
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) % denominator


def c4_documents(source, role, seed, coverage):
    native = "train" if role == "train" else "validation"
    count = 1024 if native == "train" else 8
    names = [f"en/c4-{native}.{i:05d}-of-{count:05d}.json.gz" for i in range(count)]
    random.Random(stable_seed(seed, "c4", native)).shuffle(names)
    for name in names:
        record = {"rows_read": 0}
        coverage.setdefault("files", {})[name] = record
        with requests.get(hf_url(source, name), stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with gzip.GzipFile(fileobj=response.raw) as stream:
                for i, line in enumerate(stream):
                    record["rows_read"] = i + 1
                    row = json.loads(line)
                    identity = row.get("url") or " ".join(row["text"].split())
                    fold = partition(identity, 2)
                    if role != "train" and fold != (0 if role == "validation" else 1):
                        continue
                    yield {**row, "id": "c4:" + hashlib.sha256(identity.encode()).hexdigest(),
                           "source_file": name, "source_row": i, "native_split": native}


def book_documents(source, role, seed, coverage):
    listing = Path(hf_hub_download(source["repo"], f"data/{role}_files.txt", repo_type="dataset", revision=source["revision"]))
    coverage["split_listing_sha256"] = sha256(listing)
    names = listing.read_text().splitlines()
    random.Random(stable_seed(seed, "pg19", role)).shuffle(names)
    for name in names:
        encoded = quote(name, safe="")
        address = f"https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o/{encoded}"
        cache = Path(coverage["asset_metadata_cache"]) / (name.replace("/", "_") + ".json")
        info = read_json(cache)
        if info is None:
            metadata = requests.get(address, timeout=60)
            metadata.raise_for_status()
            info = metadata.json()
            atomic_json(cache, info)
        generation = info["generation"]
        response = requests.get(address + f"?alt=media&generation={generation}", timeout=120)
        response.raise_for_status()
        content = response.content
        assert len(content) == int(info["size"])
        assert base64.b64encode(hashlib.md5(content).digest()).decode() == info["md5Hash"]
        coverage.setdefault("files", {})[name] = {"generation": generation, "bytes": len(content),
                                                   "sha256": hashlib.sha256(content).hexdigest()}
        book_id = Path(name).stem
        yield {"id": "gutenberg:" + book_id, "url": "https://www.gutenberg.org/ebooks/" + book_id,
               "text": content.decode("utf-8"), "source_file": name, "source_generation": generation,
               "source_sha256": hashlib.sha256(content).hexdigest(), "native_split": role}


def documents(name, role, c, coverage, project):
    source = c["sources"][name]
    kind = source["adapter"]
    if kind == "existing_fineweb":
        saved = Path(project) / c["fineweb_source"]["registry_records"]
        collections = [role] if role != "train" else [f"cycle_{i:02d}" for i in range(1, 11)]
        index = read_json(Path(project) / c["fineweb_source"]["artifact_index"])
        for collection in collections:
            relative = f"data/{collection}/documents.jsonl"
            path = saved / collection / "documents.jsonl"
            if sha256(path) != index["files"][relative]["sha256"]:
                raise RuntimeError("Historical FineWeb selection checksum differs")
            coverage.setdefault("files", {})[relative] = {"sha256": sha256(path)}
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                row["source_token_limit"] = row["used_source_tokens"]
                row["source_genuine_end"] = not row["trimmed"]
                yield row
        return
    seed = stable_seed(c["seed"], name, role)
    if kind == "c4":
        iterator = c4_documents(source, role, c["seed"], coverage)
    elif kind == "pg19":
        yield from book_documents(source, role, c["seed"], coverage)
        return
    elif kind == "wikitext":
        count = 2 if role == "train" else 1
        files = [f"wikitext-103-raw-v1/{role}-{i:05d}-of-{count:05d}.parquet" for i in range(count)]
        iterator = wiki_documents(parquet_rows(source, files, coverage))
    elif kind == "ag_news":
        native = "test" if role == "test" else "train"
        iterator = parquet_rows(source, [f"data/{native}-00000-of-00001.parquet"], coverage)
        def selected(rows):
            for row in rows:
                normalized = " ".join(row["text"].split())
                validation = partition(normalized, 16) == 0
                if role != "test" and validation != (role == "validation"):
                    continue
                yield {"text": row["text"], "id": f"ag-news:{native}:{row['source_row']}",
                       "url": None, "source_file": row["source_file"], "source_row": row["source_row"],
                       "native_split": native, "topic_label": row["label"]}
        iterator = selected(iterator)
    else:
        raise ValueError("Unknown corpus source adapter")
    yield from shuffled(iterator, c["shuffle_buffer"], seed)
