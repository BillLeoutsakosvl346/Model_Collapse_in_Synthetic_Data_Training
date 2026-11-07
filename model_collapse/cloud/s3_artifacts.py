"""Checksum-addressed bulk recovery on a private Runpod network volume.

Small indexes stay in pinned private Hub commits. Credentials are inherited
through the environment and never become part of an artifact or descriptor.
"""
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from ..common import STATE, load_secrets, sha256

BULK_SUFFIXES = (".npy", ".safetensors", ".pt", "documents.jsonl")


def validate_store(store):
    if not isinstance(store, dict) or set(store) != {"backend", "bucket", "region", "prefix"}:
        raise ValueError("Bulk storage requires an exact Runpod S3 descriptor")
    if store["backend"] != "runpod_s3":
        raise ValueError("Unknown bulk backup backend")
    if not re.fullmatch(r"[a-z0-9]{5,32}", store["bucket"]):
        raise ValueError("Invalid Runpod volume bucket")
    if store["region"] not in {"EU-RO-1", "EU-CZ-1", "US-KS-2", "US-CA-2", "US-IL-1"}:
        raise ValueError("Unsupported S3 region")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", store["prefix"]):
        raise ValueError("Unsafe backup prefix")
    return store


def client(store):
    import boto3
    from botocore.config import Config
    validate_store(store)
    load_secrets()
    access, secret = (os.environ.get(k) for k in ("RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY"))
    if not access or not secret:
        raise RuntimeError("Runpod S3 credentials are missing")
    return boto3.client("s3", region_name=store["region"],
        endpoint_url="https://s3api-" + store["region"].lower() + ".runpod.io",
        aws_access_key_id=access, aws_secret_access_key=secret,
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"},
                      connect_timeout=20, read_timeout=120, s3={"addressing_style": "path"}))


def reference(store, checksum):
    validate_store(store)
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Invalid bulk checksum")
    return {**store, "key": store["prefix"] + "/objects/" + checksum}


def check_capacity(store, capacity_gb, required_bytes, *, reserve_bytes=3_000_000_000, api=None):
    """Count the entire physical bucket before admitting additional checkpoints."""
    from .hf_storage import admission
    api = client(store) if api is None else api
    used = 0
    for page in api.get_paginator("list_objects_v2").paginate(Bucket=store["bucket"]):
        used += sum(item["Size"] for item in page.get("Contents", []))
    return admission(int(capacity_gb * 1_000_000_000), used, required_bytes, reserve_bytes)


def validate_reference(ref, checksum):
    if not isinstance(ref, dict) or set(ref) != {"backend", "bucket", "region", "prefix", "key"}:
        raise ValueError("Invalid bulk artifact reference")
    store = {k: v for k, v in ref.items() if k != "key"}
    if ref != reference(store, checksum):
        raise ValueError("Bulk key does not match its checksum")
    return store


def upload(path, meta, store, *, connection=None):
    from botocore.exceptions import ClientError
    ref = reference(store, meta["sha256"])
    api = connection or client(store)
    try:
        head = api.head_object(Bucket=store["bucket"], Key=ref["key"])
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code")) not in {"404", "NoSuchKey", "NotFound"}:
            raise
    else:
        # Runpod's S3 gateway does not preserve arbitrary x-amz-meta headers.
        # Object identity comes from the pinned SHA-addressed index; full cold
        # restoration verifies the bytes rather than trusting S3 metadata/ETag.
        if head["ContentLength"] != meta["bytes"]:
            raise RuntimeError("Existing content-addressed backup is inconsistent")
        return ref
    if Path(path).stat().st_size != meta["bytes"] or sha256(path) != meta["sha256"]:
        raise RuntimeError("Bulk upload input changed")
    api.upload_file(str(path), store["bucket"], ref["key"], ExtraArgs={"Metadata": {"sha256": meta["sha256"]}})
    head = api.head_object(Bucket=store["bucket"], Key=ref["key"])
    if head["ContentLength"] != meta["bytes"]:
        raise RuntimeError("Bulk upload read-back failed")
    return ref


def download(meta, *, connection=None):
    ref = meta["storage"]
    store = validate_reference(ref, meta["sha256"])
    api = connection or client(store)
    folder = STATE / "bulk-cache"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / meta["sha256"]
    if target.is_file() and target.stat().st_size == meta["bytes"] and sha256(target) == meta["sha256"]:
        return target
    temporary = folder / (meta["sha256"] + "." + uuid.uuid4().hex + ".partial")
    try:
        api.download_file(store["bucket"], ref["key"], str(temporary))
        if temporary.stat().st_size != meta["bytes"] or sha256(temporary) != meta["sha256"]:
            raise RuntimeError("Bulk restore checksum mismatch")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def probe(store):
    """Upload and cold-download actual bytes before renting compute."""
    import hashlib
    payload = os.urandom(4096)
    checksum = hashlib.sha256(payload).hexdigest()
    folder = STATE / "bulk-probes"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / checksum
    path.write_bytes(payload)
    meta = {"sha256": checksum, "bytes": len(payload)}
    meta["storage"] = upload(path, meta, store)
    restored = download(meta)
    if restored.read_bytes() != payload:
        raise RuntimeError("Bulk recovery round-trip failed")
    return {"store": store, "bytes": len(payload), "sha256": checksum, "cold_round_trip": True}
