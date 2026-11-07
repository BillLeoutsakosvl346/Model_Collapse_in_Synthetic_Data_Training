"""Read-only storage admission for the dedicated private research namespace.

The CLI uses the same capacity calculation as the completion launcher. Its quota
must come from the namespace's recorded plan, and every repository consuming
that namespace's allowance must be included.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from model_collapse.cloud.hf_storage import admission, check_storage, unique_lfs_bytes


def main():
    from huggingface_hub import HfApi, get_token
    from model_collapse.common import atomic_json, load_secrets
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--quota-gb", type=float, required=True)
    parser.add_argument("--required-gb", type=float, default=40)
    parser.add_argument("--reserve-gb", type=float, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    amounts = (args.quota_gb, args.required_gb, args.reserve_gb)
    if any(not math.isfinite(value) or value < 0 for value in amounts):
        parser.error("Storage sizes must be finite and nonnegative")
    load_secrets()
    result = check_storage({"model": args.model_repo, "dataset": args.dataset_repo}, args.namespace,
                           *(int(value * 1e9) for value in amounts),
                           HfApi(token=os.environ.get("HF_TOKEN") or get_token()))
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
