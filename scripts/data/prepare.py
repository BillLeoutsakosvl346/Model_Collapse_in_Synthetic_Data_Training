"""Prepare pinned train/validation/test pools without starting any model training."""
import argparse
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from model_collapse.common import PROJECT, load_secrets
from scripts.data.registry import prepare, preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "corpora/cross_corpus_v1.yaml")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    load_secrets()
    c = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(c["model"], revision=c["model_revision"])
    prepare(args.out, c, tokenizer, PROJECT)
    preview(args.out, tokenizer)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
