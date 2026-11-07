"""Offline documentation/configuration/evidence checks. Never reads credentials or cloud state."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

PROJECT = Path(__file__).resolve().parents[2]


def tracked_paths(root):
    raw = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root)
    return sorted({Path(root) / x for x in raw.decode("utf-8").split("\0") if x and (Path(root) / x).is_file()})


def markdown_errors(root, paths):
    root = Path(root).resolve()
    errors = []
    for path in paths:
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"(?ms)^```[^\n]*\n.*?^```[^\n]*$", "", text)
        for target in re.findall(r"!?\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\)", text):
            target = target.removeprefix("<").removesuffix(">")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            destination = (path.parent / unquote(parsed.path)).resolve()
            if not destination.is_relative_to(root) or not destination.exists():
                errors.append(f"{path.relative_to(root)}: missing/outside relative link {target}")
    return errors


def publication_errors(root, paths):
    root = Path(root).resolve()
    errors = []
    for path in paths:
        if path.name != "publication_manifest.json" or not path.is_relative_to(root / "reports"):
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for rel, meta in manifest.get("files", {}).items():
            target = (path.parent / rel).resolve()
            if not target.is_relative_to(path.parent):
                errors.append(f"{path.relative_to(root)}: unsafe evidence path {rel}")
                continue
            if not target.is_file():
                errors.append(f"{path.relative_to(root)}: absent evidence {rel}")
                continue
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            # Earlier publications record only a checksum, without a byte count.
            meta = {"sha256": meta} if isinstance(meta, str) else meta
            if actual != meta["sha256"] or target.stat().st_size != meta.get("bytes", target.stat().st_size):
                errors.append(f"{path.relative_to(root)}: evidence checksum/size differs {rel}")
    return errors


def config_errors(root):
    from model_collapse.common import config
    root = Path(root)
    errors, checked = [], 0
    for folder in ("configs", "experiments", "comparisons"):
        for path in sorted((root / folder).rglob("*.yaml")):
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and value.get("kind") in ("full", "smoke"):
                try:
                    config(path)
                    checked += 1
                except (ValueError, KeyError, TypeError) as e:
                    errors.append(f"{path.relative_to(root)}: {e}")
    pairs = [("full.yaml", "pretrained_reset/full.yaml"), ("smoke.yaml", "pretrained_reset/smoke.yaml"),
             ("full-continued.yaml", "continued_from_real_s0/full.yaml"),
             ("smoke-continued.yaml", "continued_from_real_s0/smoke.yaml")]
    for old, named in pairs:
        legacy = yaml.safe_load((root / "configs" / old).read_text(encoding="utf-8"))
        current = yaml.safe_load((root / "experiments/opt125m_wikitext2" / named).read_text(encoding="utf-8"))
        current = {k: v for k, v in current.items() if k not in ("schema_version", "experiment_id")}
        if legacy != current:
            errors.append(f"Legacy/named scientific settings differ: {old} / {named}")
    catalogue_path = root / "experiments/catalog.yaml"
    catalogue = yaml.safe_load(catalogue_path.read_text(encoding="utf-8"))
    def paths(value):
        if isinstance(value, dict):
            for v in value.values():
                yield from paths(v)
        elif isinstance(value, list):
            for v in value:
                yield from paths(v)
        elif isinstance(value, str) and not urlsplit(value).scheme and re.search(r"\.(md|yaml|json|csv)$", value):
            yield value
    for rel in paths(catalogue):
        if not (catalogue_path.parent / rel).is_file():
            errors.append(f"Catalogue target missing: {rel}")
    for key, c in catalogue["conditions"].items():
        if c.get("launch_record") and c.get("full_run_id"):
            receipt = json.loads((catalogue_path.parent / c["launch_record"]).read_text(encoding="utf-8"))
            if receipt["run_id"] != c.get("full_pair_run_id", c["full_run_id"]):
                errors.append(f"Catalogue/launch run ID differs: {key}")
    return errors, checked


def changed_evidence_errors(root, base):
    """Existing scientific publications/dated receipts may only be extended by new files."""
    if not base or set(base) == {"0"}:
        return []
    raw = subprocess.check_output(["git", "diff", "--name-status", "--no-renames", base, "--", "reports", "docs/validation"],
                                  cwd=root, text=True)
    errors = []
    for line in raw.splitlines():
        status, name = line.split("\t", 1)
        if name == "reports/README.md":
            continue  # Mutable navigation, not experiment evidence.
        if status != "A":
            errors.append(f"Published evidence changed relative to {base}: {status} {name}; publish a new revision instead")
    return errors


def check(root=PROJECT, *, base=None):
    root = Path(root).resolve()
    paths = tracked_paths(root)
    errors = markdown_errors(root, paths) + publication_errors(root, paths)
    config_issues, configurations = config_errors(root)
    errors += config_issues + changed_evidence_errors(root, base)
    # Track citations/location guides, never their private local source documents.
    indexed = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode("utf-8").split("\0")
    for name in indexed:
        if name == ".env" or name.startswith((".state/", "runs/", "literature/papers/")) or (name.startswith("docs/project/") and name.endswith(".docx")):
            errors.append(f"Private material is tracked: {name}")
    return {"markdown_files": sum(p.suffix.lower() == ".md" for p in paths), "configurations": configurations,
            "publication_manifests": sum(p.name == "publication_manifest.json" for p in paths),
            "link_scope": "Relative file/directory targets outside fenced code; external URLs and fragment anchors are not fetched/validated.",
            "paid_operations": False, "errors": errors, "passed": not errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Git revision against which existing evidence/receipt changes are forbidden")
    args = parser.parse_args()
    result = check(base=args.base)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
