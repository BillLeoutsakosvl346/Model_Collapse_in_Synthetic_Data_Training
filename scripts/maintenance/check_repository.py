"""Offline documentation/configuration/evidence checks. Never reads credentials or cloud state."""
from __future__ import annotations

import argparse
import hashlib
import html
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
            if parsed.scheme or parsed.netloc:
                continue
            destination = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path.resolve()
            if not destination.is_relative_to(root) or not destination.exists():
                errors.append(f"{path.relative_to(root)}: missing/outside relative link {target}")
            elif (parsed.fragment and destination.suffix.lower() == ".md"
                  and not path.is_relative_to(root / "reports") and "history" not in path.parts):
                if unquote(parsed.fragment) not in heading_anchors(destination.read_text(encoding="utf-8")):
                    errors.append(f"{path.relative_to(root)}: missing heading {target}")
    return errors


def heading_anchors(text):
    """GitHub-style ATX heading slugs and explicit HTML anchors."""
    text = re.sub(r"(?ms)^```[^\n]*\n.*?^```[^\n]*$", "", text)
    anchors = set(re.findall(r'<(?:a|span)\b[^>]*\b(?:id|name)=[\"\']([^\"\']+)', text))
    counts = {}
    for title in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$", text):
        title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", title)
        slug = re.sub(r"[^\w\-\s]", "", html.unescape(re.sub(r"<[^>]+>", "", title)).lower())
        slug = re.sub(r"\s", "-", slug)
        suffix = counts.get(slug, 0)
        counts[slug] = suffix + 1
        anchors.add(slug if not suffix else f"{slug}-{suffix}")
    return anchors


def publication_errors(root, paths):
    root = Path(root).resolve()
    errors = []
    for path in paths:
        if path.name != "publication_manifest.json" or not path.is_relative_to(root / "reports"):
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            errors.append(f"{path.relative_to(root)}: unreadable publication manifest")
            continue
        inventory = manifest.get("files", manifest.get("artifacts")) if isinstance(manifest, dict) else None
        if not isinstance(inventory, dict) or not inventory:
            errors.append(f"{path.relative_to(root)}: publication requires a nonempty files inventory")
            continue
        for rel, meta in inventory.items():
            meta = {"sha256": meta} if isinstance(meta, str) else meta
            if (not isinstance(rel, str) or not isinstance(meta, dict)
                    or not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("sha256", "")))
                    or ("bytes" in meta and (type(meta["bytes"]) is not int or meta["bytes"] < 0))):
                errors.append(f"{path.relative_to(root)}: invalid publication entry {rel}")
                continue
            target = (path.parent / rel).resolve()
            if not target.is_relative_to(path.parent):
                errors.append(f"{path.relative_to(root)}: unsafe evidence path {rel}")
                continue
            if not target.is_file():
                errors.append(f"{path.relative_to(root)}: absent evidence {rel}")
                continue
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            # Earlier publications record only a checksum, without a byte count.
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
        record = c.get("full_launch_record", c.get("launch_record"))
        run_id = c.get("full_comparison_run_id", c.get("full_pair_run_id", c.get("full_run_id")))
        if record and run_id:
            receipt = json.loads((catalogue_path.parent / record).read_text(encoding="utf-8"))
            ids = {receipt.get("run_id"), receipt.get("resources", {}).get("run_id")}
            ids.update(job.get("run_id") for job in receipt.get("full_jobs", []))
            if run_id not in ids:
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


def private_path(name):
    """Indexed-file policy; Git ignores alone do not protect force-added secrets."""
    path = Path(name.replace("\\", "/"))
    lower = name.replace("\\", "/").lower()
    basename = path.name.lower()
    return (basename != ".env.example" and (basename == ".env" or basename.startswith(".env."))
            or any(part.lower() in (".state", ".ssh", ".runpod") for part in path.parts)
            or lower.startswith(("runs/", "literature/papers/", "checkpoints/", "cache/", "data/"))
            or path.suffix.lower() in (".safetensors", ".ckpt", ".pt", ".pth", ".pem", ".key")
            or (len(path.parts) == 1 and path.suffix.lower() in (".pdf", ".docx"))
            or (lower.startswith("docs/project/") and path.suffix.lower() == ".docx"))


def check(root=PROJECT, *, base=None):
    root = Path(root).resolve()
    paths = tracked_paths(root)
    errors = markdown_errors(root, paths) + publication_errors(root, paths)
    config_issues, configurations = config_errors(root)
    errors += config_issues + changed_evidence_errors(root, base)
    # Track citations/location guides, never their private local source documents.
    indexed = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode("utf-8").split("\0")
    for name in indexed:
        if private_path(name):
            errors.append(f"Private material is tracked: {name}")
    return {"markdown_files": sum(p.suffix.lower() == ".md" for p in paths), "configurations": configurations,
            "publication_manifests": sum(p.name == "publication_manifest.json" for p in paths),
            "link_scope": "Relative targets and current-document Markdown heading anchors; external URLs are not fetched. Historical report/history anchors are preserved.",
            "paid_operations": False, "errors": errors, "passed": not errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Git revision against which existing evidence/receipt changes are forbidden")
    args = parser.parse_args()
    result = check(base=args.base)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
