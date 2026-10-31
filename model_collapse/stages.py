"""Individual local experiment stages; remote guarded orchestration uses pipeline.py."""
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from .common import atomic_json, config, read_json
from .data import load_blocks, prepare
from .engine import evaluate, generate, load_model, train, weights_hash
from .workflow.lineage import training_lineage


def stage(name, root, generation=0, config_path=None):
    root = Path(root)
    c = read_json(root / "config.json")
    if c is None:
        if name != "prepare" or not config_path:
            raise ValueError("Prepare a configured run first")
        c = config(config_path)
        from .common import load_secrets
        load_secrets()
        if c.get("initialization") == "reset_s0":
            from .workflow import resolve_branch_source
            c = resolve_branch_source(c)
        elif c.get("initial_dataset"):
            from .workflow.initial_data import resolve_initial_dataset
            c = resolve_initial_dataset(c)
        atomic_json(root / "config.json", c)
    if name == "prepare":
        tok = AutoTokenizer.from_pretrained(c["model"], revision=c["model_revision"])
        result = prepare(root, c, tok)
        if c.get("initialization") == "reset_s0":
            from .workflow.branch import import_shared_prefix
            import_shared_prefix(root, c)
        return result
    selected = root / f"seed_{c['seed']}" / f"gen_{generation:02d}"
    if name == "train":
        if generation < 0:
            raise ValueError("Train generation must be nonnegative")
        if c.get("initialization") == "reset_s0" and generation < 2:
            raise ValueError("Cycles 0 and 1 are imported shared evidence; new reset training starts at cycle 2")
        blocks = load_blocks(root, "train") if generation == 0 else np.load(root / f"synthetic_for_gen_{generation:02d}/tokens.npy", mmap_mode="r")
        if c.get("initialization") == "continued":
            if generation == 0:
                from .continued import initialize
                result = initialize(root, c)
                if c.get("initial_dataset"):
                    from .workflow.initial_data import import_initial_dataset
                    import_initial_dataset(root, c)
                return result
        if generation:
            blocks, expected, parent_checkpoint, lineage = training_lineage(root, generation, c, None)
            if c.get("initialization") in ("continued", "reset_s0"):
                return train(root, generation, blocks, load_blocks(root, "validation"), c, expected,
                             initial_checkpoint=parent_checkpoint, lineage=lineage)
        base = load_model(c)
        initial = weights_hash(base)
        del base
        lineage = None
        if generation:
            blocks, initial, _, lineage = training_lineage(root, generation, c, initial)
        return train(root, generation, blocks, load_blocks(root, "validation"), c, initial, lineage=lineage)
    model = load_model(c, None if generation < 0 else selected / "checkpoint")
    if name == "evaluate":
        result = {s: evaluate(model, load_blocks(root, s), c) for s in ("validation", "test")}
        atomic_json((root / f"seed_{c['seed']}" / "pretrained" if generation < 0 else selected) / "standalone_evaluation.json", result)
        return result
    if generation < 0:
        raise ValueError("Generate requires a trained nonnegative generation")
    if c.get("initialization") == "reset_s0" and generation < 2:
        raise ValueError("Shared generation artifacts are immutable; new generation starts at cycle 2")
    prompts = load_blocks(root, "train")
    _, manifest = generate(root, generation, model, prompts, c, read_json(selected / "training.json")["selected_weights_sha256"])
    return manifest
