"""Explicit training decoders; omission preserves historical five-beam kwargs."""
from __future__ import annotations

import hashlib
import json
import math

SAMPLING_POLICY = "sha256_seed_cycle_purpose_batch_v1"


def decoding_settings(c):
    block, new, beams = c.get("block_size", 64), c.get("new_tokens", 64), c.get("num_beams", 5)
    penalty = c.get("repetition_penalty", 1.0)
    mode = c.get("generation_decoder", "beam")
    if block != 64 or new != 64 or not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("Generation requires 64-token blocks/continuations and a finite positive repetition penalty")
    if mode not in ("beam", "greedy", "sampling") or type(beams) is not int or beams != (5 if mode == "beam" else 1):
        raise ValueError("Decoder requires five beams for beam search or one for greedy/sampling")
    sample = mode == "sampling"
    if c.get("do_sample", sample) is not sample:
        raise ValueError("do_sample differs from the declared decoder")
    result = {"num_beams": beams, "do_sample": sample, "min_new_tokens": new, "max_new_tokens": new,
              "repetition_penalty": penalty, "no_repeat_ngram_size": 0, "forced_eos_token_id": None, "use_cache": True}
    if sample:
        expected = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
        if any(c.get(k) != v or isinstance(c.get(k), bool) for k, v in expected.items()):
            raise ValueError("Pure sampling requires explicit temperature=1, top_k=0 and top_p=1")
        result.update(expected)
    elif any(k in c for k in ("temperature", "top_k", "top_p")):
        raise ValueError("Sampling filters are valid only with sampling")
    return result


def decoder_config(c, mode):
    """Resolve each arm without changing its parent's five-beam probe policy."""
    result = {k: v for k, v in c.items() if k not in ("generation_decoder", "do_sample", "temperature", "top_k", "top_p")}
    if mode == "beam":
        result["num_beams"] = 5
    elif mode in ("greedy", "sampling"):
        result.update(generation_decoder=mode, num_beams=1, do_sample=mode == "sampling")
        if mode == "sampling":
            result.update(temperature=1.0, top_k=0, top_p=1.0)
    else:
        raise ValueError("Unknown training decoder")
    decoding_settings(result)
    return result


def sampling_seed(c, generation, offset, provenance):
    value = {"policy": SAMPLING_POLICY, "seed": c["seed"], "generation": generation,
             "offset": offset, "purpose": c.get("generation_purpose", "training"),
             "parent": provenance["parent_weights_sha256"], "prompts": provenance["prompt_tokens_sha256"],
             "batch": c["generation_batch"]}
    return int.from_bytes(hashlib.sha256(json.dumps(value, sort_keys=True).encode()).digest()[:8], "big") % (2**63 - 1)
