"""Read historical result schemas without changing the original files."""
from __future__ import annotations

import csv
from pathlib import Path

METRIC_DEFINITIONS = {
    "corpus_perplexity": "exp(sum shifted-token NLL / scored targets), not arithmetic mean passage perplexity; 63 targets per independent 64-token block.",
    "distinct_n": "Unique within-continuation n-grams / all within-continuation n-gram occurrences, pooled over the fixed probe; no n-grams cross row boundaries.",
    "repeated_4gram_rate": "Occurrences beyond the first of each 4-gram / all within-continuation 4-grams, pooled across outputs; includes phrases shared between outputs.",
    "duplicate_continuation_rate": "1 - unique complete continuation rows / probe rows.",
    "token_entropy_nats": "Empirical unigram token-frequency entropy of the generated probe, including special tokens; not predictive uncertainty.",
    "frozen_s0_nll": "Mean shifted-token NLL of continuation-only text under frozen real-data G0/S0; not conditioned on the original prompt and not KL divergence.",
    "test_block_nll_p95": "95th percentile of mean NLL per real test block; not a specifically identified rare-token measure.",
    "test_sequence_perplexity_sd": "Population standard deviation across real passages, not uncertainty across independent seeds.",
    "probe_scope": "The same first 256 real training prompts (or the actual saved smaller probe); descriptive measurements, not a random corpus sample.",
    "runtime": "Original recorded train/generation seconds retained; new-work seconds are zero for imported shared stages, with their source identity preserved separately.",
    "within_output_diagnostics": "Compute empirical token entropy and distinct/repeated n-grams within each 64-token output, then average outputs; pooled and within-output statistics answer different questions.",
    "newline_diagnostics": "Count pinned OPT token 50118 (newline), outputs with at least 32 newlines, and entirely newline outputs. Formatting is included in the unfiltered diversity metrics.",
}

_INT_FIELDS = {"generation", "cycle", "selected_epoch", "training_blocks", "training_tokens_per_epoch",
               "repeated_4gram_denominator", "probe_count", "peak_allocated_gpu_bytes"}
_STRING_FIELDS = {"initial_weights_sha256", "selected_weights_sha256", "input_tokens_sha256", "initialization",
                  "stage_origin", "source_run_id", "source_stage", "execution_identity"}


def read_metrics_csv(path):
    """Normalize ``generation``/``cycle`` and numeric types in memory only.

    Original metric names and values are retained. ``generation`` is the
    canonical stage index; absence in an old schema never invents measurements.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        raw = list(csv.DictReader(handle))
    result = []
    for values in raw:
        row = {}
        for key, value in values.items():
            if value in (None, ""):
                row[key] = None
            elif key in _STRING_FIELDS or key.endswith("sha256"):
                row[key] = value
            elif key in _INT_FIELDS:
                row[key] = int(value)
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
        stage = row.get("generation", row.get("cycle"))
        if stage is None:
            raise ValueError("Metrics CSV needs generation or cycle")
        if row.get("cycle") is not None and row.get("generation") is not None and row["cycle"] != row["generation"]:
            raise ValueError("Conflicting cycle and generation indices")
        row["generation"] = int(stage)
        result.append(row)
    return result
