"""Probe statistics; n-grams never cross continuation boundaries."""
from collections import Counter

import numpy as np


def diversity(values):
    values = np.asarray(values)
    result = {}
    for n in (1, 2, 3, 4):
        grams = Counter(tuple(row[i:i + n]) for row in values.tolist() for i in range(len(row) - n + 1))
        total = sum(grams.values())
        if n < 4:
            result[f"distinct_{n}"] = len(grams) / total
        else:
            result["repeated_4gram_rate"] = (total - len(grams)) / total
            result["repeated_4gram_denominator"] = total
    result["duplicate_continuation_rate"] = 1 - len(set(map(tuple, values.tolist()))) / len(values)
    counts = np.asarray(list(Counter(values.flatten().tolist()).values()), dtype=float)
    p = counts / counts.sum()
    result["token_entropy_nats"] = float(-(p * np.log(p)).sum())
    result["probe_count"] = len(values)
    return result


def probe_diagnostics(values, newline_token_id=50118):
    """Week-3 within-output/formatting diagnostics, using the pinned OPT newline ID."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 64 or not len(values):
        raise ValueError("Probe diagnostics require nonempty 64-token continuations")
    entropies, repetition, distinct = [], [], {n: [] for n in (1, 2, 3, 4)}
    for row in values.tolist():
        frequencies = np.asarray(list(Counter(row).values()), dtype=float) / len(row)
        entropies.append(float(-(frequencies * np.log(frequencies)).sum()))
        for n in distinct:
            grams = {tuple(row[i:i+n]) for i in range(len(row)-n+1)}
            distinct[n].append(len(grams) / (len(row)-n+1))
        repetition.append(1 - distinct[4][-1])
    newlines = (values == newline_token_id).sum(axis=1)
    return {"mean_within_entropy_nats": float(np.mean(entropies)),
            "mean_within_repeated_4gram_rate": float(np.mean(repetition)),
            "fraction_outputs_with_repeated_4gram": float(np.mean(np.asarray(repetition) > 0)),
            **{f"mean_within_distinct_{n}": float(np.mean(v)) for n, v in distinct.items()},
            "newline_token_50118_count": int(newlines.sum()),
            "newline_token_50118_share": float(newlines.sum() / values.size),
            "newline_heavy_outputs_ge32": int((newlines >= 32).sum()),
            "all_newline_outputs": int((newlines == 64).sum())}
