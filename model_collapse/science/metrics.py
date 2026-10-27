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
