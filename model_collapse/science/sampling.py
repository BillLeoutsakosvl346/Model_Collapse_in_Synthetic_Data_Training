"""Pure mixture proportions shared by the distinct WikiText/FineWeb samplers."""
import math
from decimal import Decimal, ROUND_HALF_UP


def synthetic_fraction(c):
    value = c.get("synthetic_fraction", 1.0)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("synthetic_fraction must be a finite number between zero and one")
    return float(value)


def mixture_counts(total, fraction):
    fraction = synthetic_fraction({"synthetic_fraction": fraction})
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("Mixture total must be a positive integer")
    n_synthetic = int((Decimal(str(fraction)) * total).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return total - n_synthetic, n_synthetic
