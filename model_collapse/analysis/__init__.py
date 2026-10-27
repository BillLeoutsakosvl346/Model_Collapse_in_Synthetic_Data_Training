"""CPU-only analysis of saved experiment evidence; no cloud or model loading."""

from .records import METRIC_DEFINITIONS, read_metrics_csv

__all__ = ["METRIC_DEFINITIONS", "read_metrics_csv"]
