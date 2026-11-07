"""Compatibility alias for :mod:`model_collapse.cloud.execution`.

The module object is shared so established callers and test patches retain
the same globals. New implementation belongs in the named subpackage.
"""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("model_collapse.cloud.execution")
