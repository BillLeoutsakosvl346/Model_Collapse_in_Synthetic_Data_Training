# Manual operational tools

These tools are outside routine tests because they can change provider resources and incur real charges. Never add them to automatic test discovery, installation hooks or report generation.

The legacy tool remains at [../check_shutdown_timeout.py](../check_shutdown_timeout.py), preserving its existing path. It provisions a paid Pod, starts the watchdog, interrupts its timeout path and checks shutdown behaviour. Its historical purpose does not grant authority for another run. Inspect its current source and the task's ownership/budget constraints before explicitly invoking it.

Prefer the integrated `python -m model_collapse smoke --config ...` acceptance workflow for changed training/import/supervision code. That workflow must prove artifact recovery and stopped compute, not only that a stop request returned.

Current procedures and recovery requirements are in [operations](../../docs/operations.md). No manual operational probe is run merely by reading this directory.
