# Manual operational tools

These tools are outside routine tests because they can change provider resources and incur real charges. Never add them to automatic test discovery, installation hooks or report generation.

`cross_corpus_evaluation.py --execute` runs a capped evaluation-only comparison of two existing FineWeb checkpoints on the frozen five-source [corpus bank](../../corpora/README.md). It uses the shared ledger, resource registry, SSH deployment, private recovery and independent watchdog. It retrieves results and removes its own temporary Pod/volume after checksum verification. It performs zero training updates.

The legacy tool remains at [../check_shutdown_timeout.py](../check_shutdown_timeout.py), preserving its existing path. It provisions a paid Pod, starts the watchdog, interrupts its timeout path and checks shutdown behaviour. Its historical purpose does not grant authority for another run. Inspect its current source and the task's ownership/budget constraints before explicitly invoking it.

Prefer the integrated `python -m model_collapse smoke --config ...` acceptance workflow for changed training/import/supervision code. That workflow must prove artifact recovery and stopped compute, not only that a stop request returned.

Current procedures and recovery requirements are in [operations](../../docs/operations.md). No manual operational probe is run merely by reading this directory.

## Scope and compatibility

These adapters encode particular dated authorizations or recovery actions. They retain established paths and, where needed, legacy global patches. A source move in the local candidate does not update a deployed job or transfer an old GPU certificate. Consult the named receipt/current state before invoking any adapter; ordinary future workflows should use explicit configurations and shared cloud helpers.

The ordinary and high-fraction staged launchers now share `launch.lock`. Their admission/deployment phase is serialized; independent training observation may overlap only within the recorded parallel scope.

## Private backup capacity

`python -m scripts.operations.hf_storage_admission --help` describes a read-only historical large-object capacity check for a dedicated private namespace. Include all repositories using its quota and the full remaining checkpoint/optimizer requirement.

`python -m scripts.operations.resume_completion64 RUN_ID` is the guarded entry point for the recorded storage-repaired comparison. It checks the verified backup repair and storage headroom before invoking the accepted cloud launcher. Consult current state and provider status first. An active comparison continues automatically.

## Authorized parallel full mixtures

`python -m scripts.operations.launch_parallel_completion_mixtures` applies the recorded parallel-full authorization for the 50/50 and 20/80 comparison. It verifies the accepted runtime/protocol and exact protected original run/launch/Pod/volume association before calling the existing launcher. The worker bundle retains its accepted identity. Recorded full launches prevent duplicate starts.

Use `python -m scripts.operations.launch_parallel_completion_mixtures --resume RUN_ID` for an unfinished recorded full comparison when parallel admission is still needed. Standard status, fetch, report and cleanup commands retain their normal interfaces. Consult current state and the [full authorization](../../docs/validation/2026-11-07_completion-mixtures-parallel-full-authorization.json) before invoking paid work.

The [80% FineWeb launcher](launch_completion_mixed80.py) starts the single mixed80 condition after its standard acceptance and budget checks. It permits fifteen minutes for first-stage startup verification, bounded by the independent watchdog, and rejects duplicate full launches.
