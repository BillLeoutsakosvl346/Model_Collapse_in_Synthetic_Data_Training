# Offline verification

Run the full suite from an editable checkout with `python -m pytest -q`.

| Directory | Responsibility |
| --- | --- |
| `science/` | Token objectives, data packing, corpus separation and both mixture algorithms |
| `workflow/` | Initialization, checkpoint selection, optimizer continuity, lineage and interrupted-stage recovery |
| `cloud/` | Fake-provider admission, shutdown, ownership, accounting and operational adapters |
| `artifacts/` | Backup filtering, safe paths, checksums, cold recovery and S3 adapters |
| `analysis/` | Historical schema readers, comparisons and publication construction |
| `repository/` | CLI compatibility, package boundaries, documentation and evidence integrity |

The shared `conftest.py` blocks external socket connections, disables credential loading and redirects operational state into each test's temporary directory. Provider tests supply explicit fakes. Subprocess tests must carry the same offline environment; no test command authorizes paid compute.

Keep condition-specific scientific assertions explicit. Shared fixtures should cover mechanics, preserving each test's masks, initialization, optimization and checkpoint policy. Full GPU acceptance remains an explicitly authorized operational task outside offline test discovery.
