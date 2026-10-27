# Acceptance evidence export

`export_acceptance.py` publishes a verified reset-branch smoke test after metadata fetch, local acceptance and temporary-resource cleanup have succeeded:

```powershell
python scripts/analysis/export_acceptance.py RUN_ID --offline-tests COUNT
```

It reads local records, verifies their checksums, exports small evidence and regenerates figures. It does not make cloud requests or copy model/token tensors. `--offline-tests` records an already completed suite; it does not run tests. `--superseded` labels successful acceptance of an earlier runtime which must not authorize changed code.

Published evidence is immutable: the exporter refuses to overwrite an existing acceptance. Every exported role has a checksum in `publication_manifest.json`; Git attributes preserve those bytes on Windows. General analysis uses `python -m model_collapse report RUN_ID [--out OUTPUT_DIRECTORY]` and writes a new destination.

See [operations](../../docs/operations.md) and the [validated setup handoff](../../docs/implementation-handoff.md).
