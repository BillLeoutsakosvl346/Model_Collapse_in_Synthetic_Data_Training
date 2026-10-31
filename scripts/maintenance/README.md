# Repository checks

```powershell
python scripts/maintenance/check_repository.py
python scripts/maintenance/check_repository.py --base GIT_REVISION
```

Checks relative Markdown file/directory targets, catalogue references/run identities, full/smoke configuration validation, legacy/named scientific equivalence, private-file exclusions and publication-manifest checksums. It makes no cloud requests, reads no credentials and never provisions anything. External links and heading fragments are outside this checker’s scope.

With `--base`, existing files under `reports/` and dated receipts under `docs/validation/` cannot be modified/deleted; new files are allowed. `reports/README.md` is mutable navigation. Publish scientific/editorial revisions separately rather than silently rewriting evidence. The CI workflow supplies the push/PR base revision.

Runtime source hashes and working-tree byte preservation are additionally checked in the dated repository-cleanup receipt. This script validates the repository; it does not certify training or live cloud state.
