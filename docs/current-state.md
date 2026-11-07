# Current recorded state

Snapshot assembled during the 7 November 2026 repository cleanup. **Latest provider observation: 7 November, 15:32 UTC/London. No live provider check was made for cleanup.** Earlier observations are preserved in the [byte-exact chronology](history/2026-11-07_pre-cleanup-current-state.txt).

## Authorized work and limits

| Scope | Recorded authorization |
| --- | --- |
| Cumulative spending ceiling | USD 64.03635, existing credit only |
| High-fraction campaign | USD 11.75 including its separately reserved storage |
| Total setup allowance | USD 3.50 |
| Combined high-fraction validation cap | USD 0.90; recorded conservative spend USD 0.7843 |
| Parallel exception | Only the three named 85/90/95% FineWeb jobs, matching RTX PRO 4500 Blackwell |
| Cleanup task | Local organisation and offline verification; no provisioning, paid validation, resource deletion or deployment |

[Campaign authorization](validation/2026-11-07_completion-high-fractions-authorization.json), [testing sublimits](validation/2026-11-07_completion-high-fractions-testing-authorization.json). Other allocation-specific reserves and historical receipts remain binding within their scopes; older cumulative ceilings are superseded. No payment settings may be changed.

## Latest recorded execution

The [full-launch receipt](validation/2026-11-07_completion-high-fractions-full-launch.json) observed all three jobs RUNNING, at cycles 3/2/2, with detached supervision, independent watchdogs, actual training and pinned backup indexes verified. It does not establish later completion. Avoid duplicate launches or resuming active/completed compute.

| Run | Pod | Working volume, 60 GB |
| --- | --- | --- |
| `20261107T150834Z-full-completion64-mixed85` | `0ciiothsflf4ik` | `b82vo80i1g` |
| `20261107T151427Z-full-completion64-mixed90` | `alhrmqkpvmiajn` | `tftzsocrpv` |
| `20261107T151610Z-full-completion64-mixed95` | `z1090zdvysy3q8` | `jlo0o9gsvx` |

Preserve all three working volumes and shared 80 GB S3 store `mng23iliig`. Worker runtime is `eb6b3104dbe395bbe192ccd14afbc74de6d24d9d22578486804f0c8c1a490e7f`; [host sources](validation/2026-11-07_completion-high-fractions-host-sources.json) are separately pinned. The [handoff](fineweb-high-fractions-handoff.md) records accepted protocols and operations. The [cleanup verification](validation/2026-11-07_repository-organisation.json) records 576 passing offline tests and a **new, offline-tested candidate runtime**; they do not modify deployed bundles or inherit their GPU acceptance.

## Completed work and recovery obligations

| Condition | Completion evidence | Storage to preserve / recovery scope |
| --- | --- | --- |
| FineWeb 80% synthetic | [Completion](validation/2026-11-07_completion-mixed80-completion-status.json) | Pod `uvhfubv6zpf6sm`, volume `ygb6njmthn`; endpoint metadata recovered, bulk restore/re-evaluation pending |
| FineWeb 0/20/50/100% | [Completion and analysis](validation/2026-11-07_fineweb-four-fraction-completion-analysis.json) | Pods `udbovqu03vp3h0` / `6fgu2hphecnsbe`, volumes `s9pmtnqkir` / `ljwf63rdw7`; small statistics verified, full checkpoint restoration separate |
| FineWeb fresh-real pilot | [Completion](validation/2026-11-06_fineweb-full-completion-status.json) | Pod `xf5ml3no72cg0z`, volume `mq5uhckkxr`; bulk restoration/re-evaluation and authored analysis pending |
| WikiText real-only | [Completion](validation/2026-11-02_real-only-completion-status.json) | Stopped Pod/persistent disk `0kj0h4jzxtkrbi`; published analysis complete, bulk recovery pending |
| WikiText mixed campaign and earlier runs | [Catalogue](../experiments/catalog.yaml), [reports](../reports/README.md) | Preserve all historical recovery resources and their ownership records; published metadata is not bulk recovery |

All additional historical retention obligations remain preserved in the [earlier state snapshot](history/2026-11-07_pre-cleanup-current-state.txt) and [earlier agent instructions](history/2026-11-07_pre-cleanup-agent-instructions.txt). Neither local folders nor metadata indexes authorize deleting cloud storage. Resource ownership and exact current costs come from private registry/ledger plus a separately requested live check.

## Where to continue

- Scientific definitions and exact configuration paths: [catalogue](../experiments/catalog.yaml).
- Published metrics, figures and delivery/recovery scope: [reports index](../reports/README.md).
- New execution: [operations](operations.md); verify the candidate runtime before any new paid launch.
- Repository organisation and compatibility: [architecture](architecture.md).

Refresh this snapshot in place with dated evidence; append chronology to history/receipts instead of adding contradictory current sections.
