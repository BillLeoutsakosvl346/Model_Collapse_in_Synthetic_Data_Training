# Current recorded state

Snapshot refreshed after 99% completion verification and nine-fraction analysis on 7 November 2026, 21:01 UTC/London. Earlier observations are preserved in the [byte-exact chronology](history/2026-11-07_pre-cleanup-current-state.txt).

## Authorized work and limits

| Scope | Recorded authorization |
| --- | --- |
| Cumulative spending ceiling | USD 64.03635, existing credit only |
| High-fraction campaign | USD 11.75 including its separately reserved storage |
| Total setup allowance | USD 3.85, explicitly increased for the 99.9% validation |
| Combined high-fraction validation cap | USD 0.90; recorded conservative spend USD 0.7843 |
| Standalone 99% condition | USD 5 allocation including USD 2 reserve; USD 0.30 combined validation cap; accepted validation USD 0.21584 |
| Standalone 99.9% condition | USD 5 allocation including USD 2 reserve; USD 0.30 combined validation cap; smoke then full launch authorized after checks pass |
| Parallel exception | Only the three named 85/90/95% FineWeb jobs, matching RTX PRO 4500 Blackwell |
| Cleanup task | Local organisation and offline verification; no provisioning, paid validation, resource deletion or deployment |

[Campaign authorization](validation/2026-11-07_completion-high-fractions-authorization.json), [testing sublimits](validation/2026-11-07_completion-high-fractions-testing-authorization.json). Other allocation-specific reserves and historical receipts remain binding within their scopes; older cumulative ceilings are superseded. No payment settings may be changed.

## Latest recorded execution

The user has [authorized](validation/2026-11-07_completion-mixed99p9-authorization.json) **99.9% synthetic / 0.1% real** preparation, validation and full launch after acceptance. Its [definition](../comparisons/fineweb_completion64_mixed99p9/README.md) preserves the matched full protocol with 9,990 synthetic / 10 real targets per cycle. The declared smoke uses 1,000 pairs (999 synthetic / one real) to represent both sources. The 607-test suite passed, followed by 112 focused checks after fixing fractional graph labels; fresh GPU acceptance is pending. The new [30 GB backup volume](validation/2026-11-07_completion-mixed99p9-backup-volume.json) `vx78hsoirz` passed private round-trip and capacity admission. No 99.9% compute has launched. The total ceiling remains USD 64.03635; existing completed-run recovery storage is preserved.

The [completion receipt](validation/2026-11-07_completion-high-fractions-completion-status.json) verifies all three completed ten-cycle jobs, 790 updates each, final metric checksums and pinned backups. Provider states are EXITED; watchdog reason is complete. Final test PPL is 32.09515 / 33.68304 / 38.56880 for 85/90/95%. Avoid duplicate launches or resuming completed compute. Full model restoration/re-evaluation remains separate.

| Run | Pod | Working volume, 60 GB |
| --- | --- | --- |
| `20261107T150834Z-full-completion64-mixed85` | `0ciiothsflf4ik` | `b82vo80i1g` |
| `20261107T151427Z-full-completion64-mixed90` | `alhrmqkpvmiajn` | `tftzsocrpv` |
| `20261107T151610Z-full-completion64-mixed95` | `z1090zdvysy3q8` | `jlo0o9gsvx` |

Preserve all three working volumes and shared 80 GB S3 store `mng23iliig`. Worker runtime is `eb6b3104dbe395bbe192ccd14afbc74de6d24d9d22578486804f0c8c1a490e7f`; [host sources](validation/2026-11-07_completion-high-fractions-host-sources.json) are separately pinned. The [handoff](fineweb-high-fractions-handoff.md) records accepted protocols and operations. The [cleanup verification](validation/2026-11-07_repository-organisation.json) records 576 passing offline tests and a **new, offline-tested candidate runtime**; they do not modify deployed bundles or inherit their GPU acceptance.

The user subsequently authorized standalone **99% synthetic** setup, GPU acceptance and full launch. Offline verification passed 580 tests and independent review. [GPU acceptance](validation/2026-11-07_completion-mixed99-setup.json) `20261107T180907Z-smoke-completion64-mixed99` passed all sixteen checks, including three terminal epochs, exact mixture reconstruction, optimizer/model handoffs, interrupted recovery, detached progress, cold restoration of 157 files / 4.51 GB with identical held-out loss, and actual watchdog/provider shutdown. Temporary Pod `2qun5v3w6c3yqa` and volume `w657u28bn3` were recovered and removed; conservative acceptance cost USD 0.21584. Accepted runtime is `68033f3641fff5dd51ab10b7f32639ed42c0f7d3251c0ccd568cdda6a5ec6b11`, protocol `28a50992262d1057fab089d43cb7e64273a58f4652fbba3b4a3636cfc91d1a3b`.

Full run **`20261107T183326Z-full-completion64-mixed99` completed all ten cycles / ten terminal epochs / 790 updates at 20:17:39 UTC/London**. The [completion and comparison receipt](validation/2026-11-07_completion-mixed99-completion-analysis.json) verifies 79 pinned small files, all continued model/optimizer/generator handoffs, 9,900 synthetic / 100 real pairs and 640,000 scored targets per cycle. Test PPL changed 29.176318 → 82.677301; final validation PPL is 81.753347. Watchdog reason is `complete`; provider shutdown is verified at 20:17:50. Pod `wbsu8iw7f32vi3` is EXITED. Preserve its 60 GB working volume `6ii1cd7qbw` and 30 GB backup volume `6spnjhl8sv`. Do not resume or duplicate completed compute. The [startup receipt](validation/2026-11-07_completion-mixed99-full-launch.json) and [capacity-race reconciliation](validation/2026-11-07_completion-mixed99-capacity-retry.json) remain historical evidence.

Its [authorization](validation/2026-11-07_completion-mixed99-authorization.json) allocated USD 5 including USD 2 reserved and at most USD 0.30 validation, within the unchanged cumulative/setup ceilings. The completion receipt records the new square nine-fraction plots, 99 metric rows, pinned inputs and output hashes; the prior eight-fraction analysis is unchanged. All scientific settings/bank/evaluation identities match; 85/90/95/99% used RTX PRO 4500, while the remaining conditions used RTX PRO 4000. Full checkpoint/dataset restoration and re-evaluation remain separate. Retained storage continues billing; seven days of the new working and backup volumes total approximately USD 1.45.

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
