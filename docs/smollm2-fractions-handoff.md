# SmolLM2 fraction sweep: setup handoff

**11 November 2026: implementation and data prepared; actual GPU acceptance blocked by Runpod capacity. All seven full experiments remain unlaunched.**

The [definition and configurations](../comparisons/smollm2_135m_fineweb_completion64_fractions/README.md) implement 0/50/80/95/99/99.9/100% synthetic targets using base SmolLM2-135M. Every condition starts independently from the same pretrained weights, uses ten fresh chunks, generates its selected targets from its own preceding checkpoint, and carries its own model/AdamW/scheduler/RNG. Historical OPT builders, data and publications remain unchanged.

## Completed checks

- **104 final focused tests passed in 17.58 seconds.** An earlier focused run passed 98 tests in 37.62 seconds. The final profile/S3/budget checks passed **46 tests in 7.41 seconds**, with overlapping coverage; these totals must not be added. The complete historical suite was not run.
- One independent reviewer checked model compatibility, tokenization, masks, mixture placement, endpoints, optimizer handoffs, coordinator failure handling, recovery and spending safeguards. [Material findings were resolved](validation/2026-11-11_smollm2-independent-review.json).
- A tiny CPU Llama test confirms native BOS=0, visible prompts, exactly 64 scored continuation targets, and agreement with a direct next-token loss calculation. This does not replace the actual SmolLM2 GPU test.
- The complete native bank passed file checksums, reconstruction of every token pair, document/passage separation and document-stratified probe checks. Indexed original documents were verified before retokenization.
- Private bulk uploads verify local hashes and provider-reported sizes. The pinned private Hub index was downloaded and checksum-verified. A 4 KB S3 probe was independently cold-downloaded. Full-bank/model/optimizer cold restoration was **not** executed.
- CLI/configuration, deployment-bundle, documentation and published-evidence checks passed. Runtime identity remains separate from configuration and Git history.

## Frozen bank

| Collection | Pairs | Scored real continuation tokens |
| --- | ---: | ---: |
| Each of ten training chunks | 10,000 | 640,000 |
| Validation | 7,447 | 476,608 |
| Test | 7,303 | 467,392 |

The original 1,000-document assignments per held-out split are preserved; 983 documents per split yield complete 128-token windows. All held-out identities remain excluded, including short documents. The fixed probe selects one prompt from each of 256 different validation documents. No extension documents were needed: all native training chunks could be filled from their verified historical allocations.

Local bank: `data/smollm2_completion64_v2/` (approximately 159 MB). `examples.json` contains readable prompts and real continuations. Generated examples await actual GPU execution.

Pinned backup: `smollm2-completion64-bank-75c3da6d89516ffd` in the existing private experiment repositories. Dataset commit: `61b426c93d46c2e5caafa24b9c07f1fc342aa3d5`. Manifest SHA-256: `8c475f058606fed0610d7555128139ea445b2f4753c2a1abc8eccd44e06c9837`. Both named configurations include the complete immutable reference. The [machine-readable receipt](validation/2026-11-11_smollm2-setup.json) records remaining hashes and scope.

Retain Standard volume **`1b2vuplxer`**, 10 GB in EU-RO-1: the prepared bank depends on its private bulk objects. Approximate storage cost is **USD 0.70/month until deleted**, using [Runpod's published storage rate](https://docs.runpod.io/storage/s3-api). Do not delete it as temporary test storage.

## Capacity blocker and spending

Seven actual placement requests created no Pod: five REST requests (preferred RTX PRO 4500, one RTX 5090 and one persistent-disk variation) and two GraphQL requests (4500 in EU-RO-1 and RTX A6000 in EU-SE-1). The alternative API explicitly returned `SUPPLY_CONSTRAINT`. Other attempts stopped during preflight without allocating resources. Catalog stock did not establish successful placement.

All six temporary network work volumes were confirmed absent by HTTP 404. The native bank's 10 GB volume remains. Provider reconciliation found **no active GPU**. Account reads allow eight concurrently running GPUs; there is no evidence that the twenty stopped historical Pods caused the problem. No historical Pod or recovery volume was changed.

**GPU compute spend: USD 0.** The machine-readable receipt records conservative storage accrual through verification; it is not an attribution of account debits. The USD 0.50 validation cap, 900-second combined paid-compute limit, USD 3 setup allocation including USD 2 reserve, USD 5.05 historical setup ceiling and USD 80 cumulative ceiling remain enforced. A requested USD 15 task allocation / USD 90 cumulative ceiling remains **unapproved**. The planned 400 GB of full working/backup capacity alone costs approximately USD 6.44 for seven days; neither full-size volume exists yet.

## Remaining acceptance and conditionally authorized launch

The user subsequently authorized full launch once acceptance and affordability pass; monetary limits remain unchanged. The next actual GPU smoke must execute only **mixed50**, two fresh chunks of 128 pairs, one optimizer update each; 32 validation/test pairs and 32 probe prompts. It must verify changed weights, stage-2 model/optimizer continuity, exact generation/scattering/masks, terminal reload, one representative private restoration with miniature evaluation, detached progress and ordinary automatic provider shutdown. No interruption matrix or deliberate budget exhaustion is required. GPU execution of the other six fractions is outside this short acceptance scope.

```powershell
$comparison = "comparisons/smollm2_135m_fineweb_completion64_fractions"

# Retry the bounded compatibility test when matching capacity is available:
python -m model_collapse smoke --config "$comparison/smoke.yaml"

# Authorized after actual acceptance and affordable admission:
python -m model_collapse launch --config "$comparison/full.yaml" --generations 10

python -m model_collapse status COMPARISON_ID
python -m model_collapse resume COMPARISON_ID
python -m model_collapse fetch COMPARISON_ID --metadata-only
python -m model_collapse report COMPARISON_ID
python -m model_collapse cleanup COMPARISON_ID
```

No native-model throughput was measured, so **there is no measured full-run runtime/cost forecast yet**. Full execution consists of 70 terminal training stages, 790 updates per condition and 523,900 generated training continuations. Its planned 200 GB working and 200 GB backup volumes have not been provisioned. The launcher rejects absent/undersized backup capacity and requires actual GPU acceptance and affordable admission. The USD 3 setup-only allocation does not establish full-sweep affordability.

Within SmolLM2, all seven conditions share one tokenizer and evaluation bank. Cross-model comparisons must label tokenizer/pretraining differences and emphasize baseline-relative trajectories. SmolLM2 pretraining includes FineWeb-Edu; experimental held-out separation does not prove pretraining novelty. One seed and decreasing learning rate remain explicit limitations.
