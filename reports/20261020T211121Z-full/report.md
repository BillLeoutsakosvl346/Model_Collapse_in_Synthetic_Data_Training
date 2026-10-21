# Week 1 full experiment

Experiment: `20261020T211121Z-full`. Seed: 42.

Completed G0-G10 at 2026-10-21T07:12:51.486807+01:00. Elapsed time from provisioning to completion: 9.02 hours. The observed account balance decrease at the preceding status check was approximately USD 2.23, including retained storage through that observation. GPU compute stopped automatically; storage is retained and continues to accrue a small charge. See execution_receipt.json for the measurement basis.

Test perplexity rose from 33.8839 (G0, real training) to 62.4912 (G1), then broadly plateaued around 52-54 through G10 (52.8282). Repeated 4-grams increased from 48.87% at G0 to 59.05% at G10. Neither perplexity nor all diversity metrics deteriorate monotonically. One seed with documented reconstruction choices supports preliminary observations, not a statistically established reproduction.

Metadata consistency audit: all 11 generation records and all 10 parent-data handoffs passed; see consistency_audit.json. This audit verifies saved records and their checksums, without downloading or re-evaluating the 11 large model checkpoints.

Figures: [overview](results_overview.png), [perplexity](perplexity.png), [diversity](diversity.png), [G0 reference scores](reference_scores.png), [runtime](runtime.png). Data: [metrics CSV](metrics.csv).

![Perplexity and repetition across generations](results_overview.png)

Untouched pretrained test corpus perplexity: 115.4316.

| Generation | Validation PPL | Test PPL | Distinct-2 | Repeated 4-grams |
|---|---:|---:|---:|---:|
| G0 | 34.2136 | 33.8839 | 0.2923 | 0.4887 |
| G1 | 62.9982 | 62.4912 | 0.3361 | 0.5150 |
| G2 | 54.4245 | 53.8417 | 0.4058 | 0.4429 |
| G3 | 53.3268 | 52.8088 | 0.3821 | 0.4688 |
| G4 | 53.0158 | 52.4957 | 0.3814 | 0.4818 |
| G5 | 52.7082 | 52.1732 | 0.3535 | 0.5223 |
| G6 | 53.1723 | 52.8445 | 0.3801 | 0.4865 |
| G7 | 53.0834 | 52.4909 | 0.3390 | 0.5394 |
| G8 | 53.7194 | 53.2654 | 0.3382 | 0.5413 |
| G9 | 53.7934 | 53.2547 | 0.3274 | 0.5570 |
| G10 | 53.5398 | 52.8282 | 0.3058 | 0.5905 |

Corpus PPL = exp(sum of NLL / count of scored shifted targets); each real block scores 63 targets. No average of batch perplexities.

Distinct-n counts unique n-grams divided by all n-grams within individual continuations. Repeated 4-gram rate counts occurrences beyond the first divided by all within-continuation 4-grams. Exact duplicate rate = 1 − unique continuations / continuation count. Special tokens remain in token metrics.

All training starts from the pinned original pretrained model. Prompts are fixed real training blocks; only continuation token IDs are used as subsequent training targets. Validation selects the best epoch; test data only reports results.

Reconstruction differences: split-wide concatenation (authors group map batches); explicit fresh-base initialization; BF16 autocast; corrected token-weighted corpus PPL; repetition penalty 1.0 for the paper's main condition (released code hardcodes 3.0). No real training targets retained after G0.

Paper: https://doi.org/10.1038/s41586-024-07566-y ; authors' code: https://github.com/iliaishacked/curse_recurse

## Comparison with the paper

The relevant comparison is Figure 1b: five epochs with no original real training targets retained after G0. Its approximate test-perplexity trajectory is a real-data baseline near 34, a first synthetic-generation spike near 60-62, and later values around 52-55. Our trajectory has the same broad shape and scale. The untouched pretrained baseline is also similar: approximately 115 in the paper and 115.4316 here. Figure values are approximate visual readings, not digitized measurements. The paper reports mean perplexity and dispersion; our headline measure is token-weighted corpus perplexity, so these are not interchangeable exact numerical targets.

This is preliminary qualitative agreement. The paper uses five independent seeds; this run uses one. Preprocessing, library versions, precision and scoring differences are documented above. Increased repetition is compatible with the paper's observations, but repetition alone does not establish loss of the original distribution's tails or complete collapse.

## Interpreting the plateau

Each generation trains on the preceding model's synthetic continuations. Separately, every generation initializes its weights from the same original pretrained OPT checkpoint. Fixed real training blocks supply generation prompts; only the 64 new continuation tokens become subsequent training targets. The original real validation and test blocks remain fixed throughout.

Test perplexity scores the probabilities of actual real-text next tokens given their correct real-text prefixes. It does not directly measure repetition while a model generates freely from its own outputs. Average next-token prediction and generated-text diversity can therefore move differently. An average can also conceal improvements on common contexts alongside deterioration on rare ones.

The real-to-synthetic transition is a plausible source of the initial performance drop. Later training and generation may approach a relatively stable, biased distribution, producing a plateau despite changes in repetition. These are possible mechanisms, not causes established by this run. Original pretrained initialization, fixed real prompts, limited fine-tuning and selection by real validation loss also constrain this experiment. Checkpoints are selected from the lowest-validation-loss epoch, rather than always taking epoch five; selected epochs are recorded in [the audit](consistency_audit.json).

The supported claim is a lasting loss relative to G0 plus increased repetition overall. Every synthetic generation has worse test perplexity than G0, but successive generations do not always have worse perplexity than their predecessor. The finite-run curve does not demonstrate inevitable or complete collapse.

## Reproducibility and artifact access

The [records directory](records/) contains the checksum-verified configuration, frozen split manifest, untouched-model metrics, all 11 generation metrics and training histories, and all 11 generation-output manifests. The final generation-output manifest describes only the fixed 256-prompt probe after G10; it is not a training dataset for an additional G11 model.

[Artifact index](artifact_index.json) lists 3,192 cloud-backed files totalling 5,758,754,784 bytes, including 11 selected model checkpoints. [Private model backups](https://huggingface.co/Bill11235813/iib-model-collapse-week1-models/tree/main/20261020T211121Z-full) and [private dataset/report backups](https://huggingface.co/datasets/Bill11235813/iib-model-collapse-week1-data/tree/main/20261020T211121Z-full) require the owner's authorization. Ordinary Git history includes small reports and provenance, rather than the bulk checkpoints or token arrays. The original cloud-generated report is preserved as [report_cloud_original.md](report_cloud_original.md).

Regenerate the overview from the checked-in CSV, without downloading models or contacting cloud services:

```powershell
.\.venv\Scripts\python.exe reports/20261020T211121Z-full/reproduce_overview.py
```

## Pre-run benchmark forecasts (not actual runtime or spend)

Measured on NVIDIA RTX 2000 Ada Generation at compute plus running storage rate $0.2456/hour. These are conservative estimates, not completed full runs; setup validation spend and retained-storage reserve are additional.

- G5: 7.13 hours; $1.75.
- G10: 13.77 hours; $3.38.
