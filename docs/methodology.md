# Shared scientific methodology

This protocol reconstructs selected ideas from [Shumailov et al. (2024)](https://doi.org/10.1038/s41586-024-07566-y), rather than claiming exact numerical reproduction. Named configurations define scientific conditions; dated records establish what actually ran. See the [experiment catalogue](../experiments/catalog.yaml), [comparison index](../comparisons/README.md) and [literature](../literature/README.md).

## Invariants: model, data and evaluation

| Setting | Value |
| --- | --- |
| Model/tokenizer | `facebook/opt-125m`, revision `27dcfa74d334bc871f3234de431e71c6eeba5dd6` |
| Dataset | `Salesforce/wikitext`, `wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3` |
| Block/context | 64 tokens; 63 shifted prediction targets per independent block |
| Seed | 42 |
| Full real training split | 37,946 blocks / 2,428,544 tokens / 2,390,598 scored targets |
| Frozen validation split | 3,922 blocks / 251,008 tokens / 247,086 scored targets |
| Frozen test split | 4,494 blocks / 287,616 tokens / 283,122 scored targets |

Preserve row strings, blank rows and newlines. Apply OPT's default special-token behaviour per row, concatenate independently within each original split, form non-overlapping blocks, and discard only the final incomplete block (58/41/29 tokens for train/validation/test). Never concatenate across splits. Validation/test blocks never supply training targets or generation prompts.

Evaluate untouched pretrained OPT separately from the real-data-fine-tuned baseline **S0/G0** (`gen_00`). Frozen real evaluation arrays and their hashes remain fixed. Use token-weighted corpus NLL/perplexity; test never selects checkpoints or tunes settings. Smoke subsets validate the pipeline and do not establish collapse or mitigation effectiveness.

## Generation and training inputs

Each original real training block supplies the same fixed 64-token prompt at every cycle. The preceding selected model generates exactly 64 new tokens using five deterministic beams, `do_sample=false`, repetition penalty 1.0, no no-repeat constraint and KV caching. Equal minimum/maximum continuation length suppresses EOS before the required length. Preserve token IDs directly; prompts are excluded from synthetic targets and no decoding/re-encoding changes them.

A full synthetic **pool** has 37,946 continuation blocks / 2,428,544 tokens. In all-synthetic conditions that pool is the next training dataset. In mixed conditions only the selected synthetic rows enter the training mixture; the full pool and actual targets have separate manifests/hashes. Decoded examples are inspection aids with unverified factual content.

The first 256 fixed prompts also form the measurement probe. Probe output at Sk describes D(k+1)-style text; real-text evaluation at Sk describes the model after learning from Dk. Preserve this one-cycle offset. Whether a final unused full dataset is generated is condition-specific: the 50/50 run omits full D11 and retains the final probe.

## Training and checkpoint policy

Full stages use five epochs of full-parameter fine-tuning: AdamW, LR `2e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay 0.01; cosine decay by epoch towards `1e-6`; effective batch 128, microbatch 32, accumulation 4; BF16 autocast with FP32 parameters. Restart optimizer and scheduler each cycle, including continued-weight training. Real validation is evaluated after every epoch; no validation-based early stopping is used.

| Selection policy | Checkpoint passed to evaluation, generation and the next cycle |
| --- | --- |
| `lowest_fixed_real_validation_loss` (omitted setting/default) | Newly trained epoch with minimum frozen real validation NLL; the untrained parent is not a rollback candidate |
| `final_epoch` | Terminal epoch, regardless of validation loss; final-epoch-5 conditions always retain epoch 5 |

Record all epoch losses/LRs, executed and selected epochs, minimum/selected validation loss and checkpoint hashes. Historical Week 1/2 stages executed five epochs but selected epoch 1 throughout. Their carried checkpoints therefore do not inherit fifty synthetic epochs. Final-epoch-5 continued S10 inherits fifty synthetic epochs; fixed-S0 A10 inherits its own five from S0 while retaining recursive data history.

## Initialization policy and shared beginnings

| Policy | Initializer at synthetic cycle k | Synthetic source generator |
| --- | --- | --- |
| `fresh_base` / pretrained reset | Identical original pretrained OPT | Preceding selected model |
| `continued` | Preceding selected checkpoint, starting from real-trained S0 | That same preceding checkpoint |
| `reset_s0` | Fixed real-trained S0 | Preceding selected checkpoint in the reset branch |

Two comparisons have different shared artifacts:

- [Week 2 shared start](../comparisons/shared_s0_s1_reset_vs_continue/README.md): import historical continued S0, selected-epoch-1 S1 and exact D2. Reset first performs new cycle 2 from S0 on D2; continued uses S1 on that same D2.
- [Week 3 final epoch 5](../comparisons/shared_start_final_epoch5/README.md): import historical S0/D1, train a **new epoch-5 S1**, then generate **new D2**. Both arms share that beginning; reset cycles 2-10 initialize from S0 while continued carries epoch-5 weights.

Imported stages retain their measurement identities, environment, timings and original costs. They are references to existing work, not new executions. Later datasets are branch-specific. Never reinterpret original-pretrained reset as fixed-S0 reset.

## Continued 50/50 real/synthetic training

The [mixed condition](../experiments/opt125m_wikitext2/continued_mixed_50_final5/README.md) imports exact historical S0 and its D1, then trains a new mixed-data S1. It never reuses an all-synthetic-trained S1. Continue selected epoch-5 weights for ten cycles.

| Source per full training mixture | Blocks | Tokens |
| --- | ---: | ---: |
| Real WikiText training split | 18,973 | 1,214,272 |
| Preceding selected model's synthetic continuations | 18,973 | 1,214,272 |
| Total | 37,946 | 2,428,544 |

Sample without replacement within each source per cycle; real blocks may recur in later cycles. Separate PCG64 sampling/shuffle streams derive from seed and cycle. Freeze the mixture across all five epochs and interruption/recovery. Save source indices, source labels, permutation, source hashes, generator identity, tokens and atomic completion manifest.

`synthetic_fraction` supports 0-1, rounding synthetic counts to the nearest block, half upwards; real rows fill the remainder. Record the effective fraction. Omitted/1.0 preserves all-synthetic behaviour. Generating the full synthetic pool before sampling is retained in this implementation. The validated condition is 0.50; support for other fractions is not GPU acceptance or launch authorization for them.

The fixed-total-block mixture is related to [Seddik et al. (2024)](https://arxiv.org/abs/2404.05090v1), but it is not their exact theoretical setup or a numerical reproduction claim. A separate 20% condition has not been accepted by the 50% certificate.

## Evidence and limitations

Keep configuration, runtime and environment identities separate. Record pinned commits, initializer/generator/data/output hashes, selected epochs, retries, timings, resource IDs and cost provenance. [Metrics](metrics.md) defines probe scope and denominators; reports distinguish primary raw-token measurements from post-hoc formatting sensitivity.

Reconstruction differences include whole-split packing rather than the authors' tokenizer-map batches, repetition penalty 1.0 rather than the released call's 3.0, explicit initialization policies, modern libraries, BF16, token-weighted perplexity and one seed. Do not change these to chase a published curve. Different GPUs/libraries can affect evaluation and deterministic beams despite matching seeds. Historical cross-GPU comparisons and the later matched-environment comparisons must be described separately. Non-monotonic loss and mixed diversity do not establish a causal mechanism or significance.
