# Autonomous Week 1 model collapse experiment

**Current execution boundary:** implementation and smoke validation only (G0-G2, 256 training blocks, one epoch, maximum USD 3). Do not execute the full-data priorities below until the user separately launches that experiment. Use README.md for the implemented workflow. No extra seeds or ablations are automatically authorized for this setup task.

Implement and execute a reproducible study of recursive synthetic-data fine-tuning with OPT-125M and WikiText-2. Deliver measured results, plots, checkpoints, synthetic datasets, a clear account of what ran, and a repository that can repeat it. Carry the work through execution and analysis, not just code generation.

## Scope and authorization

- Repository: `BillLeoutsakosvl346/Model_Collapse_in_Synthetic_Data_Training`.
- Budget: at most **USD 25 of incremental Runpod costs for this task**, including setup, failed attempts, GPU compute, container disks, volumes, and a stated allowance for any retained storage. The account balance is not the spending authorization.
- **No fixed wall-clock limit.** Finish sooner when the work is done. Compute budget-derived shutdown deadlines from the current price; do not impose an unrelated time limit or idle to fill time.
- Build, test, run, analyze, save results, and clean up task-owned compute autonomously. Resolve routine implementation decisions without waiting for the user.
- One GPU Pod at a time. Do not buy credits, enable automatic top-ups, provision clusters, or create warm serverless workers.
- Do not modify or stop pre-existing account resources. Record the IDs of every resource created by this task and operate only on those IDs.
- Do not add novel mitigation, filtering, real/synthetic mixtures, LoRA, quantization, or decoding ablations to the main experiment.
- Check all required credentials before spending. If access is missing or denied, finish useful local work and identify the exact blocker. Never solicit keys in ordinary chat output, retry a permission denial with broader credentials, or invent a successful connection.
- Use programmatic Runpod and Hugging Face access from the cloud environment and Pod. Do not depend on the user's local browser or sign-ins. Read `CLOUD_SETUP.md` for credential names, optional private artifact repositories, and raw-token versus proxy-secret behavior. Hugging Face supplies model/data access and optional private backups; do not provision paid Hugging Face compute or upgrades.

## Research context and sources

This Cambridge Engineering IIB project investigates model collapse in language models recursively fine-tuned on synthetic data. Later work may study early detection and mitigation. Week 1 establishes a reliable baseline, evaluation protocol, and recursive loop.

Primary paper: Shumailov et al., *AI models collapse when trained on recursively generated data*, Nature 631, 755-759 (2024), DOI [10.1038/s41586-024-07566-y](https://doi.org/10.1038/s41586-024-07566-y).

Other primary sources:

- [Authors' current code](https://github.com/iliaishacked/curse_recurse)
- [Archived code cited by the paper](https://zenodo.org/records/10866595)
- [OPT-125M model card](https://huggingface.co/facebook/opt-125m)
- [WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext)
- [Perplexity evaluation documentation](https://huggingface.co/docs/transformers/en/perplexity)

Read the language-model experiment and relevant code before finalizing the implementation. The paper describes 64-token blocks, 64-token continuations, five-beam search, five epochs in the condition retaining no real training data, fixed original training prompts at every generation, and real validation data for checkpoint selection. The authors ran multiple seeds; one seed here is initially preliminary evidence.

The released code has settings that do not fully specify the paper's runs. In particular, its generation call hardcodes `repetition_penalty=3.0`, while the paper describes repetition penalties in a separate ablation. Follow the explicit protocol below and document discrepancies. Paper prose about initialization/checkpoint selection also needs careful interpretation: the user explicitly requests reinitializing every fine-tuning run from the same pretrained checkpoint. Do not silently substitute continued training on the previous generation's weights.

The local proposal and paper may not exist in the cloud checkout. This brief carries the agreed project scope; fetch primary sources as needed. Do not publish the user's proposal automatically.

## Execution priorities

1. Build and validate a tiny end-to-end smoke test, clearly separated from reported scientific runs.
2. Run the untouched pretrained evaluation, full-data real baseline G0, and full-data synthetic generations G1 and G2 for seed 42.
3. Extend that same reproducible chain to G10 if the measured remaining cost fits the budget. G10 means ten synthetic generations after the real-data baseline; report this convention explicitly.
4. If sufficient budget remains after completing the first chain and reserving cleanup/storage costs, repeat the same protocol for seeds 43 and 44. Complete comparable chains rather than scattering many incomplete runs. Do not spend unnecessarily to exhaust the budget.
5. Analyze every completed full-data generation. If budget prevents G10, report the completed subset honestly, retain resumable state, and provide the command and estimated cost to continue.

Do not reduce the main dataset, epoch count, context length, or beam count merely to obtain a faster or more dramatic result. Use a smaller dataset/epoch count only for a separately labeled smoke test. Any necessary scientific deviation requires a new, clearly named configuration and an explanation.

## Experimental protocol

### Data and reproducibility

- Base model: `facebook/opt-125m`. Resolve and record its exact revision and tokenizer revision. Cache the original checkpoint and reload it for every generation.
- Dataset: `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, using its original train, validation, and test splits. Record the resolved version/revision, dataset fingerprint, preprocessing, and resulting token/block counts.
- Use the OPT tokenizer. Select and document newline handling, special-token behavior, concatenation, block boundaries, and dropped remainders. Inspect the authors' preprocessing and preserve it where practical. Freeze the resulting data manifest for the entire experiment.
- Pack training data into 64-token blocks. Never concatenate across train/validation/test boundaries.
- Create a fixed prompt set from the real training split only. Reuse it at every generation. Never use real validation or test text as synthetic-data prompts or as training targets.
- Keep validation and test data, their order, tokenizer, evaluation context, and scoring masks fixed. Use real validation loss for checkpoint selection. Test metrics are reporting measurements, not a basis for changing settings or choosing among experiments.
- Seed Python, NumPy, PyTorch, data order, and any stochastic operations. Record precision, deterministic settings and limitations, library versions, CUDA/driver, GPU, code commit, and the full resolved configuration.

### Training and generation

- Evaluate untouched pretrained OPT before fine-tuning; label it `pretrained`, separately from G0.
- G0: full-parameter causal-language-model fine-tuning on real WikiText-2 training blocks.
- Initial settings: five epochs per generation, AdamW, learning rate `2e-5`, effective batch size 128, cosine decay toward `1e-6`, and explicitly recorded optimizer defaults. These are reconstruction choices based on public code, not a guarantee of identical historical runs.
- Use BF16 or FP16 as supported and stable. Record the choice as an implementation detail/deviation. Test finite losses and gradients before a long run. Keep scientific settings consistent across generations.
- Use a microbatch that fits, with gradient accumulation to preserve the effective batch size. Keep generation batch size independently configurable. A memory-related microbatch adjustment is allowed; log it.
- Save the selected checkpoint using the lowest real validation loss under a fixed evaluation cadence, initially once per epoch. Reset optimizer and scheduler state between generations.
- To create the dataset for G(g+1), use the selected Gg model in evaluation mode with the fixed original real training prompts. Generate exactly 64 new tokens after each 64-token prompt with `num_beams=5`, `do_sample=false`, and `repetition_penalty=1.0`. Explicitly configure early-EOS/length handling to make the 64-token requirement reliable. Do not add an unreported top-k, top-p, repetition, or no-repeat constraint.
- Store only the 64 new token IDs as the next training targets, excluding the real prompt. Preserve the same number of training blocks and token budget across generations. Keep token IDs to avoid decode/re-encode drift; save decoded samples for inspection too.
- G1 and later: reload the identical original pretrained OPT weights, then fine-tune on the previous model's generated continuations. Do not start from G0 or the previous synthetic checkpoint.
- Use batched generation with KV caching. Keep model/data caches on persistent storage. Measure optimization changes before introducing complexity; this small model does not require a distributed training stack or a custom serving framework.

### Measurements

For every completed generation, save raw values plus a tidy CSV/JSON summary:

- Validation and real test negative log-likelihood, valid scored-token count, and corpus perplexity. Sum token NLL and divide by the exact count of scored, shifted, non-padding targets before exponentiating. Do not average batch perplexities.
- Train/validation loss by epoch or step, selected epoch/checkpoint, and train/evaluation/generation duration.
- Per-sequence loss/perplexity or a reproducible per-sequence evaluation export. Label arithmetic mean sequence perplexity separately from corpus perplexity. Compute an actual standard deviation when reporting one; `exp(std(loss))` is not a perplexity standard deviation.
- Distinct-1, distinct-2, and distinct-3; repeated 4-gram rate with its explicit denominator; duplicate continuation rate; token frequency/entropy summaries. Compare equal-length and equal-count output sets with identical definitions.
- Scores of synthetic continuations under a frozen G0 reference model, using a clearly specified scoring context, to support paper-style drift histograms. Keep this reference fixed rather than replacing it with the latest model.
- Continuations for a fixed documented training-prompt sample across generations. Exclude prompt tokens from generated-text diversity measurements.
- Training tokens/second, generation continuations/second and new tokens/second, peak allocated GPU memory, actual batch sizes, retries, resource IDs, quoted prices, and measured/estimated incremental costs distinguished by provenance.

The paper's reported perplexity values are reference context, not target values to manufacture. Do not change tokenization, context, evaluation, or checkpoints to chase a published number. Degradation may be noisy or weak; report what happened without claiming monotonic collapse is guaranteed.

## Repository and artifact organization

Implement a small modular Python repository with a configurable full-pipeline entry point and separate preparation, training, generation, evaluation, and plotting functions/scripts. Prefer a tested minimal PyTorch/Transformers implementation over a large framework rewrite.

Provide documented commands for installation, a tiny smoke test, a full run, resuming an interrupted run, and rebuilding plots/report from saved metrics without rerunning training. Pin the actual working dependency versions and model/data revisions; reuse the official Pod image's working PyTorch installation where practical.

Use a run layout equivalent to:

```text
configs/
src/ or scripts/
tests/
runs/<experiment_id>/
  manifest.json
  resources.json
  cost_ledger.jsonl
  events.jsonl
  seed_42/
    pretrained/metrics.json
    gen_00/
      config.json
      metrics.json
      checkpoint/
      samples.jsonl
      training_history.jsonl
    gen_01/ ... gen_10/
    synthetic_for_gen_01/ ... synthetic_for_gen_10/
reports/week1/
  report.md
  metrics.csv
  figures/
```

Write completion manifests atomically after verifying the corresponding files. Resume only from verified completed stages with matching code/config/data hashes. Do not overwrite a completed run or mix smoke-test data with full-data results. Keep all selected generation weights and synthetic datasets; optimizer state need only be retained where needed for resuming an unfinished training stage.

Commit code, configurations, README, tests, report, summary metrics, and reasonably sized figures to a working branch. Return a PR or commit link with a clear description of the actual result. Keep credentials, bulk datasets, caches, checkpoints, and large archives out of ordinary Git history. Do not publish large artifacts publicly without explicit authorization; provide a verified private artifact bundle or retained storage location instead.

## Cloud execution and budget protection

### Preflight before creating paid resources

1. Verify GitHub read/write and Runpod control-plane access without printing credentials. Read the starting account balance, live GPU price/availability, image/template configuration, and storage prices. Do not infer API access from a browser login. Resolve the public Hugging Face model and dataset programmatically. If `HF_TOKEN` is supplied, verify its identity and the actual permissions needed by the chosen private artifact destinations; a successful identity read does not prove write permission. Honor optional `HF_MODEL_REPO` and `HF_DATASET_REPO` destinations. Before relying on private uploads, complete a tiny task-owned upload/download/checksum probe. If the token can create repositories and no destinations are configured, create clearly named private task repositories and record their returned IDs. Check existing storage limits; do not incur an unapproved Hugging Face charge.
2. Read the current official Runpod tools/docs and inspect live command help. Prefer connected MCP tools for supported operations and current CLI/API for the rest. The local machine's installed skills are not automatically in Cloud; install the official skills or read official guides if needed.
3. Choose a single GPU that supports this full-parameter workload. The user prioritizes speed within budget. A stocked RTX 4090 is a reasonable initial candidate; consider a cheaper suitable GPU or a 5090 only from current prices, compatibility, and expected/measured cost. Do not use old advertised rates as a deployment quote.
4. Prefer an official PyTorch Pod image and Secure Cloud, unless the user has explicitly requested another cloud type. Match the host CUDA floor to the selected image, and co-locate any volume with actual GPU stock. Set only the needed ports and credentials.
5. Confirm the code transfer, execution, status/log retrieval, backup, and shutdown paths work in this cloud environment. SSH is optional, not assumed. An HTTPS control-plane launch of a pinned startup script is acceptable. Make the exact code commit available to the Pod before provisioning it.
6. Decide where every large artifact will remain recoverable after GPU shutdown. Verify the destination and permissions. Avoid an ephemeral container disk as the only copy. A network volume is acceptable, but its continuing storage cost must be reserved and reported.
7. Establish a conservative cost ledger and an independent budget watchdog. If a safe control/shutdown path cannot be verified, do not start a long paid run.

### Paid execution

- Start with a tiny end-to-end run and a short warmed-up benchmark of both training and five-beam generation. Include setup and failed attempts in the cost ledger. Extrapolate from the actual frozen dataset size, five epochs, and generation batches before deciding how many full generations fit.
- The experiment must run as a detached Pod-side batch job or equivalent independent job. It must survive SSH/client disconnection and the Codex agent stopping. Do not make continued training depend on a local terminal, an active browser, or the agent continuing to poll.
- Save logs, state, metrics, and checkpoint progress continuously on persistent storage. Generate the final figures and a factual machine-produced report from the pipeline itself; an agent can add interpretation afterward.
- Stop beginning new stages when accrued cost plus the conservative next-stage forecast and shutdown/storage reserve would exceed USD 25. Costs must include every task-owned Pod and disk/volume, including failed attempts and idle time.
- Enforce a price-derived maximum paid runtime with an independent watchdog and a conservative margin for API delays, shutdown retries, and storage. This cutoff exists solely to enforce the dollar budget. Reconcile against current billing reads when available and label estimates. Do not rely on a stale CLI `--stop-after` or `--terminate-after` flag without current evidence that the backend enforces it.
- Test the shutdown mechanism with a short bounded validation before the long run, verifying that the GPU actually stops. Record the evidence. A `finally` block or successful stop request alone is not sufficient proof of stopped compute; use an independent control/read-back path as well.
- Stop and preserve partial results on failure or budget exhaustion. Avoid repeated blind retries or silently launching duplicate Pods. Reconcile any timed-out create with a fresh resource read before retrying.

### Completion and cleanup

- Verify artifact integrity with sizes/checksums and a manifest. Provide an artifact index and a usable retrieval route that does not require the GPU to remain running.
- Stop the task's GPU promptly on completion, failure, or the budget cutoff. Read back its final state. After verified backup, remove task-created compute that no longer needs to exist. Never delete the only copy of results.
- Prefer exporting all results outside paid Runpod storage and removing task-owned storage afterward. If a volume must remain for recovery, retain only that storage, report its ID, live rate, and retrieval/deletion instructions, and reserve a clearly stated retention allowance within the budget. Do not claim zero ongoing cost while retaining a paid volume.
- If independent shutdown or backup verification fails, report that exact unresolved state prominently. Do not describe the run as safely finished while a GPU may still be billing.

## Verification and final deliverables

Use meaningful checks of token-loss aggregation and masking, data separation, exact continuation length/prompt removal, fixed base initialization, resume behavior, and the budget/shutdown path. Run an actual small end-to-end job before scaling. Inspect generated plots for legible axes, correct generation order, and consistency with saved metrics. Tests must cover relevant failure modes rather than restating implementation details.

The final report must include:

1. What actually ran: model/data revisions, seeds, completed generations, full-data versus smoke-test status, epochs, optimizer/precision, decoding, checkpoint rule, and evaluation definition.
2. A generation table showing validation/test loss and perplexity, diversity/repetition, run time, and cost provenance. Missing generations remain missing, not filled with inferred values.
3. Figures for real-text loss/perplexity, diversity and repetition, and frozen-reference output-score distributions when collected. With repeated seeds, show each run and an appropriate mean/spread. With one seed, label it preliminary and omit misleading across-seed error bars.
4. Fixed-prompt examples across generations and a short evidence-based interpretation of whether/how degradation appeared. Explain deviations and plausible measurement limitations without claiming proof beyond the data.
5. Actual tests and checks, failures/retries, resource/cost ledger, remaining budget, and actual versus estimated costs.
6. Links/paths for code, configurations, metrics, figures, every checkpoint and synthetic dataset, the artifact manifest, and exact rerun/resume commands.
7. Final read-back of task-created resource states, any continuing storage charges, and precise recovery/cleanup instructions.

Return the measured outcome even if the result is negative, incomplete, or budget-limited. A running job, a planned experiment, or an untested script is not a completed scientific result. If the agent session ends while the independent job continues, explicitly mark that state as pending and provide the verified job ID, artifact location, budget cutoff, and status/recovery command; never fabricate the future result.
