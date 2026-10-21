# Project instructions

This Cambridge Engineering IIB project studies model collapse in language models trained recursively on synthetic data. Week 1 establishes a reproducible baseline with OPT-125M and WikiText-2. Novel mitigation methods are out of scope.

Setup and validation completed, including a tiny real GPU G0-G2 smoke experiment, recovery, private backups and shutdown, below the USD 3 setup allowance. The user subsequently explicitly authorized the full-data experiment `20261020T211121Z-full`; G0-G10 completed on 21 October 2026 and GPU compute stopped automatically. Results are in `reports/20261020T211121Z-full/`, with bulk checkpoints and datasets in private Hugging Face backups. Current work is analysis, reporting and publication to the existing GitHub repository. Do not launch additional paid experiments unless explicitly requested by the user. This supersedes older automatic-execution priorities in WEEK1_CLOUD_TASK.md.

Read [WEEK1_CLOUD_TASK.md](WEEK1_CLOUD_TASK.md) for the scientific protocol. Use README.md for the implemented local-to-Runpod workflow; Codex Cloud is optional.

The user's current instruction is a maximum of **USD 25 in incremental Runpod costs for this task**, with **no fixed time limit**. Preserve a conservative reserve for storage and shutdown. Do not confuse this budget with permission to spend the entire account balance. Prefer completing a sound experiment below the ceiling.

Use live GPU prices and availability. Provision at most one GPU Pod at a time. Verify credentials, an artifact recovery route, and a shutdown mechanism before a long paid run. Budget protection must survive the agent session ending.

Never expose or commit credentials. Do not commit the local proposal or paper automatically. Never invent measurements, claim a launched job has finished, or describe a single run as statistically established evidence.

Only modify or clean up cloud resources created for this task, recording their returned IDs. Preserve other account resources. Save work and report concrete blockers when credentials, permissions, budget, or infrastructure prevent completion.
