# Current workflow
The implemented workflow builds locally and launches a detached Runpod batch job. Follow README.md. Codex Cloud and browser control are optional; no OpenAI API key is required for the experiment. The current setup task stops after smoke validation. The remaining Cloud setup notes are an optional alternative, not an instruction to start a full run automatically.

# Codex Cloud setup for Week 1

Create a cloud environment for `BillLeoutsakosvl346/Model_Collapse_in_Synthetic_Data_Training`, using the branch containing `WEEK1_CLOUD_TASK.md`. The cloud task will implement the experiment and run GPU work on Runpod. The laptop can be switched off after the cloud environment has been published and the task has actually started.

## Account access

1. In Codex, choose **Work in > Cloud**, create or select an environment, and connect this GitHub repository. Allow the cloud task to push a working branch and create a pull request or otherwise return its changes.
2. Supply `RUNPOD_API_KEY` privately in the cloud environment. The task needs GPU catalog and account reads, management of its own Pod and storage, and the ability to stop its Pod. Give it only the necessary permissions supported by Runpod. Do not paste the key into a task prompt, repository file, or commit.
3. Enable the network destinations the chosen implementation actually uses. Runpod API calls and remote artifact retrieval require internet access; allow access to the relevant Runpod HTTPS API, GitHub, package registries, and any chosen artifact store. Test the requests during setup. An allowed domain does not supply credentials.
4. Publish the prepared environment, then start the task with the prompt below. Local browser sign-ins, local skills, and uncommitted local files do not automatically transfer to Cloud.

OPT-125M and WikiText-2 are publicly downloadable. A Hugging Face token is optional for those downloads. A write token is needed only if the task uses a private Hugging Face repository as an artifact backup destination. Use a private destination for large artifacts unless public distribution has been explicitly authorized.

## Configure both services without browser automation

The cloud task should use API/SDK/CLI calls for Runpod and Hugging Face. A laptop browser session is not part of the runtime. Runpod's hosted MCP server is an optional control-plane interface; direct API/CLI access is sufficient. No Hugging Face MCP connection is needed to download models or upload artifacts with `huggingface_hub`.

Configure these values privately in the cloud environment or its Personal vault, and ensure the environment requests the matching keys:

| Key | Purpose | Required access |
| --- | --- | --- |
| `RUNPOD_API_KEY` | Discover GPUs and prices; create, inspect, and stop this task's Pod and storage | The necessary Runpod read/write permissions; read-only access cannot provision or stop compute |
| `HF_TOKEN` | Authenticate Hugging Face calls and back up selected checkpoints and synthetic datasets privately | Write access to the chosen private artifact repositories; public OPT/WikiText downloads alone do not require it |
| `HF_MODEL_REPO` | Optional explicit model-backup destination in `owner/name` form | Existing private repository accessible to the token |
| `HF_DATASET_REPO` | Optional explicit dataset-backup destination in `owner/name` form | Existing private repository accessible to the token |

Create a dedicated Runpod key from [Credentials > API keys](https://console.runpod.io/user/credentials), and a dedicated Hugging Face token from [Access Tokens](https://huggingface.co/settings/tokens). For narrowly scoped Hugging Face access, create the private artifact repositories first and grant the token write access only to those destinations. Alternatively, the agent may create private task repositories if the supplied token already authorizes that operation. Repository IDs are non-secret; token values are private.

Use private environment-variable values when a program or remote Pod must read the actual token. Use destination-scoped network secrets when all credential use stays in supported HTTPS requests from Codex Cloud. The distinction below matters when the job forwards credentials to a remote process.

Before the long run, require the cloud agent to:

1. Authenticate to Runpod and read the account/Pod state and GPU catalog.
2. Authenticate to Hugging Face, resolve OPT-125M and WikiText-2, and verify any configured private backup destinations.
3. If using Hugging Face backup, upload a tiny probe file to a task-owned location, download it again, and compare its checksum before trusting a large checkpoint upload. Do not overwrite or delete existing user data.
4. Verify that remote execution, status retrieval, artifact backup, and GPU shutdown can work without the laptop or an open browser.

Once these checks pass, the browser is only needed to view progress or results. An expired/insufficient token or a network restriction is a setup blocker, not a reason to request ongoing local browser access. Use Hugging Face Hub storage within existing account limits; do not enable paid Hugging Face inference, training jobs, subscriptions, or upgrades.

Network secrets and direct environment variables have different behavior. Current Codex Cloud network secrets give programs a placeholder that is substituted in allowed HTTPS requests. If a remote Runpod process needs a credential for self-shutdown, do not forward that placeholder to the Pod. Use an appropriate private environment variable or another tested remote shutdown design. Never log its value.

Cloud networking can differ from the laptop. Do not assume outbound SSH works. Verify the intended execution and monitoring path; a Pod startup job controlled over HTTPS is an alternative when SSH is unavailable.

## Prompt to paste into a new cloud task

> Read AGENTS.md and WEEK1_CLOUD_TASK.md and execute the Week 1 experiment end to end. Implement and test the repository, verify Runpod access, choose a suitable GPU using live prices and a short benchmark, run the real-data baseline and recursive synthetic generations, save all artifacts systematically, create the plots and evidence-based report, and release the GPU. Work autonomously on routine decisions. The maximum incremental Runpod spend for this task is USD 25, including disks and storage; there is no fixed time limit. Complete as much of the specified experiment as the budget supports, prioritizing one full-data ten-generation run. Do not stop after writing code or launching training. The remote job must save results and enforce its budget even if the agent session ends. Return actual results, costs, artifact locations, validation evidence, and remaining resource state. If a required credential or capability is missing, preserve your work and report the exact blocker before starting paid compute.

## Before leaving the laptop

- Confirm the environment is published and the cloud task is submitted and running, rather than still waiting for a secret or approval.
- The cloud agent must verify its remote job launch, checkpoint storage, monitoring path, and budget shutdown before describing the experiment as running unattended.
- Codex Cloud is a remote coding environment, not a guarantee of an indefinitely running agent turn. The experiment itself must be an independent remote job.

## Official documentation

- [Codex Cloud environments](https://learn.chatgpt.com/docs/environments/cloud-environments)
- [Local and cloud execution modes](https://learn.chatgpt.com/docs/environments/modes)
- [Runpod agent setup](https://docs.runpod.io/agent-setup.md)
- [Runpod credentials](https://docs.runpod.io/get-started/credentials)
- [Runpod hosted MCP server](https://docs.runpod.io/get-started/mcp-servers)
- [Hugging Face tokens and scopes](https://huggingface.co/docs/hub/security-tokens)
- [Hugging Face environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)

These instructions prepare a future cloud task. They do not mean training has already run or that a cloud environment has already been published.
