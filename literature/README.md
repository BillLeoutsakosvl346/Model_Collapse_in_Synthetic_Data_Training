# Research literature

Public citations and our research notes belong here. Local PDF copies are private and ignored under `papers/`; cloning the repository does not download them. The project proposal has a separate home in [project documents](../docs/project/README.md).

| Reference | Relationship to this project | Local PDF |
| --- | --- | --- |
| Shumailov et al. (2024), *AI models collapse when trained on recursively generated data*, Nature. [DOI](https://doi.org/10.1038/s41586-024-07566-y) | Motivation for recursive synthetic-data experiments; our exact reconstruction choices are in the [methodology](../docs/methodology.md). | `papers/shumailov_2024_model_collapse.pdf` |
| Seddik et al. (2024), *How Bad is Training on Synthetic Data? A Statistical Analysis of Language Model Collapse*. [arXiv:2404.05090v1](https://arxiv.org/abs/2404.05090v1) | Related analysis of synthetic-only and mixed real/synthetic training; motivates the separately defined [50/50 condition](../experiments/opt125m_wikitext2/continued_mixed_50_final5/README.md). Our run is not a claim to reproduce its numerical results. | `papers/seddik_2024_synthetic_data.pdf` |

[BibTeX references](references.bib) contain the bibliographic records. Titles/authors were checked against the supplied PDFs; the Seddik identifier was also checked against the primary arXiv record. These are the supplied editions, not a claim that all later revisions have been reviewed.

Future reading notes can use `notes/<citation-key>.md`, separating the paper's protocol/findings, our interpretation and differences from our experiments. Store new experiment measurements in [reports](../reports/README.md), not literature notes.
