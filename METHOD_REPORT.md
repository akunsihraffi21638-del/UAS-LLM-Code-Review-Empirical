# Empirical Methodology and Report

## Research Questions

- **RQ1:** Is GPT assistance associated with pull-request resolution time?
- **RQ2:** Is GPT assistance associated with differences across PR phases?
- **RQ3:** For what code-review tasks is GPT assistance used?

## Data

The baseline is the paper’s public companion repository. It contains the baseline
`at_resolution.csv`, phase JSON, pull timelines, and R scripts. This replication
uses those published artifacts and does not claim independent reconstruction of
all 25,473 original candidates.

## OA-to-Experiment Link

The OA vector-database material establishes embeddings, similarity distances,
native vector storage, SQL similarity search, approximate nearest-neighbor
indexes, RAG, and agentic RAG. The experiment applies the RAG retrieval pattern
to evidence passages with a local sparse-vector TF-IDF index and cosine
similarity. A production deployment could replace TF-IDF with dense embeddings
and a native vector database such as Oracle 23ai.

## Procedure

1. Acquire the companion repository and inspect schemas.
2. Filter RQ1 rows with complete numeric fields.
3. Compare GPT-assisted and non-assisted PR resolution distributions.
4. Apply Mann-Whitney U, nearest-neighbor matching, and log-linear OLS sensitivity analysis.
5. Extract phase durations from the baseline phase JSON.
6. Calculate phase medians and Mann-Whitney tests.
7. Generate a keyword-frequency RQ3 proxy.
8. Chunk OA/paper text, vectorize it, retrieve passages, and log provenance.

## Reporting Rules

The final report must distinguish descriptive associations from causation.
It must disclose keyword false positives/negatives, manual-label uncertainty,
selection bias, confounding, platform/API differences, and the inability to
observe latent developer intent.

## Measured Baseline-Artifact Results

From the companion repository's RQ1 artifact:

| Measure | GPT-assisted | Non-assisted |
|---|---:|---:|
| N | 1,368 | 1,036 |
| Median merge/resolution time | 20.845 h | 51.713 h |

The median difference was -30.868 hours (GPT minus non-assisted), with a Mann-Whitney
p-value of 3.674e-16 and a bootstrap median-difference 95% interval of -28.683
to -14.012 hours. The exploratory matched and log-linear sensitivity model produced
a negative log1p coefficient of -0.660 (p=4.890e-17). These results are associations
in an observational baseline artifact, not causal estimates.

The RQ2 paired-distance artifact produced medians of 1.159 h versus 0.001 h for
review, 38.341 h versus 18.703 h for waiting before change, 0.0795 h versus 0 h
for change, and 18.330 h versus 3.064 h for resolution. These phase results differ
from the paper's qualitative summary and must be interpreted with the matching,
sample, and event-mapping limitations visible in the companion artifact.

RQ3 is reported as an event-frequency proxy only: the companion `events_gpt.csv`
contains too little purpose-labelled data for a valid task distribution claim.

## Reproduce

Install dependencies, run `experiment.py`, and run `rag_vector_index.py`.
Commit the source and generated JSON/PNG files, but preserve source licensing
and size constraints before publishing a large dataset.
