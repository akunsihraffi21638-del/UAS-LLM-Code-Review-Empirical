# UAS LLM Code Review Experiment

This scratch workspace contains a controlled replication/extension of the paper
“The Impact of Large Language Models (LLMs) on Code Review Process.”

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python experiment.py --baseline baseline --output outputs
python rag_vector_index.py --corpus rag_corpus --query "How does vector retrieval support RAG for code review evidence?" --k 5 --output outputs
```

## Method

1. Preserve the companion repository as the baseline evidence source.
2. Load the baseline RQ1 merged-PR CSV and project-phase JSON.
3. Clean missing numeric fields and compare GPT versus non-assisted groups.
4. Report median and Mann-Whitney U tests.
5. Fit an exploratory log-linear OLS model and nearest-neighbor matched comparison.
6. Produce RQ2 phase medians and RQ3 keyword frequencies.
7. Build a local TF-IDF sparse vector index over OA and paper extracts for retrieval demonstrations.

Association, not causal effect, is the output of this observational design.
