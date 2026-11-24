$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m pip install --disable-pip-version-check -r requirements.txt
if (-not (Test-Path "baseline")) {
    git clone --depth 1 https://github.com/acollant/GPT-Assistance-PR.git baseline
}
python experiment.py --baseline baseline --output outputs
python rag_vector_index.py --corpus rag_corpus --query "How do embeddings, vector similarity, and RAG support code review evidence retrieval?" --k 5 --output outputs
python -m unittest test_experiment.py
