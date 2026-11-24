# Prompts and Run Log

## Dataset labeling prompt
You are reviewing a GitHub pull request event. Classify whether the event explicitly uses GPT/ChatGPT/LLM assistance for code-review insight or decision support.

Return only:
- `assistance`: GPT/ChatGPT/LLM is explicitly used for code-review or decision support.
- `non_assistance`: GPT/ChatGPT/LLM is absent or is used only for implementation/API access.
- `uncertain`: evidence is insufficient.

Evidence fields: pull request title, body, comments, filenames, timeline events, and shared links. Do not infer assistance from project technology or generic "AI" wording.

## RAG research prompt
Use only the retrieved OA vector-database/RAG passages and retrieved paper passages to answer the assigned question. Cite the source filename and section/page when available. Separate reported findings from independent analysis. Do not claim causation unless randomized or strongly identified evidence exists.

## RQ prompts
- RQ1: Is GPT assistance associated with shorter pull-request resolution time?
- RQ2: Is GPT assistance associated with shorter submission, review, waiting-before-change, and waiting-after-change phases?
- RQ3: Which code-review tasks are most frequently associated with GPT-assisted events?

## Reproducibility rule
Record model name, model version, retrieval corpus, retrieval count, prompt version, temperature, seed, timestamp, and any manual review decisions. Never treat keyword matches as ground truth.
