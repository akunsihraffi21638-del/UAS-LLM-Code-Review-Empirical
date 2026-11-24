import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk(path: Path, max_words: int = 180):
    text = path.read_text(encoding="utf-8", errors="ignore")
    words = text.split()
    for i in range(0, len(words), max_words):
        chunk_words = words[i:i + max_words]
        yield {
            "source": path.name,
            "chunk": i // max_words,
            "text": " ".join(chunk_words),
        }


def build(corpus_dir: Path, query: str, k: int, output: Path):
    documents = []
    for path in sorted(corpus_dir.glob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            documents.extend(chunk(path))
    if not documents:
        raise SystemExit("No .txt or .md corpus documents found")

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([d["text"] for d in documents])
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked = scores.argsort()[::-1][:k]
    results = [{"rank": i + 1, "score": float(scores[idx]), **documents[idx]} for i, idx in enumerate(ranked)]

    output.mkdir(parents=True, exist_ok=True)
    (output / "rag_retrieval.json").write_text(json.dumps({
        "query": query,
        "model": "local TF-IDF sparse vector index",
        "metric": "cosine similarity",
        "results": results,
    }, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    build(args.corpus, args.query, args.k, args.output)
