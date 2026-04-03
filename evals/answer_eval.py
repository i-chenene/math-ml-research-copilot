import json
import os
import re
from pathlib import Path

import requests


def run() -> None:
    backend = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
    dataset_path = Path("evals/dataset/sample_eval.json")
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))

    citation_ok = 0
    keyword_ok = 0
    for row in rows:
        response = requests.post(
            f"{backend}/qa/ask",
            json={"question": row["question"], "top_k": 5},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        answer_text = payload["answer"].lower()

        has_citation = bool(re.search(r"\[[^\]]+ p\.\d+\]", answer_text)) or bool(
            payload["citations"]
        )
        has_keyword = any(term.lower() in answer_text for term in row.get("expected_terms", []))
        citation_ok += int(has_citation)
        keyword_ok += int(has_keyword)

    n = len(rows) or 1
    print(f"Citation coverage: {citation_ok / n:.2f}")
    print(f"Keyword coverage: {keyword_ok / n:.2f}")
    print("TODO: replace this heuristic with rubric-based grading and model-judge scoring.")


if __name__ == "__main__":
    run()
