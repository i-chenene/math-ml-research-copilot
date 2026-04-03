import json
import os
from pathlib import Path

import requests


def run() -> None:
    backend = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
    dataset_path = Path("evals/dataset/sample_eval.json")
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    k = 5
    hits = 0
    details = []

    for row in rows:
        response = requests.post(
            f"{backend}/retrieval/search",
            json={"query": row["question"], "top_k": k},
            timeout=60,
        )
        response.raise_for_status()
        chunks = response.json()["chunks"]
        expected_doc = row["expected_doc_name_contains"].lower()
        hit = any(expected_doc in chunk["document_name"].lower() for chunk in chunks)
        hits += int(hit)
        details.append({"id": row["id"], "hit": hit, "returned": len(chunks)})

    score = hits / len(rows) if rows else 0.0
    print(f"Hit@{k}: {score:.2f} ({hits}/{len(rows)})")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    run()
