# Math/ML Research Copilot (MVP)

Local-first research assistant for machine learning and mathematics papers.

## Why This Project

Most AI paper assistants look good in demos but fail on trust and reproducibility.
This MVP focuses on practical research behavior:

- ingest real papers (PDF)
- retrieve relevant evidence locally
- answer questions with explicit citations
- compare papers in a structured format
- include lightweight evaluation scripts from day one

## What It Does

- PDF ingestion and text chunking with page metadata
- local semantic retrieval with FAISS
- QA grounded in retrieved chunks with citations like `[doc_name p.3]`
- paper comparison on objective/method/datasets/metrics/strengths/limitations
- retrieval and answer-quality eval skeletons

## Demo In 60 Seconds

1. Upload 1-2 PDFs from Streamlit.
2. Ask a question over your corpus.
3. Read an answer with citations.
4. Compare two papers side-by-side in JSON output.

Demo video: `TODO`  
Screenshots:

- Upload and document list: `docs/screenshots/upload.png`
- QA with citations: `docs/screenshots/qa.png`
- Comparison output: `docs/screenshots/compare.png`

## Demo Checklist

Before sharing the project publicly:

- [ ] Add a 60-90s demo video link
- [ ] Add 3 screenshots in `docs/screenshots/`
- [ ] Show one QA example with citations in README
- [ ] Mention one known limitation and one next step

## Architecture

- FastAPI backend: `app/backend/main.py`
- Streamlit frontend: `app/frontend/streamlit_app.py`
- Core services:
  - PDF extraction: `app/services/pdf_ingestion.py`
  - chunking: `app/services/chunking.py`
  - embeddings/provider abstraction: `app/services/providers.py`
  - persistence (SQLite): `app/services/repository.py`
  - vector index (FAISS): `app/services/vector_store.py`
  - ingestion/retrieval/QA/comparison/index rebuild orchestration
- Evals: `evals/`

More details: `docs/architecture.md`.

## Quickstart (Windows-friendly)

```bash
python -m pip install -e ".[dev]"
copy .env.example .env
python -m uvicorn app.backend.main:app --reload --port 8000
```

In a second terminal:

```bash
python -m streamlit run app/frontend/streamlit_app.py
```

## Optional Setup with `uv`

```bash
uv venv
uv pip install -e ".[dev]"
copy .env.example .env
uv run uvicorn app.backend.main:app --reload --port 8000
uv run streamlit run app/frontend/streamlit_app.py
```

## API Endpoints

- `GET /health`
- `POST /documents/upload`
- `GET /documents`
- `POST /retrieval/search`
- `POST /qa/ask`
- `POST /papers/compare`
- `POST /index/rebuild`
- `POST /documents/arxiv` (stub with TODO)

## Tests and Lint

```bash
python -m ruff check .
python -m pytest -q
```

## Evals

Run backend first, then:

```bash
python evals/retrieval_eval.py
python evals/answer_eval.py
```

## Known Limitations

- PDF extraction can miss formulas, figures, and dense table content.
- Default `LLM_PROVIDER=mock` is local and deterministic but not semantic.
- arXiv ingestion endpoint is a clean stub.
- eval scripts are lightweight heuristics, not full benchmark tooling.

## Roadmap

1. Real arXiv ingestion (URL/ID -> PDF -> existing pipeline).
2. Math-aware extraction and section-aware chunking.
3. Stronger eval dataset with retrieval and faithfulness metrics.
4. Confidence guardrails for weak-evidence answers.

## What I Learned

- Citation-first UX is crucial for trust in research copilots.
- Service boundaries make experimentation much easier.
- Even small evals improve decision quality during iteration.
