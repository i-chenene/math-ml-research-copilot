# Math/ML Research Copilot (MVP)

Local-first research assistant for machine learning and mathematics papers.

## Goal

This project provides a recruiter-grade MVP scaffold that supports:

- PDF ingestion and text chunking with page tracking
- local indexing with FAISS
- retrieval and question answering with citations
- paper-to-paper structured comparison
- simple evaluation scripts for retrieval and answer quality

## Architecture

- FastAPI backend: `app/backend/main.py`
- Streamlit frontend: `app/frontend/streamlit_app.py`
- Services:
  - PDF extraction: `app/services/pdf_ingestion.py`
  - chunking: `app/services/chunking.py`
  - embeddings + LLM provider abstraction: `app/services/providers.py`
  - persistence (SQLite): `app/services/repository.py`
  - FAISS index: `app/services/vector_store.py`
  - retrieval/QA/comparison/index rebuild
- Evals: `evals/`

More details: `docs/architecture.md`.

## Setup

### Option A: `uv` (recommended)

```bash
uv venv
uv pip install -e ".[dev]"
cp .env.example .env
```

### Option B: `pip`

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
cp .env.example .env
```

## Run

Start API:

```bash
uv run uvicorn app.backend.main:app --reload --port 8000
```

Start frontend in another terminal:

```bash
uv run streamlit run app/frontend/streamlit_app.py
```

## API Endpoints

- `GET /health`
- `POST /documents/upload` (multipart PDFs)
- `GET /documents`
- `POST /retrieval/search`
- `POST /qa/ask`
- `POST /papers/compare`
- `POST /index/rebuild`
- `POST /documents/arxiv` (stub)

## Tests and Lint

```bash
uv run ruff check .
uv run pytest -q
```

## Evals

Run backend first, then:

```bash
uv run python evals/retrieval_eval.py
uv run python evals/answer_eval.py
```

## Known limitations

- PDF extraction may miss formulas/figures and complex layouts.
- Mock provider is deterministic but not semantic; use `LLM_PROVIDER=openai` for real embeddings/answers.
- arXiv ingestion is a clean TODO stub.
- Eval scripts are intentionally lightweight heuristics.

## Roadmap

1. Add robust arXiv fetch + PDF ingestion endpoint.
2. Upgrade chunking (section-aware split, equation-aware parsing).
3. Add richer eval set and judge-based answer scoring.
4. Add compare mode that uses structured extraction prompts with confidence scores.
