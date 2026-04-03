from dataclasses import dataclass

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.models.schemas import (
    AskRequest,
    AskResponse,
    CompareRequest,
    CompareResponse,
    DocumentOut,
    HealthResponse,
    RetrievalRequest,
    RetrievalResponse,
    UploadResponse,
    UploadResult,
)
from app.services.chunking import TextChunker
from app.services.comparison import PaperComparisonService
from app.services.indexing import IndexService
from app.services.ingestion import IngestionService
from app.services.pdf_ingestion import PDFIngestionError, PDFIngestionService
from app.services.providers import OpenAICompatibleProvider
from app.services.qa import QAService
from app.services.repository import SQLiteRepository
from app.services.retrieval import RetrievalService
from app.services.vector_store import FaissStore

FILES_PARAM = File(...)


@dataclass
class ServiceContainer:
    ingestion: IngestionService
    retrieval: RetrievalService
    qa: QAService
    comparison: PaperComparisonService
    repository: SQLiteRepository
    index_service: IndexService


def build_container() -> ServiceContainer:
    settings = get_settings()
    repository = SQLiteRepository(db_path=settings.sqlite_path)
    provider = OpenAICompatibleProvider(settings)
    vector_store = FaissStore(
        index_path=settings.faiss_path, embedding_dim=settings.embedding_dim
    )
    chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
    retrieval = RetrievalService(
        repository=repository, vector_store=vector_store, provider=provider
    )
    return ServiceContainer(
        ingestion=IngestionService(
            repository=repository,
            pdf_service=PDFIngestionService(),
            chunker=chunker,
            provider=provider,
            vector_store=vector_store,
        ),
        retrieval=retrieval,
        qa=QAService(retrieval_service=retrieval, provider=provider),
        comparison=PaperComparisonService(repository=repository),
        repository=repository,
        index_service=IndexService(
            repository=repository, provider=provider, vector_store=vector_store
        ),
    )


app = FastAPI(title="Math/ML Research Copilot API", version="0.1.0")
container = build_container()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = FILES_PARAM) -> UploadResponse:
    uploaded: list[UploadResult] = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported in MVP.")
        payload = await file.read()
        try:
            doc_id, count = container.ingestion.ingest_pdf(name=file.filename, pdf_bytes=payload)
        except PDFIngestionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        uploaded.append(UploadResult(document_id=doc_id, name=file.filename, chunks_created=count))
    return UploadResponse(uploaded=uploaded)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents() -> list[DocumentOut]:
    return container.repository.list_documents()


@app.post("/retrieval/search", response_model=RetrievalResponse)
def retrieval_search(payload: RetrievalRequest) -> RetrievalResponse:
    settings = get_settings()
    top_k = payload.top_k or settings.top_k
    chunks = container.retrieval.retrieve(query=payload.query, top_k=top_k)
    return RetrievalResponse(chunks=chunks)


@app.post("/qa/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    settings = get_settings()
    top_k = payload.top_k or settings.top_k
    return container.qa.ask(question=payload.question, top_k=top_k)


@app.post("/papers/compare", response_model=CompareResponse)
def compare(payload: CompareRequest) -> CompareResponse:
    try:
        return container.comparison.compare(payload.document_a_id, payload.document_b_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/index/rebuild")
def rebuild_index() -> dict[str, int]:
    rebuilt = container.index_service.rebuild()
    return {"chunks_indexed": rebuilt}


@app.post("/documents/arxiv")
def ingest_arxiv_stub(url: str) -> dict[str, str]:
    return {
        "status": "stub",
        "message": (
            f"arXiv ingestion for {url} is not implemented yet. "
            "TODO: download PDF and pass bytes through existing ingestion service."
        ),
    }
