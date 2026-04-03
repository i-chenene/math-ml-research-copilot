from pathlib import Path

import fitz

from app.core.config import Settings
from app.services.chunking import TextChunker
from app.services.ingestion import IngestionService
from app.services.pdf_ingestion import PDFIngestionService
from app.services.providers import OpenAICompatibleProvider
from app.services.repository import SQLiteRepository
from app.services.vector_store import FaissStore


def _build_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "We propose a method for gradient-based optimization in neural nets.",
    )
    payload = doc.write()
    doc.close()
    return payload


def test_ingestion_pipeline(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite3"
    index_path = tmp_path / "test.index"
    settings = Settings(
        llm_provider="mock",
        sqlite_path=db_path,
        faiss_path=index_path,
        data_dir=tmp_path,
        embedding_dim=64,
        chunk_size=200,
        chunk_overlap=20,
    )
    settings.ensure_paths()

    repo = SQLiteRepository(db_path=db_path)
    provider = OpenAICompatibleProvider(settings)
    store = FaissStore(index_path=index_path, embedding_dim=settings.embedding_dim)

    service = IngestionService(
        repository=repo,
        pdf_service=PDFIngestionService(),
        chunker=TextChunker(chunk_size=200, chunk_overlap=20),
        provider=provider,
        vector_store=store,
    )
    doc_id, chunks_count = service.ingest_pdf(name="toy.pdf", pdf_bytes=_build_pdf_bytes())

    assert doc_id > 0
    assert chunks_count >= 1
    docs = repo.list_documents()
    assert docs[0].name == "toy.pdf"
