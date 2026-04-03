from app.services.chunking import TextChunker
from app.services.pdf_ingestion import PDFIngestionService
from app.services.providers import OpenAICompatibleProvider
from app.services.repository import SQLiteRepository
from app.services.vector_store import FaissStore


class IngestionService:
    def __init__(
        self,
        repository: SQLiteRepository,
        pdf_service: PDFIngestionService,
        chunker: TextChunker,
        provider: OpenAICompatibleProvider,
        vector_store: FaissStore,
    ) -> None:
        self.repository = repository
        self.pdf_service = pdf_service
        self.chunker = chunker
        self.provider = provider
        self.vector_store = vector_store

    def ingest_pdf(
        self,
        name: str,
        pdf_bytes: bytes,
        source_ref: str | None = None,
    ) -> tuple[int, int]:
        page_texts = self.pdf_service.extract_page_text(pdf_bytes)
        chunks = self.chunker.chunk_pages(page_texts)
        if not chunks:
            raise ValueError("No chunks were generated from PDF")

        doc_id = self.repository.create_document(
            name=name, source_type="pdf", source_ref=source_ref
        )
        chunk_ids = self.repository.create_chunks(
            [(doc_id, chunk.page, chunk.chunk_index, chunk.text) for chunk in chunks]
        )
        embeddings = self.provider.embed_texts([chunk.text for chunk in chunks])
        self.vector_store.add(chunk_ids, embeddings)
        return doc_id, len(chunk_ids)
