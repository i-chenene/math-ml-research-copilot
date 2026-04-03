from app.models.schemas import RetrievalChunk
from app.services.providers import OpenAICompatibleProvider
from app.services.repository import SQLiteRepository
from app.services.vector_store import FaissStore


class RetrievalService:
    def __init__(
        self,
        repository: SQLiteRepository,
        vector_store: FaissStore,
        provider: OpenAICompatibleProvider,
    ) -> None:
        self.repository = repository
        self.vector_store = vector_store
        self.provider = provider

    def retrieve(self, query: str, top_k: int) -> list[RetrievalChunk]:
        query_embedding = self.provider.embed_texts([query])[0]
        hits = self.vector_store.search(query_embedding, top_k=top_k)
        if not hits:
            return []

        ids = [chunk_id for chunk_id, _ in hits]
        scores = {chunk_id: score for chunk_id, score in hits}
        rows = self.repository.get_chunks_by_ids(ids)

        return [
            RetrievalChunk(
                chunk_id=int(row["chunk_id"]),
                document_id=int(row["document_id"]),
                document_name=str(row["document_name"]),
                page=int(row["page"]) if row["page"] is not None else None,
                score=scores.get(int(row["chunk_id"]), 0.0),
                text=str(row["text"]),
            )
            for row in rows
        ]
