from app.services.providers import OpenAICompatibleProvider
from app.services.repository import SQLiteRepository
from app.services.vector_store import FaissStore


class IndexService:
    def __init__(
        self,
        repository: SQLiteRepository,
        provider: OpenAICompatibleProvider,
        vector_store: FaissStore,
    ) -> None:
        self.repository = repository
        self.provider = provider
        self.vector_store = vector_store

    def rebuild(self) -> int:
        rows = self.repository.get_all_chunks()
        self.vector_store.reset()
        if not rows:
            return 0
        ids = [int(row["id"]) for row in rows]
        texts = [str(row["text"]) for row in rows]
        vectors = self.provider.embed_texts(texts)
        self.vector_store.add(ids, vectors)
        return len(ids)
