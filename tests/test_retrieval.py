from pathlib import Path

from app.core.config import Settings
from app.services.providers import OpenAICompatibleProvider
from app.services.repository import SQLiteRepository
from app.services.retrieval import RetrievalService
from app.services.vector_store import FaissStore


def test_retrieval_returns_chunks(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite3"
    index_path = tmp_path / "test.index"
    settings = Settings(
        llm_provider="mock",
        sqlite_path=db_path,
        faiss_path=index_path,
        data_dir=tmp_path,
        embedding_dim=64,
    )
    settings.ensure_paths()

    repo = SQLiteRepository(db_path=db_path)
    provider = OpenAICompatibleProvider(settings)
    store = FaissStore(index_path=index_path, embedding_dim=settings.embedding_dim)

    doc_id = repo.create_document(name="paper_a.pdf", source_type="pdf")
    chunk_ids = repo.create_chunks(
        [
            (doc_id, 1, 0, "This paper studies optimization methods for deep learning."),
            (doc_id, 2, 1, "Experiments are on CIFAR-10 benchmark."),
        ]
    )
    embeddings = provider.embed_texts(
        [
            "This paper studies optimization methods for deep learning.",
            "Experiments are on CIFAR-10 benchmark.",
        ]
    )
    store.add(chunk_ids, embeddings)

    retrieval = RetrievalService(repository=repo, vector_store=store, provider=provider)
    result = retrieval.retrieve("Which benchmark is used?", top_k=2)

    assert result
    assert all(chunk.document_name == "paper_a.pdf" for chunk in result)
