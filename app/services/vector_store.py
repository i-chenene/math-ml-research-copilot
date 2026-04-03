from pathlib import Path

import faiss
import numpy as np


class FaissStore:
    def __init__(self, index_path: Path, embedding_dim: int) -> None:
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.index = self._load_or_create()

    def _new_index(self) -> faiss.IndexIDMap2:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedding_dim))

    def _load_or_create(self) -> faiss.IndexIDMap2:
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))  # type: ignore[return-value]
        return self._new_index()

    def reset(self) -> None:
        self.index = self._new_index()
        self.save()

    def add(self, ids: list[int], embeddings: np.ndarray) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        ids_arr = np.array(ids, dtype=np.int64)
        self.index.add_with_ids(embeddings, ids_arr)
        self.save()

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        scores, ids = self.index.search(query, top_k)
        results: list[tuple[int, float]] = []
        for idx, score in zip(ids[0], scores[0], strict=False):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
