import hashlib

import numpy as np
from openai import OpenAI

from app.core.config import Settings


class OpenAICompatibleProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = settings.llm_provider.lower()
        self.client = None
        if self.provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
            self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    @staticmethod
    def _mock_vector(text: str, dim: int) -> np.ndarray:
        # Deterministic local embedding for offline runs and tests.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(0, 1, dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if self.provider == "openai" and self.client is not None:
            response = self.client.embeddings.create(
                model=self.settings.embedding_model, input=texts
            )
            vectors = [item.embedding for item in response.data]
            return np.array(vectors, dtype=np.float32)
        vectors = [self._mock_vector(text, self.settings.embedding_dim) for text in texts]
        return np.array(vectors, dtype=np.float32)

    def answer_from_context(self, question: str, context: str) -> str:
        if self.provider == "openai" and self.client is not None:
            response = self.client.chat.completions.create(
                model=self.settings.chat_model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful research copilot. Use only provided context. "
                            "If evidence is weak, say so explicitly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Question:\n"
                            f"{question}\n\n"
                            "Context chunks:\n"
                            f"{context}\n\n"
                            "Write a concise answer and cite sources using [doc_name p.X]."
                        ),
                    },
                ],
            )
            return response.choices[0].message.content or ""
        return ""
