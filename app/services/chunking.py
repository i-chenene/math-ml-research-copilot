from dataclasses import dataclass


@dataclass(slots=True)
class ChunkPayload:
    page: int | None
    text: str
    chunk_index: int


class TextChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> list[str]:
        normalized = " ".join(text.split())
        if not normalized:
            return []

        chunks: list[str] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(normalized):
                break
            start += step
        return chunks

    def chunk_pages(self, page_texts: list[tuple[int, str]]) -> list[ChunkPayload]:
        payloads: list[ChunkPayload] = []
        counter = 0
        for page_number, text in page_texts:
            for chunk in self._split_text(text):
                payloads.append(ChunkPayload(page=page_number, text=chunk, chunk_index=counter))
                counter += 1
        return payloads
