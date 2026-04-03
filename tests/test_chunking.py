from app.services.chunking import TextChunker


def test_chunking_creates_overlap() -> None:
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    text = "abcdefghijklmnopqrstuvwxyz0123456789"
    chunks = chunker.chunk_pages([(1, text)])

    assert len(chunks) >= 2
    assert chunks[0].page == 1
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
