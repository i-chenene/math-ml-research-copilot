import fitz


class PDFIngestionError(RuntimeError):
    pass


class PDFIngestionService:
    def extract_page_text(self, pdf_bytes: bytes) -> list[tuple[int, str]]:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:  # pragma: no cover - fitz exceptions vary
            raise PDFIngestionError("Failed to open PDF") from exc

        page_texts: list[tuple[int, str]] = []
        with doc:
            for idx, page in enumerate(doc, start=1):
                page_text = page.get_text("text")
                if page_text.strip():
                    page_texts.append((idx, page_text))

        if not page_texts:
            raise PDFIngestionError("No extractable text found in PDF")
        return page_texts
