import re

from app.models.schemas import CompareResponse, CompareSide
from app.services.repository import SQLiteRepository


def _pick_sentence(text: str, patterns: list[str], fallback: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        lowered = sentence.lower()
        if any(pattern in lowered for pattern in patterns):
            return sentence.strip()
    return fallback


class PaperComparisonService:
    def __init__(self, repository: SQLiteRepository) -> None:
        self.repository = repository

    def _build_side(self, document_id: int) -> CompareSide:
        doc = self.repository.get_document(document_id)
        if doc is None:
            raise ValueError(f"Document {document_id} not found")
        chunks = self.repository.get_chunks_for_document(document_id, limit=120)
        merged = " ".join(chunk["text"] for chunk in chunks)
        short = merged[:2500]

        objective = _pick_sentence(
            short,
            ["objective", "we propose", "we present", "our goal", "aim"],
            "Objective not clearly detected from extracted text.",
        )
        method = _pick_sentence(
            short,
            ["method", "approach", "architecture", "algorithm", "model"],
            "Method details were limited in retrieved excerpts.",
        )
        datasets = _pick_sentence(
            short,
            ["dataset", "datasets", "benchmark", "cifar", "imagenet", "mnist"],
            "Datasets were not explicitly identified.",
        )
        metrics = _pick_sentence(
            short,
            ["accuracy", "f1", "auc", "bleu", "mse", "mae", "metric"],
            "Metrics were not explicitly identified.",
        )
        strengths = (
            "Text includes a clear technical narrative and sufficient extractable sections."
            if len(merged) > 1500
            else "Paper text extraction is short; interpretation may miss details."
        )
        limitations = (
            "This comparison is extraction-based and may miss figures/tables "
            "or math notation from PDF."
        )

        return CompareSide(
            document_id=document_id,
            document_name=str(doc["name"]),
            objective=objective,
            method=method,
            datasets=datasets,
            metrics=metrics,
            strengths=strengths,
            limitations=limitations,
        )

    def compare(self, document_a_id: int, document_b_id: int) -> CompareResponse:
        paper_a = self._build_side(document_a_id)
        paper_b = self._build_side(document_b_id)
        summary = (
            f"{paper_a.document_name} and {paper_b.document_name} were compared "
            "using extracted text. "
            "Use this output as a first-pass map before reading full papers."
        )
        return CompareResponse(paper_a=paper_a, paper_b=paper_b, high_level_summary=summary)
