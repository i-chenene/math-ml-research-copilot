import re

from app.models.schemas import AskResponse
from app.services.providers import OpenAICompatibleProvider
from app.services.retrieval import RetrievalService


def _extractive_answer(question: str, contexts: list[str]) -> str:
    query_terms = {token for token in re.findall(r"[a-zA-Z]{3,}", question.lower())}
    candidate_sentences: list[str] = []
    for ctx in contexts:
        for sentence in re.split(r"(?<=[.!?])\s+", ctx):
            lowered = sentence.lower()
            if query_terms and any(term in lowered for term in query_terms):
                candidate_sentences.append(sentence.strip())
    if not candidate_sentences:
        candidate_sentences = [ctx[:220].strip() for ctx in contexts[:2]]
    return " ".join(candidate_sentences[:2]).strip()


class QAService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        provider: OpenAICompatibleProvider,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.provider = provider

    def ask(self, question: str, top_k: int) -> AskResponse:
        chunks = self.retrieval_service.retrieve(question, top_k=top_k)
        if not chunks:
            return AskResponse(
                answer="I could not find enough evidence in the indexed corpus to answer reliably.",
                citations=[],
                evidence_strength="weak",
                retrieved_chunks=[],
            )

        context_lines = []
        citations: list[str] = []
        for chunk in chunks:
            page_label = f"p.{chunk.page}" if chunk.page is not None else "p.?"
            citation = f"[{chunk.document_name} {page_label}]"
            citations.append(citation)
            context_lines.append(f"{citation} {chunk.text}")

        llm_answer = self.provider.answer_from_context(
            question=question,
            context="\n\n".join(context_lines),
        )
        if not llm_answer.strip():
            summary = _extractive_answer(question, [chunk.text for chunk in chunks])
            llm_answer = (
                f"Based on retrieved excerpts: {summary}\n\n"
                "Confidence note: this is an extractive answer from local context."
            )

        best_score = max(chunk.score for chunk in chunks)
        evidence_strength = "strong" if best_score >= 0.35 else "weak"
        if evidence_strength == "weak":
            llm_answer += "\n\nEvidence appears limited; verify by reading cited sections."

        return AskResponse(
            answer=llm_answer,
            citations=list(dict.fromkeys(citations)),
            evidence_strength=evidence_strength,
            retrieved_chunks=chunks,
        )
