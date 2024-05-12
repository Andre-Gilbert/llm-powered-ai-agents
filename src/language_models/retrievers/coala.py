"""Cognitive Architecture for Language Agent (CoALA) retriever."""

from pydantic import BaseModel

from language_models.retrievers.utils import format_documents
from language_models.vector_stores.faiss import FAISSVectorStore


class CoALARetriever(BaseModel):
    """Class that implements CoALA."""

    semantic_vector_store: FAISSVectorStore
    episodic_vector_store: FAISSVectorStore
    fetch_k: int = 1
    score_threshold: float = 0.0

    def get_relevant_documents(self, query: str) -> str:
        """Gets relevant documents."""
        semantic_documents = self.semantic_vector_store.similarity_search(query, self.fetch_k, self.score_threshold)
        semantic_documents = [document for document, _ in semantic_documents]
        episodic_documents = self.episodic_vector_store.similarity_search(query, self.fetch_k, self.score_threshold)
        episodic_documents = [document for document, _ in episodic_documents]
        formatted_semantic_documents = format_documents(semantic_documents)
        formatted_episodic_documents = format_documents(episodic_documents)
        return f"Context:\n\n{formatted_semantic_documents}\n\nPrevious tasks:\n\n{formatted_episodic_documents}"
