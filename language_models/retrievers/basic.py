"""Basic retriever."""

from pydantic import BaseModel

from language_models.retrievers.utils import format_documents
from language_models.vector_stores.faiss import FAISSVectorStore


class BasicRetriever(BaseModel):
    """Class that implements naive RAG."""

    vector_store: FAISSVectorStore
    score_threshold: float = 0.0

    def get_relevant_documents(self, user_text: str, fetch_k: int = 5) -> str:
        """Gets relevant documents."""
        documents = self.vector_store.similarity_search(user_text, fetch_k, self.score_threshold)
        documents = [document for document, _ in documents]
        return format_documents(documents)
