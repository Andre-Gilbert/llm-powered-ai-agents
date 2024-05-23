"""Contextual compression retriever."""

from langchain_core.documents import Document
from pydantic import BaseModel

from language_models.agents.react import ChatMessage, ChatMessageRole
from language_models.models.llm import OpenAILanguageModel
from language_models.retrievers.utils import format_documents
from language_models.vector_stores.faiss import FAISSVectorStore

_PROMPT_TEMPLATE = """Given the following question and context, respond with YES if the context is relevant to the question and NO if it isn't.

Question:

{question}

Context:

{context}
"""


class ContextualCompressionRetriever(BaseModel):
    """Class that implements a contextual compression retriever."""

    llm: OpenAILanguageModel
    vector_store: FAISSVectorStore
    score_threshold: float = 0.0

    def _parse_output(self, output: str) -> bool:
        """Parses LLM output."""
        cleaned_upper_text = output.strip().upper()
        if "YES" in cleaned_upper_text and "NO" in cleaned_upper_text:
            raise ValueError(f"Ambiguous response. Both 'YES' and 'NO' in received: {output}.")
        elif "YES" in cleaned_upper_text:
            return True
        elif "NO" in cleaned_upper_text:
            return False
        else:
            raise ValueError(f"Expected output value to include either 'YES' or 'NO'. Received {output}.")

    def _compress_documents(self, user_text: str, documents: list[Document]) -> list[Document]:
        """Filters relevant documents."""
        compressed_documents = []
        for document in documents:
            prompt = _PROMPT_TEMPLATE.format(question=user_text, context=document.page_content)
            output = self.llm.get_completion([ChatMessage(role=ChatMessageRole.USER, content=prompt)])
            try:
                include_doc = self._parse_output(output)
            except ValueError:
                include_doc = False
            if include_doc:
                compressed_documents.append(document)
        return compressed_documents

    def get_relevant_documents(self, user_text: str, fetch_k: int = 5) -> str:
        """Gets relevant documents."""
        documents = self.vector_store.similarity_search(user_text, fetch_k, self.score_threshold)
        documents = [document for document, _ in documents]
        compressed_documents = self._compress_documents(user_text, documents)
        return format_documents(compressed_documents)
