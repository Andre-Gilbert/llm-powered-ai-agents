"""Retriever utils."""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(
    documents: list[Document],
    separators: list[str] | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Splits the given documents.

    Args:
        docs: A list of documents to split.
        separators: A list of separators used to split the documents. Defaults to None.
        chunk_size: The size per document. Defaults to 2000.
        chunk_overlap: The overlap of characters per document. Defaults to 100.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


def format_documents(documents: list[Document]) -> str:
    """Formats the documents for the LLM prompt."""
    return "\n\n".join(document.page_content for document in documents)
