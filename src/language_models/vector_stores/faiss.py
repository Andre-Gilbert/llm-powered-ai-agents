"""FAISS vector store."""

from __future__ import annotations

import operator
import pickle
from enum import Enum
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document
from pydantic import BaseModel

from language_models.models.embedding import SentenceTransformerEmbeddingModel


class DistanceMetric(str, Enum):
    """Distance metrics for calculating distances between vectors."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    COSINE_SIMILARITY = "COSINE_SIMILARITY"


class FAISSVectorStore(BaseModel):
    """Class that implements RAG using Meta FAISS.

    Attributes:
        embeddings: The embeddings to use when generating queries.
        index: The FAISS index.
        documents: Mapping of indices to document.
        distance_metric: The distance metric for calculating distances between vectors.
        _normalize_L2: Whether the vectors should be normalized before storing.
    """

    embedding_model: SentenceTransformerEmbeddingModel
    index: Any = None
    documents: dict[int, Any] = {}
    distance_metric: DistanceMetric = DistanceMetric.COSINE_SIMILARITY
    _normalize_L2: bool = True

    def add_documents(self, documents: list[Document]) -> None:
        """Adds documents to the FAISS index."""
        texts = [document.page_content for document in documents]
        embeddings = self.embedding_model.embed_texts(texts)
        vectors = np.array(embeddings, dtype=np.float32)
        if self.index is None:
            if self.distance_metric == DistanceMetric.EUCLIDEAN_DISTANCE:
                self.index = faiss.IndexFlatL2(vectors.shape[1])
            else:
                self.index = faiss.IndexFlatIP(vectors.shape[1])
        if self._normalize_L2:
            faiss.normalize_L2(vectors)
        self.index.add(vectors)
        document_id = len(self.documents)
        for document in documents:
            self.documents[document_id] = document
            document_id += 1

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding_model: SentenceTransformerEmbeddingModel,
        distance_metric: DistanceMetric = DistanceMetric.COSINE_SIMILARITY,
    ) -> FAISSVectorStore:
        """Creates a FAISS index from texts.

        Args:
            documents: A list of documents used for creating the FAISS index.
            embedding_model: The embeddings to use when generating queries.
            distance_metric: The distance metric for calculating distances between vectors.

        Returns:
            An instance of the FAISS index.
        """
        normalize_L2 = distance_metric == DistanceMetric.COSINE_SIMILARITY
        vector_store = cls(
            embedding_model=embedding_model,
            distance_metric=distance_metric,
            _normalize=normalize_L2,
        )
        vector_store.add_documents(documents)
        return vector_store

    def save_local(self, folder_path: str, index_filename: str) -> None:
        """Saves the FAISS index and configuration to disk.

        Args:
            folder_path: The folder path to save the index and configuration to.
            index_filename: The filename used for saving.
        """
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)
        faiss.write_index(self.index, str(path / f"{index_filename}.faiss"))
        with open(path / f"{index_filename}.pkl", "wb") as file:
            pickle.dump((self.documents, self.distance_metric, self._normalize_L2), file)

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        index_filename: str,
        embedding_model: SentenceTransformerEmbeddingModel,
    ) -> FAISSVectorStore:
        """Loads the FAISS index and configuration from disk.

        Args:
            folder_path: The folder path to save the index and configuration to.
            index_filename: The filename used for loading.
            embedding_model: The embeddings to use when generating queries.

        Returns:
            An instance of the FAISS index.
        """
        path = Path(folder_path)
        index = faiss.read_index(str(path / f"{index_filename}.faiss"))
        with open(path / f"{index_filename}.pkl", "rb") as file:
            documents, distance_metric, normalize_L2 = pickle.load(file)
        return cls(
            embedding_model=embedding_model,
            index=index,
            documents=documents,
            distance_metric=distance_metric,
            _normalize_L2=normalize_L2,
        )

    def similarity_search(
        self,
        query: str,
        fetch_k: int = 3,
        score_threshold: float = 0.0,
    ) -> list[tuple[Document, int, float]]:
        """Gets relevant context.

        Args:
            query: The query to embed.
            fetch_k: The number of documents to fetch.
            score_threshold: Only include value above threshold.

        Returns:
            A list of documents most similar to the query text and L2 distance in float for each.
        """
        embedding = self.embedding_model.embed_query(query)
        vector = np.array([embedding], dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, fetch_k)
        documents = [
            (self.documents[index], score) for index, score in zip(indices[0], scores[0]) if index in self.documents
        ]
        if score_threshold:
            cmp = operator.le if self.distance_metric == DistanceMetric.EUCLIDEAN_DISTANCE else operator.gt
            documents = [(document, score) for document, score in documents if cmp(score, score_threshold)]
        return documents
