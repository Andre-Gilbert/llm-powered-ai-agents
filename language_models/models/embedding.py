"""Sentence transformer embeddings."""

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingModel(BaseModel):
    """Class that implements a HuggingFace transformer."""

    model: str

    def __init__(self, **data):
        super().__init__(**data)
        self.transformer = SentenceTransformer(self.model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.transformer.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            query: The query to embed.

        Returns:
            Embedding for the query.
        """
        return self.embed_texts([query])[0]
