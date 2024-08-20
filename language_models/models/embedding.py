"""Sentence transformer embeddings."""

from pydantic import BaseModel, Field, model_validator
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingModel(BaseModel):
    """Class that implements a HuggingFace transformer."""

    model: str
    transformer: SentenceTransformer = Field(init=False)

    @model_validator(mode="before")
    def initialize_transformer(cls, values: dict) -> dict:
        """Ensure the SentenceTransformer is initialized."""
        if "transformer" not in values:
            model = values.get("model")
            if not model:
                raise ValueError("A model name must be provided.")
            values["transformer"] = SentenceTransformer(model)
        return values

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

    class Config:
        arbitrary_types_allowed = True
