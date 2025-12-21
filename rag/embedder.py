import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, batch_size=32):
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # good for cosine similarity
            show_progress_bar=False
        )
        return vectors

    def encode_query(self, query: str):
        vec = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        # vec is already 1D when encoding a single string
        return vec
