
from typing import List
from sentence_transformers import SentenceTransformer

class DocEmbedding:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim

    def embed(self, chunks: List[str]) -> list:
        return self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)