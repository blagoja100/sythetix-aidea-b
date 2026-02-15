
from typing import List
from logger import Logger
from sentence_transformers import SentenceTransformer

logger = Logger("DocEmbedder")

class DocEmbedding:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim

    def embed(self, file_name: str, chunks: List[str]) -> list:
        logger.log(f"Generating embeddings for {len(chunks)} chunks in file '{file_name}'...")
        return self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)