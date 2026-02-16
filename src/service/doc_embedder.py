import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List
from tools.logger import Logger
from sentence_transformers import SentenceTransformer

logger = Logger("DocEmbedder")

class DocEmbedding:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim

    def embed(self, file_name: str, chunks: List[str]) -> list:
        logger.log(f"Generating embeddings for {len(chunks)} chunks in file '{file_name}'...")
        return self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)