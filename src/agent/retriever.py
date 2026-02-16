import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from tools.logger import Logger

logger = Logger("Retriever")

class Retriever:
    def __init__(self, top_k=5):
        self.retriever = None
        
        logger.log("Initializing Retriever...")
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", 
            encode_kwargs={"normalize_embeddings": True})
        
        logger.log("Embedding function initialized.")
        self.vector_store = OpenSearchVectorSearch(
            opensearch_url="http://admin:admin@localhost:9200",
            index_name="kb_index",
            embedding_function=self.embedding_fn
        )

        logger.log("Vector store initialized.")
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        logger.log("Retriever initialized.")
        
    def get_relevant_documents(self, query):
        return self.retriever.invoke(query)
