import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#Create a vector index in OpenSearch
#This index is your knowledge base for RAG.
import json
from opensearchpy import OpenSearch
from uuid import uuid4
from tools.logger import Logger

logger = Logger("DocIndexer")

class IndexClient:
    def __init__(self):
        self.client = OpenSearch(
            hosts=[{"host": "localhost", "port": 9200}],
            http_auth=("admin", "admin"),
            use_ssl=False,
        )
        self.index_name = "kb_index"

    def get_client(self):
        return self.client
    def get_index_name(self):
        return self.index_name


class IndexInitializer:
    def __init__(self, client=None):
        _idx_client = IndexClient()
        self.client = _idx_client.get_client() if client is None else client
        self.index_name = _idx_client.get_index_name()

    def create_index(self):
        logger.log("Connected to OpenSearch")
        logger.log(json.dumps(self.client.info(), indent=2))

        logger.log(f"Creating index... '{self.index_name}'")

        index_body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 512
                }
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "metadata": {"type": "object"},
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": 384,  # must match your embedding model
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "faiss",
                            "parameters": {
                                "ef_construction": 512,
                                "m": 16
                            }
                        }
                    }
                }
            }
        }

        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=index_body)
            logger.log(json.dumps(index_body, indent=2))    
            logger.log(f"Index '{self.index_name}' created.")
        else:
            logger.log(f"Index '{self.index_name}' already exists.")

class IndexManager:
    def __init__(self, client=None):
        _idx_client = IndexClient()
        self.client = _idx_client.get_client() if client is None else client
        self.index_name = _idx_client.get_index_name()
    
    def index_chunk(self, content, embedding, metadata=None):
        doc = {
            "text": content,
            "metadata": metadata or {},
            "vector_field": embedding.tolist()
        }
        self.client.index(index=self.index_name, id=str(uuid4()), body=doc)
        logger.log(f"Indexed chunk with metadata: {metadata}, content length: {len(content)}, embedding dim: {len(embedding)}")