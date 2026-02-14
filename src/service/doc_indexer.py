#Create a vector index in OpenSearch
#This index is your knowledge base for RAG.
import json
from opensearchpy import OpenSearch
from uuid import uuid4

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
        # Create an instance of the helper class to get config
        _idx_client = IndexClient()
        self.client = _idx_client.get_client() if client is None else client
        self.index_name = _idx_client.get_index_name()

    def create_index(self):
        print("Connected to OpenSearch")
        print(json.dumps(self.client.info(), indent=2))

        print(f"Creating index... '{self.index_name}'")

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
                    "content": {"type": "text"},
                    "metadata": {"type": "object"},
                    "embedding": {
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
            print(json.dumps(index_body, indent=2))    
            print(f"Index '{self.index_name}' created.")
        else:
            print(f"Index '{self.index_name}' already exists.")

class IndexManager:
    def __init__(self, client=None):
        _idx_client = IndexClient()
        self.client = _idx_client.get_client() if client is None else client
        self.index_name = _idx_client.get_index_name()
    
    def index_chunk(self, content, embedding, metadata=None):
        doc = {
            "content": content,
            "metadata": metadata or {},
            "embedding": embedding.tolist()
        }
        self.client.index(index=self.index_name, id=str(uuid4()), body=doc)