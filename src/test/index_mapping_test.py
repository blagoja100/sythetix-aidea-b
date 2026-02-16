import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from service.doc_indexer import IndexManager

indexManager = IndexManager()
print(indexManager.client.indices.get_mapping(index="kb_index"))