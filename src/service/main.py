from doc_indexer import IndexInitializer, IndexManager
from doc_loader import TxtFileLoader
from doc_embedder import DocEmbedding

if __name__ == "__main__":

    # Initialize Index
    initializer = IndexInitializer()
    initializer.create_index()
    
    # Load and chunk documents
    loader = TxtFileLoader()
    chunks = loader.load_txt_file_chunks()

    # Embed chunks
    embedder = DocEmbedding()
    embeddings = embedder.embed(chunks)

    indexManager = IndexManager()    
    for content, emb in zip(chunks, embeddings):
        indexManager.index_chunk(content, emb)