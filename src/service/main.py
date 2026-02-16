from doc_indexer import IndexInitializer, IndexManager
from doc_loader import TxtFileLoader
from doc_embedder import DocEmbedding
from tools.logger import Logger

logger = Logger("Main")

if __name__ == "__main__":

    # Initialize Index
    initializer = IndexInitializer()
    initializer.create_index()
    
    # Load and chunk documents
    loader = TxtFileLoader()
    file_chunks = loader.load_txt_file_chunks()

    # Embed chunks
    if len(file_chunks) == 0:
        logger.log("No chunks to embed. Exiting.")
        exit(0)
        
    embedder = DocEmbedding()    
    indexManager = IndexManager()

    for filename, chunks in file_chunks.items():
        embeddings = embedder.embed(filename, chunks)
        for content, emb in zip(chunks, embeddings):
            indexManager.index_chunk(content, emb, metadata={"filename": filename})