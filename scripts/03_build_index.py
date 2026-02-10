"""
Script 3: Build Embedding Index

This script:
1. Loads processed chunks
2. Generates embeddings using sentence-transformers
3. Builds FAISS index for fast retrieval
4. Saves everything to disk
"""

import sys
import json
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    CHUNKS_DIR,
    EMBEDDINGS_DIR,
    INDEXES_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE
)
from src.retrieval.embeddings import EmbeddingGenerator
from src.retrieval.dense_retriever import FAISSRetriever
from src.retrieval.sparse_retriever import BM25Retriever

def main():
    print("Embedding & Index Builder")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Batch size: {EMBEDDING_BATCH_SIZE}")

    chunks_path = CHUNKS_DIR / "chunks.json"
    print(f"\nLoading chunks from: {chunks_path}")
    
    with open(chunks_path, 'r') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")

    print("\nInitializing embedding model...")
    generator = EmbeddingGenerator(
        model_name=EMBEDDING_MODEL,
        cache_dir=EMBEDDINGS_DIR
    )
    print("\nGenerating embeddings...")
    embeddings = generator.encode_chunks(chunks, batch_size=EMBEDDING_BATCH_SIZE)
    
    print(f"Generated embeddings: {embeddings.shape}")
    chunk_ids = [chunk['chunk_id'] for chunk in chunks]

    embeddings_path = EMBEDDINGS_DIR / "chunk_embeddings.npz"
    generator.save_embeddings(embeddings, chunk_ids, embeddings_path)

    print("\nBuilding FAISS index...")
    retriever = FAISSRetriever(embedding_dim=generator.embedding_dim)
    retriever.build_index(embeddings, chunk_ids, index_type="flatip")

    index_path = INDEXES_DIR / "faiss_index.bin"
    metadata_path = INDEXES_DIR / "faiss_metadata.json"
    retriever.save(index_path, metadata_path)

    print("\nBuilding BM25 index...")
    bm25_retriever = BM25Retriever()
    bm25_retriever.build_index(chunks)

    bm25_path = INDEXES_DIR / "bm25_index.pkl"
    bm25_retriever.save(bm25_path)
    print(f"BM25 Index: {bm25_path}")

    print("Index Build Complete")
    print(f"Embeddings: {embeddings_path}")
    print(f"FAISS Index: {index_path}")
    print(f"Ready for retrieval!")


if __name__ == "__main__":
    main()