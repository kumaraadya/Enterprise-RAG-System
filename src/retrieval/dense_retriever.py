"""
Dense Retrieval with FAISS

FAISS (Facebook AI Similarity Search):
- Fast approximate nearest neighbor search
- Works entirely in memory (no database needed)
- Perfect for our scale (~1000s of chunks)
"""
import numpy as np
import faiss
from typing import List, Tuple
from pathlib import Path
import json

"""
    Dense retrieval using FAISS index.
    
    Supports:
    - Building index from embeddings
    - Similarity search (cosine via normalized vectors)
    - Saving/loading indexes
"""
class FAISSRetriever:    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunk_ids = None

    """
        Build FAISS index from embeddings.
        
        Index types:
        - "flatip": Exact search with inner product (fast for <100K vectors)
        - "hnsw": Approximate search with HNSW graph (faster for >100K)
        
        Args:
            embeddings: NumPy array (num_chunks, embedding_dim)
            chunk_ids: List of chunk IDs (same order as embeddings)
            index_type: Type of FAISS index
    """
    def build_index(self, embeddings: np.ndarray, chunk_ids: List[str], 
                   index_type: str = "flatip"):
        assert embeddings.shape[0] == len(chunk_ids), "Mismatch in counts"
        assert embeddings.shape[1] == self.embedding_dim, "Dimension mismatch"

        self.chunk_ids = chunk_ids
        if index_type == "flatip":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.index.add(embeddings.astype('float32'))
        
        print(f"FAISS index built: {index_type}")
        print(f"   Vectors: {self.index.ntotal}")
        print(f"   Dimension: {self.embedding_dim}")