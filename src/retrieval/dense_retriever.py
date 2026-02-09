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

    """
        Search for top-k most similar chunks.
        
        Args:
            query_embedding: Query vector (1D array)
            k: Number of results to return
            
        Returns:
            (chunk_ids, scores) where scores are cosine similarities
    """
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:        
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        retrieved_ids = [self.chunk_ids[idx] for idx in indices[0]]
        retrieved_scores = scores[0].tolist()
        
        return retrieved_ids, retrieved_scores
    
    """
        Save index and metadata to disk.
        
        Args:
            index_path: Path for FAISS index file
            metadata_path: Path for metadata JSON
    """
    def save(self, index_path: Path, metadata_path: Path):
        faiss.write_index(self.index, str(index_path))
        metadata = {
            "embedding_dim": self.embedding_dim,
            "num_vectors": self.index.ntotal,
            "chunk_ids": self.chunk_ids
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Index saved to: {index_path}")
        print(f"Metadata saved to: {metadata_path}")

    """
        Load index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON
    """
    def load(self, index_path: Path, metadata_path: Path):

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.embedding_dim = metadata['embedding_dim']
        self.chunk_ids = metadata['chunk_ids']
        
        print(f"Index loaded from: {index_path}")
        print(f"Vectors: {self.index.ntotal}")