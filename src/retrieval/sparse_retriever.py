"""
Sparse Retrieval with BM25

BM25 (Best Matching 25):
- Classic information retrieval algorithm
- Based on term frequency and inverse document frequency
- Excellent for exact keyword matches
"""

from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict
import json
from pathlib import Path
import pickle

"""
    Sparse retrieval using BM25 algorithm.    
    How it works:
    1. Tokenizes documents and queries
    2. Ranks by term frequency and rarity
    3. Returns top-k matches
"""
class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunk_ids = None
        self.tokenized_chunks = None
    """
        Build BM25 index from chunks.        
        Args:
            chunks: List of chunk dicts with 'text' and 'chunk_id'
    """
    def build_index(self, chunks: List[Dict]):
        print("Building BM25 index...")

        texts = [chunk['text'] for chunk in chunks]
        self.chunk_ids = [chunk['chunk_id'] for chunk in chunks]
        self.tokenized_chunks = [text.lower().split() for text in texts]

        self.bm25 = BM25Okapi(self.tokenized_chunks)

        print(f"BM25 index built with {len(self.chunk_ids)} documents")

    """
        Search for top-k documents using BM25 scoring.        
        Args:
            query: Search query
            k: Number of results            
        Returns:
            (chunk_ids, scores)
    """
    def search(self, query: str, k: int = 10) -> Tuple[List[str], List[float]]:        
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = scores.argsort()[-k:][::-1]

        retrieved_ids = [self.chunk_ids[idx] for idx in top_k_indices]
        retrieved_scores = [scores[idx] for idx in top_k_indices]
        
        return retrieved_ids, retrieved_scores
    
    """
        Save BM25 index to disk.
        
        Args:
            output_path: Path for pickle file
    """
    def save(self, output_path: Path):        
        data = {
            "bm25": self.bm25,
            "chunk_ids": self.chunk_ids,
            "tokenized_chunks": self.tokenized_chunks
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"BM25 index saved to: {output_path}")
    
    """
        Load BM25 index from disk.
        
        Args:
            input_path: Path to pickle file
    """
    def load(self, input_path: Path):        
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        self.bm25 = data['bm25']
        self.chunk_ids = data['chunk_ids']
        self.tokenized_chunks = data['tokenized_chunks']
        
        print(f"BM25 index loaded from: {input_path}")
        print(f"Documents: {len(self.chunk_ids)}")

"""
    Combines dense and sparse retrieval.
    
    Uses Reciprocal Rank Fusion (RRF) to merge results.
"""
class HybridRetriever:    
    """
        Initialize hybrid retriever.
        
        Args:
            dense_retriever: FAISS retriever instance
            sparse_retriever: BM25 retriever instance
            alpha: Weight for dense (1-alpha for sparse). 0.5 = equal weight
    """
    def __init__(self, dense_retriever, sparse_retriever, alpha: float = 0.5):        
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.alpha = alpha

    """
        Hybrid search combining dense and sparse results.        
        Algorithm:
        1. Get top-k from dense retrieval
        2. Get top-k from sparse retrieval
        3. Merge using Reciprocal Rank Fusion        
        Args:
            query_text: Text query (for BM25)
            query_embedding: Vector query (for FAISS)
            k: Total results to return            
        Returns:
            (chunk_ids, combined_scores)
    """
    def search(self, query_text: str, query_embedding, k: int = 20) -> Tuple[List[str], List[float]]:
        dense_ids, dense_scores = self.dense.search(query_embedding, k=k)
        sparse_ids, sparse_scores = self.sparse.search(query_text, k=k)

        combined_scores = {}

        for rank, (chunk_id, score) in enumerate(zip(dense_ids, dense_scores)):
            combined_scores[chunk_id] = self.alpha * (1 / (rank + 1))

        for rank, (chunk_id, score) in enumerate(zip(sparse_ids, sparse_scores)):
            if chunk_id in combined_scores:
                combined_scores[chunk_id] += (1 - self.alpha) * (1 / (rank + 1))
            else:
                combined_scores[chunk_id] = (1 - self.alpha) * (1 / (rank + 1))

        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_ids = [item[0] for item in sorted_items[:k]]
        final_scores = [item[1] for item in sorted_items[:k]]
        
        return final_ids, final_scores