"""
Cross-Encoder Reranking Module
Implements:
1. Cross-encoder inference for reranking
2. Fine-tuning capability for domain adaptation
3. Batch processing for efficiency
"""
import torch
from sentence_transformers import CrossEncoder
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm

"""
    Reranker using cross-encoder architecture.    
    Why cross-encoder?
    - Sees query and document together
    - Can model interactions between them
    - More accurate than bi-encoder
    - Trade-off: Slower (can't pre-compute)
"""
class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-macro-MiniLM-L-6-v2"):
        print(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        print("Cross-encoder loaded")

    """
        Rerank retrieved chunks using cross-encoder.        
        Args:
            query: User query
            chunks: Retrieved chunk dictionaries
            top_n: Number of top chunks to return
            batch_size: Batch size for inference
            
        Returns:
            (reranked_chunks, scores)
    """
    def rerank(self, query: str, chunks: List[Dict], top_n: int = 5, 
               batch_size: int = 16) -> Tuple[List[Dict], List[float]]:
        if not chunks:
            return [], []
        
        pairs = [[query, chunk['text']] for chunk in chunks]
        scores = self.model.predict(
            pairs,
            batch_size = batch_size,
            show_progress_bar = False
        )
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        top_chunks = [item[0] for item in scored_chunks[:top_n]]
        top_scores = [float(item[1]) for item in scored_chunks[:top_n]]

        return top_chunks, top_scores
    
    """
        Rerank given chunk IDs.        
        Args:
            query: User query
            chunk_ids: List of chunk IDs to rerank
            chunks_dict: Mapping from chunk_id to chunk dict
            top_n: Number to return            
        Returns:
            (reranked_ids, scores)
    """   
    def rerank_by_ids(self, query:str, chunk_ids: List[str], chunks_dict: Dict[str, Dict],
                       top_n: int = 5) -> Tuple[Lists[str, List[float]]]:
        chunks = [chunks_dict[cid] for cid in chunk_ids if cid in chunks_dict]

        reranked_chunks, scores = self.rerank(query, chunks, top_n)

        reranked_ids = [chunk['chunk_id'] for chunk in reranked_chunks]

        return reranked_ids, scores

"""
    Fine-tune cross-encoder on custom data.
"""   
class CrossEncoderTrainer:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
    """
        Prepare data for ranking loss.        
        Creates triplets: (query, positive, negative)        
        Args:
            queries: List of queries
            positive_chunks: Relevant chunks for each query
            negative_chunks: Irrelevant chunks for each query            
        Returns:
            List of training samples
    """
    def prepare_training_data(self, queries: List[str], positive_chunks: List[str], 
                              negative_chunks: List[str]) -> List[Tuple]:
        samples = []

        for query, pos, neg in zip(queries, positive_chunks, negative_chunks):
            samples.append((query, pos, 1.0))
            samples.append((query, neg, 0.0))

            return samples
    
        
