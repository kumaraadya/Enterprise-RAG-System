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
    
    
        
