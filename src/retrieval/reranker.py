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
                       top_n: int = 5) -> Tuple[List[str], List[float]]:
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

    """
        Fine-tune cross-encoder.        
        Uses binary classification loss:
        - Score = 1 for relevant pairs
        - Score = 0 for irrelevant pairs        
        Args:
            train_samples: List of (query, chunk, label) tuples
            epochs: Training epochs
            batch_size: Batch size
            warmup_steps: Warmup steps for learning rate
            output_path: Where to save fine-tuned model
    """  
    def train(self, train_examples: List[Tuple], epochs: int = 3,
              batch_size: int = 16, warmup_steps: int = 100, output_path: str = "models/cross-encoder-finetuned"):
        from sentence_transformers import InputExample
        from torch.utils.data import DataLoader

        train_examples = [
            InputExample(texts=[query, chunk], label=float(label))
            for query, chunk, label in train_examples
            ]

        train_dataloader = DataLoader(
            train_examples,
            shuffle = True,
            batch_size=batch_size
        )

        self.model_fit(
            train_dataloader=train_dataloader,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True
        )
        print(f"Model fine tuned and saved to: {output_path}")  

    """
        Create weak training labels from chunks.
    
        Weak supervision strategy:
        - Positive: Chunks from same document/section
        - Negative: Random chunks from different documents
    
        This isn't perfect but provides signal for fine-tuning.
    
        Args:
            chunks: All chunks with metadata
            num_samples: Number of training samples to create
        
        Returns:
            (queries, positive_chunks, negative_chunks)
    """
    def create_weak_training_data(
            chunks: List[Dict], num_samples: int = 1000
    )->Tuple[List[str], List[str], List[str]]:
        import random

        queries = []
        positives = []
        negatives = []

        doc_to_chunks = {}
        for chunk in chunks:
            doc_id = chunk['doc_id']
            if doc_id not in doc_to_chunks:
                doc_to_chunks[doc_id] = []
            doc_to_chunks[doc_id].append(chunk)

        documents = list(doc_to_chunks.keys())

        for _ in range(num_samples):
            doc_id = random.choice(documents)
            doc_chunks = doc_to_chunks[doc_id]

            if len(doc_chunks) < 2:
                continue

            chunk1, chunk2 = random.sample(doc_chunks, 2)

            query = chunk1['text'][:200]
            positive = chunk2['text']

            other_docs = [d for d in documents if d != doc_id]
            neg_doc = random.choice(other_docs)
            neg_chunk = random.choice(doc_to_chunks[neg_doc])
            negative = neg_chunk['text']

            queries.append(query)
            positives.append(positive)
            negatives.append(negative)

        return queries, positives, negatives       
