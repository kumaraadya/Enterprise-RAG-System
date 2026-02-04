"""
Embedding Generation Module

Handles:
1. Loading embedding models
2. Generating embeddings for chunks
3. Caching embeddings to disk
4. Batch processing for efficiency
"""
import numpy as np
from typing import List, Dict
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import hashlib

"""
    Generates and caches embeddings using sentence-transformers.
    
    Why sentence-transformers?
    - Pre-trained on semantic similarity tasks
    - Optimized for retrieval
    - Easy to use
"""
class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_dir: Path = None):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            show_progress: Show tqdm progress bar
            
        Returns:
            NumPy array of shape (len(texts), embedding_dim)
    """
    def encode_batch(self, texts: List[str], batch_size: int = 32,
                     show_progress: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size = batch_size,
            show_progress_bar = show_progress,
            convert_to_numpy = True,
            normalize_embeddings = True
        )

        return embeddings 
    
    """
        Generate embeddings for chunk dictionaries.
        
        Args:
            chunks: List of chunk dicts with 'text' field
            batch_size: Batch size for encoding
            
        Returns:
            NumPy array of embeddings
    """
    def encode_chunks(self, chunks: List[Dict], batch_size: int = 32) -> np.ndarray:
        texts = [chunk['text'] for chunk in chunks]
        return self.encode_batch(texts, batch_size=batch_size)
    
    """
        Save embeddings to disk with metadata.
        
        Format: NPZ file with:
        - embeddings: The actual vectors
        - chunk_ids: Corresponding chunk IDs
        - metadata: Model info, shape, etc.
    """
    def save_embeddings(self, embeddings: np.ndarray, chunk_ids: List[str],
                        output_path: Path):
        metadata = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "num_embeddings": len(embeddings),
            "checksum": self._compute_checksum(embeddings)
        }
        np.savez_compressed(
            output_path,
            embeddings =embeddings,
            chunk_ids = chunk_ids,
            metadata = json.dumps(metadata)
        )

        print(f"Embeddings save to: {output_path}")
        print(f"Shape: {embeddings.shape}")

    """
        Load embeddings from disk.
        
        Returns:
            (embeddings array, chunk_ids list, metadata dict)
    """
    def load_embeddings(self, input_path: Path) -> tuple:
        data = np.load(input_path, allow_pickle =  True)
        embeddings = data['embeddings']
        chunk_ids = data['chunk_ids'].tolist()
        metadata = json.loads(str(data['metadata']))

        print(f"Loaded embeddings: {embeddings.shape}")
        print(f"Model: {metadata['model_name']}")

        return embeddings, chunk_ids, metadata
    
    """Compute MD5 checksum for verification."""
    @staticmethod
    def _compute_checksum(arr: np.ndarray) -> str:       
        return hashlib.md5(arr.tobytes()).hexdigest()[:16]
    
def generate_query_embedding(query: str, model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embedding = model.encode(
        query,
        convert_to_numpy = True,
        normalize_embeddings = True
    )
    return embedding