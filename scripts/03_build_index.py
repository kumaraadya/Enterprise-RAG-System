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