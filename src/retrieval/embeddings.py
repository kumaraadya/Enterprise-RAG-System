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