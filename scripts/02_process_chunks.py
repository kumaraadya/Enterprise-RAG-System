"""
Script 2: Process Documents into Chunks

This script:
1. Reads cleaned documents
2. Applies chunking strategy
3. Saves chunks with metadata
4. Creates chunk inventory
"""
import sys
import json
from pathlib import Path
from typing import List
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from src.config import CLEANED_DATA_DIR, CHUNKS_DIR, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.data.chunking import TokenChunker, choose_chunker

"""
    Process a single document into chunks.
    
    Args:
        file_path: Path to cleaned text file
        chunker: Chunking strategy instance
        doc_metadata: Metadata to attach to each chunk
        
    Returns:
        List of chunk dictionaries
"""
def process_document(file_path: Path, chunker, doc_metadata: dict) -> List[dict]:    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    doc_id = file_path.stem    
    chunks = chunker.chunk_text(text, doc_id, doc_metadata)
    
    return [chunk.to_dict() for chunk in chunks]