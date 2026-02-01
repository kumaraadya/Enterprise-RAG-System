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

def main():
    print("Document Chunking Processor")
    print(f"Chunk size: {CHUNK_SIZE} tokens")
    print(f"Overlap: {CHUNK_OVERLAP} tokens")

    manifest_path = DATA_DIR / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    
    print(f"\nFound {len(manifest)} documents to process")

    chunker = choose_chunker(
        strategy="token",
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    all_chunks = []

    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing documents"):
        file_path = Path(row['local_clean_path'])
        
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        doc_metadata = {
            "company": row['company'],
            "cik": row['cik'],
            "filing_date": row['filing_date'],
            "form_type": row['form'],
            "source_url": row['source_url'],
            "accession_number": row['accession_number']
        }

        try:
            chunks = process_document(file_path, chunker, doc_metadata)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue

    output_path = CHUNKS_DIR / "chunks.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)

    chunk_df = pd.DataFrame(all_chunks)
    
    stats_path = CHUNKS_DIR / "chunk_statistics.csv"
    stats = {
        "total_chunks": len(chunk_df),
        "total_documents": chunk_df['doc_id'].nunique(),
        "avg_tokens_per_chunk": chunk_df['token_count'].mean(),
        "min_tokens": chunk_df['token_count'].min(),
        "max_tokens": chunk_df['token_count'].max(),
        "companies": chunk_df['metadata'].apply(lambda x: x['company']).nunique()
    }
    
    pd.DataFrame([stats]).to_csv(stats_path, index=False)

    print("Chunking Complete")
    print(f"Total chunks created: {len(chunk_df)}")
    print(f"Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
    print(f"Output: {output_path}")
    print(f"Statistics: {stats_path}")

if __name__ == "__main__":
    main()