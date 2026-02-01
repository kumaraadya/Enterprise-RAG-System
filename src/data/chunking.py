"""
Document Chunking Strategies

Implements multiple chunking approaches:
1. Fixed-size chunking (token-based with overlap)
2. Sentence-aware chunking (don't split mid-sentence)

Each chunk includes metadata for citations.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import tiktoken

"""
    Represents a single document chunk with metadata.
    
    Attributes:
        chunk_id: Unique identifier
        doc_id: Source document ID
        chunk_index: Position in document
        text: The actual chunk content
        start_char: Character offset in original document
        end_char: End character offset
        token_count: Number of tokens in chunk
        metadata: Additional fields (company, date, page, etc.)
"""
@dataclass
class Chunk:    
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    token_count: int
    metadata: Dict[str, Any]
    
    """Convert to dictionary for serialization."""
    def to_dict(self) -> dict:        
        return asdict(self)
    
"""
    Token-based chunking with overlap.
    
    Uses tiktoken (GPT tokenizer) to accurately count tokens.
    Creates overlapping chunks to prevent context loss at boundaries.
"""
class TokenChunker: 
    """
        Initialize chunker.
        
        Args:
            chunk_size: Target tokens per chunk
            overlap: Number of overlapping tokens between chunks
            model: Model name for tokenizer selection
    """
    def __init__(self, chunk_size: int = 512, overlap: int = 128, model: str = "gpt-3.5-turbo"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.encoding_for_model(model)

    """
        Chunk a document into overlapping token-based segments.
        
        Args:
            text: Full document text
            doc_id: Unique document identifier
            metadata: Document metadata to attach to chunks
            
        Returns:
            List of Chunk objects
    """
    def chunk_text(self, text: str, doc_id: str, metadata: Dict[str, Any]) -> List[Chunk]:
        tokens = self.encoding.encode(text)
        
        chunks = []
        chunk_index = 0

        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))

            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)

            start_char = len(self.encoding.decode(tokens[:start_idx]))
            end_char = len(self.encoding.decode(tokens[:end_idx]))

            chunk = Chunk(
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                doc_id=doc_id,
                chunk_index=chunk_index,
                text=chunk_text.strip(),
                start_char=start_char,
                end_char=end_char,
                token_count=len(chunk_tokens),
                metadata=metadata.copy()
            )
            
            chunks.append(chunk)
            chunk_index += 1

            start_idx += (self.chunk_size - self.overlap)

            if start_idx >= len(tokens):
                break
        
        return chunks

"""
    Sentence-aware chunking that respects sentence boundaries.
    
    This is better for semantic coherence but may have variable chunk sizes.
"""    
class SentenceChunker:    
    """
        Initialize sentence-aware chunker.
        
        Args:
            target_tokens: Target size (actual size may vary)
            model: Model for tokenizer
    """
    def __init__(self, target_tokens: int = 512, model: str = "gpt-3.5-turbo"):        
        self.target_tokens = target_tokens
        self.encoding = tiktoken.encoding_for_model(model)

    """
        Split text into sentences (simple heuristic).
        
        Uses regex to split on periods, exclamation, question marks
        while avoiding common abbreviations.
    """
    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    """
        Chunk text while respecting sentence boundaries.
        
        Algorithm:
        1. Split into sentences
        2. Accumulate sentences until target size
        3. Create chunk when threshold reached
    """
    def chunk_text(self, text: str, doc_id: str, metadata: Dict[str, Any]) -> List[Chunk]:        
        sentences = self.split_sentences(text)
        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        chunk_index = 0
        char_offset = 0

        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))

            if current_token_count + sentence_tokens > self.target_tokens and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)

                chunk = Chunk(
                    chunk_id = f"{doc_id}_chunk_{chunk_index}",
                    doc_id = doc_id,
                    chunk_index = chunk_index,
                    text = chunk_text,
                    start_char = char_offset,
                    token_count=current_token_count,
                    metadata=metadata.copy()
                )

                chunks.append(chunk)
                chunk_index += 1
                char_offset += len(chunk_text) + 1

                current_chunk_sentences = []
                current_token_count = 0

            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens

        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            
            chunk = Chunk(
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                doc_id=doc_id,
                chunk_index=chunk_index,
                text=chunk_text,
                start_char=char_offset,
                end_char=char_offset + len(chunk_text),
                token_count=current_token_count,
                metadata=metadata.copy()
            )
            
            chunks.append(chunk)
        
        return chunks

"""
    Factory function to create chunker based on strategy.
    
    Args:
        strategy: "token" or "sentence"
        **kwargs: Parameters passed to chunker
        
    Returns:
        Initialized chunker instance
"""    
def choose_chunker(strategy: str = "token", **kwargs) -> TokenChunker | SentenceChunker:    
    if strategy == "token":
        return TokenChunker(**kwargs)
    elif strategy == "sentence":
        return SentenceChunker(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'token' or 'sentence'.")

