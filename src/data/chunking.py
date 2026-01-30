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