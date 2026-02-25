"""
Retrieval Evaluation Metrics

Implements standard IR metrics:
1. Recall@K: % of relevant docs in top-k
2. MRR (Mean Reciprocal Rank): Position of first relevant doc
3. nDCG (Normalized Discounted Cumulative Gain): Ranking quality
"""

import numpy as np
from typing import List, Dict, Set

