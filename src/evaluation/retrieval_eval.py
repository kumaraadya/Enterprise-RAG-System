"""
Retrieval Evaluation Metrics

Implements standard IR metrics:
1. Recall@K: % of relevant docs in top-k
2. MRR (Mean Reciprocal Rank): Position of first relevant doc
3. nDCG (Normalized Discounted Cumulative Gain): Ranking quality
"""

import numpy as np
from typing import List, Dict, Set

"""
    Calculate Recall@K.
    
    Formula: |Retrieved ∩ Relevant| / |Relevant|
    
    Args:
        retrieved: List of retrieved chunk IDs (ordered)
        relevant: Set of relevant chunk IDs (ground truth)
        k: Cutoff position
        
    Returns:
        Recall value [0, 1]
"""
def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_at_k = set(retrieved[:k])
    hits = retrieved_at_k.intersection(relevant)

    return len(hits) / len(relevant)

"""
    Calculate MRR (Mean Reciprocal Rank).
    
    MRR = 1 / rank of first relevant item
    
    Args:
        retrieved: List of retrieved chunk IDs
        relevant: Set of relevant chunk IDs
        
    Returns:
        MRR value [0, 1]
"""
def mean_reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    for i, chunk_id in enumerate(retrieved, 1):
        if chunk_id in relevant:
            return 1.0 / i
        
    return 0.0

"""
    Calculate nDCG@K (Normalized Discounted Cumulative Gain).
    
    Measures ranking quality, giving more weight to top positions.
    
    Args:
        retrieved: List of retrieved chunk IDs
        relevant: Set of relevant chunk IDs
        k: Cutoff position
        
    Returns:
        nDCG value [0, 1]
"""
def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved[:k], 1):
        if chunk_id in relevant:
            dcg += 1.0 / np.log2(i + 1)

    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

    if idcg == 0:
        return 0.0
    
    return dcg / idcg

"""
        Evaluate a single query.
        
        Args:
            retrieved: Retrieved chunk IDs
            relevant: Ground truth relevant IDs
            k_values: K values to test
            
        Returns:
            Dict of metric scores
"""
class RetrievalEvaluator:
    def evaluate_query(self, retrieved: List[str], relevant: Set[str],
                       k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        metrics = {}
        metrics['mrr'] = mean_reciprocal_rank(retrieved, relevant)

        for k in k_values:
            metrics[f'recall@{k}'] = recall_at_k(retrieved, relevant, k)
            metrics[f'ndcg@{k}'] = ndcg_at_k(retrieved, relevant, k)

        return metrics
    
    """
        Evaluate multiple queries and average results.
        
        Args:
            queries: List of query dicts with 'query' and 'relevant_chunks'
            retrieval_results: Dict mapping query -> retrieved chunk IDs
            k_values: K values to test
            
        Returns:
            Dict of averaged metrics
    """
    def evaluate_queries(self, queries: List[Dict], retrieval_results: Dict[str, List[str]],
                         k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        all_metrics = []

        for query_data in queries:
            query = query_data['query']
            relevant = set(query_data.get('relevant_chunks', []))

            if query not in retrieval_results:
                continue

            retrieved = retrieval_results[query]
            metrics = self.evaluate_query(retrieved, relevant, k_values)
            all_metrics.append(metrics)

        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics