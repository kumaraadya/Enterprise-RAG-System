"""
Script 5: System Evaluation

Runs complete evaluation:
1. Retrieval metrics (Recall, MRR, nDCG)
2. Answer quality metrics
3. Generates evaluation report
"""

import sys
import json
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    DATA_DIR, CHUNKS_DIR, INDEXES_DIR,
    EMBEDDING_MODEL, TOP_K_RETRIEVAL
)

from src.retrieval.embeddings import generate_query_embedding
from src.retrieval.dense_retriever import FAISSRetriever
from src.evaluation.retrieval_eval import RetrievalEvaluator

def main():
    print("System Evaluation")
    eval_path = DATA_DIR / "eval_queries.json"
    if not eval_path.exists():
        print("Evaluation queries not found. Run scripts/create_eval_set.py first.")
        return
    
    with open(eval_path, 'r') as f:
        eval_queries = json.load(f)

    print(f"Loaded {len(eval_queries)} evaluation queries")

    print(f"\nLoading retrieval index...")
    retriever = FAISSRetriever(embedding_dim = 384)
    index_path = INDEXES_DIR / "faiss_index.bin"
    metadata_path = INDEXES_DIR / "faiss_metadata.json"
    retriever.load(index_path, metadata_path)

    retrieval_results = {}

    print("\nRunning Retrieval...")
    for query_data in eval_queries:
        query = query_data['query']

        query_emb = generate_query_embedding(query, EMBEDDING_MODEL)
        chunk_ids, scores = retriever.search(query_emb, k = TOP_K_RETRIEVAL)
        retrieval_results[query] = chunk_ids
        print(f"  {query[:50]}... -> {len(chunk_ids)} results")

    print("\nEvaluating retrieval...")
    evaluator = RetrievalEvaluator()

    queries_with_truth = [q for q in eval_queries if q.get('relevant_chunks')]

    if queries_with_truth:
        metrics = evaluator.evaluate_queries(queries_with_truth,
                                             retrieval_results,
                                             k_values = [5, 10, 20])
        
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.3f}")

        results_dir = Path("eval_results")
        results_dir.mkdir(exist_ok=True)

        results_df = pd.DataFrame([metrics])
        results_df.to_csv(results_dir / "retrieval_metrics.csv", index=False)
        
        print(f"\nResults saved to: {results_dir}")
    else:
        print("\nNo ground truth labels found in evaluation queries.")
        print("Please manually label relevant_chunks in eval_queries.json")

    print("Evaluation complete!")


if __name__ == "__main__":
    main()