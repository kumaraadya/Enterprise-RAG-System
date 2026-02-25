"""
Create Evaluation Dataset

Manually create a small set of queries with known relevant chunks.
This is used to measure retrieval and answer quality.
"""
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import DATA_DIR

def create_eval_queries():
    eval_queries = [
        {
            "query": "What are Apple's main revenue sources?",
            "relevant_chunks": [
                "Apple_0000320193_2023-11-03_10K_000032019323000106_chunk_86",
                "Apple_0000320193_2023-11-03_10K_000032019323000106_chunk_68",
                "Apple_0000320193_2023-11-03_10K_000032019323000106_chunk_67",
                "Apple_0000320193_2023-11-03_10K_000032019323000106_chunk_66",
            ],
            "expected_answer_contains": ["iPhone", "services", "revenue"],
            "category": "financial"
        },
        {
            "query": "What risks does Tesla identify in their business?",
            "relevant_chunks": [
                "Tesla_0001318605_2024-01-29_10K_000162828024002390_chunk_64",
                "Tesla_0001318605_2024-01-29_10K_000162828024002390_chunk_60",
                "Tesla_0001318605_2025-01-30_10K_000162828025003063_chunk_70",
                "Tesla_0001318605_2026-01-29_10K_000162828026003952_chunk_70"
            ],
            "expected_answer_contains": ["risk", "competition", "production"],
            "category": "risk"
        },
        {
            "query": "How much did Microsoft invest in AI in 2023?",
            "relevant_chunks": [
                "Microsoft_0000789019_2023-07-27_10K_000095017023035122_chunk_114",
                "Microsoft_0000789019_2023-07-27_10K_000095017023035122_chunk_47",
                "Microsoft_0000789019_2023-07-27_10K_000095017023035122_chunk_198",
                "Microsoft_0000789019_2025-07-30_10K_000095017025100235_chunk_123"
            ],
            "expected_answer_contains": ["investment", "AI", "artificial intelligence"],
            "category": "financial"
        },
        {
            "query": "What is Amazon's strategy for AWS?",
            "relevant_chunks": [
                "Amazon_0001018724_2023-02-03_10K_000101872423000004_chunk_33",
                "Amazon_0001018724_2023-02-03_10K_000101872423000004_chunk_32",
                "Amazon_0001018724_2024-02-02_10K_000101872424000008_chunk_71",
                "Amazon_0001018724_2024-02-02_10K_000101872424000008_chunk_34"
            ],
            "expected_answer_contains": ["cloud", "AWS", "strategy"],
            "category": "business"
        },
        {
            "query": "What are NVIDIA's main products?",
            "relevant_chunks": [
                "NVIDIA_0001045810_2023-02-24_10K_000104581023000017_chunk_29",
                "NVIDIA_0001045810_2023-02-24_10K_000104581023000017_chunk_34",
                "NVIDIA_0001045810_2023-02-24_10K_000104581023000017_chunk_30",
                "NVIDIA_0001045810_2023-02-24_10K_000104581023000017_chunk_31",
            ],
            "expected_answer_contains": ["GPU", "graphics", "chips"],
            "category": "business"
        }
    ]

    output_path = DATA_DIR/"eval_queries.json"
    with open(output_path, 'w') as f:
        json.dump(eval_queries, f, indent = 2)

    print(f"Created {len(eval_queries)} evaluation queries")
    print(f"Saved to {output_path}")
    
if __name__ == "__main__":
    create_eval_queries()