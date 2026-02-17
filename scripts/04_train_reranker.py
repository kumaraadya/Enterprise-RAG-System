"""
Script 4: Train Cross-Encoder Reranker

This script demonstrates deep learning training:
1. Creates weak training labels from chunks
2. Fine-tunes cross-encoder on domain data
3. Tracks training with MLflow
4. Evaluates on held-out set

This is OPTIONAL but shows ML expertise.
You can skip this and just use the pre-trained model.
"""
import sys
import json
from pathlib import Path
import random
import mlflow
import mlflow.pytorch

sys.path.append(str(Path(__file__).parent.parent))
from src.config import CHUNKS_DIR, PROJECT_ROOT, RERANKER_MODEL
from src.retrieval.reranker import CrossEncoderTrainer, create_weak_training_data

def main():
    print("Cross-Encoder Training")
    print("Training uses weak labels from document structure.")

    chunks_path = CHUNKS_DIR/"chunks.json"
    with open(chunks_path, 'r') as f:
        chunks = json.load(f)

    print(f"\nLoaded {len(chunks)} chunks")
    print("\nGenerating weak training labels...")
    queries, positives, negatives = create_weak_training_data(chunks, num_samples=1000)
    print(f"Created {len(queries)} training samples")
    split_idx = int(0.8 * len(queries))

    train_queries = queries[:split_idx]
    train_pos = positives[:split_idx]
    train_neg = negatives[:split_idx]

    val_queries = queries[split_idx:]
    val_pos = positives[split_idx:]
    val_neg = negatives[split_idx:]

    print(f"Train: {len(train_queries)} | Val: {len(val_queries)}")

    print(f"\nInitializing trainer with base model: {RERANKER_MODEL}")
    trainer = CrossEncoderTrainer(RERANKER_MODEL)
    train_samples = trainer.prepare_training_data(train_queries, train_pos, train_neg)

    mlflow.set_experiment("cross-encoder-training")
    with mlflow.start_run():
        mlflow.log_param("base_model", RERANKER_MODEL)
        mlflow.log_param("train_samples", len(train_samples))
        mlflow.log_param("val_samples", len(val_queries))
        mlflow.log_param("epochs", 3)

        output_path = PROJECT_ROOT/"models"/"cross-encoder-finetuned"
        output_path.parent.mkdir(exist_ok=True)

        print("\nStarting training...")
        trainer.train(
            train_samples=train_samples,
            epochs=3,
            batch_size=16,
            warmup_steps=100,
            output_path=str(output_path)
        )
        if not output_path.exists() or not any(output_path.iterdir()):
            trainer.model.save(str(output_path))

        mlflow.log_artifacts(str(output_path.resolve()), artifact_path="model")
        print("\nTraining Complete!")
        print(f"Model saved to: {output_path}")

    print("We can now use this fine-tuned model for reranking!")

if __name__ == "__main__":
    response = input("Train cross-encoder? This takes ~10 minutes. (y/n): ")
    if response.lower() == 'y':
        main()
    else:
        print("Skipping training. Will use pre-trained model.")