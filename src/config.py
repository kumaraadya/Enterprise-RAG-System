"""
Configuration file for the entire RAG system.
All constants, paths, and settings in one place for easy management.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
INDEXES_DIR = DATA_DIR / "indexes"
LOGS_DIR = PROJECT_ROOT / "logs"

for dir_path in [RAW_DATA_DIR, CLEANED_DATA_DIR, CHUNKS_DIR, 
                 EMBEDDINGS_DIR, INDEXES_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Aadya Kumar (itsme.aadya7@gmail.com)"

YEARS_PER_COMPANY =3

COMPANIES = {
    "Apple": "0000320193",
    "Microsoft": "0000789019",
    "Amazon": "0001018724",
    "Alphabet": "0001652044",
    "Meta": "0001326801",
    "Tesla": "0001318605",
    "NVIDIA": "0001045810",
    "Netflix": "0001065280",
    "Intel": "0000050863",
    "IBM": "0000051143",
}

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{accession_nodashes}/{primary_doc}"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 32

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_BATCH_SIZE = 16

TOP_K_RETRIEVAL = 20
TOP_N_RERANKED = 5

LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o-mini"

MAX_TOKENS = 1000
TEMPERATURE = 0.1

EVAL_QUERIES_PATH = DATA_DIR / "eval_queries.json"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"

API_HOST = "0.0.0.0"
API_PORT = 8000

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
