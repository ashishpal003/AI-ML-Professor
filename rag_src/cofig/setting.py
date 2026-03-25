from pathlib import Path

class Settings:
    PDF_DIR = "data"

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    MIN_CHUNK_LENGTH = 50
    K = 8
    CACHE_TTL = 600

    EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    RERANKER_MODEL = "BAAI/bge-reranker-base"
    
    CHAT_MODEL = "llama3:8b"
    TEMEPERATURE = 0.2

    VECTOR_DB_PATH = "vectorstore/faiss_store"

    MLFLOW_EXPERIMENT = "rag_ingestion"

    SEMANTIC_THRESHOLD = 0.85

    EVAL_PROBABILITY = 0.5 # 25% of requests