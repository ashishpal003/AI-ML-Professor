from pathlib import Path

class Settings:
    PDF_DIR = "data"

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    MIN_CHUNK_LENGTH = 50

    EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    VECTOR_DB_PATH = "vectorstore/faiss_store"

    MLFLOW_EXPERIMENT = "rag_ingestion"