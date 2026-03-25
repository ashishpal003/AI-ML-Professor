from rag_src.ingestion.splitter import TextSplitter
from rag_src.ingestion.embedder import VectorStoreBuilder
from rag_src.cofig.setting import Settings
from langchain_community.document_loaders import PyMuPDFLoader
import tempfile
import os

class UploadPipeline:

    def __init__(self, settings: Settings):
        self.settings = settings

        self.splitter = TextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            min_chunk_length=self.settings.MIN_CHUNK_LENGTH
        )

        self.embedder = VectorStoreBuilder(settings.EMBEDDINGS_MODEL)

    def run(self, file_bytes, filename: str):
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp.flush
            loader = PyMuPDFLoader(tmp.name)
            docs = loader.load()

        chunks = self.splitter.split(docs)

        self.embedder.upsert(
            documents=chunks,
            path=self.settings.VECTOR_DB_PATH
        )

        return {
            "filename": filename,
            "chunks_added": len(chunks)
        }