from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from rag_src.utils.exceptions import MyException
from rag_src.utils.logger import get_logger
import sys
import os

logger = get_logger(__name__)

class VectorStoreLoader:

    def __init__(self, model_name: str, db_path: str):
        self.model_name = model_name
        self.db_path = db_path

        try:
            self.embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            logger.error(f"Embedding load failed: {e}")
            raise MyException(e, sys)
        
    def load(self):
        try:
            if not os.path.exists(self.db_path):
                logger.warning("Vector DB is not found. Returning None.")
                return None
            
            logger.info(f"Loading FAISS index from {self.db_path}")

            return FAISS.load_local(
                self.db_path,
                self.embedding,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"Vector store loading failed: {e}")
            raise MyException(e, sys)
