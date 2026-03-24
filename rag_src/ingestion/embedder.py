from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag_src.utils.exceptions import MyException
from rag_src.utils.logger import get_logger
import sys

import os
import shutil

logger = get_logger(__name__)

class VectorStoreBuilder:

    def __init__(self, model_name: str):
        self.model_name = model_name

        try:
            self.embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                encode_kwargs={"normalize_embeddings": True}
            )

            logger.info(f"loadded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Embedding model load failed: {e}")
            raise MyException(e, sys)
        
    def build(self, documents):
        try:
            logger.info("Building FAISS vector store")
            return FAISS.from_documents(documents, self.embedding)
        
        except Exception as e:
            logger.error(f"Vector store build failed: {e}")
            raise MyException(e, sys)
    
    def save(self, vectorstore: FAISS, path: str):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            
            vectorstore.save_local(path)
            logger.info(f"Vectorstore saved at {path}")

        except Exception as e:
            logger.error(f"Saving vectorstore failed: {e}")
            raise MyException(e, sys)
        
    def load_or_create(self, path: str):
        if os.path.exists(path):
            return FAISS.load_local(
                path,
                self.embedding,
                allow_dangerous_deserialization=True
            )
        return None
    
    def add_documents(self, documents, path: str):
        try:
            db = self.load_or_create(path)

            if db:
                logger.info("Updating existng vectorstore")
                db.add_documents(documents)
            else:
                logger.info("Creating new vectorstore")
                db = FAISS.from_documents(documents, self.embedding)
            
            db.save_local(path)
            return db
        
        except Exception as e:
            raise MyException(e, sys)