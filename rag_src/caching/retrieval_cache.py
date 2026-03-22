import hashlib
from typing import List, Optional

from langchain_core.documents import Document
from rag_src.utils.logger import get_logger
from rag_src.utils.exceptions import MyException
import sys

logger = get_logger(__name__)

class RetrievalCache:
    
    def __init__(self):
        self.cache = {}

    def _key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[List[Document]]:
        try:
            key = self._key(query=query)

            if key in self.cache:
                logger.info("Retrieval cache HIT")
                return self.cache[key]
            
            logger.info("Retrieval cache MISS")
            return None
        
        except Exception as e:
            logger.error(f"Retrieval cache get failed: {e}")
            raise MyException(e, sys)
        
    def set(self, query: str, docs: List[Document]):
        try:
            key = self._key(query=query)
            self.cache[key] = docs

            logger.info("Retrieval cache updated")

        except Exception as e:
            logger.error(f"Retrieval cache set failed: {e}")
            raise MyException(e, sys)