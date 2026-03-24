import hashlib
from typing import List, Optional

from langchain_core.documents import Document
from rag_src.utils.logger import get_logger
from rag_src.utils.exceptions import MyException
import sys
import json

logger = get_logger(__name__)

class RetrievalCache:
    
    def __init__(self, redis_client):
        self.redis = redis_client

    def _key(self, query: str) -> str:
        return f"retrieval:{hashlib.md5(query.encode()).hexdigest()}"
    
    def get(self, query: str):
        try:
            key = self._key(query=query)
            data = self.redis.get(key)

            if data:
                logger.info("Retrieval cache HIT")
                return [
                    Document(page_content=d["page_content"], metadata=d["metadata"])
                    for d in json.loads(data)
                ]
            
            logger.info("Retrieval cache MISS")
            return None
        
        except Exception as e:
            logger.error(f"Retrieval cache get failed: {e}")
            raise MyException(e, sys)
        
    def set(self, query: str, docs):
        try:
            key = self._key(query=query)
            
            serialized = json.dumps([
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in docs
            ])

            self.redis.set(key, serialized, ex=600) ## TTL = 30 minutes

        except Exception as e:
            logger.error(f"Retrieval cache set failed: {e}")
            raise MyException(e, sys)