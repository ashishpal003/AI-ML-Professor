from rag_src.caching.semantic_cache import SemanticCache
from rag_src.caching.retrieval_cache import RetrievalCache

from rag_src.cofig.setting import Settings
from rag_src.utils.logger import get_logger
from rag_src.utils.exceptions import MyException
import sys

logger = get_logger(__name__)

class CacheManager:

    def __init__(self, settings: Settings):
        try:
            self.settings = settings
            self.semantic_cache = SemanticCache(model_name=self.settings.EMBEDDINGS_MODEL, threshold=self.settings.SEMANTIC_THRESHOLD)
            self.retrieval_cahce = RetrievalCache()

            logger.info("CacheManager initialized")
        
        except Exception as e:
            logger.error(f"CacheManager init failed: {e}")

    # semantic cache
    def get_semantic(self, query: str):
        return self.semantic_cache.get(query=query)
    
    def set_semantic(self, query: str, response: str):
        self.semantic_cache.set(query, response)

    # retrieval cache
    def get_retrieval(self, query: str):
        return self.retrieval_cahce.get(query=query)
    
    def set_retrieval(self, query: str, docs):
        self.retrieval_cahce.set(query, docs)