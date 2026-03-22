from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from rag_src.utils.logger import get_logger
from rag_src.utils.exceptions import MyException
import sys

logger = get_logger(__name__)

class SemanticCache:

    def __init__(self, model_name: str, threshold: float = 0.85):
        try:
            self.embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                encode_kwargs={"normalize_embeddings": True}
            )

            self.vectorstore: FAISS = None
            self.responses = []
            self.threshold = threshold

            logger.info("Semantic cache initialized")

        except Exception as e:
            logger.error(f"Semantic cache init failed: {e}")
            raise MyException(e, sys)
    
    def get(self, query: str):
        try:
            if self.vectorstore is None:
                logger.info("Semantic cache empty")
                return None
            
            results = self.vectorstore.similarity_search_with_score(query=query, k=1)

            if not results:
                return None
            
            doc, score = results[0]

            # FAISS returns L2 distance -> convert to similarity
            similarity = 1 / (1 + score)

            logger.info(f"Semantic Similarity: {similarity:.3f}")

            if similarity >= self.threshold:
                idx = doc.metadata["idx"]
                logger.info("Semantic cache HIT")
                return self.responses[idx]
            
            logger.info("Semantic cache MISS")
            return None
        
        except Exception as e:
            logger.error(f"Semantic cache get failed: {e}")
            raise MyException(e, sys)

    def set(self, query: str, response: str):
        try:
            idx = len(self.responses)
            self.responses.append(response)

            doc = Document(
                page_content=query,
                metadata={"idx": idx}
            )

            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(
                    [doc],
                    self.embedding
                )
            else:
                self.vectorstore.add_documents([doc])

            logger.info("Semantic cache updated")

        except Exception as e:
            logger.error(f"Semantic cache set failed: {e}")
    