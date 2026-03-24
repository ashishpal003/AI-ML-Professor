from sentence_transformers import CrossEncoder
from rag_src.utils.exceptions import MyException
from rag_src.utils.logger import get_logger
import sys

logger = get_logger(__name__)

class Reranker:

    def __init__(self, model_name: str="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            logger.info(f"Loading reranker model: {model_name}")
            self.model = CrossEncoder(model_name=model_name)

        except Exception as e:
            logger.error(f"Reranker model load failed: {e}")
            raise MyException(e, sys)
        
    def rerank(self, query: str, documents):
        try:
            logger.info("Reranking documents")

            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.model.predict(pairs)

            ranked = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True
            )

            # rerranked_docs = [doc for doc, _ in ranked]

            reranked_docs = [
                {
                    "doc": doc,
                    "score": float(score)
                }
                for doc, score in ranked
            ]

            logger.info("Reranking completed")

            return reranked_docs
        
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise MyException(e, sys)