
from langchain_community.vectorstores import FAISS
from rag_src.utils.exceptions import MyException
from rag_src.utils.logger import get_logger
import sys

logger = get_logger(__name__)

class Retriever:

    def __init__(self, vectorstore: FAISS, k: int=5):
        self.vectorstore = vectorstore
        self.k = k

    def retriever(self, query: str):
        try:
            logger.info(f"Retrieving top {self.k} docs from query {query}")

            docs = self.vectorstore.similarity_search(query=query, k=self.k)

            logger.info(f"Retrieved {len(docs)} documents")

            return docs
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise MyException(e, sys)