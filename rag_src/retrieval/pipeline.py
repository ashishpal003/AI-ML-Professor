from rag_src.retrieval.vectorstore_loader import VectorStoreLoader
from rag_src.retrieval.retriever import Retriever
from rag_src.retrieval.reranker import Reranker
from rag_src.cofig.setting import Settings

from rag_src.observability.mlflow_tracker import MLFlowTracker
from rag_src.utils.logger import get_logger
from rag_src.utils.exceptions import MyException
import time
import sys

logger = get_logger(__name__)

class RetrievalPipeline:

    def __init__(self, settings: Settings):
        self.settings = settings

        self.loader = VectorStoreLoader(
            settings.EMBEDDINGS_MODEL,
            settings.VECTOR_DB_PATH
        )

        self.vectorstore = self.loader.load()

        self.retriever = Retriever(vectorstore=self.vectorstore, k=self.settings.K)
        self.reranker = Reranker(model_name=self.settings.RERANKER_MODEL)
    
    def run(self, query: str):
        logger.info("Starting retrieval pipeline")

        try:
            with MLFlowTracker(experiment_name="rag_retrieval") as tracker:

                start_time = time.time()
                self.vectorstore = self.loader.load()
                if self.vectorstore is None:
                    logger.warning("Vectorstore is empty. Retrieval disabled.")

                if self.vectorstore is None:
                    return []

                self.retriever.vectorstore = self.vectorstore

                docs = self.retriever.retriever(query=query)
                reranked_docs = self.reranker.rerank(query, docs)

                latency = time.time() - start_time

                # log params
                tracker.log_params({
                    "query": query,
                    "top_k": self.retriever.k
                })

                # Log metrics
                tracker.log_metrics({
                    "latency_sec": latency,
                    "num_docs_retrieved": len(docs)
                })

                logger.info(f"Retrieval completed in {latency:.2f}s")

                return reranked_docs
            
        except Exception as e:
            logger.critical(f"Retrieval pipeline failed: {e}", exc_info=True)
            raise MyException(e, sys)
