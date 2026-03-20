from rag_src.cofig.setting import Settings
from rag_src.ingestion.loader import DocumentLoader
from rag_src.ingestion.splitter import TextSplitter
from rag_src.ingestion.embedder import VectorStoreBuilder

from rag_src.utils.exceptions import MyException
from rag_src.utils.logger import get_logger
from rag_src.observability.mlflow_tracker import MLFlowTracker
import sys

import numpy as np
import tempfile

logger = get_logger(__name__)

class IngestionPipeline:

    def __init__(self, settings: Settings):
        self.settings = settings

        self.loader = DocumentLoader(self.settings.PDF_DIR)
        self.splitter = TextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            min_chunk_length=self.settings.MIN_CHUNK_LENGTH
        )

        self.embedder = VectorStoreBuilder(model_name=self.settings.EMBEDDINGS_MODEL)

    def run(self):
        logger.info("Starting ingestion pipeline")

        try:
            with MLFlowTracker(experiment_name=self.settings.MLFLOW_EXPERIMENT) as tracker:

                docs = self.loader.load()
                chunks = self.splitter.split(documents=docs)

                vectorstore = self.embedder.build(documents=chunks)
                self.embedder.save(vectorstore=vectorstore, path=self.settings.VECTOR_DB_PATH)

                avg_len = np.mean([len(c.page_content) for c in chunks])

                tracker.log_params({
                    "chunk_size": self.settings.CHUNK_SIZE,
                    "chunk_overlap": self.settings.CHUNK_OVERLAP,
                    "embedding_model": self.settings.EMBEDDINGS_MODEL
                })

                tracker.log_metrics({
                    "num_docs": len(docs),
                    "num_chunks": len(chunks),
                    "avg_chunk_length": avg_len
                })

                # save sample chunks
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                    for c in chunks[:5]:
                        f.write(c.page_content + "\n\n---\n\n")

                    tracker.log_artifact(f.name)

                logger.info("Ingestion completed successfully")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise MyException(e, sys)