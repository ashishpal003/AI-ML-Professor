from rag_src.retrieval.pipeline import RetrievalPipeline
from rag_src.generation.llm import LLMService
from rag_src.generation.prompt_builder import PromptBuilder
from rag_src.generation.memory import ConversationMemory
from rag_src.cofig.setting import Settings

from rag_src.utils.logger import get_logger
from rag_src.utils.exceptions import MyException

from rag_src.observability.mlflow_tracker import MLFlowTracker

import time
import sys

logger = get_logger(__name__)

class RAGPipeline:

    def __init__(self, settings: Settings):
        self.settings = settings

        self.retrieval_pipeline = RetrievalPipeline(self.settings)
        self.llm = LLMService(model_name=self.settings.CHAT_MODEL)
        self.prompt_builder = PromptBuilder()
        self.memory = ConversationMemory()

    def run(self, query: str) -> str:
        logger.info("Starting conversational RAG pipeline")

        try:
            with MLFlowTracker("rag_generation_chat") as tracker:

                start_time = time.time()

                # Retrieval
                docs = self.retrieval_pipeline.run(query=query)

                context = "\n\n".join(
                    [d.page_content for d in docs[:3] if d.page_content]
                )
                logger.info(f"context: {context}")

                # memory
                histroy = self.memory.get_history()
                logger.info(f"chat_histroy: {histroy}")

                messages = self.prompt_builder.build(
                    query=query,
                    context=context,
                    history=histroy
                )
                logger.info(f"prompt: {messages}")

                response = self.llm.generate(messages=messages)

                # update memory
                self.memory.add(query=query, response=response)

                latency = time.time() - start_time

                # 📊 MLflow
                tracker.log_params({
                    "query": query,
                    "context_docs": min(3, len(docs))
                })

                tracker.log_metrics({
                    "latency_sec": latency,
                    "context_length": len(context),
                    "response_length": len(response)
                })

                logger.info(f"RAG completed in {latency:.2f}s")

                return response
            
        except Exception as e:
            logger.critical(f"RAG pipeline failed: {e}", exc_info=True)
            raise MyException(e, sys)