from rag_src.retrieval.pipeline import RetrievalPipeline

from rag_src.generation.llm import LLMService
from rag_src.generation.prompt_builder import PromptBuilder
from rag_src.generation.memory import ConversationMemory

from rag_src.evaluation.deepeval_evaluator import DeepEvalEvaluator
from deepeval.models import OllamaModel

from rag_src.cofig.setting import Settings

from rag_src.caching.cache_manager import CacheManager

from rag_src.utils.logger import get_logger
from rag_src.utils.exceptions import MyException
from rag_src.utils.retry import retry
from rag_src.utils.diagnostics import Diagnostics

from rag_src.observability.mlflow_tracker import MLFlowTracker

import time
import sys
import random

logger = get_logger(__name__)

class RAGPipeline:

    def __init__(self, settings: Settings):
        self.settings = settings

        self.retrieval_pipeline = RetrievalPipeline(self.settings)
        self.llm = LLMService(model_name=self.settings.CHAT_MODEL)
        self.prompt_builder = PromptBuilder()
        self.memory = ConversationMemory()
        self.cache = CacheManager(settings=self.settings)

        ## set Ollama model with DeepEvals base class
        llm = OllamaModel(self.settings.CHAT_MODEL)
        self.evaluator = DeepEvalEvaluator(llm=llm)

    @retry(max_retries=2)
    def _safe_llm_call(self, messages):
        return self.llm.generate(messages)

    def run(self, query: str) -> str:
        logger.info("Starting conversational RAG pipeline")

        try:
            with MLFlowTracker("rag_generation_chat") as tracker:

                start_time = time.time()

                # Semantic Cache
                semantic_hit = self.cache.get_semantic(query=query)

                if semantic_hit:
                    logger.info("Returining SEMANTIC cached response")

                    tracker.log_metrics({
                        "semantic_cache_hit": 1,
                        "retrieval_cahce_hit": 0
                    })

                    return semantic_hit
                
                # Retrieval Cache
                docs = self.cache.get_retrieval(query=query)

                retrieval_hit = 1 if docs else 0

                if not docs:
                    # Retrieval
                    docs = self.retrieval_pipeline.run(query=query)
                    self.cache.set_retrieval(query, docs)

                context = "\n\n".join(
                    [d.page_content for d in docs[:3] if d.page_content]
                )
                # logger.info(f"context: {context}")

                Diagnostics.log_context_stats(context)

                # memory
                histroy = self.memory.get_history()
                # logger.info(f"chat_histroy: {histroy}")

                messages = self.prompt_builder.build(
                    query=query,
                    context=context,
                    history=histroy
                )
                # logger.info(f"prompt: {messages}")

                response = self._safe_llm_call(messages=messages)

                Diagnostics.log_response_stats(response)
                self.cache.set_semantic(query, response)

                # evaluation only on X% of calls
                eval_scores = {"faithfulness": 0.0, "answer_relevancy": 0.0}
                if random.random() < self.settings.EVAL_PROBABILITY:
                    # This forces the script to wait until the evaluation is finished
                    eval_scores = self.evaluator.evaluate(query, response, context)

                else:
                    logger.info("Skipping evaluation (sampling)")

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
                    "response_length": len(response),

                    # Cache metrics
                    "semantic_cache_hit": int(semantic_hit is not None),
                    "retrieval_cache_hit": retrieval_hit,

                    # Evaluation metrics
                    "faithfulness": eval_scores["faithfulness"],
                    "answer_relevancy": eval_scores["answer_relevancy"]
                })

                logger.info(f"RAG completed in {latency:.2f}s")

                return response
            
        except Exception as e:
            logger.critical(f"RAG pipeline failed: {e}", exc_info=True)
            raise MyException(e, sys)