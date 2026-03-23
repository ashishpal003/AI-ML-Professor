from rag_src.retrieval.pipeline import RetrievalPipeline

from rag_src.generation.llm import LLMService
from rag_src.generation.prompt_builder import PromptBuilder
from rag_src.generation.memory import ConversationMemory
from rag_src.generation.memory_store import MemoryStore

from rag_src.evaluation.deepeval_evaluator import DeepEvalEvaluator
from deepeval.models import OllamaModel

from rag_src.cofig.setting import Settings

from rag_src.caching.cache_manager import CacheManager

from rag_src.query_transform.rewriter import QueryRewriter
from rag_src.query_transform.multi_query import MultiQueryGenerator

from rag_src.utils.logger import get_logger
from rag_src.utils.exceptions import MyException
from rag_src.utils.retry import retry
from rag_src.utils.diagnostics import Diagnostics

from rag_src.observability.mlflow_tracker import MLFlowTracker

import time
import sys
import random
import asyncio

logger = get_logger(__name__)

class AsyncRAGPipeline:

    def __init__(self, settings: Settings):
        self.settings = settings

        self.retrieval_pipeline = RetrievalPipeline(self.settings)
        self.llm = LLMService(model_name=self.settings.CHAT_MODEL)
        self.prompt_builder = PromptBuilder()
        self.memory_store = MemoryStore()
        self.cache = CacheManager(settings=self.settings)

        ## set Ollama model with DeepEvals base class
        llm = OllamaModel(self.settings.CHAT_MODEL)
        self.evaluator = DeepEvalEvaluator(llm=llm)
        self.rewriter = QueryRewriter(self.settings.CHAT_MODEL)
        self.multi_query = MultiQueryGenerator(self.settings.CHAT_MODEL)

    # Retry wrapper
    async def _safe_llm_call(self, messages):
        delay = 1
        for attempt in range(3):
            try:
                return await self.llm.agenerate(messages)
            except Exception as e:
                logger.warning(f"LLM attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep(delay)
                delay *=2

    # Background evaluation
    async def _run_evaluation(self, query: str, response: str, context: str):
        try:
            scores = await asyncio.to_thread(
                self.evaluator.evaluate,
                query,
                response,
                context
            )

            logger.info(f"Async evaluation scores: {scores}")

        except Exception as e:
            logger.error(f"Async evaluation failed: {e}")

    # Async Pipeline
    async def run(self, query: str, session_id: str) -> str:

        logger.info("Starting async conversational RAG pipeline")

        try:
            with MLFlowTracker("rag_generation_chat") as tracker:

                start_time = time.time()

                memory = self.memory_store.get_memory(session_id)

                # semantic cache
                semantic_hit = self.cache.get_semantic(query=query)

                if semantic_hit:
                    logger.info("Returning SEMANTIC cache response")

                    tracker.log_metrics({
                        "semantic_cache_hit": 1,
                        "retrieval_cache_hit": 0
                    })

                    return semantic_hit
                
                # retrieval cache
                docs = self.cache.get_retrieval(query=query)
                retrieval_hit = 1 if docs else 0

                if not docs:
                    docs = await asyncio.to_thread(
                        self.retrieval_pipeline.run,
                        query
                    )
                    self.cache.set_retrieval(query, docs)

                # context
                context = "\n\n".join(
                    [d.page_content for d in docs[:3] if d.page_content]
                )

                Diagnostics.log_context_stats(context)

                # history
                history = memory.get_history()

                # prompt
                messages = self.prompt_builder.build(
                    query=query,
                    context=context,
                    history=history
                )

                # LLM
                response = await self._safe_llm_call(messages)

                Diagnostics.log_response_stats(response)

                # cache
                self.cache.set_semantic(query, response)

                # evaluation
                if random.random() < self.settings.EVAL_PROBABILITY:
                    asyncio.create_task(
                        self._run_evaluation(query, response, context)
                    )
                else:
                    logger.info("Skipping evaluation (sampling)")

                memory.add(query=query, response=response)

                latency = time.time() - start_time

                # MLFlow
                tracker.log_params({
                    "query": query,
                    "context_docs": min(3, len(docs))
                })

                tracker.log_metrics({
                    "latency_sec": latency,
                    "context_length": len(context),
                    "response_length": len(response),
                    "semantic_cache_hit": int(semantic_hit is not None),
                    "retrieval_cache_hit": retrieval_hit
                })

                logger.info(f"RAG completed in {latency:.2f}s")

                return response
            
        except Exception as e:
            logger.critical(f"Async RAG pipeline failed: {e}", exc_info=True)
            raise MyException(e, sys)
        
    async def stream(self, query: str, session_id: str):
        logger.info("Starting STREAMING RAG pipeline")

        try:
            memory = self.memory_store.get_memory(session_id)

            # semantic cache
            semantic_hit = self.cache.get_semantic(query=query)
            if semantic_hit:
                yield semantic_hit
                return
            
            # query rewrite 
            rewritten_query = await self.rewriter.rewrite(query=query)

            if not rewritten_query:
                rewritten_query = query

            # retrieval cache - base on rewritten query
            docs = self.cache.get_retrieval(query=rewritten_query)

            if not docs:

                # multi query
                queries = await self.multi_query.generate(rewritten_query)

                # parallel retrieval
                tasks = [
                    asyncio.to_thread(self.retrieval_pipeline.run, q)
                    for q in queries
                ]

                results = await asyncio.gather(*tasks)

                # flatten
                all_docs = [doc for sublist in results for doc in sublist]

                # deduplicate
                unique_docs = list({d.page_content: d for d in all_docs}.values())

                # final reranking
                docs = self.retrieval_pipeline.reranker.rerank(
                    rewritten_query,
                    unique_docs
                )

                # retrieval cache - set
                self.cache.set_retrieval(rewritten_query, docs)

            # build context
            context = "\n\n".join(
                [d.page_content for d in docs[:3] if d.page_content]
            )

            # history
            history = memory.get_history()

            # build prompt
            messages = self.prompt_builder.build(
                query=query,
                context=context,
                history=history
            )

            # stream response
            full_response = ""

            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, "content"):
                    token = chunk.content
                else:
                    token = str(chunk)
                
                full_response += token
                yield token

            # cache
            self.cache.set_semantic(query, full_response)
            
            # memory update
            memory.add(query=query, response=full_response)

            # async evaluation
            asyncio.create_task(
                self._run_evaluation(query, full_response, context)
            )            
        
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield "Error generating response."