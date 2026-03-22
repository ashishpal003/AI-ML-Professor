# from ragas import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from rag_src.utils.logger import get_logger
from rag_src.utils.exceptions import MyException
import sys

logger = get_logger(__name__)

class DeepEvalEvaluator:

    def __init__(self, llm):
        """
        model -> ChatOllama or LangChain LLM wrapper
        """
        
        try:
            # initialize metrics object
            self.faithfulness = FaithfulnessMetric(model=llm)
            self.relevancy = AnswerRelevancyMetric(model=llm)

            logger.info("DeepEval evaluator initialized")
        
        except Exception as e:
            logger.error(f"DeepEval init failed: {e}")
            raise MyException(e, sys)

    def evaluate(self, query: str, response: str, context: str) -> dict:
        try:
            test_case = LLMTestCase(
                input=query,
                actual_output=response,
                retrieval_context=[context]
            )

            faithfulness_score = self.faithfulness.measure(test_case)
            relevancy_score = self.relevancy.measure(test_case)

            scores = {
                "faithfulness": float(faithfulness_score),
                "answer_relevancy": float(relevancy_score)
            }

            logger.info(f"DeepEval scores: {scores}")

            return scores
        
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0
            }