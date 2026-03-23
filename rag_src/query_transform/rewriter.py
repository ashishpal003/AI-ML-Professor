from rag_src.generation.llm import LLMService
from rag_src.utils.logger import get_logger
from rag_src.utils.exceptions import MyException
from langchain_core.messages import HumanMessage

logger = get_logger(__name__)

class QueryRewriter:

    def __init__(self, model_name):
        self.llm = LLMService(model_name=model_name)

    async def rewrite(self, query: str) -> str:
        prompt = f"""
Rewrite the following query to make it more specific and retrieval-friendly.
Please do as asked in the format provided and do not add any description about the rewritten query.

Original Query:
{query}

Rewritten Query:
"""
        try:
            rewritten = await self.llm.agenerate([HumanMessage(content=prompt)])
            logger.info(f"Rewritten query: {rewritten}")
            return rewritten.strip()

        except Exception as e:
            logger.warning(f"Rewrite failed: {e}")
            return query #fallback