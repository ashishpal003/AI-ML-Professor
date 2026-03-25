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
Rewrite the following query to make it a bit more specific and retrieval-friendly.
Make sure the rewritten query does not change in its meaning.
Please do not add any extra topic or keywords from your end.

Qriginal Query:
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