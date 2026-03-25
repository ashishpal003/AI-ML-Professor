from rag_src.generation.llm import LLMService
from langchain_core.messages import HumanMessage
from rag_src.utils.logger import get_logger

logger = get_logger(__name__)

class MultiQueryGenerator:

    def __init__(self, model_name: str):
        self.llm = LLMService(model_name=model_name)

    async def generate(self, query: str):
        prompt = f"""
Generate 3 different rephrasings of this query for better document retrieval.
Please provide a just a list of rephrased query and nothing extra.
Also please dont add sentences like 'Here are three rephrasings of the query for better document retrieval:' or any thing similar with in the list"

### Example
Query:
Define Machine Learning (ML) as a subfield of Artificial Intelligence (AI), explaining its core concepts, goals, and applications in areas such as computer vision, natural language processing, and predictive analytics.

list:
- What are the fundamental principles and objectives of machine learning within the broader context of artificial intelligence?
- Explain the key ideas, purposes, and uses of machine learning in disciplines like computer vision, natural language processing, and predictive modeling.
- Describe the core concepts, aims, and areas of application for machine learning as a subfield of artificial intelligence.

## New query
Query:
{query}

list:
"""
        try:
            response = await self.llm.agenerate([HumanMessage(content=prompt)])

            queries = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]

            logger.info(f"List of queries: {queries}")
            return list(set(queries))[:5]
        except Exception:
            return [query]