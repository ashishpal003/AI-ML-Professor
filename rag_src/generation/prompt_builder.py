from langchain_core.messages import HumanMessage, SystemMessage

from rag_src.utils.exceptions import MyException
from rag_src.utils.logger import get_logger
import sys

logger = get_logger(__name__)

class PromptBuilder:
    
    def build(self, query: str, context: str, history: str):
        try:
            logger.info("Building chat messages")

            system_prompt = f"""
You are an expert professor helping a student.

Guidelines:
- Explain concepts clearly and step-by-step
- Use examples when helpful
- Be concise but insightful
- If answer is not in context, say "I don't know"

Use the context below to answer.

Context:
{context}
"""
            human_prompt = f"""
Chat History:
{history}

Student Question:
{query}
"""
            return [
                SystemMessage(content=system_prompt.strip()),
                HumanMessage(content=human_prompt.strip())
            ]
        
        except Exception as e:
            logger.error(f"Prompt building failed: {e}")
            raise MyException(e, sys)