from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from rag_src.utils.exceptions import MyException
from rag_src.utils.logger import get_logger
import sys

logger = get_logger(__name__)

class LLMService:

    def __init__(self, model_name="llama3", temperature=0.2):
        try:
            logger.info(f"Initialize ChatOllam: {model_name}")

            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature
            )
        
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            raise MyException(e, sys)
        
    def generate(self, messages: list) -> str:
        try:
            logger.info("Invoking ChatOllama")

            response = self.llm.invoke(messages)

            if not response or not response.content:
                raise ValueError("Emplty response from LLM")
            
            return response.content
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise MyException(e, sys)