from rag_src.utils.logger import get_logger
import sys

logger = get_logger(__name__)

class ConversationMemory:

    def __init__(self, max_turns: int=5):
        self.history = []
        self.max_turns = max_turns

    def add(self, query: str, response: str):
        try:
            logger.info("Updating conversation memory")

            self.history.append({
                "user": query,
                "assistant": response
            })

            self.history = self.history[-self.max_turns:]

        except Exception as e:
            logger.error(f"Memory update failed: {e}")

    def get_history(self) -> str:
        formatted = []

        for turn in self.history:
            formatted.append(f"User: {turn['user']}")
            formatted.append(f"Assistant: {turn['assistant']}")

        return "\n".join(formatted)