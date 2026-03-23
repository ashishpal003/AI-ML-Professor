from rag_src.utils.logger import get_logger
from rag_src.generation.memory import ConversationMemory
import sys

logger = get_logger(__name__)

class MemoryStore:

    def __init__(self):
        self.store = {}
    
    def get_memory(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = ConversationMemory()

        return self.store[session_id]