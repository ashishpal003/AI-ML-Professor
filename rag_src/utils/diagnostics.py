from rag_src.utils.logger import get_logger
from rag_src.utils.exceptions import MyException

logger = get_logger(__name__)

class Diagnostics:
    @staticmethod
    def log_context_stats(context: str):
        try:
            length = len(context)
            words = len(context.split())

            logger.info(
                f"Context stats -> chars: {length}, words: {words}"
            )
        
        except Exception as e:
            logger.error(f"Context diagnostics failed: {e}")

    @staticmethod
    def log_response_stats(response: str):
        try:
            length = len(response)
            words = len(response.split())

            logger.info(
                f"Response stats → chars: {length}, words: {words}"
            )

        except Exception as e:
            logger.error(f"Response diagnostics failed: {e}")