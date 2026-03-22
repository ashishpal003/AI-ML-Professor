import time
from functools import wraps
from rag_src.utils.logger import get_logger
from rag_src.utils.exceptions import MyException
import sys

logger = get_logger(__name__)

def retry(max_retries=3, delay=1, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(1, max_retries+1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    logger.warning(
                        f"Attempt {attempt} failed: {e}"
                    )

                    if attempt == max_retries:
                        logger.error("Max retries reached")
                        raise MyException(e, sys)
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator