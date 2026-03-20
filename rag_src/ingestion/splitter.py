from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_src.utils.exceptions import MyException
from rag_src.utils.logger import get_logger
import sys

logger = get_logger(__name__)

class TextSplitter:
    
    def __init__(self, chunk_size, chunk_overlap, min_chunk_length):
        self.min_chunk_length = min_chunk_length

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def split(self, documents):
        try:
            logger.info("Splitting documents into chunks")

            chunks = self.splitter.split_documents(documents)

            filtered_chunks = [
                c for c in chunks if len(c.page_content.strip()) > self.min_chunk_length
            ]
            
            logger.info(f"Generated {len(filtered_chunks)} valid chunks")

            return filtered_chunks
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            raise MyException(e, sys)
        