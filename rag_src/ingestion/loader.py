from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from rag_src.utils.exceptions import MyException
from rag_src.utils.logger import get_logger
import sys

logger = get_logger(__name__)

class DocumentLoader:

    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def load(self):
        try:
            logger.info(f"loading documents from {self.folder_path}")

            loader = DirectoryLoader(
                path=self.folder_path,
                glob="*.pdf",
                loader_cls=PyMuPDFLoader,
                show_progress=True
            )

            documents = loader.load()

            logger.info(f"loader {len(documents)} pages")

            return documents
        
        except Exception as e:
            logger.error(f"Document loading failed: {e}")
            raise MyException(e, sys)



