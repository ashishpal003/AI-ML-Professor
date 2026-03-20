from rag_src.cofig.setting import Settings
from rag_src.ingestion.pipeline import IngestionPipeline

if __name__ == "__main__":
    pipeline = IngestionPipeline(settings=Settings)
    pipeline.run()