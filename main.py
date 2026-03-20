from rag_src.cofig.setting import Settings
from rag_src.ingestion.pipeline import IngestionPipeline
from rag_src.retrieval.pipeline import RetrievalPipeline

if __name__ == "__main__":
    # pipeline = IngestionPipeline(settings=Settings)
    # pipeline.run()

    retrieval_pipeline = RetrievalPipeline(settings=Settings)

    query = "What is supervised machine learning?"

    result = retrieval_pipeline.run(query=query)

    for i, doc in enumerate(result[:3]):
        print(f"\n--- Result {i+1} ---\n")
        print(doc.page_content[:300])