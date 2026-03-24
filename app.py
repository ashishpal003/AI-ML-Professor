from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi import UploadFile, File
from rag_src.ingestion.upload_pipeline import UploadPipeline

import asyncio

from rag_src.generation.async_pipeline import AsyncRAGPipeline
from rag_src.cofig.setting import Settings
from rag_src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="AI-ML Professor")

## Initialize pipeline
settings = Settings()
rag_pipline = AsyncRAGPipeline(settings)
upload_pipeline = UploadPipeline(settings)

class QueryRequest(BaseModel):
    query: str
    session_id: str

# non streaming endpoint for testing
@app.post("/query")
async def query_rag(request: QueryRequest):
    response = await rag_pipline.run(request.query, request.session_id)
    return {"response": response}

# Streaming Endpoint (Main Feature)
@app.post("/stream")
async def stream_rag(request: QueryRequest):

    async def event_generator():
        try:
            async for chunk in rag_pipline.stream(request.query, request.session_id):

                # Handle LangChain chunk objects if needed
                if hasattr(chunk, "content"):
                    chunk = chunk.content

                yield chunk

                # Optional: small delay added
                await asyncio.sleep(0.02)

        except Exception as e:
            logger.error(f"Streaming endpoint error: {e}")
            yield "Error generating response"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/clear-cache")
def clear_cache():
    rag_pipline.cache.redis.client.flushdb()
    return {"message": "Cache cleared"}

@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        content = await file.read()

        result = upload_pipeline.run(content, file.filename)

        return {
            "message": "File uploaded successfully",
            "details": result
        }
    
    except Exception as e:
        return {"error": str(e)}