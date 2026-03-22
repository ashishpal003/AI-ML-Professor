from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import asyncio

from rag_src.generation.async_pipeline import AsyncRAGPipeline
from rag_src.cofig.setting import Settings
from rag_src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="AI-ML Professor")

## Initialize pipeline
settings = Settings()
rag_pipline = AsyncRAGPipeline(settings)

class QueryRequest(BaseModel):
    query: str

# non streaming endpoint for testing
@app.post("/query")
async def query_rag(request: QueryRequest):
    response = await rag_pipline.run(request.query)
    return {"response": response}

# Streaming Endpoint (Main Feature)
@app.post("/stream")
async def stream_rag(request: QueryRequest):

    async def event_generator():
        try:
            async for chunk in rag_pipline.stream(request.query):

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
        media_type="text/plain"
    )