from rag_src.cofig.setting import Settings
from rag_src.generation.async_pipeline import AsyncRAGPipeline

import asyncio

async def main():

    rag = AsyncRAGPipeline(Settings)

    print("🎓 AI Professor is ready (type 'exit' to quit)\n")

    while True:
        query = input("Ask: ")

        if query.lower() == "exit":
            break

        try:
            response =await rag.run(query=query)
            print("\n👨‍🏫", response)

            pending = asyncio.all_tasks() - {asyncio.current_task()}
            if pending:
                await asyncio.wait(pending, timeout=5)

        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

    

    