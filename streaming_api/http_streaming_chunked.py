# main.py
from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse
import asyncio
from datetime import datetime

app = FastAPI()


@app.get("/stream")
async def stream(request: Request):
    async def gen():
        # cleanup & disconnect-safe
        try:
            for i in range(10):
                if await request.is_disconnected():
                    break
                chunk = f"[{datetime.now().isoformat()}] item {i}\n"
                yield chunk
                await asyncio.sleep(1)
        finally:
            # put any resource cleanup here
            pass

    return StreamingResponse(gen(), media_type="text/plain")
