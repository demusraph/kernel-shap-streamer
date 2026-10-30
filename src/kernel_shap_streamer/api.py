from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
import asyncio
from .core import ShapleyEngine
from .models import ModelWrapper

app = FastAPI(title="KernelSHAP Streamer")

@app.post("/explain/stream")
async def stream_explanation(instance: list[float], engine: ShapleyEngine):
    """
    SSE endpoint for real-time waterfall updates.
    """
    async def event_generator():
        phi = engine.explain(np.array(instance))
        yield f"data: {json.dumps({'attribution': phi.tolist()})}\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")