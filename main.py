from fastapi import FastAPI
from src.kernel_shap_streamer.models import ExplanationRequest, ExplanationResponse
from src.kernel_shap_streamer.core import KernelSHAPEngine
import numpy as np

app = FastAPI(title="KernelSHAP Streamer")

# Mock Model Proxy
def mock_model(x: np.ndarray) -> np.ndarray:
    return np.sum(x, axis=1)

engine = KernelSHAPEngine(model_proxy=mock_model)

@app.post("/explain", response_model=ExplanationResponse)
async def explain(request: ExplanationRequest):
    # In production, fetch background from cache
    background = np.random.randn(100, len(request.features))
    return engine.explain(request, background)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)