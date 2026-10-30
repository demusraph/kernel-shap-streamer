import asyncio
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- Data Models ---

class SHAPMetric(BaseModel):
    feature_name: str
    shap_value: float
    timestamp: str

@dataclass
class StreamState:
    active_connections: List[WebSocket] = None

    def __post_init__(self):
        self.active_connections = []

# --- Application Logic ---

class KernelSHAPServer:
    def __init__(self):
        self.app = FastAPI(title="KernelSHAP Streamer")
        self.state = StreamState()
        self.db_path = Path("shap_metrics.db")
        self._init_db()
        self._setup_routes()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id TEXT PRIMARY KEY,
                    feature_name TEXT,
                    shap_value REAL,
                    timestamp TEXT
                )
            """)

    async def broadcast_metric(self, feature_name: str, value: float):
        metric = SHAPMetric(
            feature_name=feature_name, 
            shap_value=value, 
            timestamp=datetime.now().isoformat()
        )
        
        # Persist to DB
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO metrics VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), metric.feature_name, metric.shap_value, metric.timestamp)
            )
        
        # Broadcast to websockets
        payload = json.dumps(metric.model_dump())
        if self.state.active_connections:
            await asyncio.gather(*[ws.send_text(payload) for ws in self.state.active_connections])

    def _setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            return "<html><body><h1>KernelSHAP Streamer</h1></body></html>"

        @self.app.websocket("/ws/metrics")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.state.active_connections.append(websocket)
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.state.active_connections.remove(websocket)