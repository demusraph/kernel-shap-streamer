import unittest
import numpy as np
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from kernel_shap_streamer.core import KernelSHAPExplainer
from kernel_shap_streamer.models import ExplanationRequest
from kernel_shap_streamer.web import KernelSHAPServer

class TestKernelSHAPStress(unittest.TestCase):
    """
    High-performance stress tests for KernelSHAP Streamer.
    """

    def setUp(self):
        # Mock model: simple linear combination
        self.model = lambda x: np.sum(x, axis=1)
        self.explainer = KernelSHAPExplainer(model_predict=self.model)
        self.server = KernelSHAPServer()
        self.client = TestClient(self.server.app)

    def test_large_dataset_throughput(self):
        """Test explainer performance with high nsamples."""
        request = ExplanationRequest(
            features={"f1": 1.0, "f2": 2.0, "f3": 3.0},
            background_data=[{"f1": 0.0, "f2": 0.0, "f3": 0.0} for _ in range(10)],
            nsamples=1000
        )
        
        response = self.explainer.explain(request)
        
        self.assertEqual(len(response.attributions), 3)
        self.assertTrue(response.converged)
        self.assertGreater(response.latency_ms, 0)

    def test_concurrent_explanation_requests(self):
        """Test thread safety of the explainer under concurrent load."""
        def run_explain():
            req = ExplanationRequest(
                features={"a": 1.0, "b": 2.0},
                background_data=[{"a": 0.0, "b": 0.0}],
                nsamples=100
            )
            return self.explainer.explain(req)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_explain) for _ in range(20)]
            results = [f.result() for f in futures]
            
        self.assertEqual(len(results), 20)
        for res in results:
            self.assertTrue(res.converged)

    def test_web_dashboard_endpoint_load(self):
        """Test FastAPI dashboard responsiveness."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("KernelSHAP Streamer", response.text)

    def test_broadcast_persistence_integrity(self):
        """Test database write integrity during broadcast simulation."""
        async def simulate_broadcast():
            await self.server.broadcast_metric("test_feat", 0.99)
            
        asyncio.run(simulate_broadcast())
        
        # Verify persistence
        import sqlite3
        conn = sqlite3.connect(self.server.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT shap_value FROM metrics WHERE feature_name = 'test_feat'")
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 0.99)

    def test_stress_websocket_connection_handling(self):
        """Test handling of multiple concurrent websocket connection attempts."""
        # Simulate rapid connection/disconnection
        def connect_and_close():
            with self.client.websocket_connect("/ws/metrics") as websocket:
                self.assertIsNotNone(websocket)
        
        threads = [threading.Thread(target=connect_and_close) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        self.assertEqual(len(self.server.state.active_connections), 0)

if __name__ == "__main__":
    unittest.main()