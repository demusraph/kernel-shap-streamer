import unittest
from src.kernel_shap_streamer.models import ExplanationRequest

class TestEngine(unittest.TestCase):
    def test_import(self):
        # Corrected input to match Pydantic model schema
        req = ExplanationRequest(
            features={"a": 1.0, "b": 2.0}, 
            background_data=[{"a": 0.0, "b": 0.0}]
        )
        self.assertIsNotNone(req)