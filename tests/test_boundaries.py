import unittest
import numpy as np
from typing import Dict
from kernel_shap_streamer.core import KernelSHAPExplainer
from kernel_shap_streamer.models import ExplanationRequest

class TestKernelSHAPBoundaries(unittest.TestCase):
    def setUp(self):
        # Dummy model: returns sum of inputs
        self.model = lambda x: np.sum(x, axis=1)
        self.explainer = KernelSHAPExplainer(self.model)

    def test_minimum_nsamples_boundary(self):
        """Test behavior at the lower bound of nsamples (10)."""
        req = ExplanationRequest(
            features={"a": 1.0, "b": 2.0},
            background_data=[{"a": 0.0, "b": 0.0}],
            nsamples=10
        )
        resp = self.explainer.explain(req)
        self.assertEqual(len(resp.attributions), 2)
        self.assertTrue(resp.converged)

    def test_maximum_nsamples_boundary(self):
        """Test behavior at the upper bound of nsamples (1000)."""
        req = ExplanationRequest(
            features={"a": 1.0, "b": 2.0},
            background_data=[{"a": 0.0, "b": 0.0}],
            nsamples=1000
        )
        resp = self.explainer.explain(req)
        self.assertEqual(len(resp.attributions), 2)

    def test_single_feature_boundary(self):
        """Test behavior with exactly one feature."""
        req = ExplanationRequest(
            features={"a": 1.0},
            background_data=[{"a": 0.0}],
            nsamples=50
        )
        resp = self.explainer.explain(req)
        self.assertEqual(len(resp.attributions), 1)
        self.assertEqual(resp.attributions[0].feature_name, "a")

    def test_uniform_background_values(self):
        """Test behavior when background data is identical."""
        req = ExplanationRequest(
            features={"a": 1.0, "b": 1.0},
            background_data=[{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}],
            nsamples=50
        )
        resp = self.explainer.explain(req)
        self.assertTrue(resp.converged)
        self.assertIsInstance(resp.attributions[0].shap_value, float)

    def test_zero_payoff_model(self):
        """Test model that always returns zero."""
        zero_model = lambda x: np.zeros(x.shape[0])
        explainer = KernelSHAPExplainer(zero_model)
        req = ExplanationRequest(
            features={"a": 1.0},
            background_data=[{"a": 0.0}],
            nsamples=50
        )
        resp = explainer.explain(req)
        self.assertAlmostEqual(resp.base_value, 0.0, places=5)

    def test_negative_feature_values(self):
        """Test handling of negative input features."""
        req = ExplanationRequest(
            features={"a": -10.0, "b": -5.0},
            background_data=[{"a": 0.0, "b": 0.0}],
            nsamples=50
        )
        resp = self.explainer.explain(req)
        self.assertEqual(len(resp.attributions), 2)

    def test_kernel_weight_singularity_handling(self):
        """Verify kernel weights don't crash on edge coalition sizes."""
        d = 5
        z = np.zeros((1, d))
        weights = self.explainer._get_kernel_weights(z, d)
        self.assertEqual(weights[0], 1e6)
        
        z_full = np.ones((1, d))
        weights_full = self.explainer._get_kernel_weights(z_full, d)
        self.assertEqual(weights_full[0], 1e6)

    def test_contribution_percentage_normalization(self):
        """Ensure contribution percentages sum to 1.0."""
        req = ExplanationRequest(
            features={"a": 1.0, "b": 2.0, "c": 3.0},
            background_data=[{"a": 0.0, "b": 0.0, "c": 0.0}],
            nsamples=100
        )
        resp = self.explainer.explain(req)
        total_pct = sum(a.contribution_pct for a in resp.attributions)
        self.assertAlmostEqual(total_pct, 1.0, places=5)

    def test_empty_background_data_constraint(self):
        """Verify Pydantic validation for empty background data."""
        with self.assertRaises(Exception):
            ExplanationRequest(
                features={"a": 1.0},
                background_data=[],
                nsamples=50
            )

    def test_latency_measurement_validity(self):
        """Ensure latency is recorded as a positive float."""
        req = ExplanationRequest(
            features={"a": 1.0},
            background_data=[{"a": 0.0}],
            nsamples=50
        )
        resp = self.explainer.explain(req)
        self.assertGreaterEqual(resp.latency_ms, 0.0)

if __name__ == "__main__":
    unittest.main()