import unittest
import numpy as np
from typing import Dict
from kernel_shap_streamer.core import KernelSHAPExplainer
from kernel_shap_streamer.models import ExplanationRequest

class TestKernelSHAPCore(unittest.TestCase):
    def setUp(self):
        # Simple linear model: y = 2*x1 + 3*x2
        def mock_predict(data: np.ndarray) -> np.ndarray:
            return 2.0 * data[:, 0] + 3.0 * data[:, 1]
        
        self.explainer = KernelSHAPExplainer(model_predict=mock_predict)
        self.features = {"x1": 1.0, "x2": 1.0}
        self.background = [
            {"x1": 0.0, "x2": 0.0},
            {"x1": 0.5, "x2": 0.5}
        ]
        self.request = ExplanationRequest(
            features=self.features,
            background_data=self.background,
            nsamples=100
        )

    def test_kernel_weights_shape(self):
        """Verify kernel weights calculation produces correct dimensions."""
        z = np.array([[0, 1], [1, 0], [1, 1]])
        weights = self.explainer._get_kernel_weights(z, 2)
        self.assertEqual(len(weights), 3)
        self.assertTrue(np.all(weights > 0))

    def test_kernel_weights_boundary_conditions(self):
        """Verify kernel weights handle empty and full coalitions with high penalty."""
        z = np.array([[0, 0], [1, 1]])
        weights = self.explainer._get_kernel_weights(z, 2)
        self.assertEqual(weights[0], 1e6)
        self.assertEqual(weights[1], 1e6)

    def test_explain_output_structure(self):
        """Verify the explanation response contains all required fields."""
        response = self.explainer.explain(self.request)
        self.assertTrue(hasattr(response, "base_value"))
        self.assertTrue(hasattr(response, "attributions"))
        self.assertEqual(len(response.attributions), 2)

    def test_attribution_sum_logic(self):
        """Verify contribution percentages sum to approximately 1.0."""
        response = self.explainer.explain(self.request)
        total_pct = sum(attr.contribution_pct for attr in response.attributions)
        self.assertAlmostEqual(total_pct, 1.0, places=5)

    def test_latency_measurement(self):
        """Verify latency is recorded as a positive float."""
        response = self.explainer.explain(self.request)
        self.assertGreaterEqual(response.latency_ms, 0.0)

    def test_convergence_flag(self):
        """Verify the converged flag is set to True."""
        response = self.explainer.explain(self.request)
        self.assertTrue(response.converged)

    def test_feature_name_mapping(self):
        """Verify that output feature names match input keys."""
        response = self.explainer.explain(self.request)
        names = [attr.feature_name for attr in response.attributions]
        self.assertIn("x1", names)
        self.assertIn("x2", names)

    def test_shap_value_magnitude(self):
        """Verify SHAP values are non-zero for a non-trivial model."""
        response = self.explainer.explain(self.request)
        for attr in response.attributions:
            self.assertNotEqual(attr.shap_value, 0.0)

    def test_input_validation_nsamples(self):
        """Verify Pydantic validation for nsamples constraints."""
        with self.assertRaises(ValueError):
            ExplanationRequest(
                features={"a": 1},
                background_data=[{"a": 0}],
                nsamples=5  # Below min 10
            )

    def test_background_data_requirement(self):
        """Verify Pydantic validation for background data length."""
        with self.assertRaises(ValueError):
            ExplanationRequest(
                features={"a": 1},
                background_data=[],  # Empty list violates min_length=1
                nsamples=100
            )

if __name__ == "__main__":
    unittest.main()