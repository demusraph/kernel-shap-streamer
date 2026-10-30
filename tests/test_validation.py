import unittest
from pydantic import ValidationError
from src.kernel_shap_streamer.models import ExplanationRequest, FeatureImportance

class TestValidation(unittest.TestCase):
    """
    Exhaustive validation suite for KernelSHAP Streamer data models.
    Ensures schema enforcement, boundary constraints, and type safety.
    """

    def setUp(self):
        self.valid_features = {"age": 25.0, "income": 50000.0}
        self.valid_background = [{"age": 20.0, "income": 40000.0}, {"age": 30.0, "income": 60000.0}]

    def test_valid_request_schema(self):
        """Test that a perfectly formed request passes validation."""
        req = ExplanationRequest(
            features=self.valid_features,
            background_data=self.valid_background,
            nsamples=100
        )
        self.assertEqual(req.nsamples, 100)
        self.assertEqual(len(req.features), 2)

    def test_nsamples_lower_bound(self):
        """Test that nsamples < 10 raises ValidationError."""
        with self.assertRaises(ValidationError):
            ExplanationRequest(
                features=self.valid_features,
                background_data=self.valid_background,
                nsamples=9
            )

    def test_nsamples_upper_bound(self):
        """Test that nsamples > 1000 raises ValidationError."""
        with self.assertRaises(ValidationError):
            ExplanationRequest(
                features=self.valid_features,
                background_data=self.valid_background,
                nsamples=1001
            )

    def test_empty_background_data(self):
        """Test that background_data cannot be empty (min_length=1)."""
        with self.assertRaises(ValidationError):
            ExplanationRequest(
                features=self.valid_features,
                background_data=[],
                nsamples=100
            )

    def test_invalid_feature_types(self):
        """Test that non-numeric feature values raise ValidationError."""
        with self.assertRaises(ValidationError):
            ExplanationRequest(
                features={"age": "twenty-five"},
                background_data=self.valid_background,
                nsamples=100
            )

    def test_feature_importance_model(self):
        """Test FeatureImportance model instantiation and type casting."""
        fi = FeatureImportance(
            feature_name="age",
            shap_value=0.5,
            contribution_pct=0.25
        )
        self.assertEqual(fi.feature_name, "age")
        self.assertIsInstance(fi.shap_value, float)
        self.assertEqual(fi.contribution_pct, 0.25)

    def test_missing_required_fields(self):
        """Test that missing required fields in ExplanationRequest raise ValidationError."""
        with self.assertRaises(ValidationError):
            # Missing background_data
            ExplanationRequest(features=self.valid_features)

    def test_nsamples_boundary_edge_cases(self):
        """Test exact boundary conditions for nsamples (10 and 1000)."""
        req_min = ExplanationRequest(
            features=self.valid_features,
            background_data=self.valid_background,
            nsamples=10
        )
        req_max = ExplanationRequest(
            features=self.valid_features,
            background_data=self.valid_background,
            nsamples=1000
        )
        self.assertEqual(req_min.nsamples, 10)
        self.assertEqual(req_max.nsamples, 1000)

    def test_malformed_background_data_structure(self):
        """Test that background_data with invalid inner types raises ValidationError."""
        with self.assertRaises(ValidationError):
            ExplanationRequest(
                features=self.valid_features,
                background_data=[{"age": "invalid_string"}],
                nsamples=100
            )

    def test_feature_importance_type_coercion(self):
        """Test that FeatureImportance coerces integers to floats correctly."""
        fi = FeatureImportance(
            feature_name="test",
            shap_value=1,
            contribution_pct=0
        )
        self.assertIsInstance(fi.shap_value, float)
        self.assertEqual(fi.shap_value, 1.0)

if __name__ == "__main__":
    unittest.main()