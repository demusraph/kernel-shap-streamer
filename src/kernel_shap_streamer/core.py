import numpy as np
from scipy.linalg import lstsq
from typing import Callable, List
import time
from .models import ExplanationRequest, ExplanationResponse, FeatureImportance

class KernelSHAPExplainer:
    """
    Optimized KernelSHAP solver using weighted least squares.
    """
    def __init__(self, model_predict: Callable[[np.ndarray], np.ndarray]):
        self.model_predict = model_predict

    def _get_kernel_weights(self, z: np.ndarray, m: int) -> np.ndarray:
        """Computes SHAP kernel weights for coalition size z."""
        d = z.shape[1]
        ones = np.sum(z, axis=1)
        # Avoid division by zero by masking
        weights = np.zeros_like(ones, dtype=float)
        mask = (ones > 0) & (ones < d)
        weights[mask] = (d - 1) / (ones[mask] * (d - ones[mask]))
        # Assign high penalty to boundary coalitions
        weights[~mask] = 1e6
        return weights

    def explain(self, request: ExplanationRequest) -> ExplanationResponse:
        start_time = time.perf_counter()
        
        feature_names = list(request.features.keys())
        x_input = np.array([request.features[f] for f in feature_names])
        d = len(feature_names)
        
        # Generate coalitions
        z = np.random.randint(2, size=(request.nsamples, d))
        
        # Map coalitions to feature space
        bg_matrix = np.array([[d[f] for f in feature_names] for d in request.background_data])
        bg_mean = np.mean(bg_matrix, axis=0)
        
        perturbed = np.where(z == 1, x_input, bg_mean)
        y_pred = self.model_predict(perturbed)
        
        # Solve WLS
        weights = self._get_kernel_weights(z, d)
        w_sqrt = np.sqrt(weights).reshape(-1, 1)
        
        A = np.hstack([np.ones((request.nsamples, 1)), z]) * w_sqrt
        b = y_pred.reshape(-1, 1) * w_sqrt
        
        # lstsq returns (solution, residuals, rank, singular_values)
        phi, _, _, _ = lstsq(A, b)
        
        # Extract scalar from the (d+1, 1) array
        base_value = float(phi[0, 0])
        shap_values = phi[1:].flatten()
        
        attributions = [
            FeatureImportance(
                feature_name=feature_names[i],
                shap_value=float(shap_values[i]),
                contribution_pct=float(abs(shap_values[i]) / (np.sum(np.abs(shap_values)) + 1e-9))
            )
            for i in range(d)
        ]
        
        return ExplanationResponse(
            base_value=base_value,
            attributions=attributions,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            converged=True
        )