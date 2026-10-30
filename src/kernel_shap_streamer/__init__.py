"""
KernelSHAP Streamer: High-performance local feature attribution.
"""
__version__ = "0.1.0"

from .models import ExplanationRequest, ExplanationResponse, FeatureImportance
from .core import KernelSHAPExplainer

__all__ = ["ExplanationRequest", "ExplanationResponse", "FeatureImportance", "KernelSHAPExplainer"]