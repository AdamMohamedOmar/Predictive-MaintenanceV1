"""Windowing, feature extraction, and dataset building pipeline."""

from .windowing import sliding_windows, count_windows
from .extractor import extract_features, feature_names
from .dataset_builder import build_dataset, load_dataset, LABEL_TO_ID, FAULT_TYPES
from .normalizer import BaselineNormalizer, normalised_feature_names

__all__ = [
    "sliding_windows",
    "count_windows",
    "extract_features",
    "feature_names",
    "build_dataset",
    "load_dataset",
    "LABEL_TO_ID",
    "FAULT_TYPES",
    "BaselineNormalizer",
    "normalised_feature_names",
]
