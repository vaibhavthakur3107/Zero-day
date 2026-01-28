"""
Base detector class for Zero-Day Attack Detection System.

Provides the abstract base class and common functionality for all
anomaly detection models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

import numpy as np
from sklearn.base import BaseEstimator

from config import ISOLATION_FOREST, ONE_CLASS_SVM, LOF

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """
    Abstract base class for anomaly detection models.

    This class defines the common interface and functionality for all
    detection models used in the zero-day attack detection system.

    Attributes:
        model: The underlying sklearn-compatible model.
        contamination: Expected proportion of anomalies in data.
        is_trained: Whether the model has been trained.
    """

    def __init__(
        self,
        model_name: str,
        contamination: float = 0.15,
        random_state: Optional[int] = 42
    ):
        """
        Initialize the base detector.

        Args:
            model_name: Name of the model for display purposes.
            contamination: Expected proportion of anomalies.
            random_state: Random state for reproducibility.
        """
        self.model_name = model_name
        self.contamination = contamination
        self.random_state = random_state
        self.model: Optional[BaseEstimator] = None
        self.is_trained = False

        logger.info(f"Initialized {model_name} detector")

    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """
        Create the underlying sklearn model instance.

        Must be implemented by subclasses.

        Returns:
            BaseEstimator: The sklearn model instance.
        """
        pass

    def fit(self, X: np.ndarray) -> 'BaseDetector':
        """
        Train the detector on the given data.

        Args:
            X: Training data (benign samples only).

        Returns:
            self: The fitted detector instance.
        """
        if self.model is None:
            self.model = self._create_model()

        logger.info(f"Training {self.model_name} on {X.shape[0]} samples")

        self.model.fit(X)
        self.is_trained = True

        logger.info(f"{self.model_name} training completed")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies in the data.

        Args:
            X: Data to predict on.

        Returns:
            np.ndarray: Binary predictions (0=normal, 1=anomaly).
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.model_name} must be trained before prediction")

        predictions = self.model.predict(X)
        # Convert to binary: -1 (anomaly) -> 1, 1 (normal) -> 0
        return (predictions == -1).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for the data.

        Args:
            X: Data to compute scores for.

        Returns:
            np.ndarray: Anomaly scores (higher = more anomalous).
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.model_name} must be trained before scoring")

        scores = self.model.decision_function(X)
        return scores

    def get_params(self) -> Dict[str, Any]:
        """
        Get model hyperparameters.

        Returns:
            Dict[str, Any]: Model parameters.
        """
        if self.model is not None:
            return self.model.get_params()
        return {}


class IsolationForestDetector(BaseDetector):
    """
    Isolation Forest anomaly detector.

    Isolation Forest is based on the principle that anomalies are
    few and different, making them easier to isolate. It works by
    randomly selecting a feature and then randomly selecting a split
    value between the max and min values of the selected feature.

    The anomaly score is based on the path length required to isolate
    a sample - shorter paths indicate anomalies.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.15,
        max_samples: Any = 'auto',
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: Optional[int] = 42
    ):
        """
        Initialize the Isolation Forest detector.

        Args:
            n_estimators: Number of base estimators in the ensemble.
            contamination: Proportion of outliers in the data set.
            max_samples: Number of samples to draw for training.
            max_features: Number of features to draw for training.
            bootstrap: Whether to use bootstrap sampling.
            random_state: Random state for reproducibility.
        """
        super().__init__(
            model_name="Isolation Forest",
            contamination=contamination,
            random_state=random_state
        )

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap

        # Use config defaults if parameters not specified
        self.params = ISOLATION_FOREST.copy()
        self.params.update({
            'n_estimators': n_estimators,
            'contamination': contamination,
            'max_samples': max_samples,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'random_state': random_state
        })

    def _create_model(self) -> BaseEstimator:
        """Create and return the Isolation Forest model instance."""
        from sklearn.ensemble import IsolationForest

        return IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1
        )


class OneClassSVMDetector(BaseDetector):
    """
    One-Class SVM anomaly detector.

    One-Class SVM learns a decision boundary around the normal data
    using support vectors. Points outside this boundary are classified
    as anomalies. It works well for high-dimensional data but can be
    computationally expensive for large datasets.

    Uses the RBF kernel by default to capture complex boundaries.
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        nu: float = 0.15,
        gamma: str = 'scale',
        cache_size: int = 500,
        random_state: Optional[int] = 42
    ):
        """
        Initialize the One-Class SVM detector.

        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly').
            nu: Upper bound on fraction of outliers.
            gamma: Kernel coefficient for 'rbf', 'poly', 'sigmoid'.
            cache_size: Cache size for kernel computation (MB).
            random_state: Random state for reproducibility.
        """
        super().__init__(
            model_name="One-Class SVM",
            contamination=nu,
            random_state=random_state
        )

        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.cache_size = cache_size

        # Use config defaults if parameters not specified
        self.params = ONE_CLASS_SVM.copy()
        self.params.update({
            'kernel': kernel,
            'nu': nu,
            'gamma': gamma,
            'cache_size': cache_size
        })

    def _create_model(self) -> BaseEstimator:
        """Create and return the One-Class SVM model instance."""
        from sklearn.svm import OneClassSVM

        return OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma,
            cache_size=self.cache_size
        )


class LOFDetector(BaseDetector):
    """
    Local Outlier Factor anomaly detector.

    LOF measures the local deviation of a sample with respect to its
    k-nearest neighbors. It identifies samples that are significantly
    less dense than their neighbors as anomalies. This makes it
    particularly effective for datasets with varying densities.

    Unlike other detectors, LOF doesn't learn a global boundary but
    compares each point's local density to its neighbors.
    """

    def __init__(
        self,
        n_neighbors: int = 30,
        contamination: float = 0.15,
        novelty: bool = True,
        random_state: Optional[int] = 42
    ):
        """
        Initialize the LOF detector.

        Args:
            n_neighbors: Number of neighbors to use.
            contamination: Proportion of outliers.
            novelty: If True, enables decision_function for new data.
            random_state: Random state for reproducibility.
        """
        super().__init__(
            model_name="Local Outlier Factor",
            contamination=contamination,
            random_state=random_state
        )

        self.n_neighbors = n_neighbors
        self.novelty = novelty

        # Use config defaults if parameters not specified
        self.params = LOF.copy()
        self.params.update({
            'n_neighbors': n_neighbors,
            'contamination': contamination,
            'novelty': novelty
        })

    def _create_model(self) -> BaseEstimator:
        """Create and return the LOF model instance."""
        from sklearn.neighbors import LocalOutlierFactor

        return LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=self.novelty,
            n_jobs=-1
        )


# =============================================================================
# Detector Factory
# =============================================================================

def create_detector(detector_type: str, **kwargs) -> BaseDetector:
    """
    Factory function to create detector instances.

    Args:
        detector_type: Type of detector ('isolation_forest', 'one_class_svm', 'lof').
        **kwargs: Additional parameters for the detector.

    Returns:
        BaseDetector: Configured detector instance.

    Raises:
        ValueError: If unknown detector type is specified.
    """
    detector_mapping = {
        'isolation_forest': IsolationForestDetector,
        'one_class_svm': OneClassSVMDetector,
        'lof': LOFDetector
    }

    if detector_type not in detector_mapping:
        raise ValueError(
            f"Unknown detector type: {detector_type}. "
            f"Available types: {list(detector_mapping.keys())}"
        )

    return detector_mapping[detector_type](**kwargs)
