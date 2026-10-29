"""Baseline Machine Learning Models for Intrusion Detection System (Week 7 - Ronak).

Implements standard benchmark classifiers (Random Forest, Decision Tree, Logistic Regression,
Gaussian Naive Bayes) supporting binary and multiclass network flow classification.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import joblib


class BaselineModelSuite:
    """
    Standard tabular baseline model suite establishing benchmark detection baselines
    prior to deep learning (VAE + FT-Transformer) integration.
    """

    DEFAULT_MODELS = ["random_forest", "decision_tree", "logistic_regression", "naive_bayes"]
    EXTENDED_MODELS = ["svm", "gradient_boosting", "mlp"]
    AVAILABLE_MODELS = DEFAULT_MODELS + EXTENDED_MODELS

    def __init__(
        self,
        random_state: int = 42,
        models_to_include: Optional[List[str]] = None,
    ):
        """
        Args:
            random_state: Deterministic seed.
            models_to_include: List of model names to initialize. If None, default 4 models are loaded.
        """
        self.random_state = random_state
        self.model_names = models_to_include or list(self.DEFAULT_MODELS)
        self.models: Dict[str, Any] = {}
        self.fitted_models: Dict[str, bool] = {}

        self._init_models()

    def _init_models(self) -> None:
        """Instantiates default configurations for benchmark models."""
        for name in self.model_names:
            if name == "random_forest":
                self.models[name] = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=15,
                    random_state=self.random_state,
                    n_jobs=1,
                )
            elif name == "decision_tree":
                self.models[name] = DecisionTreeClassifier(
                    max_depth=15,
                    random_state=self.random_state,
                )
            elif name == "logistic_regression":
                self.models[name] = LogisticRegression(
                    max_iter=500,
                    random_state=self.random_state,
                )
            elif name == "naive_bayes":
                self.models[name] = GaussianNB()
            elif name == "svm":
                self.models[name] = LinearSVC(
                    random_state=self.random_state,
                    dual="auto",
                    max_iter=1000,
                )
            elif name == "gradient_boosting":
                self.models[name] = GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=5,
                    random_state=self.random_state,
                )
            elif name == "mlp":
                self.models[name] = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=300,
                    random_state=self.random_state,
                )
            else:
                raise ValueError(f"Unknown baseline model: {name}. Supported: {self.AVAILABLE_MODELS}")
            self.fitted_models[name] = False

    def fit_single(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Any:
        """Fits a specific baseline model."""
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not in suite.")
        self.models[model_name].fit(X, y)
        self.fitted_models[model_name] = True
        return self.models[model_name]

    def fit_all(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fits all configured baseline models on training data."""
        for name in self.model_names:
            self.fit_single(name, X, y)
        return self.models

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Generates class predictions using the specified model."""
        if not self.fitted_models.get(model_name, False):
            raise RuntimeError(f"Model '{model_name}' is not fitted.")
        return self.models[model_name].predict(X)

    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Generates class probability distributions using the specified model."""
        if not self.fitted_models.get(model_name, False):
            raise RuntimeError(f"Model '{model_name}' is not fitted.")
        if hasattr(self.models[model_name], "predict_proba"):
            return self.models[model_name].predict_proba(X)
        raise AttributeError(f"Model '{model_name}' does not support predict_proba.")

    def save_model(self, model_name: str, file_path: str) -> None:
        """Serializes a fitted model to disk."""
        if not self.fitted_models.get(model_name, False):
            raise RuntimeError(f"Cannot save unfitted model '{model_name}'.")
        joblib.dump(self.models[model_name], file_path)

    def load_model(self, model_name: str, file_path: str) -> None:
        """Loads a model artifact from disk."""
        self.models[model_name] = joblib.load(file_path)
        self.fitted_models[model_name] = True
