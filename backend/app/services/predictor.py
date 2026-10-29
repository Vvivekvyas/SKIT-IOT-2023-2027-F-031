"""
Prediction service: loads the ML team's hybrid model once and runs it for the API.

torch is imported only inside load(), never at module import time. If torch or a
trained model is missing, the service stays "not ready" and POST /predict answers
503, while the rest of the API keeps working.

Where the model comes from (checked in this order):
  1. PREDICT_MODEL_PATH  - a trained checkpoint file (format: docs/api/PREDICT_API.md)
  2. PREDICT_DEMO_MODE   - untrained random weights for demos; never allowed in production
  3. neither             - not ready (503)

Feature values are never logged and never stored.
"""
import logging
import math
import os
import sys
from pathlib import Path

from app.core.config import settings
from app.schemas.prediction import MAX_FEATURES

logger = logging.getLogger("ids.predict")

NORMAL_LABELS = {"benign", "benign_final", "normal"}
DEMO_FEATURES = 10
DEMO_CLASSES = ["BENIGN", "DDoS", "PortScan", "Mirai"]


class FeatureError(ValueError):
    """The request's features do not fit the model. The message is safe to show."""


class ModelUnavailable(RuntimeError):
    """No model is loaded."""


def is_normal(label: str) -> bool:
    return label.strip().lower() in NORMAL_LABELS


def _repo_root() -> Path:
    if settings.PIPELINE_ROOT:
        return Path(settings.PIPELINE_ROOT).resolve()
    # backend/app/services/predictor.py -> repo root is three levels above "services"
    return Path(__file__).resolve().parents[3]


def _import_ml():
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from ml.autoencoder.autoencoder import DeepAutoencoder
    from ml.inference.hybrid_predictor import HybridPredictor
    from ml.models.ft_transformer import FTTransformer

    return FTTransformer, DeepAutoencoder, HybridPredictor


class PredictionService:
    def __init__(self) -> None:
        self._predictor = None
        self._center = None
        self._scale = None
        self.model_version: str | None = None
        self.feature_names: list[str] = []
        self.class_names: list[str] = []
        self.unavailable_reason = "The model has not been loaded yet."

    @property
    def ready(self) -> bool:
        return self._predictor is not None

    # ---------- loading ----------

    def load(self) -> None:
        """Never raises: a broken model must not stop the API from starting."""
        self._predictor = None
        try:
            self._load()
        except Exception:
            logger.exception("Prediction model failed to load")
            self._predictor = None
            self.unavailable_reason = "The model could not be loaded. See the server log."

    def _load(self) -> None:
        try:
            import numpy  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            self.unavailable_reason = "PyTorch is not installed on the server."
            logger.warning("Prediction disabled: PyTorch is not installed")
            return

        path = settings.PREDICT_MODEL_PATH
        if path:
            if not os.path.isfile(path):
                self.unavailable_reason = "The configured model file was not found."
                logger.warning("Prediction disabled: model file not found at %s", path)
                return
            self._load_checkpoint(path)
        elif settings.PREDICT_DEMO_MODE:
            if settings.ENV == "production":
                self.unavailable_reason = "Demo mode is not allowed in production."
                logger.error("PREDICT_DEMO_MODE is ignored because ENV=production")
                return
            self._load_demo()
        else:
            self.unavailable_reason = "No trained model is installed yet."
            logger.info("Prediction disabled: no PREDICT_MODEL_PATH and demo mode is off")

    def _finish(self, predictor, names, classes, version) -> None:
        self._predictor = predictor
        self.feature_names = names
        self.class_names = classes
        self.model_version = version
        self.unavailable_reason = ""
        logger.info(
            "Prediction model ready: version=%s features=%d classes=%d",
            version, len(names), len(classes),
        )

    def _load_demo(self) -> None:
        import torch

        FTTransformer, DeepAutoencoder, HybridPredictor = _import_ml()
        torch.manual_seed(0)
        names = [f"feature_{i}" for i in range(1, DEMO_FEATURES + 1)]
        clf = FTTransformer(
            num_features=len(names), num_classes=len(DEMO_CLASSES),
            embedding_dim=16, num_heads=2, num_layers=1,
        )
        ae = DeepAutoencoder(input_dim=len(names), latent_dim=4, hidden_dims=[16])
        predictor = HybridPredictor(clf, ae, DEMO_CLASSES, feature_names=names)
        self._center = self._scale = None
        self._finish(predictor, names, list(DEMO_CLASSES), "demo-untrained")
        logger.warning("DEMO MODE: predictions come from UNTRAINED random weights")

    def _load_checkpoint(self, path: str) -> None:
        import numpy as np
        import torch

        FTTransformer, DeepAutoencoder, HybridPredictor = _import_ml()
        # weights_only=True refuses arbitrary pickled code inside the file.
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

        names = [str(n) for n in ckpt["feature_names"]]
        classes = [str(c) for c in ckpt["class_names"]]
        if not 1 <= len(names) <= MAX_FEATURES or len(set(names)) != len(names) or not classes:
            raise ValueError("Checkpoint feature/class metadata is invalid")

        clf = FTTransformer(
            num_features=len(names), num_classes=len(classes),
            **ckpt.get("classifier_config", {}),
        )
        clf.load_state_dict(ckpt["classifier_state"])
        ae = DeepAutoencoder(input_dim=len(names), **ckpt.get("autoencoder_config", {}))
        ae.load_state_dict(ckpt["autoencoder_state"])
        if "anomaly_threshold" in ckpt:
            ae.anomaly_threshold = float(ckpt["anomaly_threshold"])

        predictor = HybridPredictor(
            clf, ae, classes,
            confidence_threshold=float(ckpt.get("confidence_threshold", 0.85)),
            anomaly_flag_threshold=float(ckpt.get("anomaly_flag_threshold", 0.70)),
            feature_names=names,
        )

        center, scale = ckpt.get("scaler_center"), ckpt.get("scaler_scale")
        if (center is None) != (scale is None):
            raise ValueError("scaler_center and scaler_scale must be given together")
        self._center = self._scale = None
        if center is not None:
            c = np.asarray(center, dtype=np.float32)
            s = np.asarray(scale, dtype=np.float32)
            if c.shape != (len(names),) or s.shape != c.shape:
                raise ValueError("Scaler length does not match the feature list")
            self._center, self._scale = c, np.where(s == 0, 1.0, s).astype(np.float32)

        self._finish(predictor, names, classes, str(ckpt.get("model_version", os.path.basename(path))))

    # ---------- inference ----------

    def _to_row(self, features: dict):
        import numpy as np

        missing = [n for n in self.feature_names if n not in features]
        if missing:
            raise FeatureError(
                f"Missing {len(missing)} required feature(s), for example: {', '.join(missing[:5])}"
            )
        values = []
        for name in self.feature_names:
            value = features[name]
            if value is None or isinstance(value, str):
                raise FeatureError(f"Feature '{name}' must be a number")
            number = float(value)  # bool -> 0.0 / 1.0
            if not math.isfinite(number):
                raise FeatureError(f"Feature '{name}' must be a finite number")
            values.append(number)
        row = np.asarray(values, dtype=np.float32)
        if self._center is not None:
            row = (row - self._center) / self._scale
        return row.reshape(1, -1)

    def predict(self, features: dict) -> dict:
        """Returns {prediction, confidence, anomaly_score, model_version}."""
        if not self.ready:
            raise ModelUnavailable(self.unavailable_reason)
        result = self._predictor.predict(self._to_row(features))[0]
        return {
            "prediction": str(result["prediction"]),
            "confidence": float(result["confidence"]),
            "anomaly_score": float(result["anomaly_score"]),
            "model_version": self.model_version,
        }


prediction_service = PredictionService()
