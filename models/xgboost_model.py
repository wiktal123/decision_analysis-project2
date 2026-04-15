from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.validation import check_is_fitted

import xgboost as xgb


class XGBoostModel(BaseEstimator, ClassifierMixin):
    def __init__(self, num_classes: int | None = None, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.kwargs = kwargs
        self._label_encoder = LabelEncoder()
        self.pipeline: Pipeline | None = None
        self.backend: str | None = None

    def _build_estimator(self, n_classes: int):
        params = dict(self.kwargs)
        params.setdefault("random_state", 42)
        params.setdefault("n_estimators", 300)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("max_depth", 4)
        params.setdefault("subsample", 0.9)
        params.setdefault("colsample_bytree", 0.9)

        if n_classes > 2:
            params.setdefault("objective", "multi:softprob")
            params.setdefault("eval_metric", "mlogloss")
            params["num_class"] = n_classes
        else:
            params.setdefault("objective", "binary:logistic")
            params.setdefault("eval_metric", "logloss")

        self.backend = "xgboost"
        return xgb.XGBClassifier(**params)

    def _build_pipeline(self, X, estimator):
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = [col for col in X.columns if col not in categorical_cols]
        preprocess = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_cols,
                ),
                ("numeric", "passthrough", numeric_cols),
            ]
        )
        return Pipeline([("preprocess", preprocess), ("model", estimator)])

    def fit(self, X, y, **kwargs) -> "XGBoostModel":
        y_encoded = self._label_encoder.fit_transform(np.asarray(y))
        n_classes = self.num_classes if self.num_classes is not None else len(self._label_encoder.classes_)
        estimator = self._build_estimator(n_classes=n_classes)
        self.pipeline = self._build_pipeline(X, estimator)
        self.pipeline.fit(X, y_encoded, **kwargs)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, attributes=["pipeline", "classes_"])
        y_pred_encoded = self.pipeline.predict(X)
        return self._label_encoder.inverse_transform(np.asarray(y_pred_encoded, dtype=int))

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, attributes=["pipeline", "classes_"])
        return self.pipeline.predict_proba(X)
