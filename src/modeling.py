"""Clasificadores calibrados en entrenamiento; selección posterior en validación."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_processing import CustomerPreprocessor, SplitData, build_preprocessor


@dataclass
class TrainedModel:
    name: str
    pipeline: CalibratedClassifierCV
    proba_validation: np.ndarray
    metrics: dict[str, float]


def build_model(classifier) -> CalibratedClassifierCV:
    # Cada fold ajusta también imputación, p99, escalado y codificación.
    pipeline = Pipeline([
        ("features", CustomerPreprocessor()),
        ("preprocessor", build_preprocessor()),
        ("clf", classifier),
    ])
    return CalibratedClassifierCV(
        estimator=pipeline, method="sigmoid",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=1,
    )


def technical_metrics(y_true, proba) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "f1_at_0.5": float(f1_score(y_true, proba >= 0.5, zero_division=0)),
    }


def train_all(split: SplitData) -> dict[str, TrainedModel]:
    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20, n_jobs=-1, random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=42,
        ),
    }
    results = {}
    for name, classifier in classifiers.items():
        pipeline = build_model(classifier)
        pipeline.fit(split.X_train, split.y_train)
        proba = pipeline.predict_proba(split.X_validation)[:, 1]
        results[name] = TrainedModel(name, pipeline, proba, technical_metrics(split.y_validation, proba))
    return results


def summary_table(models: dict[str, TrainedModel]) -> pd.DataFrame:
    return pd.DataFrame([{"model": m.name, **m.metrics} for m in models.values()])
