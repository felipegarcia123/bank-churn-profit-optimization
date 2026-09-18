"""Validación y transformaciones ajustadas exclusivamente con entrenamiento."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

TARGET_COL = "churned"
ID_COL = "customer_id"
VALUE_COL = "annual_value_usd"
CATEGORICAL_FEATURES = ["geography", "gender"]
RAW_NUMERIC_FEATURES = [
    "credit_score", "age", "tenure_years", "balance", "n_products",
    "has_credit_card", "is_active_member", "estimated_salary",
]
ENGINEERED_NUMERIC_FEATURES = [
    "has_zero_balance", "balance_to_salary_ratio", "products_per_tenure_year",
]
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
RAW_FEATURES = RAW_NUMERIC_FEATURES + CATEGORICAL_FEATURES
_RENAME_MAP = dict(zip(
    ["CustomerId", "CreditScore", "Geography", "Gender", "Age", "Tenure",
     "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"],
    [ID_COL, "credit_score", "geography", "gender", "age", "tenure_years",
     "balance", "n_products", "has_credit_card", "is_active_member", "estimated_salary", TARGET_COL],
))
NIM_RATE = 0.025
FEE_PER_PRODUCT = 80.0
INTERCHANGE_PER_CARD = 150.0


def prepare_dataframe(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Normaliza el CSV; acepta nulos en predictores, pero no tipos inválidos."""
    df = df.drop(columns=["RowNumber", "Surname"], errors="ignore").rename(columns=_RENAME_MAP).copy()
    if df.empty:
        raise ValueError("La base de clientes está vacía.")
    if df.columns.duplicated().any():
        raise ValueError("Hay nombres de columnas duplicados tras normalizar el esquema.")
    required = set(RAW_FEATURES + ([TARGET_COL] if require_target else []))
    if missing := required - set(df.columns):
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    for col in RAW_NUMERIC_FEATURES + ([TARGET_COL] if TARGET_COL in df else []):
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"La columna {col} debe contener números.") from exc
        if np.isinf(df[col].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"La columna {col} contiene valores infinitos.")
        if (df[col].dropna() < 0).any():
            raise ValueError(f"La columna {col} no admite valores negativos.")
    for col in ["has_credit_card", "is_active_member", TARGET_COL]:
        if col in df and not df[col].dropna().isin([0, 1]).all():
            raise ValueError(f"La columna {col} solo admite 0 o 1.")
    if TARGET_COL in df and df[TARGET_COL].isna().any():
        raise ValueError("La columna churned no admite valores vacíos.")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    if ID_COL in df and (df[ID_COL].isna().any() or df[ID_COL].duplicated().any()):
        raise ValueError("Los identificadores de cliente deben ser únicos y no vacíos.")
    return df


def _compute_annual_value(df: pd.DataFrame) -> pd.Series:
    """Proxy de ingreso anual, no CLV de vida completa ni beneficio contable."""
    return (NIM_RATE * df["balance"] + FEE_PER_PRODUCT * df["n_products"]
            + INTERCHANGE_PER_CARD * df["has_credit_card"]).round(2)


def engineer_features(df: pd.DataFrame, segment_edges=None) -> pd.DataFrame:
    """Transformaciones deterministas; los segmentos admiten empates y lotes pequeños."""
    df = df.copy()
    df[VALUE_COL] = _compute_annual_value(df)
    df["has_zero_balance"] = (df["balance"] == 0).astype(int)
    df["balance_to_salary_ratio"] = (df["balance"] / (df["estimated_salary"] + 1)).round(3)
    df["products_per_tenure_year"] = (df["n_products"] / df["tenure_years"].clip(lower=1)).round(3)
    low, high = (1000.0, 3000.0) if segment_edges is None else segment_edges
    df["value_segment"] = np.select(
        [df[VALUE_COL] <= low, df[VALUE_COL] <= high], ["Low", "Mid"], default="High"
    )
    return df


class CustomerPreprocessor(TransformerMixin, BaseEstimator):
    """Persiste imputación, límites p99 y cortes de valor aprendidos en fit."""

    def fit(self, X, y=None):
        df = prepare_dataframe(X, require_target=False)
        self.medians_ = df[RAW_NUMERIC_FEATURES].median()
        if self.medians_.isna().any():
            raise ValueError("No se puede ajustar una columna numérica completamente vacía.")
        self.modes_ = {}
        for col in CATEGORICAL_FEATURES:
            modes = df[col].mode()
            if modes.empty:
                raise ValueError(f"No se puede ajustar {col}: todos los valores están vacíos.")
            self.modes_[col] = modes.iloc[0]
        filled = df.fillna({**self.medians_.to_dict(), **self.modes_})
        self.caps_ = filled[["balance", "estimated_salary"]].quantile(0.99)
        for col, cap in self.caps_.items():
            filled[col] = filled[col].clip(upper=cap)
        self.segment_edges_ = _compute_annual_value(filled).quantile([0.5, 0.8]).to_numpy()
        return self

    def transform(self, X):
        check_is_fitted(self, ["medians_", "modes_", "caps_", "segment_edges_"])
        df = prepare_dataframe(X, require_target=False)
        df = df.fillna({**self.medians_.to_dict(), **self.modes_})
        for col, cap in self.caps_.items():
            df[col] = df[col].clip(upper=cap)
        return engineer_features(df, self.segment_edges_)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ], remainder="drop")


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    ids_test: pd.Series


def split_data(df: pd.DataFrame, random_state: int = 42) -> SplitData:
    """60/20/20 estratificado; no calcula estadísticas de transformación."""
    df = prepare_dataframe(df).reset_index(drop=True)
    if ID_COL not in df:
        df[ID_COL] = [f"ROW_{i:06d}" for i in range(len(df))]
    train_val, test = train_test_split(df, test_size=0.2, stratify=df[TARGET_COL], random_state=random_state)
    train, validation = train_test_split(
        train_val, test_size=0.25, stratify=train_val[TARGET_COL], random_state=random_state
    )
    return SplitData(
        *(part[RAW_FEATURES].reset_index(drop=True) for part in [train, validation, test]),
        *(part[TARGET_COL].astype(int).reset_index(drop=True) for part in [train, validation, test]),
        test[ID_COL].reset_index(drop=True),
    )


def load_data(path: str) -> pd.DataFrame:
    return prepare_dataframe(pd.read_csv(path))


def full_pipeline(path: str) -> SplitData:
    return split_data(load_data(path))
