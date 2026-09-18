"""Datos sintéticos deterministas: CI no necesita datos ni modelos privados."""
import numpy as np
import pandas as pd


def customers(n=180):
    rng = np.random.default_rng(42)
    age = rng.integers(18, 80, n)
    active = rng.integers(0, 2, n)
    churn = ((age > 48) & (active == 0)).astype(int)
    return pd.DataFrame({
        "CustomerId": np.arange(1000, 1000 + n),
        "CreditScore": rng.integers(400, 850, n),
        "Geography": rng.choice(["France", "Spain", "Germany"], n),
        "Gender": rng.choice(["Female", "Male"], n),
        "Age": age, "Tenure": rng.integers(0, 11, n),
        "Balance": rng.uniform(1000, 200000, n),
        "NumOfProducts": rng.integers(1, 5, n),
        "HasCrCard": rng.integers(0, 2, n), "IsActiveMember": active,
        "EstimatedSalary": rng.uniform(1000, 200000, n), "Exited": churn,
    })
