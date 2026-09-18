"""Dashboard local: base en data/raw, política económica y exportación de llamadas."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src import ARTIFACT_VERSION
from src.data_processing import ID_COL, TARGET_COL, VALUE_COL, prepare_dataframe
from src.financial_evaluator import (
    EconomicConfig, compare_against_baselines, contact_mask,
    cost_benefit_breakdown, priority_call_list,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", str(PROJECT_DIR / "models/best_model.pkl")))
DATA_PATH = Path(os.getenv("CHURN_DATA_PATH", str(PROJECT_DIR / "data/raw/Churn_Modelling.csv")))


@st.cache_resource
def load_artifact(path: str, modified_at_ns: int):
    # La fecha en la clave invalida el cache al regenerar el modelo.
    artifact = joblib.load(path)
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("El modelo usa un formato anterior. Ejecuta python -m src.train_pipeline.")
    return artifact


def money(value):
    return f"US${value:,.0f}"


st.set_page_config(page_title="Retention Copilot", page_icon="💰", layout="wide")
st.title("💰 Retention Copilot")
st.caption("Prioriza campañas de retención por beneficio incremental esperado.")

if not MODEL_PATH.is_file():
    st.error("Falta el modelo local. Ejecuta: python -m src.train_pipeline")
    st.stop()
try:
    artifact = load_artifact(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
except Exception as exc:
    st.error(f"No se pudo cargar el modelo: {exc}")
    st.stop()

st.sidebar.header("Parámetros de la campaña")
default = artifact["economic_config"]
cost = st.sidebar.number_input("Costo por contacto (US$)", min_value=0.01,
                               value=float(default["retention_cost"]), step=1.0)
success = st.sidebar.slider("Tasa de éxito de retención", min_value=0.0, max_value=1.0,
                            value=float(default["retention_success_rate"]), step=0.05,
                            help="Supuesto de clientes que se retienen entre quienes habrían abandonado.")
cfg = EconomicConfig(cost, success)
manual = st.sidebar.checkbox("Aplicar además un riesgo mínimo")
threshold = st.sidebar.slider("Riesgo mínimo de abandono", min_value=0.0, max_value=1.0,
                              value=0.2, step=0.01, disabled=not manual)
threshold = threshold if manual else None
st.sidebar.caption("Se recomienda contactar solo si probabilidad × ingreso anual × éxito supera el costo.")
st.sidebar.markdown(f"**Modelo activo:** `{artifact['model_name']}`")

st.subheader("1. Base de clientes")
st.caption(f"Fuente local: {DATA_PATH.name}")
if not DATA_PATH.is_file():
    st.error("No se encontró la base de clientes. Coloca Churn_Modelling.csv en data/raw/.")
    st.stop()
try:
    df = prepare_dataframe(pd.read_csv(DATA_PATH), require_target=False)
    features = artifact["value_processor"].transform(df)
    proba = artifact["pipeline"].predict_proba(df)[:, 1]
except Exception as exc:
    st.error(f"No se pudo procesar la base de clientes: {exc}")
    st.stop()
st.success(f"Cargados {len(df):,} clientes.")
st.info(
    "Simulación con supuestos de costo y éxito de retención. El valor por cliente es una estimación "
    "de ingreso anual, no CLV de vida completa. Los resultados no son ingresos observados."
)
if TARGET_COL in df:
    st.caption("Esta base contiene etiquetas históricas y puede incluir clientes de entrenamiento. "
               "La evaluación reservada está en reports/executive_report.json.")

values = features[VALUE_COL].to_numpy()
mask = contact_mask(proba, values, cfg, threshold)
saved = float(cfg.retention_success_rate * (proba[mask] * values[mask]).sum())
campaign_cost = float(mask.sum() * cfg.retention_cost)
net = saved - campaign_cost

st.subheader("2. Recomendación de campaña")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes a contactar", f"{mask.sum():,}")
c2.metric("Ingreso anual preservado (esperado)", money(saved))
c3.metric("Costo de campaña", money(campaign_cost))
c4.metric("Beneficio incremental esperado", money(net),
          f"ROI: {net / campaign_cost:.0%}" if campaign_cost else "Sin campaña")

if TARGET_COL in df:
    with st.expander("Simulación retrospectiva con etiquetas históricas"):
        breakdown = cost_benefit_breakdown(df[TARGET_COL].to_numpy(), proba, values, threshold, cfg)
        comparison = compare_against_baselines(df[TARGET_COL].to_numpy(), proba, values, cfg, threshold)
        st.caption(f"Beneficio simulado con las etiquetas: {money(breakdown.net_profit)}. "
                   "La tasa de éxito sigue siendo un supuesto; no mide efecto causal.")
        st.dataframe(comparison.scenarios, width="stretch", hide_index=True)

st.subheader("3. Lista priorizada de llamadas")
ids = df[ID_COL] if ID_COL in df else pd.Series([f"ROW_{i:06d}" for i in range(len(df))])
calls = priority_call_list(ids, proba, values, cfg, threshold=threshold,
                           extra_cols=features[["geography", "value_segment", "tenure_years"]])
if calls.empty:
    st.info("Ningún cliente supera el costo de contacto con estos parámetros.")
else:
    top_k = len(calls) if len(calls) == 1 else st.slider(
        "Mostrar top-K llamadas", min_value=1, max_value=len(calls), value=min(50, len(calls)),
    )
    st.dataframe(calls.head(top_k), width="stretch", hide_index=True)
    st.subheader("4. Beneficio esperado según el número de contactos")
    st.line_chart(calls.set_index("priority_rank")[["cumulative_expected_profit_usd", "cumulative_campaign_cost_usd"]])
    st.caption("Todas las llamadas recomendadas tienen beneficio esperado positivo. "
               "Con presupuesto limitado, prioriza las primeras filas.")
st.download_button("Descargar lista priorizada (CSV)", calls.to_csv(index=False).encode("utf-8"),
                   file_name="priority_calls.csv", mime="text/csv", disabled=calls.empty)
