"""Beneficio incremental esperado frente a no intervenir; supuestos explícitos."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EconomicConfig:
    retention_cost: float = 15.0
    retention_success_rate: float = 0.30
    currency: str = "USD"

    def __post_init__(self):
        if not np.isfinite(self.retention_cost) or self.retention_cost <= 0:
            raise ValueError("El costo de retención debe ser finito y mayor que cero.")
        if not np.isfinite(self.retention_success_rate) or not 0 <= self.retention_success_rate <= 1:
            raise ValueError("La tasa de éxito debe estar entre 0 y 1.")


def _inputs(proba, customer_value, y_true=None):
    p, v = np.asarray(proba, dtype=float), np.asarray(customer_value, dtype=float)
    if p.ndim != 1 or v.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(v).all():
        raise ValueError("Probabilidades y valores deben ser vectores finitos de igual longitud.")
    if ((p < 0) | (p > 1)).any() or (v < 0).any():
        raise ValueError("Las probabilidades deben estar entre 0 y 1 y los valores no ser negativos.")
    if y_true is not None:
        y = np.asarray(y_true)
        if y.shape != p.shape or not np.isin(y, [0, 1]).all():
            raise ValueError("El target debe ser un vector binario de igual longitud.")
    return p, v


def expected_profit(proba, customer_value, cfg: EconomicConfig) -> np.ndarray:
    p, v = _inputs(proba, customer_value)
    return cfg.retention_success_rate * p * v - cfg.retention_cost


def contact_mask(proba, customer_value, cfg: EconomicConfig, threshold=None) -> np.ndarray:
    """Contactar solo si el beneficio esperado es positivo; corte de riesgo opcional."""
    p, v = _inputs(proba, customer_value)
    mask = expected_profit(p, v, cfg) > 0
    if threshold is not None:
        if not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1.0, 2.0):
            raise ValueError("El threshold debe estar entre 0 y 1 (o el límite de no contacto).")
        mask &= p >= threshold
    return mask


def per_customer_profit(y_true, proba, customer_value, threshold, cfg) -> np.ndarray:
    """Delta frente a no actuar: no contactar tiene valor incremental cero."""
    p, v = _inputs(proba, customer_value, y_true)
    mask = contact_mask(p, v, cfg, threshold)
    return np.where(mask, np.asarray(y_true) * cfg.retention_success_rate * v - cfg.retention_cost, 0.0)


@dataclass
class CostBenefitBreakdown:
    threshold: float | None
    tp: int
    fp: int
    fn: int
    tn: int
    contacted: int
    revenue_saved: float
    campaign_cost: float
    revenue_lost_fn: float
    net_profit: float
    roi: float

    def as_dict(self):
        return asdict(self)


def cost_benefit_breakdown(y_true, proba, customer_value, threshold, cfg) -> CostBenefitBreakdown:
    p, v = _inputs(proba, customer_value, y_true)
    mask, churn = contact_mask(p, v, cfg, threshold), np.asarray(y_true) == 1
    saved = float(cfg.retention_success_rate * v[mask & churn].sum())
    cost = float(cfg.retention_cost * mask.sum())
    net = float(per_customer_profit(y_true, p, v, threshold, cfg).sum())
    return CostBenefitBreakdown(
        threshold, int((mask & churn).sum()), int((mask & ~churn).sum()),
        int((~mask & churn).sum()), int((~mask & ~churn).sum()), int(mask.sum()),
        saved, cost, float(v[~mask & churn].sum()), net, net / cost if cost else 0.0,
    )


@dataclass
class ThresholdSweep:
    grid: pd.DataFrame
    best_threshold: float
    best_breakdown: CostBenefitBreakdown


def optimize_threshold(y_true, proba, customer_value, cfg, n_points=101) -> ThresholdSweep:
    """Exploración SOLO en validación; incluye explícitamente no contactar."""
    thresholds = np.append(np.linspace(0, 1, n_points), np.nextafter(1.0, 2.0))
    grid = pd.DataFrame([
        cost_benefit_breakdown(y_true, proba, customer_value, float(t), cfg).as_dict()
        for t in thresholds
    ])
    best = grid.sort_values(["net_profit", "contacted"], ascending=[False, True]).iloc[0]
    threshold = float(best["threshold"])
    return ThresholdSweep(grid, threshold, cost_benefit_breakdown(y_true, proba, customer_value, threshold, cfg))


def priority_call_list(customer_ids, proba, customer_value, cfg, threshold=None, top_n=None, extra_cols=None):
    p, v = _inputs(proba, customer_value)
    mask = contact_mask(p, v, cfg, threshold)
    ids = np.asarray(customer_ids)
    if ids.shape != p.shape:
        raise ValueError("Debe existir un identificador por predicción.")
    df = pd.DataFrame({
        "customer_id": ids,
        "churn_probability": p,
        "annual_value_usd": v,
        "revenue_at_risk_usd": p * v,
        "expected_profit_if_called_usd": expected_profit(p, v, cfg),
    })
    if extra_cols is not None:
        if len(extra_cols) != len(df):
            raise ValueError("Las columnas adicionales deben tener una fila por cliente.")
        df = pd.concat([df, extra_cols.reset_index(drop=True)], axis=1)
    df = df.loc[mask].sort_values("expected_profit_if_called_usd", ascending=False, kind="stable").reset_index(drop=True)
    if top_n is not None:
        if top_n < 0:
            raise ValueError("top_n no puede ser negativo.")
        df = df.head(top_n).copy()
    df["cumulative_expected_profit_usd"] = df["expected_profit_if_called_usd"].cumsum()
    df["cumulative_campaign_cost_usd"] = np.arange(1, len(df) + 1) * cfg.retention_cost
    df["priority_rank"] = np.arange(1, len(df) + 1)
    # Redondear después de decidir y acumular evita cambiar la política en los límites.
    return df.round({"churn_probability": 6, **{c: 2 for c in df if c.endswith("_usd")}})


@dataclass
class BusinessComparison:
    scenarios: pd.DataFrame
    uplift_vs_do_nothing: float
    uplift_vs_call_everyone: float
    uplift_vs_random: float


def compare_against_baselines(y_true, proba, customer_value, cfg, model_threshold=None):
    """Misma población; azar = esperanza exacta al elegir k clientes sin reemplazo."""
    p, v = _inputs(proba, customer_value, y_true)
    y = np.asarray(y_true)
    mask = contact_mask(p, v, cfg, model_threshold)
    n, k = len(p), int(mask.sum())
    total_at_risk = float((y * v).sum())
    all_saved = cfg.retention_success_rate * total_at_risk
    model_saved = float(cfg.retention_success_rate * (y[mask] * v[mask]).sum())
    rows = []
    for name, count, saved in [
        ("No actuar", 0, 0.0),
        ("Llamar a todos", n, all_saved),
        ("Azar: misma cobertura (esperanza)", k, all_saved * k / n if n else 0.0),
        ("Modelo: política económica", k, model_saved),
    ]:
        cost = float(count * cfg.retention_cost)
        rows.append({
            "scenario": name, "contacted": count, "campaign_cost_usd": cost,
            "net_profit_vs_status_quo_usd": saved - cost,
            "remaining_revenue_at_risk_usd": total_at_risk - saved,
        })
    profits = [row["net_profit_vs_status_quo_usd"] for row in rows]
    return BusinessComparison(pd.DataFrame(rows), profits[3], profits[3] - profits[1], profits[3] - profits[2])


def executive_report(breakdown: CostBenefitBreakdown, comparison: BusinessComparison, cfg: EconomicConfig) -> str:
    b = breakdown
    return "\n".join([
        "REPORTE DE EVALUACIÓN — TEST RESERVADO",
        "Beneficio incremental simulado; no representa ingresos observados de una campaña.",
        f"Costo por contacto: {cfg.retention_cost:.2f} {cfg.currency}",
        f"Éxito de retención supuesto: {cfg.retention_success_rate:.0%}",
        "Política: probabilidad calibrada × ingreso anual × éxito > costo",
        f"Clientes contactados: {b.contacted:,}",
        f"TP: {b.tp} | FP: {b.fp} | FN: {b.fn} | TN: {b.tn}",
        f"Ingreso anual preservado (simulado): {b.revenue_saved:,.2f} {cfg.currency}",
        f"Costo de campaña: {b.campaign_cost:,.2f} {cfg.currency}",
        f"Beneficio incremental: {b.net_profit:,.2f} {cfg.currency}",
        f"ROI incremental: {b.roi:.1%}",
        f"Mejora vs. llamar a todos: {comparison.uplift_vs_call_everyone:,.2f} {cfg.currency}",
        f"Mejora vs. azar (esperanza): {comparison.uplift_vs_random:,.2f} {cfg.currency}",
    ]) + "\n"
