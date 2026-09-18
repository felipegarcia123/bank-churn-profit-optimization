"""Entrenar en 60%, seleccionar en 20% y evaluar una vez en el 20% reservado."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import joblib

from src import ARTIFACT_VERSION
from src.data_processing import CustomerPreprocessor, VALUE_COL, full_pipeline
from src.financial_evaluator import (
    EconomicConfig, compare_against_baselines, cost_benefit_breakdown,
    executive_report, priority_call_list,
)
from src.modeling import technical_metrics, train_all


def run_training(data_path, out_model, out_report, out_calls, cfg):
    print("[1/4] Separando entrenamiento, validación y test (60/20/20)...", flush=True)
    split = full_pipeline(str(data_path))
    value_processor = CustomerPreprocessor().fit(split.X_train)
    validation_value = value_processor.transform(split.X_validation)[VALUE_COL].to_numpy()
    print("[2/4] Entrenando y calibrando tres modelos (CV de 3 folds)...", flush=True)
    models = train_all(split)
    validation = []
    for name, model in models.items():
        breakdown = cost_benefit_breakdown(
            split.y_validation, model.proba_validation, validation_value, None, cfg,
        )
        validation.append({"model": name, **model.metrics, "net_profit": breakdown.net_profit,
                           "contacted": breakdown.contacted})
    best_row = max(validation, key=lambda row: row["net_profit"])
    best = models[best_row["model"]]
    print(f"[3/4] Evaluación única en test: {best.name}", flush=True)
    proba = best.pipeline.predict_proba(split.X_test)[:, 1]
    test_value = value_processor.transform(split.X_test)[VALUE_COL].to_numpy()
    breakdown = cost_benefit_breakdown(split.y_test, proba, test_value, None, cfg)
    comparison = compare_against_baselines(split.y_test, proba, test_value, cfg)
    metrics = technical_metrics(split.y_test, proba)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib"]},
        "split_sizes": {"train": len(split.y_train), "validation": len(split.y_validation), "test": len(split.y_test)},
        "random_state": 42,
        "policy": "positive_expected_incremental_profit",
        "economic_config": asdict(cfg),
        "model": best.name,
        "validation_results": validation,
        "test_metrics": metrics,
        "test_breakdown": breakdown.as_dict(),
        "test_scenarios": comparison.scenarios.to_dict(orient="records"),
        "uplift_vs_call_everyone": comparison.uplift_vs_call_everyone,
        "uplift_vs_random": comparison.uplift_vs_random,
    }
    print("[4/4] Guardando modelo, reporte agregado y lista local de test...", flush=True)
    for path in [out_model, out_report, out_calls]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "artifact_version": ARTIFACT_VERSION, "model_name": best.name,
        "pipeline": best.pipeline, "value_processor": value_processor,
        "economic_config": asdict(cfg), "metadata": metadata,
    }, out_model)
    report = executive_report(breakdown, comparison, cfg)
    Path(out_report).write_text(report, encoding="utf-8")
    Path(out_report).with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
    priority_call_list(split.ids_test, proba, test_value, cfg).to_csv(out_calls, index=False)
    print(report)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/Churn_Modelling.csv")
    parser.add_argument("--out-model", default="models/best_model.pkl")
    parser.add_argument("--out-report", default="reports/executive_report.txt")
    parser.add_argument("--out-calls", default="reports/priority_calls.csv")
    parser.add_argument("--retention-cost", type=float, default=15.0)
    parser.add_argument("--retention-success", type=float, default=0.30)
    args = parser.parse_args()
    try:
        cfg = EconomicConfig(args.retention_cost, args.retention_success)
        run_training(args.data, args.out_model, args.out_report, args.out_calls, cfg)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
