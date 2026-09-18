"""Verifica exportación y recarga del artefacto sin depender del CSV real."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fixtures import customers
from src import ARTIFACT_VERSION
from src.financial_evaluator import EconomicConfig
from src.modeling import TrainedModel, build_model, technical_metrics
from src.train_pipeline import run_training


def train_small_model(split):
    pipeline = build_model(LogisticRegression(max_iter=1000)).fit(split.X_train, split.y_train)
    p = pipeline.predict_proba(split.X_validation)[:, 1]
    return {'logistic_regression': TrainedModel(
        'logistic_regression', pipeline, p, technical_metrics(split.y_validation, p),
    )}


class TrainingTests(unittest.TestCase):
    def test_export_reload_and_reserved_report(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            raw = customers(300)
            raw.to_csv(root / 'data.csv', index=False)
            with patch('src.train_pipeline.train_all', side_effect=train_small_model), redirect_stdout(StringIO()):
                report = run_training(root / 'data.csv', root / 'model.pkl', root / 'report.txt',
                                      root / 'calls.csv', EconomicConfig())
            artifact = joblib.load(root / 'model.pkl')
            self.assertEqual(artifact['artifact_version'], ARTIFACT_VERSION)
            self.assertEqual(artifact['pipeline'].predict_proba(raw.iloc[:1]).shape, (1, 2))
            self.assertEqual(report['split_sizes'], {'train': 180, 'validation': 60, 'test': 60})
            b = report['test_breakdown']
            self.assertEqual(sum(b[key] for key in ['tp', 'fp', 'fn', 'tn']), 60)
            self.assertEqual(len(pd.read_csv(root / 'calls.csv')), b['contacted'])
            saved = json.loads((root / 'report.json').read_text())
            self.assertEqual(saved, report)
            self.assertNotIn('customer_id', (root / 'report.json').read_text())
            self.assertEqual(len(saved['dataset_sha256']), 64)
