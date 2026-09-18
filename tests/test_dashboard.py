"""Regresiones de Streamlit con archivos sintéticos temporales."""
from dataclasses import asdict
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
from sklearn.linear_model import LogisticRegression
from streamlit.testing.v1 import AppTest

from fixtures import customers
from src import ARTIFACT_VERSION
from src.data_processing import CustomerPreprocessor, prepare_dataframe, RAW_FEATURES
from src.financial_evaluator import EconomicConfig
from src.modeling import build_model

ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.data_path = Path(cls.temp.name) / 'customers.csv'
        cls.model_path = Path(cls.temp.name) / 'model.pkl'
        cls.raw = customers()
        cls.raw.to_csv(cls.data_path, index=False)
        normalized = prepare_dataframe(cls.raw)
        x, y = normalized[RAW_FEATURES], normalized.churned
        model = build_model(LogisticRegression(max_iter=1000)).fit(x, y)
        joblib.dump({
            'artifact_version': ARTIFACT_VERSION, 'model_name': 'synthetic_test_model',
            'pipeline': model, 'value_processor': CustomerPreprocessor().fit(x),
            'economic_config': asdict(EconomicConfig()),
        }, cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.raw.to_csv(self.data_path, index=False)
        self.env = patch.dict(os.environ, {'CHURN_DATA_PATH': str(self.data_path),
                                          'CHURN_MODEL_PATH': str(self.model_path)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def app(self):
        return AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()

    def assert_healthy(self, app):
        self.assertFalse(app.exception)
        self.assertFalse(app.error)

    def test_local_data_and_parameter_reruns(self):
        app = self.app()
        self.assert_healthy(app)
        self.assertEqual(len(app.get('file_uploader')), 0)
        self.assertIn('180', app.success[0].value)
        saved = app.metric[1].value
        app.sidebar.slider[0].set_value(.4).run()
        self.assert_healthy(app)
        self.assertNotEqual(saved, app.metric[1].value)
        app.sidebar.number_input[0].set_value(25.).run()
        self.assert_healthy(app)
        app.sidebar.checkbox[0].check().run()
        app.sidebar.slider[1].set_value(.95).run()
        self.assert_healthy(app)
        app.sidebar.slider[0].set_value(0.).run()
        self.assert_healthy(app)
        self.assertEqual(app.metric[0].value, '0')

    def test_single_customer_without_target(self):
        self.raw.iloc[:1].drop(columns='Exited').to_csv(self.data_path, index=False)
        self.assert_healthy(self.app())

    def test_missing_local_data(self):
        with patch.dict(os.environ, {'CHURN_DATA_PATH': str(self.data_path.with_name('missing.csv'))}):
            app = self.app()
        self.assertFalse(app.exception)
        self.assertIn('No se encontró la base', app.error[0].value)

    def test_invalid_local_data(self):
        self.raw.drop(columns='Age').to_csv(self.data_path, index=False)
        app = self.app()
        self.assertFalse(app.exception)
        self.assertIn('Faltan columnas', app.error[0].value)

    def test_missing_model(self):
        with patch.dict(os.environ, {'CHURN_MODEL_PATH': str(self.model_path.with_name('missing.pkl'))}):
            app = self.app()
        self.assertFalse(app.exception)
        self.assertIn('Falta el modelo', app.error[0].value)
