import unittest

import numpy as np
import pandas as pd

from fixtures import customers
from src.data_processing import CustomerPreprocessor, NUMERIC_FEATURES, prepare_dataframe, split_data


class DataTests(unittest.TestCase):
    def test_repeated_preparation_and_batch_invariance(self):
        raw = customers()
        processor = CustomerPreprocessor().fit(raw)
        first = processor.transform(raw)
        pd.testing.assert_frame_equal(first, processor.transform(raw))
        pd.testing.assert_frame_equal(first.iloc[:1], processor.transform(raw.iloc[:1]))
        self.assertTrue(set(NUMERIC_FEATURES).issubset(first.columns))

    def test_held_out_extremes_do_not_change_fit(self):
        raw = customers()
        processor = CustomerPreprocessor().fit(raw)
        original_caps = processor.caps_.copy()
        held_out = raw.iloc[:2].copy()
        held_out.loc[held_out.index[0], "Balance"] = 1e12
        held_out.loc[held_out.index[1], "Balance"] = np.nan
        result = processor.transform(held_out)
        self.assertEqual(result.iloc[0].balance, original_caps['balance'])
        self.assertEqual(result.iloc[1].balance, processor.medians_['balance'])
        pd.testing.assert_series_equal(processor.caps_, original_caps)
        self.assertFalse(result.isna().any().any())

    def test_tied_values_and_unseen_category(self):
        raw = customers(12)
        raw['Balance'] = 0
        raw['NumOfProducts'] = 1
        raw['HasCrCard'] = 0
        processor = CustomerPreprocessor().fit(raw)
        raw.loc[0, 'Geography'] = 'Unknown'
        result = processor.transform(raw.iloc[:1])
        self.assertEqual(result.iloc[0].value_segment, 'Low')

    def test_input_validation(self):
        for column, value in [('Balance', -1), ('Age', np.inf), ('Exited', 3)]:
            with self.subTest(column=column):
                raw = customers()
                raw[column] = value
                with self.assertRaises(ValueError):
                    prepare_dataframe(raw)
        with self.assertRaises(ValueError):
            prepare_dataframe(customers().drop(columns=['Age']))
        with self.assertRaises(ValueError):
            prepare_dataframe(pd.concat([customers(), customers()]))
        with self.assertRaises(ValueError):
            prepare_dataframe(customers().iloc[:0])

    def test_split_sizes_and_no_overlap(self):
        raw = customers(300)
        split = split_data(raw)
        self.assertEqual([len(split.X_train), len(split.X_validation), len(split.X_test)], [180, 60, 60])
        # Los salarios son únicos en este fixture y permiten identificar cada fila.
        sets = [set(x.estimated_salary) for x in [split.X_train, split.X_validation, split.X_test]]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(len(set.union(*sets)), len(raw))
