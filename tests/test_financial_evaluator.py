import unittest

import numpy as np
import pandas as pd

from src.financial_evaluator import (
    EconomicConfig, compare_against_baselines, contact_mask, cost_benefit_breakdown,
    optimize_threshold, per_customer_profit, priority_call_list,
)


class FinancialTests(unittest.TestCase):
    def test_profit_definitions_agree(self):
        y, p, v = np.array([1, 1, 0, 0]), np.array([.9, .1, .8, .1]), np.full(4, 100.)
        cfg = EconomicConfig()
        b = cost_benefit_breakdown(y, p, v, None, cfg)
        self.assertAlmostEqual(per_customer_profit(y, p, v, None, cfg).sum(), b.net_profit)
        self.assertEqual((b.tp, b.fp, b.fn, b.tn), (1, 1, 1, 1))
        self.assertEqual(b.net_profit, 0)
        self.assertEqual(b.revenue_lost_fn, 100)

    def test_policy_reacts_to_economics_and_customer_value(self):
        p, v = np.array([.1, .9]), np.array([1000., 20.])
        np.testing.assert_array_equal(contact_mask(p, v, EconomicConfig()), [True, False])
        self.assertFalse(contact_mask(p, v, EconomicConfig(40)).any())
        self.assertFalse(contact_mask(p, v, EconomicConfig(15, 0)).any())

    def test_threshold_uses_full_precision_and_list_matches_policy(self):
        p, v = np.array([.49999, .50001, .8]), np.full(3, 1000.)
        cfg = EconomicConfig()
        calls = priority_call_list(pd.Series(['a', 'b', 'c']), p, v, cfg, threshold=.5)
        self.assertEqual(calls.customer_id.tolist(), ['c', 'b'])
        self.assertEqual(len(calls), contact_mask(p, v, cfg, .5).sum())
        self.assertTrue((calls.expected_profit_if_called_usd > 0).all())

    def test_optimizer_can_choose_no_campaign(self):
        sweep = optimize_threshold(np.array([0]), np.array([1.]), np.array([100.]), EconomicConfig())
        self.assertEqual(sweep.best_breakdown.contacted, 0)
        self.assertEqual(sweep.best_breakdown.net_profit, 0)

    def test_random_baseline_has_same_budget_and_expected_value(self):
        comparison = compare_against_baselines(
            np.array([1, 0]), np.array([.9, .01]), np.array([100., 100.]), EconomicConfig(),
        )
        rows = comparison.scenarios
        self.assertEqual(rows.iloc[2].contacted, rows.iloc[3].contacted)
        self.assertEqual(rows.iloc[2].campaign_cost_usd, rows.iloc[3].campaign_cost_usd)
        self.assertEqual(rows.iloc[2].net_profit_vs_status_quo_usd, 0)
        self.assertEqual(comparison.uplift_vs_random, 15)
        self.assertEqual(rows.iloc[1].remaining_revenue_at_risk_usd, 70)

    def test_invalid_economic_inputs(self):
        for cost, success in [(0, .3), (-1, .3), (15, 1.1), (15, np.nan)]:
            with self.assertRaises(ValueError):
                EconomicConfig(cost, success)
        with self.assertRaises(ValueError):
            contact_mask([np.nan], [100], EconomicConfig())
        with self.assertRaises(ValueError):
            contact_mask([.2], [100, 200], EconomicConfig())
