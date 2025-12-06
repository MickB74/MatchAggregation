import unittest
import pandas as pd
import numpy as np
from utils import calculate_financials

class TestFinancials(unittest.TestCase):
    def test_calculate_financials_basic(self):
        # Setup dummy data (10 hours)
        hours = 10
        # Scenario: Guaranteed profit
        # Strike: $30, Market: $50 -> Profit $20/MWh
        # Matched: 100 MWh
        # Deficit: 10 MWh, Grid Cost: $60/MWh
        
        matched_profile = pd.Series(np.full(hours, 10.0)) # Total 100 MWh
        deficit_profile = pd.Series(np.full(hours, 1.0))  # Total 10 MWh
        
        strike_price = 30.0
        market_price = 50.0 # Flat
        rec_price = 8.0
        
        metrics = calculate_financials(matched_profile, deficit_profile, strike_price, market_price, rec_price)
        
        # Expected Settlement: (50 - 30) * 100 = $2000
        expected_settlement = 2000.0
        self.assertAlmostEqual(metrics['settlement_value'], expected_settlement)
        
        # Expected REC Cost: 100 * 8 = $800
        expected_rec_cost = 800.0
        self.assertAlmostEqual(metrics['rec_cost'], expected_rec_cost)
        
        # Expected Net Cost:
        # Deficit Cost (at Market): 10 * 50 = 500
        # Matched Cost (at Strike): 100 * 30 = 3000
        # REC Cost: 800
        # Total: 4300
        expected_net_cost = 4300.0
        self.assertAlmostEqual(metrics['net_cost'], expected_net_cost)
        
        # Avg Cost: 4300 / 110
        self.assertAlmostEqual(metrics['avg_cost_per_mwh'], 4300.0 / 110.0)

if __name__ == '__main__':
    unittest.main()
