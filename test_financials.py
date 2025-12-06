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
        grid_price = 60.0
        
        metrics = calculate_financials(matched_profile, deficit_profile, strike_price, market_price, grid_price)
        
        # Expected Settlement: (50 - 30) * 100 = $2000
        expected_settlement = 2000.0
        self.assertAlmostEqual(metrics['settlement_value'], expected_settlement)
        
        # Expected Grid Cost: 10 * 60 = $600
        expected_grid_cost = 600.0
        self.assertAlmostEqual(metrics['grid_cost'], expected_grid_cost)
        
        # Expected Total Cost: 600 - 2000 = -1400 (Net Benefit)
        self.assertAlmostEqual(metrics['total_net_cost'], expected_grid_cost - expected_settlement)

if __name__ == '__main__':
    unittest.main()
