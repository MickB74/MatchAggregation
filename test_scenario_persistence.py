import unittest
import pandas as pd
import numpy as np
import json
from unittest.mock import MagicMock

class TestScenarioPersistence(unittest.TestCase):
    def test_profile_serialization(self):
        # 1. Create Mock Session State
        session_state = {}
        
        # 2. Populate with "Custom" Profiles (as they would appear in app)
        # Solar: pd.Series
        solar_profile = pd.Series(np.random.rand(8760), name='Solar (MW)')
        session_state['custom_solar_profile'] = solar_profile
        
        # Wind: pd.Series
        wind_profile = pd.Series(np.random.rand(8760), name='Wind (MW)')
        session_state['custom_wind_profile'] = wind_profile
        
        # Prices: pd.DataFrame with Index
        dates = pd.date_range(start='2024-01-01', periods=8760, freq='h')
        prices_df = pd.DataFrame({'Price': np.random.rand(8760) * 50}, index=dates)
        session_state['shared_market_prices'] = prices_df
        
        # 3. Simulate "Save" (Dict creation)
        export_config = {}
        
        if 'custom_solar_profile' in session_state:
             export_config['custom_solar_profile'] = session_state['custom_solar_profile'].tolist()
        
        if 'custom_wind_profile' in session_state:
             export_config['custom_wind_profile'] = session_state['custom_wind_profile'].tolist()
             
        if 'shared_market_prices' in session_state:
             df_prices = session_state['shared_market_prices']
             if 'Price' in df_prices.columns:
                 export_config['custom_battery_prices'] = df_prices['Price'].tolist()
                 
        # 4. Verify JSON Serializable
        json_str = json.dumps(export_config)
        loaded_config = json.loads(json_str)
        
        # 5. Simulate "Load" (Restore to Session State)
        new_session_state = {}
        
        if 'custom_solar_profile' in loaded_config:
            new_session_state['custom_solar_profile'] = pd.Series(loaded_config['custom_solar_profile'])
            
        if 'custom_wind_profile' in loaded_config:
            new_session_state['custom_wind_profile'] = pd.Series(loaded_config['custom_wind_profile'])
            
        if 'custom_battery_prices' in loaded_config:
            price_data = loaded_config['custom_battery_prices']
            # Reconstruct index (Simulation)
            year = 2024
            dates = pd.date_range(start=f'{year}-01-01', periods=len(price_data), freq='h')
            new_session_state['shared_market_prices'] = pd.DataFrame({'Price': price_data}, index=dates)

        # 6. Assert Equality
        # Solar
        pd.testing.assert_series_equal(
            session_state['custom_solar_profile'].reset_index(drop=True), 
            new_session_state['custom_solar_profile'].reset_index(drop=True),
            check_names=False
        )
        # Wind
        pd.testing.assert_series_equal(
             session_state['custom_wind_profile'].reset_index(drop=True), 
             new_session_state['custom_wind_profile'].reset_index(drop=True),
             check_names=False
        )
        # Prices
        pd.testing.assert_frame_equal(
            session_state['shared_market_prices'].reset_index(drop=True),
            new_session_state['shared_market_prices'].reset_index(drop=True)
        )
        
        print("Serialization Test Passed!")

if __name__ == '__main__':
    unittest.main()
