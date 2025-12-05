import pandas as pd
import numpy as np
from utils import generate_dummy_load_profile, generate_dummy_generation_profile, calculate_cfe_score, simulate_battery_storage, recommend_portfolio

def test_utils():
    print("Testing generate_dummy_load_profile...")
    load = generate_dummy_load_profile(10000, 'Flat')
    assert len(load) == 8760
    assert abs(load.sum() - 10000) < 1.0
    print("Load profile test passed.")

    print("Testing generate_dummy_generation_profile...")
    gen_solar = generate_dummy_generation_profile(10, 'Solar')
    gen_geo = generate_dummy_generation_profile(10, 'Geothermal')
    gen_nuc = generate_dummy_generation_profile(10, 'Nuclear')
    
    assert len(gen_solar) == 8760
    assert gen_geo.min() > 0 # Geothermal should be baseload
    assert gen_nuc.min() == 10 # Nuclear should be flat
    print("Generation profiles test passed.")

    print("Testing simulate_battery_storage...")
    surplus = pd.Series(np.zeros(8760))
    surplus[0] = 10 # 10 MW surplus at hour 0
    deficit = pd.Series(np.zeros(8760))
    deficit[1] = 5 # 5 MW deficit at hour 1
    
    discharge, soc = simulate_battery_storage(surplus, deficit, 10, 4)
    
    # Check if battery charged (SoC increased)
    # Initial SoC is 50% of 40MWh = 20MWh
    # Added 10MW * sqrt(0.85) approx 9.2 MWh
    assert soc[0] > 20 
    
    # Check if battery discharged
    assert discharge[1] > 0
    assert soc[1] < soc[0]
    print("Battery simulation test passed.")
    
    print("Testing recommend_portfolio...")
    rec = recommend_portfolio(load)
    assert 'Solar' in rec
    assert 'Geothermal' in rec
    assert 'Battery_MW' in rec
    print("Recommendation test passed.")

if __name__ == "__main__":
    test_utils()
