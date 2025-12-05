import pandas as pd
import numpy as np
from utils import generate_dummy_load_profile, generate_dummy_generation_profile, calculate_cfe_score, simulate_battery_storage, recommend_portfolio

def test_generate_load():
    print("Testing generate_dummy_load_profile...")
    load = generate_dummy_load_profile(10000, 'Flat')
    assert len(load) == 8760
    assert abs(load.sum() - 10000) < 1.0
    print("Load profile test passed.")

def test_generate_gen():
    print("Testing generate_dummy_generation_profile...")
    gen_solar = generate_dummy_generation_profile(10, 'Solar')
    gen_geo = generate_dummy_generation_profile(10, 'Geothermal')
    gen_nuc = generate_dummy_generation_profile(10, 'Nuclear')
    
    assert len(gen_solar) == 8760
    assert gen_geo.min() > 0 # Geothermal should be baseload
    assert gen_nuc.min() == 10 # Nuclear should be flat
    print("Generation profiles test passed.")

def test_battery_simulation():
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

def test_recommend_portfolio_cfe_target():
    print("\nTesting Recommendation Logic for 95% CFE Target...")
    load = generate_dummy_load_profile(50000, 'Data Center')
    
    # Request recommendation
    rec = recommend_portfolio(load, target_cfe=0.95)
    print("Recommendation:", rec)
    
    # Verify CFE of recommendation
    solar = generate_dummy_generation_profile(rec['Solar'], 'Solar')
    wind = generate_dummy_generation_profile(rec['Wind'], 'Wind')
    geo = generate_dummy_generation_profile(rec['Geothermal'], 'Geothermal')
    nuc = generate_dummy_generation_profile(rec['Nuclear'], 'Nuclear')
    
    total_gen = solar + wind + geo + nuc
    surplus = (total_gen - load).clip(lower=0)
    deficit = (load - total_gen).clip(lower=0)
    
    batt_discharge, _ = simulate_battery_storage(surplus, deficit, rec['Battery_MW'], rec['Battery_Hours'])
    
    total_available = total_gen + batt_discharge
    cfe, _ = calculate_cfe_score(load, total_available)
    
    print(f"Achieved CFE: {cfe:.4f}")
    assert cfe >= 0.95, f"Recommendation failed to meet target CFE. Got {cfe:.4f}"
    print("Recommendation Logic Test Passed!")

if __name__ == "__main__":
    test_generate_load()
    test_generate_gen()
    test_battery_simulation()
    test_recommend_portfolio_cfe_target()
