import pandas as pd
import numpy as np
from utils import generate_dummy_load_profile, generate_dummy_generation_profile, calculate_cfe_score, simulate_battery_storage, recommend_portfolio, calculate_portfolio_metrics

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

def test_portfolio_metrics():
    print("\nTesting Portfolio Metrics...")
    # Flat load of 10 for 8760 hours = 87600 MWh
    load = pd.Series(np.full(8760, 10.0))
    
    # Matching generation (e.g. perfect nuclear)
    # 10 MW capacity matching 10 MW load perfectly
    gen = pd.Series(np.full(8760, 10.0))
    
    metrics = calculate_portfolio_metrics(load, gen, 10.0)
    
    # Productivity: 87600 MWh / 10 MW = 8760 MWh/MW
    assert abs(metrics['productivity'] - 8760) < 1.0
    
    # LoGH: Should be 0%
    assert metrics['logh'] == 0.0
    
    # Grid Consumption: 0
    assert metrics['grid_consumption'] == 0.0
    
    # Case 2: 50% matching
    gen_half = pd.Series(np.full(8760, 5.0))
    metrics_half = calculate_portfolio_metrics(load, gen_half, 10.0) # Still 10MW capacity installed but underperforming/curtailed or just 5MW useful
    
    # Productivity: 43800 MWh / 10 MW = 4380
    assert abs(metrics_half['productivity'] - 4380) < 1.0
    
    # LoGH: Should be 100% (every hour is short)
    assert metrics_half['logh'] == 1.0
    
    print("Portfolio metrics test passed.")

if __name__ == "__main__":
    test_generate_load()
    test_generate_gen()
    test_battery_simulation()
    test_recommend_portfolio_cfe_target()
    test_portfolio_metrics()
