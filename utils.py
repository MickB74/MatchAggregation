import pandas as pd
import numpy as np

def generate_dummy_load_profile(annual_consumption_mwh, profile_type='Flat'):
    """
    Generates a dummy hourly load profile for a year (8760 hours).
    
    Args:
        annual_consumption_mwh (float): Total annual consumption in MWh.
        profile_type (str): Type of profile ('Flat', 'Office', 'Data Center').
        
    Returns:
        pd.Series: Hourly load in MW.
    """
    hours = 8760
    
    if profile_type == 'Flat':
        # Constant load
        avg_load = annual_consumption_mwh / hours
        profile = np.full(hours, avg_load)
        
    elif profile_type == 'Data Center':
        # Flat with minor noise
        avg_load = annual_consumption_mwh / hours
        noise = np.random.normal(0, 0.05 * avg_load, hours)
        profile = np.full(hours, avg_load) + noise
        
    elif profile_type == 'Office':
        # Higher during day (8am-6pm), lower at night
        # Simple simulation
        profile = np.zeros(hours)
        for h in range(hours):
            hour_of_day = h % 24
            if 8 <= hour_of_day <= 18:
                profile[h] = 1.5
            else:
                profile[h] = 0.5
        
        # Normalize to match annual consumption
        total_unscaled = np.sum(profile)
        scaling_factor = annual_consumption_mwh / total_unscaled
        profile = profile * scaling_factor
        
    else:
        # Default to flat
        avg_load = annual_consumption_mwh / hours
        profile = np.full(hours, avg_load)
        
    return pd.Series(profile, name='Load (MW)')

def generate_dummy_generation_profile(capacity_mw, resource_type='Solar'):
    """
    Generates a dummy hourly generation profile for a year.
    
    Args:
        capacity_mw (float): Installed capacity in MW.
        resource_type (str): 'Solar' or 'Wind'.
        
    Returns:
        pd.Series: Hourly generation in MW.
    """
    hours = 8760
    
    if resource_type == 'Solar':
        # Simple solar curve: peak at noon, zero at night
        profile = np.zeros(hours)
        for h in range(hours):
            hour_of_day = h % 24
            if 6 <= hour_of_day <= 18:
                # Sine wave approximation for day
                profile[h] = np.sin(np.pi * (hour_of_day - 6) / 12)
            else:
                profile[h] = 0
        
        # Add some random cloud cover noise
        noise = np.random.uniform(0.8, 1.0, hours)
        profile = profile * noise * capacity_mw
        
    elif resource_type == 'Wind':
        # Random variation with some diurnal pattern (often higher at night in ERCOT)
        # Using a sum of sines for variety + noise
        t = np.linspace(0, 8760, hours)
        base_wind = 0.4 * np.sin(t / 24 * 2 * np.pi) + 0.5 # Daily cycle
        seasonal = 0.2 * np.sin(t / 8760 * 2 * np.pi) # Seasonal
        noise = np.random.normal(0, 0.2, hours)
        
        profile = (base_wind + seasonal + noise)
        profile = np.clip(profile, 0, 1) * capacity_mw
        
    elif resource_type == 'Geothermal':
        # Baseload: Flat with very high availability (e.g. 95% capacity factor)
        # Small random fluctuations
        profile = np.full(hours, capacity_mw * 0.95)
        noise = np.random.normal(0, 0.01 * capacity_mw, hours)
        profile = profile + noise
        profile = np.clip(profile, 0, capacity_mw)

    elif resource_type == 'Nuclear':
        # Baseload: Extremely flat, near 100% (refueling outages ignored for simple demo)
        profile = np.full(hours, capacity_mw)
        
    else:
        profile = np.zeros(hours)
        
    return pd.Series(profile, name=f'{resource_type} Generation (MW)')

def calculate_cfe_score(load_profile, generation_profile):
    """
    Calculates the Carbon Free Energy (CFE) score.
    
    CFE Score = (Total Matched Energy) / (Total Load)
    Matched Energy at hour h = min(Load[h], Generation[h])
    
    Args:
        load_profile (pd.Series): Hourly load.
        generation_profile (pd.Series): Hourly renewable generation.
        
    Returns:
        float: CFE Score (0.0 to 1.0).
        pd.Series: Matched energy profile.
    """
    matched_energy = np.minimum(load_profile, generation_profile)
    total_load = load_profile.sum()
    
    if total_load == 0:
        return 0.0, matched_energy
        
    cfe_score = matched_energy.sum() / total_load
    return cfe_score, matched_energy

def simulate_battery_storage(surplus_profile, deficit_profile, capacity_mw, duration_hours):
    """
    Simulates a simple battery storage system.
    Charges from surplus, discharges to cover deficit.
    
    Args:
        surplus_profile (pd.Series): Hourly surplus energy (MW).
        deficit_profile (pd.Series): Hourly energy deficit (MW).
        capacity_mw (float): Battery power capacity (MW).
        duration_hours (float): Battery duration (hours).
        
    Returns:
        pd.Series: Battery discharge profile (MW).
        pd.Series: Battery state of charge (MWh).
    """
    hours = len(surplus_profile)
    max_energy_mwh = capacity_mw * duration_hours
    current_energy_mwh = max_energy_mwh * 0.5 # Start at 50%
    
    discharge_profile = np.zeros(hours)
    soc_profile = np.zeros(hours)
    
    # Efficiency
    round_trip_efficiency = 0.85
    charge_eff = np.sqrt(round_trip_efficiency)
    discharge_eff = np.sqrt(round_trip_efficiency)
    
    for h in range(hours):
        surplus = surplus_profile[h]
        deficit = deficit_profile[h]
        
        # Charge
        if surplus > 0 and current_energy_mwh < max_energy_mwh:
            charge_power = min(surplus, capacity_mw)
            energy_to_add = charge_power * charge_eff
            space_available = max_energy_mwh - current_energy_mwh
            
            real_energy_added = min(energy_to_add, space_available)
            current_energy_mwh += real_energy_added
            
        # Discharge
        elif deficit > 0 and current_energy_mwh > 0:
            discharge_power_needed = deficit
            max_discharge_possible = min(capacity_mw, current_energy_mwh / discharge_eff)
            
            real_discharge = min(discharge_power_needed, max_discharge_possible)
            discharge_profile[h] = real_discharge
            
            energy_removed = real_discharge / discharge_eff
            current_energy_mwh -= energy_removed
            
        soc_profile[h] = current_energy_mwh
        
    return pd.Series(discharge_profile, name='Battery Discharge (MW)'), pd.Series(soc_profile, name='Battery SoC (MWh)')

def recommend_portfolio(load_profile, target_cfe=0.95):
    """
    Simple heuristic to recommend a technology mix.
    
    Strategy:
    1. Calculate baseload (min load). Cover X% of this with Geothermal/Nuclear.
    2. Calculate remaining energy needs. Cover with Solar/Wind mix.
    3. Add Battery for smoothing.
    """
    avg_load = load_profile.mean()
    min_load = load_profile.min()
    peak_load = load_profile.max()
    total_load = load_profile.sum()
    
    recommendation = {
        'Solar': 0,
        'Wind': 0,
        'Geothermal': 0,
        'Nuclear': 0,
        'Battery_MW': 0,
        'Battery_Hours': 4
    }
    
    # 1. Baseload Coverage
    # Suggest covering 80% of min load with firm clean energy
    firm_target = min_load * 0.8
    # Split firm between Geothermal and Nuclear (e.g., 50/50 preference or just one)
    recommendation['Geothermal'] = firm_target * 0.5
    recommendation['Nuclear'] = firm_target * 0.5
    
    # 2. Variable Renewable Coverage
    # Remaining energy to cover
    firm_gen_annual = (recommendation['Geothermal'] + recommendation['Nuclear']) * 8760
    remaining_load = total_load - firm_gen_annual
    
    if remaining_load > 0:
        # Oversize variable renewables slightly to account for intermittency
        # Target 1.5x the remaining load energy
        target_variable_gen = remaining_load * 1.5
        
        # Assume 25% capacity factor for Solar, 40% for Wind
        # Split target gen 50/50 between Solar and Wind energy
        target_solar_gen = target_variable_gen * 0.5
        target_wind_gen = target_variable_gen * 0.5
        
        recommendation['Solar'] = target_solar_gen / (8760 * 0.25)
        recommendation['Wind'] = target_wind_gen / (8760 * 0.40)
        
    # 3. Battery Storage
    # Suggest battery power equal to 20% of peak load
    recommendation['Battery_MW'] = peak_load * 0.2
    
    # Round values
    for k, v in recommendation.items():
        recommendation[k] = round(v, 1)
        
    return recommendation
