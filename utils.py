# Utility functions for MatchAggregation App
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
    
    # Create a deterministic random generator based on profile type
    # Use a simple hash mapping to ensure stability across Python sessions
    seed_map = {'Flat': 42, 'Data Center': 101, 'Office': 202}
    seed = seed_map.get(profile_type, 999)
    rng = np.random.default_rng(seed)
    
    if profile_type == 'Flat':
        # Constant load
        avg_load = annual_consumption_mwh / hours
        profile = np.full(hours, avg_load)
        
    elif profile_type == 'Data Center':
        # Flat with minor noise
        avg_load = annual_consumption_mwh / hours
        noise = rng.normal(0, 0.05 * avg_load, hours)
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

def generate_dummy_generation_profile(capacity_mw, resource_type='Solar', use_synthetic=False):
    """
    Generates a dummy hourly generation profile for a year.
    Refined for ERCOT North characteristics (approximate).
    
    Args:
        capacity_mw (float): Installed capacity in MW.
        resource_type (str): 'Solar' or 'Wind'.
        use_synthetic (bool): If True, forces synthetic model even if real data file exists.
        
    Returns:
        pd.Series: Hourly generation in MW.
    """
    hours = 8760
    t = np.arange(hours)
    
    # Seasonality helper (0 to 1 scaling, peak in Summer for Solar, Spring/Fall for Wind)
    day_of_year = (t // 24)
    
    # Create a deterministic random generator based on resource type
    seed_map = {'Solar': 500, 'Wind': 600, 'Geothermal': 700, 'Nuclear': 800}
    seed = seed_map.get(resource_type, 999)
    rng = np.random.default_rng(seed)
    
    if resource_type == 'Solar':
        # 1. Try Real Data (PVWatts CSV)
        if not use_synthetic:
            import os
            pvwatts_file = 'pvwatts_hourly_Denton.csv'
            
            if os.path.exists(pvwatts_file):
                try:
                    # Robustly find the header line to handle blank lines/metadata variability
                    header_index = 0
                    with open(pvwatts_file, 'r') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines[:50]): # Check first 50 lines
                            if 'AC System Output (W)' in line:
                                header_index = i
                                break
                                
                    # Read with skiprows (skips first N rows, row N becomes header)
                    df_solar = pd.read_csv(pvwatts_file, skiprows=header_index)
                    
                    if 'AC System Output (W)' in df_solar.columns:
                        # Extract raw watts
                        raw_watts = df_solar['AC System Output (W)'].values
                        
                        # Pad/Truncate to 8760
                        if len(raw_watts) > hours:
                            raw_watts = raw_watts[:hours]
                        elif len(raw_watts) < hours:
                            raw_watts = np.pad(raw_watts, (0, hours - len(raw_watts)), 'constant')
                            
                        # Normalize (100 kW system)
                        system_size_watts = 100_000.0 
                        unit_profile = raw_watts / system_size_watts
                        
                        return pd.Series(unit_profile * capacity_mw, name='Solar Generation (MW)')
                except Exception as e:
                    print(f"Failed to load PVWatts file: {e}")
        
        # 2. Synthetic Fallback (runs if use_synthetic=True OR file missing/failed)
        profile = np.zeros(hours)
        
        for h in range(hours):
            hour_of_day = h % 24
            
            # Simple day/night check (broadened slightly for summer)
            is_daytime = 6 <= hour_of_day <= 19
            if is_daytime:
                # Center peak at 13 (1 PM)
                # (hour - 6) / 14 * pi -> maps 6 to 0, 13 to pi/2, 20 to pi
                # normalized sine wave
                day_shape = np.sin(np.pi * (hour_of_day - 6) / 13)
                if day_shape < 0: day_shape = 0
                
                # Seasonal Factor: Bassel on cosine of day of year
                # Peak at day 172 (June 21), Min at day 355/0
                current_day = day_of_year[h]
                seasonal_factor = 0.7 + 0.3 * np.cos(2 * np.pi * (current_day - 172) / 365)
                # Range: 0.4 to 1.0 multiplier? Actually solar variance is intensity + day length.
                # Scaler: 0.7 (winter) to 1.1 (summer peak intensity)
                
                # Cloud Noise
                noise = rng.uniform(0.7, 1.0)
                
                profile[h] = day_shape * seasonal_factor * noise * capacity_mw
            else:
                profile[h] = 0.0
        
        return pd.Series(profile, name='Solar Generation (MW)')
        
        # Fallback to Dummy Solar Profile
        # 1. Diurnal: Peak around 1 PM (hour 13). Sinusoidal.
        # 2. Seasonal: Peak in Summer (approx day 172). Lowest in Winter.
        
        profile = np.zeros(hours)
        
        for h in range(hours):
            hour_of_day = h % 24
            
            # Simple day/night check (broadened slightly for summer)
            is_daytime = 6 <= hour_of_day <= 19
            if is_daytime:
                # Center peak at 13 (1 PM)
                # (hour - 6) / 14 * pi -> maps 6 to 0, 13 to pi/2, 20 to pi
                # normalized sine wave
                day_shape = np.sin(np.pi * (hour_of_day - 6) / 13)
                if day_shape < 0: day_shape = 0
                
                # Seasonal Factor: Bassel on cosine of day of year
                # Peak at day 172 (June 21), Min at day 355/0
                current_day = day_of_year[h]
                seasonal_factor = 0.7 + 0.3 * np.cos(2 * np.pi * (current_day - 172) / 365)
                # Range: 0.4 to 1.0 multiplier? Actually solar variance is intensity + day length.
                # Scaler: 0.7 (winter) to 1.1 (summer peak intensity)
                
                # Cloud Noise
                noise = rng.uniform(0.7, 1.0)
                
                profile[h] = day_shape * seasonal_factor * noise * capacity_mw
            else:
                profile[h] = 0.0
        
    elif resource_type == 'Wind':
        # Wind Profile (ERCOT North)
        # 1. Diurnal: "Inverse Solar" - higher at night/evening, dip midday.
        # 2. Seasonal: High in Spring (March-May) and Fall (Oct). Lower in Summer (midday).
        # 3. Volatility: High.
        
        # Diurnal Component (Peak ~2 AM, Trough ~2 PM)
        # cos curve shifted
        hour_arg = 2 * np.pi * (t % 24 - 2) / 24 
        diurnal = 0.6 + 0.25 * np.cos(hour_arg) # Oscillates 0.35 to 0.85
        
        # Seasonal Component
        # Peak Spring (Day 100) and Fall (Day 300). Dip Summer (Day 200) and Winter (Day 0)
        # Superposition of waves
        seasonal = 0.7 + 0.3 * np.sin(2 * np.pi * (day_of_year - 50) / 365 * 2) 
        # Peaks roughly twice a year
        
        # Stochastic / Volatility
        # Weibull distribution-ish or just red noise
        # Let's use random noise correlated with time
        noise = rng.normal(0, 0.15, hours)
        
        # Combine
        raw_profile = diurnal * seasonal + noise
        
        # Clip and Scale
        # Max CF typically ~50-60% for onshore
        profile = np.clip(raw_profile, 0, 1.0) * capacity_mw
        
    elif resource_type == 'Geothermal':
        # Baseload: Flat with very high availability (e.g. 95% capacity factor)
        # Small random fluctuations
        profile = np.full(hours, capacity_mw * 0.95)
        noise = rng.normal(0, 0.01 * capacity_mw, hours)
        profile = profile + noise
        profile = np.clip(profile, 0, capacity_mw)

    elif resource_type == 'CCS Gas':
        # Baseload/Firm: Extremely flat, dispatchable, near 100%
        profile = np.full(hours, capacity_mw)
        # Verify extremely small variance
        noise = rng.normal(0, 0.005 * capacity_mw, hours)
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

def calculate_portfolio_metrics(load_profile, matched_profile, total_gen_capacity):
    """
    Calculates detailed portfolio performance metrics.
    
    Args:
        load_profile (pd.Series): Hourly load.
        matched_profile (pd.Series): Hourly matched energy.
        total_gen_capacity (float): Total installed generation capacity (MW).
        
    Returns:
        dict: Dictionary of metrics.
    """
    total_load = load_profile.sum()
    total_matched = matched_profile.sum()
    
    # CFE Score
    cfe_score = total_matched / total_load if total_load > 0 else 0.0
    
    # MW Match Productivity (MWh matched per MW capacity)
    productivity = total_matched / total_gen_capacity if total_gen_capacity > 0 else 0.0
    
    # Loss of Green Hours (LoGH)
    # Count hours where we didn't meet full load (with small tolerance for float errors)
    unmatched_mask = load_profile > (matched_profile + 0.001)
    green_hours_lost = unmatched_mask.sum()
    logh = green_hours_lost / 8760
    
    # Grid Consumption (Energy Deficit)
    grid_consumption = total_load - total_matched
    
    return {
        'cfe_score': cfe_score,
        'productivity': productivity,
        'logh': logh,
        'grid_consumption': grid_consumption
    }

def simulate_battery_storage(surplus_profile, deficit_profile, capacity_mw, duration_hours, availability_profile=None):
    """
    Simulates a simple battery storage system.
    Charges from surplus, discharges to cover deficit.
    
    Args:
        surplus_profile (pd.Series): Hourly surplus energy (MW).
        deficit_profile (pd.Series): Hourly energy deficit (MW).
        capacity_mw (float): Battery power capacity (MW).
        duration_hours (float): Battery duration (hours).
        availability_profile (pd.Series, optional): Max available capacity per hour (MW). Used for outages.
        
    Returns:
        pd.Series: Battery discharge profile (MW).
        pd.Series: Battery state of charge (MWh).
        pd.Series: Battery charge profile (MW).
    """
    hours = len(surplus_profile)
    max_energy_mwh = capacity_mw * duration_hours
    current_energy_mwh = max_energy_mwh * 0.5 # Start at 50%
    
    discharge_profile = np.zeros(hours)
    charge_profile = np.zeros(hours)
    soc_profile = np.zeros(hours)
    
    # Efficiency
    round_trip_efficiency = 0.85
    charge_eff = np.sqrt(round_trip_efficiency)
    discharge_eff = np.sqrt(round_trip_efficiency)
    
    for h in range(hours):
        surplus = surplus_profile[h]
        deficit = deficit_profile[h]
        
        # Determine available capacity for this hour
        available_cap = capacity_mw
        if availability_profile is not None:
            available_cap = availability_profile.iloc[h] if hasattr(availability_profile, 'iloc') else availability_profile[h]
        
        # Charge
        if surplus > 0 and current_energy_mwh < max_energy_mwh:
            # Can only charge up to available capacity
            charge_power = min(surplus, available_cap)
            energy_to_add = charge_power * charge_eff
            space_available = max_energy_mwh - current_energy_mwh
            
            real_energy_added = min(energy_to_add, space_available)
            real_charge_power = real_energy_added / charge_eff
            
            charge_profile[h] = real_charge_power
            current_energy_mwh += real_energy_added
            
        # Discharge
        elif deficit > 0 and current_energy_mwh > 0:
            discharge_power_needed = deficit
            # Can only discharge up to available capacity
            max_discharge_power = min(available_cap, current_energy_mwh / discharge_eff)
            
            real_discharge = min(discharge_power_needed, max_discharge_power)
            discharge_profile[h] = real_discharge
            
            energy_removed = real_discharge / discharge_eff
            current_energy_mwh -= energy_removed
            
        # Clamp to 0 to avoid floating point errors showing negative zero
        if current_energy_mwh < 0:
            current_energy_mwh = 0.0
        
        soc_profile[h] = current_energy_mwh
        
    return (pd.Series(discharge_profile, name='Battery Discharge (MW)'), 
            pd.Series(soc_profile, name='Battery SoC (MWh)'),
            pd.Series(charge_profile, name='Battery Charge (MW)'))

def calculate_battery_financials(contract_params, ops_data):
    """
    Calculates detailed battery settlement based on contract terms.
    
    Args:
        contract_params (dict):
            - capacity_mw
            - base_rate_monthly ($/MW-month)
            - guaranteed_availability (%)
            - guaranteed_rte (%)
            - vom_rate ($/MWh)
            
        ops_data (dict):
            - available_mw_profile (pd.Series)
            - discharge_mwh_profile (pd.Series)
            - charge_mwh_profile (pd.Series)
            - market_price_profile (pd.Series)
            
    Returns:
        dict: Financial results
    """
    # Unpack Inputs
    cap_mw = contract_params['capacity_mw']
    rate_mo = contract_params['base_rate_monthly']
    guar_avail = contract_params['guaranteed_availability']
    guar_rte = contract_params['guaranteed_rte']
    vom_rate = contract_params['vom_rate']
    
    avail_mw = ops_data['available_mw_profile']
    discharged = ops_data['discharge_mwh_profile']
    charged = ops_data['charge_mwh_profile']
    prices = ops_data['market_price_profile']
    
    hours_in_year = 8760 # Approximation
    intervals_per_month = 730 # Approx hours/month
    
    # --- Step 1: Capacity Payment ---
    # Actual logic usually aggregates monthly, but we'll do an annual aggregate for this demo
    # to be compatible with the single-year simulation.
    
    # Calculate Average Annual Availability
    total_avail_mw_hours = avail_mw.sum()
    max_possible_mw_hours = cap_mw * len(avail_mw)
    
    actual_availability = total_avail_mw_hours / max_possible_mw_hours if max_possible_mw_hours > 0 else 0.0
    
    # Performance Factor
    # If Actual >= Guaranteed -> 1.0
    # If Actual < Guaranteed -> Actual / Guaranteed
    if actual_availability >= guar_avail:
        perf_factor = 1.0
    else:
        perf_factor = actual_availability / guar_avail if guar_avail > 0 else 0.0
        
    # Annualize Capacity Payment
    annual_base_payment = (cap_mw * rate_mo) * 12
    final_capacity_payment = annual_base_payment * perf_factor
    
    
    # --- Step 2: RTE Adjustment ---
    total_discharged = discharged.sum()
    total_charged = charged.sum()
    
    actual_rte = total_discharged / total_charged if total_charged > 0 else 0.0
    
    rte_penalty = 0.0
    if actual_rte < guar_rte:
        # Calculate Excess Energy Loss
        # How much we SHOULD have lost (or consumed) vs how much we DID consume
        # Standard formulation: Excess_Loss = Charged * (Guar_RTE - Actual_RTE) ??
        # Wait, user formula: Excess_Energy_Loss_MWh = Total Charged MWh * (Guaranteed RTE - Actual RTE)
        # This represents the MISSING energy output relative to input.
        
        excess_loss = total_charged * (guar_rte - actual_rte)
        
        # Penalty = Excess Loss * Weighted Avg Charge Price
        # Calculate Weighted Avg Charge Price (LMP Charging)
        charge_cost = (charged * prices).sum()
        weighted_charge_price = charge_cost / total_charged if total_charged > 0 else 0.0
        
        rte_penalty = excess_loss * weighted_charge_price
        
    # --- Step 3: VOM Payment ---
    vom_payment = total_discharged * vom_rate
    
    # --- Final Settlement ---
    # Invoice = Capacity + VOM - RTE Penalty
    net_invoice = final_capacity_payment + vom_payment - rte_penalty
    
    return {
        'net_invoice': net_invoice,
        'capacity_payment': final_capacity_payment,
        'vom_payment': vom_payment,
        'rte_penalty': rte_penalty,
        'actual_availability': actual_availability,
        'actual_rte': actual_rte,
        'total_discharged': total_discharged,
        'total_charged': total_charged
    }

def calculate_buyer_pl(ops_data, capacity_mw, toll_rate_mw_mo, ancillary_rev_mw_mo, charging_cost_profile=None):
    """
    Calculates the Buyer's P&L (Tolling Model).
    
    Buyer pays fixed Toll + Charging Costs.
    Buyer earns Arbitrage Revenue + Ancillary Revenue.
    
    Args:
        ops_data (dict): Battery operations data (discharge, charge, price).
        capacity_mw (float): The battery's power capacity in MW.
        toll_rate_mw_mo (float): Fixed monthly rental per MW.
        ancillary_rev_mw_mo (float): Estimated Ancillary Service revenue per MW-month.
        charging_cost_profile (pd.Series, optional): Hourly cost ($/MWh) to charge. 
                                                     If None, uses market price (Grid charging).
    
    Returns:
        pd.DataFrame: Monthly P&L breakdown.
    """
    discharge_mwh = ops_data['discharge_mwh_profile']
    charge_mwh = ops_data['charge_mwh_profile']
    market_price = ops_data['market_price_profile']
    
    # 1. Calculate Arbitrage Revenue (Hourly)
    hourly_revenue = discharge_mwh * market_price
    
    # 2. Calculate Charging Cost (Hourly)
    if charging_cost_profile is None:
        # Default to Grid Charging at Market Price
         hourly_cost = charge_mwh * market_price
    else:
        # Constrained Charging (e.g. at Solar PPA Price)
        hourly_cost = charge_mwh * charging_cost_profile
        
    # 3. Create DataFrame
    # Create a proper DatetimeIndex for 2024 to group by month
    dt_index = pd.date_range(start='2024-01-01', periods=len(discharge_mwh), freq='h')
    
    df = pd.DataFrame({
        'Revenue_Arb': hourly_revenue.values,
        'Cost_Charge': hourly_cost.values,
        'Ancillary_Rev': 0.0, # Placeholder
        'Toll_Cost': 0.0      # Placeholder
    }, index=dt_index)
    
    # Group by Month
    monthly_pl = df.resample('ME').sum() # 'ME' is month end in newer pandas, 'M' in older. using 'M' for safety or 'ME' if pandas > 2.0?
    # Let's use 'M' to be safe with older pandas versions, or checks. Streamlit cloud usually recent.
    # 'M' is deprecated in pandas 2.2+. Let's use 'ME'.
    monthly_pl = df.resample('ME').sum()
    
    # Add Monthly Fixed Items
    # Toll = Rate * Capacity
    monthly_toll = toll_rate_mw_mo * capacity_mw
    
    # Ancillary = Rate * Capacity
    monthly_ancillary = ancillary_rev_mw_mo * capacity_mw
    
    monthly_pl['Toll_Cost'] = monthly_toll
    monthly_pl['Ancillary_Rev'] = monthly_ancillary
    
    # Net Profit
    # Revenue + Ancillary - ChargeCost - Toll
    monthly_pl['Net_Profit'] = (
        monthly_pl['Revenue_Arb'] + 
        monthly_pl['Ancillary_Rev'] - 
        monthly_pl['Cost_Charge'] - 
        monthly_pl['Toll_Cost']
    )
    
    # Add Month names for display
    monthly_pl['Month'] = monthly_pl.index.strftime('%b')
    
    return monthly_pl

def recommend_portfolio(load_profile, target_cfe=0.95, excluded_techs=None, existing_capacities=None):
    """
    Heuristic recommendation for initial portfolio based on load.
    If existing_capacities provided, builds around those values (keeps non-zero values, fills zeros).
    
    Args:
        load_profile (pd.Series): Hourly load profile
        target_cfe (float): Target CFE score (default 0.95)
        excluded_techs (list): List of technologies to exclude from recommendation
        existing_capacities (dict): Existing capacity values to build around
    
    Returns:
        dict: Recommended capacities
    """
    if excluded_techs is None:
        excluded_techs = []
    
    if existing_capacities is None:
        existing_capacities = {}
    
    avg_load = load_profile.mean()
    min_load = load_profile.min()
    peak_load = load_profile.max()
    total_load = load_profile.sum()
    
    # Initial Recommendation (Baseline)
    recommendation = {
        'Solar': existing_capacities.get('Solar', 0),
        'Wind': existing_capacities.get('Wind', 0),
        'Geothermal': existing_capacities.get('Geothermal', 0),
        'Nuclear': existing_capacities.get('Nuclear', 0),
        'CCS Gas': existing_capacities.get('CCS Gas', 0),
        'Battery_MW': existing_capacities.get('Battery_MW', 0),
        'Battery_Hours': 2
    }
    
    # Check if we have any existing non-zero capacities
    has_existing = any(v > 0 for v in existing_capacities.values() if v is not None)
    
    # 1. Baseload Coverage (Firm Clean Energy) - Only set if not already specified
    # Suggest covering 80% of average load with firm clean energy
    firm_target = avg_load * 0.80
    
    # Only set baseload if user hasn't specified any firm techs
    existing_firm = recommendation.get('CCS Gas', 0) + recommendation.get('Nuclear', 0) + recommendation.get('Geothermal', 0)
    
    if existing_firm == 0:
        # Logic to distribute firm target - Prioritize CCS Gas for baseload
        if 'CCS Gas' not in excluded_techs:
            # Use CCS Gas as primary baseload
            recommendation['CCS Gas'] = firm_target
        elif 'Nuclear' not in excluded_techs:
            # Fallback to Nuclear if CCS is excluded
            recommendation['Nuclear'] = firm_target
        elif 'Geothermal' not in excluded_techs:
            # Final fallback to Geothermal
            recommendation['Geothermal'] = firm_target


    # 2. Variable Renewable Coverage (Initial Guess)
    firm_gen_annual = (recommendation['Geothermal'] + recommendation['Nuclear'] + recommendation['CCS Gas']) * 8760
    remaining_load = total_load - firm_gen_annual
    
    # Only set VRE if user hasn't specified any
    existing_vre = recommendation.get('Solar', 0) + recommendation.get('Wind', 0)
    
    if remaining_load > 0 and existing_vre == 0:
        target_variable_gen = remaining_load * 2.0 # Increased from 1.2x to 2.0x for better coverage
        
        var_techs = [t for t in ['Solar', 'Wind'] if t not in excluded_techs]
        
        if var_techs:
            # Capacity Factors: Solar ~0.25, Wind ~0.40
            # If both present, split 50/50 energy target
            # If only one, give 100% energy target
            
            for t in var_techs:
                if t == 'Solar':
                    recommendation['Solar'] = (target_variable_gen / len(var_techs)) / (8760 * 0.25)
                elif t == 'Wind':
                    recommendation['Wind'] = (target_variable_gen / len(var_techs)) / (8760 * 0.40)
        
    # 3. Battery Storage - Only set if not already specified
    if recommendation.get('Battery_MW', 0) == 0 and 'Battery' not in excluded_techs:
        recommendation['Battery_MW'] = peak_load * 0.40 # Increased from 0.2 to 0.4
    
    # Iterative Optimization Loop
    max_iterations = 100
    current_cfe = 0.0
    
    for i in range(max_iterations):
        # Generate Profiles based on current recommendation
        solar_gen = generate_dummy_generation_profile(recommendation['Solar'], 'Solar')
        wind_gen = generate_dummy_generation_profile(recommendation['Wind'], 'Wind')
        geo_gen = generate_dummy_generation_profile(recommendation['Geothermal'], 'Geothermal')
        nuc_gen = generate_dummy_generation_profile(recommendation['Nuclear'], 'Nuclear')
        ccs_gen = generate_dummy_generation_profile(recommendation['CCS Gas'], 'CCS Gas')
        
        total_gen = solar_gen + wind_gen + geo_gen + nuc_gen + ccs_gen
        
        # Calculate Surplus/Deficit for Battery
        surplus = (total_gen - load_profile).clip(lower=0)
        deficit = (load_profile - total_gen).clip(lower=0)
        
        # Simulate Battery
        batt_discharge, _, _ = simulate_battery_storage(surplus, deficit, recommendation['Battery_MW'], recommendation['Battery_Hours'])
        
        # Calculate CFE
        total_available = total_gen + batt_discharge
        cfe_score, _ = calculate_cfe_score(load_profile, total_available)
        current_cfe = cfe_score
        
        if current_cfe >= target_cfe - 0.0001: # Float tolerance
            break
            
        # More aggressive scaling to compensate for 2-hour battery limit
        gap = target_cfe - current_cfe
        scaler = 1.03
        if gap > 0.10: scaler = 1.20  # Increased from 1.15
        elif gap > 0.05: scaler = 1.12  # Increased from 1.08
        elif gap > 0.01: scaler = 1.08  # Increased from 1.05
        
        # Scale up all technologies
        if 'Solar' not in excluded_techs:
            recommendation['Solar'] *= scaler
        if 'Wind' not in excluded_techs:
            recommendation['Wind'] *= scaler
            
        # Also scale Firm generation (critical for 100% CFE with limited battery)
        for t in ['Geothermal', 'Nuclear', 'CCS Gas']:
            if t not in excluded_techs:
                recommendation[t] *= scaler
        
        # Increase Battery Power (Duration capped at 2 hours)
        if 'Battery' not in excluded_techs:
            recommendation['Battery_MW'] *= scaler
            # CAP: Limit battery power to 1.1x Peak Load to enforce realistic sizing
            # It's rarely economic to size battery significantly larger than peak load
            if recommendation['Battery_MW'] > peak_load * 1.1:
                recommendation['Battery_MW'] = peak_load * 1.1
                
            # Keep battery duration capped at 2 hours for scenarios
            # Users can still manually set higher values if desired
            
    return recommendation

def generate_dummy_price_profile(avg_price, return_base_avg=False):
    """
    Generates an hourly market price profile (8760 hours).
    Attempts to load real ERCOT 2024 Data (HB_NORTH) from 'ercot_rtm_2024.parquet'.
    Falls back to synthetic duck curve if file missing.
    
    Args:
        avg_price: Average price (used only for synthetic fallback)
        return_base_avg: If True, returns (profile, base_average) tuple
        
    Returns:
        pd.Series or tuple: Price profile, or (profile, base_avg) if return_base_avg=True
    """
    hours = 8760
    import os
    base_average = avg_price  # Default for synthetic
    
    # 1. Try Loading Real Data
    parquet_file = 'ercot_rtm_2024.parquet'
    if os.path.exists(parquet_file):
        try:
            # Read Parquet
            df = pd.read_parquet(parquet_file)
            
            # Filter for HB_NORTH
            # Check for 'Location' or 'SettlementPoint' column depending on schema
            # We verified 'Location' and 'HB_NORTH' in step 107
            if 'Location' in df.columns and 'SPP' in df.columns:
                 df_north = df[df['Location'] == 'HB_NORTH'].copy()
                 
                 # Ensure datetime
                 if 'Time' in df_north.columns:
                    df_north['Time'] = pd.to_datetime(df_north['Time'])
                    df_north.set_index('Time', inplace=True)
                    
                    # Resample to Hourly Mean (RTM is 15-min)
                    # The parquet Time is likely interval start.
                    df_hourly = df_north['SPP'].resample('h').mean()
                    
                    # Handle Missing/NaN
                    df_hourly = df_hourly.interpolate(method='linear').bfill().ffill()
                    
                    # Normalize to 8760 (Handle Leap Year 2024)
                    # If we have more than 8760 rows, checking for Feb 29 usually best.
                    # But simple approach: Remove Feb 29 if exists, otherwise truncate end.
                    if len(df_hourly) > 8760:
                        # Check if index is datetime
                        if isinstance(df_hourly.index, pd.DatetimeIndex):
                            # Function to filter out Feb 29
                            df_hourly = df_hourly[~((df_hourly.index.month == 2) & (df_hourly.index.day == 29))]
                    
                    # Truncate/Pad to 8760
                    raw_profile = df_hourly.values
                    if len(raw_profile) > hours:
                        raw_profile = raw_profile[:hours]
                    elif len(raw_profile) < hours:
                        raw_profile = np.pad(raw_profile, (0, hours - len(raw_profile)), 'constant', constant_values=raw_profile.mean())
                    
                    profile = raw_profile
                    
                    # Clip negative prices first (to avoids mean drift after normalization)
                    # User requested negative prices be allowed
                    # profile = np.clip(raw_profile, 0, None) 
                    profile = raw_profile
                    
                    # Calculate base average from real data
                    base_average = np.mean(profile)
                    
                    # Scaling: User requested ACTUAL data, so we disable scaling here.
                    # The 'avg_price' argument is ignored for real data.
                    # current_avg = np.mean(profile)
                    # if current_avg != 0:
                    #     profile = profile * (avg_price / current_avg)
                    
                    if return_base_avg:
                        return pd.Series(profile, name='Market Price ($/MWh)'), base_average
                    return pd.Series(profile, name='Market Price ($/MWh)')
                    
        except Exception as e:
            print(f"Failed to load Real Price Data: {e} - Falling back to synthetic.")

    # 2. Synthetic Fallback
    t = np.arange(hours)
    day_of_year = (t // 24)
    
    rng = np.random.default_rng(999) # Fixed seed
    
    profile = np.zeros(hours)
    
    for h in range(hours):
        hour_of_day = h % 24
        
        # Diurnal: High Evening (17-21), Low Midday (10-15)
        # Base shape
        diurnal = 1.0 + 0.4 * np.sin(2 * np.pi * (hour_of_day - 14) / 24) 
        # Shift peak to ~20:00 (Hour 20) -> (20-14)/24 = 6/24 = 0.25 -> sin(pi/2) = 1
        # Trough ~8:00
        
        # Sharp evening peak adder
        if 16 <= hour_of_day <= 21:
            diurnal += 0.5
            
        # Depress midday (solar cannibalization effect)
        if 10 <= hour_of_day <= 15:
            diurnal -= 0.3
            
        # Seasonal
        current_day = day_of_year[h]
        seasonal = 1.0 + 0.2 * np.cos(2 * np.pi * (current_day - 172) / 365) # Summer peak
        
        # Noise
        noise = rng.normal(0, 0.1)
        
        profile[h] = diurnal * seasonal + noise
        
    # Normalize to match avg_price
    current_avg = profile.mean()
    if current_avg != 0:
        profile = profile * (avg_price / current_avg)
    base_average = avg_price
        
    # Clip negative prices (simplified) unless desired
    profile = np.clip(profile, 0, None)
    
    if return_base_avg:
        return pd.Series(profile, name='Market Price ($/MWh)'), base_average
    return pd.Series(profile, name='Market Price ($/MWh)')

def calculate_financials(matched_profile, deficit_profile, tech_profiles, tech_prices, market_price_avg, rec_price, price_scaler=1.0):
    """
    Calculates financial metrics for the portfolio using per-technology pricing and HOURLY market prices.
    """
    # Generate Hourly Market Prices and apply scaler
    market_price_profile = generate_dummy_price_profile(market_price_avg) * price_scaler
    
    # 1. Calculate PPA Cost of Matched Energy (Weighted Attribution)
    # We assume 'matched_profile' is composed of the various techs in proportion to their generation.
    # Total Available Generation at each hour
    total_gen_profile = sum(tech_profiles.values())
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        attribution_factors = matched_profile / total_gen_profile
        attribution_factors = attribution_factors.fillna(0.0) # 0/0 -> 0
    
    total_ppa_cost = 0.0
    
    for tech, profile in tech_profiles.items():
        price = tech_prices.get(tech, 0.0)
        # Matched MWh from this tech
        matched_mwh_tech = profile * attribution_factors
        tech_cost = matched_mwh_tech.sum() * price
        total_ppa_cost += tech_cost
        
    # 2. Market Value of Matched Energy (HOURLY)
    # Value = Sum(Matched[h] * MarketPrice[h])
    market_value_matched = (matched_profile * market_price_profile).sum()
    
    # 3. Settlement Value
    # PPA Settlement = Market Value - PPA Cost
    settlement_value = market_value_matched - total_ppa_cost
    
    # 4. REC Cost
    total_matched_mwh = matched_profile.sum()
    rec_cost = total_matched_mwh * rec_price
    
    # 5. Net Energy Cost
    # Cost to serve load = (Deficit * Market) + (Matched * PPA Price) + (Matched * REC)
    market_cost_deficit = (deficit_profile * market_price_profile).sum()
    
    net_cost = market_cost_deficit + total_ppa_cost + rec_cost
    
    total_deficit_mwh = deficit_profile.sum()
    total_load = total_matched_mwh + total_deficit_mwh
    avg_cost_per_mwh = net_cost / total_load if total_load > 0 else 0.0
    
    # Calculate Weighted Averages for Display
    weighted_ppa_price = total_ppa_cost / total_matched_mwh if total_matched_mwh > 0 else 0.0
    
    # Capture Value: Market Value of the generated energy (Total Available Generation)
    # Includes Battery Discharge as it is part of the supply available to match/sell.
    
    market_value_total_gen = (total_gen_profile * market_price_profile).sum()
    total_gen_mwh = total_gen_profile.sum()
    weighted_market_price = market_value_total_gen / total_gen_mwh if total_gen_mwh > 0 else 0.0
    
    return {
        'settlement_value': settlement_value,
        'rec_cost': rec_cost,
        'net_cost': net_cost,
        'avg_cost_per_mwh': avg_cost_per_mwh,
        'weighted_ppa_price': weighted_ppa_price,
        'weighted_market_price': weighted_market_price
    }


def process_uploaded_profile(uploaded_file, keywords=None):
    """
    Reads an uploaded CSV file and attempts to extract a standard hourly profile (MW).
    Supports standard CSVs and NREL PVWatts/SAM exports with metadata rows.
    
    Args:
        uploaded_file: Streamlit UploadedFile object.
        keywords (list): Optional list of column keywords to prioritize (e.g., ['load', 'demand']).
        
    Returns:
        pd.Series: Hourly profile in MW (length 8760), or None if failed.
    """
    if uploaded_file is None:
        return None
        
    try:
        # 1. Attempt to find the header row
        # Read first few lines to find a row that looks like a header
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.split('\n')
        
        header_row_index = 0
        found_header = False
        
        # Common NREL/PVWatts column signatures
        nrel_signatures = ['Year', 'Month', 'Day', 'Hour', 'AC System Output (W)']
        
        for i, line in enumerate(lines[:50]): # Check first 50 lines
            # Simple heuristic: Check for comma separation and keywords
            line_lower = line.lower()
            if 'year' in line_lower and 'month' in line_lower and 'day' in line_lower:
                header_row_index = i
                found_header = True
                break
            # Fallback for simple files
            if keywords:
                if any(k in line_lower for k in keywords):
                    header_row_index = i
                    found_header = True
                    break
        
        # Reload with identified header
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, header=header_row_index)
        
        # 2. Identify the Data Column
        target_col = None
        
        # PVWatts specific check
        pvwatts_col = 'AC System Output (W)'
        if pvwatts_col in df.columns:
            # Convert W to MW
            return df[pvwatts_col] / 1e6 # PVWatts usually outputs W, so /1e6 for MW
            
        # Generic keyword search
        if keywords:
            # Normalize column names
            df.columns = [str(c).strip() for c in df.columns]
            
            for col in df.columns:
                col_lower = col.lower()
                if any(k in col_lower for k in keywords):
                    target_col = col
                    break
                    
        # Fallback: Use the first numeric column if no keyword match
        if target_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[0]
                
        if target_col:
            # Ensure length is 8760 (truncate or pad)
            data = df[target_col].values
            if len(data) > 8760:
                data = data[:8760]
            elif len(data) < 8760:
                # Pad with zeros or repeat? Pad with zeros is safer for now.
                data = np.pad(data, (0, 8760 - len(data)), 'constant')
                
            return pd.Series(data)
            
        st.error("Could not identify a valid data column in the CSV.")
        return None
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None
