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
            
        # Clamp to 0 to avoid floating point errors showing negative zero
        if current_energy_mwh < 0:
            current_energy_mwh = 0.0
            
        soc_profile[h] = current_energy_mwh
        
    return pd.Series(discharge_profile, name='Battery Discharge (MW)'), pd.Series(soc_profile, name='Battery SoC (MWh)')

def recommend_portfolio(load_profile, target_cfe=0.95, excluded_techs=None):
    """
    Heuristic to recommend a technology mix targeting a specific CFE score (default 95%).
    
    Strategy:
    1. Start with a baseline heuristic.
    2. Iteratively scale up variable renewables and battery storage until target CFE is met.
    """
    if excluded_techs is None:
        excluded_techs = []
        
    avg_load = load_profile.mean()
    min_load = load_profile.min()
    peak_load = load_profile.max()
    total_load = load_profile.sum()
    
    # Initial Recommendation (Baseline)
    recommendation = {
        'Solar': 0,
        'Wind': 0,
        'Geothermal': 0,
        'Nuclear': 0,
        'CCS Gas': 0,
        'Battery_MW': 0,
        'Battery_Hours': 4
    }
    
    # 1. Baseload Coverage (Firm Clean Energy)
    # Suggest covering 80% of min load with firm clean energy
    firm_target = min_load * 0.8
    
    # Logic to distribute firm target
    firm_techs = [t for t in ['Geothermal', 'Nuclear', 'CCS Gas'] if t not in excluded_techs]
    if firm_techs:
        for t in firm_techs:
            recommendation[t] = firm_target / len(firm_techs)


    # 2. Variable Renewable Coverage (Initial Guess)
    firm_gen_annual = (recommendation['Geothermal'] + recommendation['Nuclear'] + recommendation['CCS Gas']) * 8760
    remaining_load = total_load - firm_gen_annual
    
    if remaining_load > 0:
        target_variable_gen = remaining_load * 1.2 # Start with 1.2x coverage
        
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
        
    # 3. Battery Storage (Initial Guess)
    if 'Battery' not in excluded_techs:
        recommendation['Battery_MW'] = peak_load * 0.2
    
    # Iterative Optimization Loop
    max_iterations = 20
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
        batt_discharge, _ = simulate_battery_storage(surplus, deficit, recommendation['Battery_MW'], recommendation['Battery_Hours'])
        
        # Calculate CFE
        total_available = total_gen + batt_discharge
        cfe_score, _ = calculate_cfe_score(load_profile, total_available)
        current_cfe = cfe_score
        
        if current_cfe >= target_cfe:
            break
            
        # Scale up if target not met
        # Increase Solar/Wind by 10%
        if 'Solar' not in excluded_techs:
            recommendation['Solar'] *= 1.1
        if 'Wind' not in excluded_techs:
            recommendation['Wind'] *= 1.1
        
        # Increase Battery Power by 5% and Duration slightly (up to 8h)
        if 'Battery' not in excluded_techs:
            recommendation['Battery_MW'] *= 1.05
            if recommendation['Battery_Hours'] < 8:
                recommendation['Battery_Hours'] += 0.5
            
    # Round values for clean output
    for k, v in recommendation.items():
        recommendation[k] = round(v, 1)
        
    return recommendation

def calculate_financials(matched_profile, deficit_profile, tech_profiles, tech_prices, market_price_avg, rec_price):
    """
    Calculates financial metrics for the portfolio using per-technology pricing.
    
    Args:
        matched_profile (pd.Series): Hourly matched renewable energy (MWh).
        deficit_profile (pd.Series): Hourly unmatched load (MWh).
        tech_profiles (dict): Dict of hourly generation profiles (pd.Series) for each tech.
            Keys should match keys in tech_prices (e.g., 'Solar', 'Wind').
        tech_prices (dict): Dict of PPA prices ($/MWh) for each tech.
        market_price_avg (float): Average Wholesale Market Price ($/MWh).
        rec_price (float): Price of RECs ($/REC or $/MWh of clean energy).
        
    Returns:
        dict: Financial metrics including Settlement Value, REC Cost, Net Cost.
    """
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
        # Matched MWh from this tech = Total Matched * (Tech Gen / Total Gen) 
        # which simplifies to: Tech Gen * (Matched / Total Gen) -> Tech Gen * attribution_factor
        matched_mwh_tech = profile * attribution_factors
        tech_cost = matched_mwh_tech.sum() * price
        total_ppa_cost += tech_cost
        
    # 2. Market Value of Matched Energy
    # Value = Matched MWh * Market Price
    total_matched_mwh = matched_profile.sum()
    market_value_matched = total_matched_mwh * market_price_avg
    
    # 3. Settlement Value
    # PPA Settlement = Market Value - PPA Cost
    # (Positive means we made money relative to PPA cost)
    settlement_value = market_value_matched - total_ppa_cost
    
    # 4. REC Cost
    rec_cost = total_matched_mwh * rec_price
    
    # 5. Net Energy Cost
    # Cost to serve load = (Deficit * Market) + (Matched * PPA Price) + (Matched * REC)
    # Note: 'Matched * PPA Price' is exactly 'total_ppa_cost' calculated above
    
    total_deficit_mwh = deficit_profile.sum()
    total_load = total_matched_mwh + total_deficit_mwh
    
    market_cost_deficit = total_deficit_mwh * market_price_avg
    
    net_cost = market_cost_deficit + total_ppa_cost + rec_cost
    
    # Calculate Weighted Averages for Display
    weighted_ppa_price = total_ppa_cost / total_matched_mwh if total_matched_mwh > 0 else 0.0
    weighted_market_price = market_value_matched / total_matched_mwh if total_matched_mwh > 0 else 0.0
    
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
