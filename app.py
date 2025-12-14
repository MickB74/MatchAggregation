import streamlit as st
import ast  # For handling single-quoted JSON-like strings
import re
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import datetime
import json
import zipfile
import io
from utils import (
    generate_dummy_load_profile, 
    generate_dummy_generation_profile, 
    calculate_cfe_score, 
    simulate_battery_storage, 
    recommend_portfolio, 
    calculate_portfolio_metrics, 
    calculate_financials, 
    process_uploaded_profile,
    generate_dummy_price_profile,
    calculate_battery_financials,
    calculate_buyer_pl
)
import project_matcher

st.set_page_config(page_title="ERCOT North Aggregation", layout="wide")

st.title("ERCOT North Renewable Energy Aggregation")
st.markdown("Aggregate load participants and optimize for 24/7 clean energy matching.")

# Session state to store participants
if 'participants' not in st.session_state:
    st.session_state.participants = []

# --- Global Settings (Sidebar - Load Scenario) ---
# --- Global Settings (Sidebar - Load Scenario) ---
# Callback to handle loading
def load_scenario():
    uploaded_file = st.session_state.get('uploaded_scenario_file')
    if uploaded_file is not None:
        try:
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            
            # Robust loading:
            content_bytes = uploaded_file.read()
            content_str = content_bytes.decode('utf-8').strip()
            
            # 1. Aggressive Cleaning: Find first '{' and last '}'
            start_idx = content_str.find('{')
            end_idx = content_str.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                content_str = content_str[start_idx : end_idx + 1]
            
            # Remove C-style comments (// ...)
            content_str = re.sub(r'//.*', '', content_str)
            
            # 2. Try standard JSON
            try:
                config = json.loads(content_str)
            except json.JSONDecodeError:
                # 3. Try ast.literal_eval
                try:
                    config = ast.literal_eval(content_str)
                except (ValueError, SyntaxError):
                    st.error("Could not parse file.")
                    return
            
            # DEBUG: Uncomment to see what was parsed
            # st.sidebar.write("DEBUG: Parsed Config:", config)

                    return

            # Apply to Session State
            # 1. Participants
            if 'participants' in config:
                st.session_state.participants = config['participants']
            
            # 2. Generation Capacities
            if 'solar_capacity' in config: st.session_state.solar_input = float(config['solar_capacity'])
            if 'wind_capacity' in config: st.session_state.wind_input = float(config['wind_capacity'])
            if 'ccs_capacity' in config: st.session_state.ccs_input = float(config['ccs_capacity'])
            if 'geo_capacity' in config: st.session_state.geo_input = float(config['geo_capacity'])
            if 'nuc_capacity' in config: st.session_state.nuc_input = float(config['nuc_capacity'])
            if 'batt_capacity' in config: st.session_state.batt_input = float(config['batt_capacity'])
            if 'batt_duration' in config: st.session_state.batt_duration_input = float(config['batt_duration'])
            
            # 3. Financials
            # 3. Financials
            if 'strike_price' in config: 
                # Backward compatibility
                st.session_state.solar_price_input = float(config['strike_price'])
                st.session_state.wind_price_input = float(config['strike_price'])
                st.session_state.ccs_price_input = float(config['strike_price'])
                st.session_state.geo_price_input = float(config['strike_price'])
                st.session_state.nuc_price_input = float(config['strike_price'])

            if 'solar_price' in config: st.session_state.solar_price_input = float(config['solar_price'])
            if 'wind_price' in config: st.session_state.wind_price_input = float(config['wind_price'])
            if 'ccs_price' in config: st.session_state.ccs_price_input = float(config['ccs_price'])
            if 'geo_price' in config: st.session_state.geo_price_input = float(config['geo_price'])
            if 'nuc_price' in config: st.session_state.nuc_price_input = float(config['nuc_price'])
            if 'batt_price' in config: st.session_state.batt_price_input = float(config['batt_price'])

            if 'market_price' in config: st.session_state.market_input = float(config['market_price'])
            if 'rec_price' in config: st.session_state.rec_input = float(config['rec_price'])
            
            # 4. Exclusions
            if 'excluded_techs' in config: st.session_state.excluded_techs_input = config['excluded_techs']
            
            st.toast("Scenario Loaded Successfully!")
            
        except Exception as e:
            st.error(f"Error parsing scenario: {e}")

# MUST be defined BEFORE the widgets that use these session state values are instantiated.
with st.sidebar:
    st.markdown("### Load Scenario")
    st.file_uploader(
        "Upload scenario_config.json", 
        type=['json', 'txt'], 
        key='uploaded_scenario_file', 
        on_change=load_scenario
    )

    st.markdown("---")

# --- Configuration Section (Top) ---
with st.expander("Configuration & Setup", expanded=True):
    tab_load, tab_gen, tab_fin = st.tabs(["1. Load Setup", "2. Generation Portfolio", "3. Financials"])
    
    # --- Tab 1: Load Setup ---
    with tab_load:
        col_load_1, col_load_2 = st.columns([1, 2])
        
        with col_load_1:
            st.markdown("#### Add Participant")
            # Only show participant form if no file is uploaded (or allow both but prioritize file?)
            # Logic: If 'uploaded_load_file' is present, we use it. But we can still build list.
            
            with st.form("add_participant"):
                next_num = len(st.session_state.participants) + 1
                p_name = st.text_input("Participant Name", f"Participant {next_num}")
                p_type = st.selectbox("Building Type", ["Data Center", "Office", "Flat"])
                p_load = st.number_input("Annual Consumption (MWh)", min_value=1000, value=50000, step=50000)
                submitted = st.form_submit_button("Add Participant")
                
                if submitted:
                    st.session_state.participants.append({
                        "name": p_name,
                        "type": p_type,
                        "load": p_load
                    })
                    st.success(f"Added {p_name}")

            if st.session_state.participants:
                if st.button("Clear Participants"):
                    st.session_state.participants = []
                    st.rerun()
            
            st.markdown("---")
            if st.button("🎲 Random Scenario (>500 GWh)"):
                # Clear existing
                st.session_state.participants = []
                
                # Logic to generate random scenario > 500k MWh
                import random
                current_total_load = 0
                count = 1
                
                # Target at least 500k
                while current_total_load < 500000:
                    # Randomly choose type
                    # Weighted towards Data Centers for higher load
                    p_type = random.choice(["Data Center", "Data Center", "Office", "Office", "Ammonia Plant"])
                    
                    if p_type == "Data Center":
                        # Large load: 100k - 300k MWh
                        load = random.randint(100000, 300000)
                    elif p_type == "Ammonia Plant":
                         # Large constant load: 200k - 400k MWh (mapped to Flat/DataCenter profile logic usually, but we use 'Flat' for now)
                         # We'll use 'Flat' profile type for Ammonia internally if 'Ammonia Plant' isn't in utils yet
                         # Actually utils only has ['Flat', 'Data Center', 'Office']
                         # Let's map Ammonia to 'Flat' effectively by just calling it 'Flat' in the backend, 
                         # OR we stick to the selectbox types.
                         # The user asked for "random scenarios", so diversity is good.
                         # Let's stick to valid types for now to ensure profile generation works:
                         # 'Data Center', 'Office', 'Flat'
                         p_type = "Flat"
                         load = random.randint(150000, 400000)
                         p_name = f"Ind. Plant {count}" 
                    else: # Office
                        # Medium load: 10k - 50k MWh
                        load = random.randint(10000, 50000)
                        
                    if p_type == "Data Center":
                         p_name = f"Data Center {count}"
                    elif p_type == "Office":
                         p_name = f"Office Park {count}"
                    elif p_type == "Flat":
                         p_name = f"Industrial {count}"

                    st.session_state.participants.append({
                        "name": p_name,
                        "type": p_type,
                        "load": load
                    })
                    
                    current_total_load += load
                    count += 1
                
                st.success(f"Generated {count-1} participants with {current_total_load:,.0f} MWh total load!")
                st.rerun()

        with col_load_2:
            st.markdown("#### Current Participants")
            if st.session_state.participants:
                p_df = pd.DataFrame(st.session_state.participants)
                st.dataframe(p_df, hide_index=True, use_container_width=True)
            else:
                st.info("No participants added yet.")
                
            st.markdown("---")
            st.markdown("#### Or Upload Aggregate Load Profile")
            uploaded_load_file = st.file_uploader("Upload CSV (Hourly load in MW)", type=['csv', 'txt'], key='uploaded_load_file')


    # --- Tab 2: Generation Portfolio ---
    with tab_gen:
        col_gen_1, col_gen_2 = st.columns([1, 1])
        
        with col_gen_1:
            st.markdown("#### Capacities")
            
            # Clear Portfolio Button for manual reset
            if st.button("🗑️ Clear Portfolio (Reset to 0)"):
                st.session_state.solar_input = 0.0
                st.session_state.wind_input = 0.0
                st.session_state.ccs_input = 0.0
                st.session_state.geo_input = 0.0
                st.session_state.nuc_input = 0.0
                st.session_state.batt_input = 0.0
                st.session_state.batt_duration_input = 2.0
                st.session_state.matched_projects = {}
                st.session_state.portfolio_recommended = False
                st.rerun()

            use_synthetic_solar = st.checkbox("Use Synthetic Solar Profile (Ignore CSV)", value=False)
            
            # Input Widgets (Keys mapped to session state)
            solar_capacity = st.number_input("Solar Capacity (MW)", min_value=0.0, step=1.0, key='solar_input')
            wind_capacity = st.number_input("Wind Capacity (MW)", min_value=0.0, step=1.0, key='wind_input')
            geo_capacity = st.number_input("Geothermal Capacity (MW)", min_value=0.0, step=1.0, key='geo_input')
            nuc_capacity = st.number_input("Nuclear Capacity (MW)", min_value=0.0, step=1.0, key='nuc_input')
            ccs_capacity = st.number_input("CCS Gas Capacity (MW)", min_value=0.0, step=1.0, key='ccs_input')
            
            # Automatically update project suggestions when capacities change
            # Build current recommendation from slider values
            current_recommendation = {
                'Solar': solar_capacity,
                'Wind': wind_capacity,
                'CCS Gas': ccs_capacity,
                'Geothermal': geo_capacity,
                'Nuclear': nuc_capacity,
                'Battery_MW': st.session_state.get('batt_input', 0.0) # Use session state for batt_input as it's in col_gen_2
            }
            
            # Check if any capacity is set (not all zeros)
            has_capacity = any(v > 0 for v in current_recommendation.values())
            
            # Update matched projects dynamically
            if has_capacity:
                matched_projects = project_matcher.match_projects_to_recommendation(current_recommendation, max_projects_per_tech=5)
                st.session_state.matched_projects = matched_projects
            else:
                # Clear matched projects if all capacities are zero
                st.session_state.matched_projects = {}
            
            st.markdown("---")

        with col_gen_2:
            st.markdown("#### Portfolio Recommendation")
            st.info("Configuration moved to Financials Tab")
            
            st.markdown("---")
            
            # Exclude Tech multiselect
            excluded_techs = st.multiselect(
                "Exclude Technologies from Recommendation",
                ['Solar', 'Wind', 'CCS Gas', 'Geothermal', 'Nuclear', 'Battery'],
                key='excluded_techs_input'
            )

            # Define callback for recommendation
            def apply_recommendation():
                # Calculate total load from participants
                temp_load = pd.Series(0.0, index=range(8760))
                if st.session_state.participants:
                    for p in st.session_state.participants:
                        temp_load += generate_dummy_load_profile(p['load'], p['type'])
                    
                    if temp_load.sum() > 0:
                        # Check whether to use existing capacities or reset
                        force_reset = st.session_state.get('force_reset_rec', False)
                        
                        if force_reset:
                            existing_capacities = {} # Ignore current values
                        else:
                            # Use existing values to build around them
                            existing_capacities = {
                                'Solar': st.session_state.get('solar_input', 0.0),
                                'Wind': st.session_state.get('wind_input', 0.0),
                                'CCS Gas': st.session_state.get('ccs_input', 0.0),
                                'Geothermal': st.session_state.get('geo_input', 0.0),
                                'Nuclear': st.session_state.get('nuc_input', 0.0),
                                'Battery_MW': st.session_state.get('batt_input', 0.0)
                            }
                        
                        # Pass excluded techs from session state (widget key='excluded_techs_input')
                        rec = recommend_portfolio(
                            temp_load, 
                            target_cfe=1.0, 
                            excluded_techs=st.session_state.excluded_techs_input,
                            existing_capacities=existing_capacities
                        )
                        st.session_state.solar_input = rec['Solar']
                        st.session_state.wind_input = rec['Wind']
                        st.session_state.ccs_input = rec['CCS Gas']
                        st.session_state.geo_input = rec['Geothermal']
                        st.session_state.nuc_input = rec['Nuclear']
                        st.session_state.batt_input = rec['Battery_MW']
                        st.session_state.batt_duration_input = rec['Battery_Hours']
                        
                        # Match projects from ERCOT queue
                        matched_projects = project_matcher.match_projects_to_recommendation(rec, max_projects_per_tech=5)
                        st.session_state.matched_projects = matched_projects
                        
                        st.session_state.portfolio_recommended = True
                    else:
                        st.session_state.portfolio_error = "Participant load is zero."
                else:
                    st.session_state.portfolio_error = "Add participants first."

            col_btn, col_chk = st.columns([1, 1])
            col_btn.button("✨ Recommend Portfolio", on_click=apply_recommendation)
            col_chk.checkbox("Force Reset (Ignore current values)", value=False, key='force_reset_rec', help="Check this to discard current slider values and generate a fresh recommendation from scratch.")
            
            # Show success/error messages after rerun
            if st.session_state.get('portfolio_recommended', False):
                st.success("Portfolio Recommended!")
                st.session_state.portfolio_recommended = False # Reset flag
            if st.session_state.get('portfolio_error', None):
                st.warning(st.session_state.portfolio_error)
                st.session_state.portfolio_error = None # Reset error
            
            # Display matched projects if available
            if st.session_state.get('matched_projects'):
                st.markdown("---")
                st.markdown("#### 📋 Suggested Projects from ERCOT Queue")
                st.markdown("*Based on recommended portfolio capacities*")
                
                matched = st.session_state.matched_projects
                
                for tech, projects in matched.items():
                    if projects:
                        with st.expander(f"**{tech}** Projects ({len(projects)} suggested)", expanded=False):
                            # Create DataFrame for display
                            proj_data = []
                            for proj in projects:
                                proj_data.append({
                                    'Project Name': proj['name'],
                                    'Capacity (MW)': f"{proj['capacity_mw']:.1f}",
                                    'County': proj['county'],
                                    'Status': proj['status'],
                                    'Proj. COD': str(proj['projected_cod'])[:10] if proj['projected_cod'] != 'Unknown' else 'TBD'
                                })
                            
                            if proj_data:
                                proj_df = pd.DataFrame(proj_data)
                                st.dataframe(proj_df, hide_index=True, use_container_width=True)
            

        st.markdown("#### Custom Profiles (Upload Unit Profiles)")
        c_prof_1, c_prof_2 = st.columns(2)
        uploaded_solar_file = c_prof_1.file_uploader("Upload Solar Profile (CSV)", type=['csv', 'txt'])
        uploaded_wind_file = c_prof_2.file_uploader("Upload Wind Profile (CSV)", type=['csv', 'txt'])


    # --- Tab 3: Financials ---
    with tab_fin:
        st.markdown("#### PPA Prices ($/MWh)")
        c_fin_1, c_fin_2, c_fin_3 = st.columns(3)
        with c_fin_1:
            solar_price = st.number_input("Solar PPA Price", min_value=0.0, value=46.5, step=1.0, key='solar_price_input', help="Q4 2024 Market: $45-47. Adjusted down ~2% due to high saturation.")
            wind_price = st.number_input("Wind PPA Price", min_value=0.0, value=54.0, step=1.0, key='wind_price_input', help="Q4 2024 Market: ~$54. Up ~3.3%. Trades at $8-10 premium over solar.")
        with c_fin_2:
            ccs_price = st.number_input("CCS Gas PPA Price", min_value=0.0, value=65.0, step=1.0, key='ccs_price_input', help="Updated 2025 Market est: $55-75 (w/ 45Q)")
            geo_price = st.number_input("Geothermal PPA Price", min_value=0.0, value=77.5, step=1.0, key='geo_price_input', help="Updated 2025 Market est: $70-85")
        with c_fin_3:
            nuc_price = st.number_input("Nuclear PPA Price", min_value=0.0, value=112.0, step=1.0, key='nuc_price_input', help="Q4 2024 Market: ~$112. Based on recent Vistra data center deal. Firm clean power premium.")
        
        st.markdown("#### Battery Configuration")
        
        # Sizing Section
        c_bat_size_1, c_bat_size_2, c_bat_size_3 = st.columns(3)
        with c_bat_size_1:
             enable_battery = st.checkbox("Enable Battery Storage", value=True)
        with c_bat_size_2:
             batt_capacity = st.number_input("Battery Power (MW)", min_value=0.0, step=1.0, key='batt_input', disabled=not enable_battery)
        with c_bat_size_3:
             batt_duration = st.number_input("Battery Duration (Hours)", min_value=0.5, value=2.0, step=0.5, key='batt_duration_input', disabled=not enable_battery)

        st.markdown("**Contract Terms**")
        c_bat_1, c_bat_2 = st.columns(2)
        with c_bat_1:
            batt_base_rate = st.number_input("Base Capacity Rate ($/MW-mo)", value=8000.0, step=500.0, help="~8/kW-mo. 2025 estimates range $6k-10k depending on duration and location.")
            batt_guar_avail = st.number_input("Guaranteed Availability (%)", value=0.98, step=0.01, min_value=0.0, max_value=1.0, help="Owner guarantees this uptime.")
        with c_bat_2:
            batt_guar_rte = st.number_input("Guaranteed RTE (%)", value=0.85, step=0.01, min_value=0.0, max_value=1.0, help="Round Trip Efficiency guarantee.")
            batt_vom = st.number_input("Variable O&M ($/MWh)", value=2.0, step=0.1, help="Wear and tear charge per MWh discharged.")
            
        simulate_outages = st.checkbox("Simulate Random Battery Outages (~2%)", value=False, help="Randomly drop availability to test performance penalties.")

        st.markdown("---")
        st.markdown("#### Market Assumptions")
        c_mkt_1, c_mkt_2, c_mkt_3, c_mkt_4 = st.columns(4)
        
        # Get base average from actual data
        _, base_market_avg = generate_dummy_price_profile(32.0, return_base_avg=True)
        market_price = base_market_avg  # Use the actual base average
        
        # Display base average (read-only)
        c_mkt_1.metric(
            "Base Avg (2024)", 
            f"${base_market_avg:.2f}",
            help="Average from 2024 ERCOT HB_NORTH data"
        )
        
        price_scaler = c_mkt_2.number_input(
            "Price Scaler", 
            min_value=0.1, 
            max_value=5.0, 
            value=1.0, 
            step=0.1, 
            key='price_scaler_input',
            help="Multiplier for 2024 prices"
        )
        
        # Show scaled price
        scaled_price = base_market_avg * price_scaler
        c_mkt_3.metric(
            "Scaled Avg",
            f"${scaled_price:.2f}",
            delta=f"{(price_scaler-1)*100:+.0f}%",
            help="Base × Scaler = Effective market price"
        )
        
        rec_price = c_mkt_4.number_input("REC Price ($/MWh)", min_value=0.0, value=3.50, step=0.5, key='rec_input', help="Market est: $2-4/MWh")


# --- Global Settings (Sidebar) ---


# Forces Dark Mode Permanently
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    /* Force text color for common elements to ensure readability */
    h1, h2, h3, p, label {
        color: #FAFAFA !important;
    }
    /* Specific overrides for inputs to be visible */
    .stTextInput input, .stNumberInput input, .stSelectbox div[role="combobox"] {
        background-color: #262730; 
        color: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)

chart_template = 'plotly_dark'
chart_bg = '#0E1117'
chart_font_color = '#FAFAFA'


# Update session state from inputs
st.session_state.solar_cap = solar_capacity
st.session_state.wind_cap = wind_capacity
st.session_state.ccs_cap = ccs_capacity
st.session_state.geo_cap = geo_capacity
st.session_state.nuc_cap = nuc_capacity
st.session_state.batt_cap = batt_capacity

# --- Main Content ---

# 1. Calculate Aggregated Load
if not st.session_state.participants and not uploaded_load_file:
    st.info("Please add load participants in the sidebar OR upload a CSV to begin.")
else:
    # Aggregate Load
    if uploaded_load_file:
        # Process Upload
        total_load_profile = process_uploaded_profile(uploaded_load_file, keywords=['load', 'mw', 'mwh', 'demand'])
        if total_load_profile is None:
             st.error("Could not parse uploaded file. Ensure column 'Load' or numeric data exists.")
             st.stop()
    else:
        # Logic for participants
        total_load_profile = pd.Series(0.0, index=range(8760))
        for p in st.session_state.participants:
            profile = generate_dummy_load_profile(p['load'], p['type'])
            total_load_profile += profile
        
    total_annual_load = total_load_profile.sum()
    
    # 2. Calculate Generation
    
    # Solar
    if uploaded_solar_file:
        solar_unit_profile = process_uploaded_profile(uploaded_solar_file, keywords=['solar', 'pv', 'generation', 'output'])
        if solar_unit_profile is not None:
             # Scale by capacity input
             solar_profile = solar_unit_profile * solar_capacity
        else:
             st.error("Error parsing Solar file.")
             st.stop()
    else:
        solar_profile = generate_dummy_generation_profile(solar_capacity, 'Solar', use_synthetic=use_synthetic_solar)

    # Wind
    if uploaded_wind_file:
        wind_unit_profile = process_uploaded_profile(uploaded_wind_file, keywords=['wind', 'turbine', 'generation', 'output'])
        if wind_unit_profile is not None:
             wind_profile = wind_unit_profile * wind_capacity
        else:
             st.error("Error parsing Wind file.")
             st.stop()
    else:
        wind_profile = generate_dummy_generation_profile(wind_capacity, 'Wind')

    # Geothermal / Nuclear (still dummy for now, rare to resize profile shape)
    ccs_profile = generate_dummy_generation_profile(ccs_capacity, 'CCS Gas')
    geo_profile = generate_dummy_generation_profile(geo_capacity, 'Geothermal')
    nuc_profile = generate_dummy_generation_profile(nuc_capacity, 'Nuclear')
    
    total_gen_profile = solar_profile + wind_profile + ccs_profile + geo_profile + nuc_profile
    
    # 3. Calculate Surplus/Deficit BEFORE Battery
    surplus = (total_gen_profile - total_load_profile).clip(lower=0)
    deficit = (total_load_profile - total_gen_profile).clip(lower=0)
    
    # 4. Simulate Battery
    
    # Generate Availability Profile (for Outage Simulation)
    availability_profile = pd.Series(batt_capacity, index=range(8760))
    if simulate_outages and batt_capacity > 0:
        # Create random outages (~2% of year = ~175 hours)
        # Use a fixed seed for reproducibility of the "random" outages in this session
        rng_outage = np.random.default_rng(42)
        outage_mask = rng_outage.random(8760) < 0.02 # 2% probability
        availability_profile[outage_mask] = 0.0 # Full outage for that hour
    
    if enable_battery:
        batt_discharge, batt_soc, batt_charge = simulate_battery_storage(surplus, deficit, batt_capacity, batt_duration, availability_profile)
    else:
        batt_discharge = pd.Series(0.0, index=range(8760))
        batt_charge = pd.Series(0.0, index=range(8760))
        batt_soc = pd.Series(0.0, index=range(8760))
        availability_profile = pd.Series(0.0, index=range(8760))
    
    # 5. Final Matching
    total_gen_with_battery = total_gen_profile + batt_discharge
    cfe_score, matched_profile = calculate_cfe_score(total_load_profile, total_gen_with_battery)
    
    # Calculate detailed metrics
    total_gen_capacity = solar_capacity + wind_capacity + ccs_capacity + geo_capacity + nuc_capacity
    metrics = calculate_portfolio_metrics(total_load_profile, matched_profile, total_gen_capacity)
    
    # Financials
    # Financials
    tech_profiles = {
        'Solar': solar_profile,
        'Wind': wind_profile,
        'CCS Gas': ccs_profile,
        'Geothermal': geo_profile,
        'Nuclear': nuc_profile,
        'Battery': batt_discharge
    }
    
    # --- Financial Analysis ---
    st.markdown("---")
    st.markdown("#### Economic Analysis")
    
    # Calculate Effective Battery Price ($/MWh) from Capacity Payment ($/kW-mo)
    # Input: batt_price ($/kW-mo)
    # Annual Cost per MW = batt_price * 1000 kW/MW * 12 months
    # Annual Cost = (batt_price * 12000) * batt_capacity
    # Effective $/MWh = Annual Cost / Total Annual Discharge MWh
    
    total_discharge_mwh = batt_discharge.sum()
    
    # Helper to generate market price series for the financial calc
    market_price_profile_series = generate_dummy_price_profile(market_price) * price_scaler
    
    batt_ops_data = {
        'available_mw_profile': availability_profile,
        'discharge_mwh_profile': batt_discharge,
        'charge_mwh_profile': batt_charge,
        'market_price_profile': market_price_profile_series
    }

    if total_discharge_mwh > 0:
        # Calculate Battery Financials Detailed
        batt_contract_params = {
            'capacity_mw': batt_capacity,
            'base_rate_monthly': batt_base_rate,
            'guaranteed_availability': batt_guar_avail,
            'guaranteed_rte': batt_guar_rte,
            'vom_rate': batt_vom
        }
        
        batt_financials = calculate_battery_financials(batt_contract_params, batt_ops_data)
        
        # Effective Price for Global Financials logic (Net Cost / MWh)
        effective_batt_price_mwh = batt_financials['net_invoice'] / total_discharge_mwh
    else:
        effective_batt_price_mwh = 0.0
        batt_financials = {
            'net_invoice': 0.0, 'capacity_payment': 0.0, 
            'vom_payment': 0.0, 'rte_penalty': 0.0,
            'actual_availability': 1.0, 'actual_rte': 0.0
        }
        
    tech_prices = {
        'Solar': solar_price,
        'Wind': wind_price,
        'CCS Gas': ccs_price,
        'Geothermal': geo_price,
        'Nuclear': nuc_price,
        'Battery': effective_batt_price_mwh # Use calculated effective price
    }
    
    fin_metrics = calculate_financials(matched_profile, deficit, tech_profiles, tech_prices, market_price, rec_price, price_scaler)
    
    # --- Dashboard ---
    
    # Metrics - Row 1
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Electricity Usage", f"{total_annual_load:,.0f} MWh")
    col2.metric("Clean Energy Generation", f"{total_gen_profile.sum():,.0f} MWh", help="Total renewable generation + nuclear")
    col3.metric("CFE Score (24/7)", f"{metrics['cfe_score']:.1%}", help="Percentage of total load met by Carbon Free Energy generation in the same hour")
    col4.metric("Battery Discharge", f"{batt_discharge.sum():,.0f} MWh")
    
    # Metrics - Row 2
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("MW Match Productivity", f"{metrics['productivity']:,.0f} MWh/MW", help="MWh of Clean Energy Matched per MW of Installed Capacity")
    col6.metric("Loss of Green Hours", f"{metrics['logh']:.1%}", help="% of hours where load is not fully matched by clean energy")
    col7.metric("Grid Consumption", f"{metrics['grid_consumption']:,.0f} MWh", help="Total energy drawn from grid (deficit)")
    col8.metric("Excess Generation", f"{surplus.sum():,.0f} MWh", help="Gross overgeneration before battery charging")
    
    # Metrics - Row 3 (Financials)
    st.subheader("Financial Overview")
    col9, col10, col11, col12 = st.columns(4)
    col9.metric("Annual PPA Settlement Value", f"${fin_metrics['settlement_value']:,.0f}", help="Annual Revenue (or Cost) from PPA Settlement: (Market - Strike) * Matched Vol")
    col10.metric("Weighted Avg PPA Price", f"${fin_metrics['weighted_ppa_price']:.2f}/MWh", help="Average cost of matched energy based on technology mix")
    col11.metric("Capture Value (2024 Base)", f"${fin_metrics['weighted_market_price']:.2f}/MWh", help="Average market value of matched energy (2024 ERCOT prices × scaler)")
    col12.metric("REC Value", f"${fin_metrics['rec_cost']:,.0f}", help="Value of RECs")

    # Battery Financials Detailed Section
    if enable_battery and batt_capacity > 0:
        st.markdown("---")
        st.subheader("🔋 Battery Settlement Detailed")
        
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        
        b_col1.metric("Net Annual Invoice", f"${batt_financials['net_invoice']:,.0f}", 
                      delta=f"-${batt_financials['rte_penalty']:,.0f} Penalty" if batt_financials['rte_penalty'] > 0 else None,
                      help="Total Payment to Owner = Capacity + VOM - Penalties")
        
        b_col2.metric("Capacity Payment", f"${batt_financials['capacity_payment']:,.0f}",
                      help=f"Base: ${batt_capacity * batt_base_rate * 12:,.0f} adjusted for Avail {batt_financials['actual_availability']:.1%}")
                      
        b_col3.metric("VOM Payment", f"${batt_financials['vom_payment']:,.0f}",
                      help=f"Usage Fee based on {batt_financials['total_discharged']:,.0f} MWh discharged")
                      
        b_col4.metric("Realized RTE", f"{batt_financials['actual_rte']:.1%}",
                      delta=f"{batt_financials['actual_rte'] - batt_guar_rte:.1%}",
                      help=f"Target: {batt_guar_rte:.1%}")
                      

    
    # --- Value Stack for Buyer (Tolling Model) ---
    st.markdown("---")
    with st.expander("💼 Buyer's P&L (Tolling Model)", expanded=True):
        st.markdown("""
        **Perspective**: The "Buyer" rents the battery for a fixed monthly toll and keeps the market revenue (Arbitrage + Ancillary).
        """)
        
        c_buy_1, c_buy_2, c_buy_3 = st.columns(3)
        
        with c_buy_1:
            toll_rate = st.number_input("Fixed Toll Rate ($/MW-mo)", value=7500.0, step=500.0, help="Monthly rent paid to owner.")
            
        with c_buy_2:
            ancillary_est = st.number_input("Est. Ancillary Revenue ($/MW-mo)", value=3000.0, step=500.0, help="Revenue from ECRS/Reg-Up etc.")
        
        with c_buy_3:
            charge_source = st.selectbox("Charging Cost Source", ["Grid (LMP)", "Solar PPA", "Wind PPA"], help="Cost assumption for charging energy.")
        
        # Prepare Data for Calculation
        cost_profile = None
        if charge_source == "Solar PPA":
            # Create a Series with the PPA price
            cost_profile = pd.Series(solar_price, index=range(8760))
        elif charge_source == "Wind PPA":
            cost_profile = pd.Series(wind_price, index=range(8760))
        
        # Use calculate_buyer_pl
        # Need to ensure inputs are float/series as expected
        buyer_pl_df = calculate_buyer_pl(
            batt_ops_data, 
            batt_capacity, 
            toll_rate, 
            ancillary_est, 
            cost_profile
        )
        
        # Summarize Results
        total_profit = buyer_pl_df['Net_Profit'].sum()
        best_month = buyer_pl_df.loc[buyer_pl_df['Net_Profit'].idxmax()]
        worst_month = buyer_pl_df.loc[buyer_pl_df['Net_Profit'].idxmin()]
        
        # Metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total Annual Profit", f"${total_profit:,.0f}", delta_color="normal" if total_profit > 0 else "inverse")
        m_col2.metric(f"Best Month ({best_month['Month']})", f"${best_month['Net_Profit']:,.0f}")
        m_col3.metric(f"Worst Month ({worst_month['Month']})", f"${worst_month['Net_Profit']:,.0f}")
        
        st.markdown("#### Monthly Profit/Loss Breakdown")
        
        # Chart
        import plotly.graph_objects as go
        
        fig_buyer = go.Figure()
        
        # Revenues (Positive)
        fig_buyer.add_trace(go.Bar(
            x=buyer_pl_df['Month'], 
            y=buyer_pl_df['Revenue_Arb'],
            name='Arbitrage Rev',
            marker_color='#2ca02c'
        ))
        
        fig_buyer.add_trace(go.Bar(
            x=buyer_pl_df['Month'], 
            y=buyer_pl_df['Ancillary_Rev'],
            name='Ancillary Rev',
            marker_color='#98df8a'
        ))
        
        # Costs (Negative)
        fig_buyer.add_trace(go.Bar(
            x=buyer_pl_df['Month'], 
            y=-buyer_pl_df['Toll_Cost'],
            name='Fixed Toll',
            marker_color='#d62728'
        ))
        
        fig_buyer.add_trace(go.Bar(
            x=buyer_pl_df['Month'], 
            y=-buyer_pl_df['Cost_Charge'],
            name='Charging Cost',
            marker_color='#ff9896'
        ))
        
        # Net Profit Line
        fig_buyer.add_trace(go.Scatter(
            x=buyer_pl_df['Month'],
            y=buyer_pl_df['Net_Profit'],
            name='Net Profit',
            line=dict(color='white', width=3, dash='dot'),
            mode='lines+markers'
        ))
        
        fig_buyer.update_layout(
            barmode='relative', 
            title='Monthly Buyer P&L (Waterfall)',
            yaxis_title='Profit / Loss ($)',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_buyer, use_container_width=True)
        
        st.info(
            "💡 **Why take this risk?** "
            "Buyers (trading houses) assume this risk to capture 'Volatile' months (e.g., August Heatwaves) where arbitrage revenue can vastly exceed the fixed toll. "
            "In 'Boring' months, they may lose money (Net Loss), but the annual potential offsets these losses."
        )

    # Charts
    st.markdown("---")
    st.subheader("Hourly Energy Balance")
    
    # Use columns to make the slider more compact
    # Create Datetime Index for charting (2024 for leap year support logic if needed, but using 8760 standard)
    # matching the 8760 length
    datetime_index = pd.date_range(start='2024-01-01', periods=8760, freq='h')

    # View Controls
    col_view_type, col_date_picker = st.columns([1, 2])
    
    with col_view_type:
        view_mode = st.radio("View Period", ["Full Year", "Select Week"], horizontal=True)
        
    start_hour = 0
    end_hour = 8760
    title_suffix = "(Full Year)"
    
    if view_mode == "Select Week":
        with col_date_picker:
            # Default to a summer week (July 1)
            default_date = datetime.date(2024, 7, 1)
            selected_date = st.date_input("Select Week Start", value=default_date, 
                                          min_value=datetime.date(2024, 1, 1), 
                                          max_value=datetime.date(2024, 12, 24))
            
            # Convert date to hour index
            # day_of_year starts at 1
            day_of_year = selected_date.timetuple().tm_yday
            start_hour = (day_of_year - 1) * 24
            end_hour = start_hour + 168 # 7 days * 24 hours
            
            # Safety clamp
            if end_hour > 8760:
                end_hour = 8760
                
            title_suffix = f"({selected_date.strftime('%b %d')} - {(selected_date + datetime.timedelta(days=7)).strftime('%b %d')})"

    # Prepare Data Slices
    # Use the datetime index for X-axis to show actual dates/times
    x_axis = datetime_index[start_hour:end_hour]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=total_load_profile[start_hour:end_hour],
                             mode='lines', name='Aggregated Load', line=dict(color='red', width=2)))
    
    # Total Generation Line
    fig.add_trace(go.Scatter(x=x_axis, y=total_gen_profile[start_hour:end_hour],
                             mode='lines', name='Total Clean Energy', line=dict(color='#2ca02c', width=2)))
    # Stacked generation profiles
    fig.add_trace(go.Scatter(x=x_axis, y=solar_profile[start_hour:end_hour], name='Solar Gen', stackgroup='one', line=dict(color='gold'), fill='tonexty'))
    fig.add_trace(go.Scatter(x=x_axis, y=wind_profile[start_hour:end_hour], name='Wind Gen', stackgroup='one', line=dict(color='lightblue'), fill='tonexty'))
    fig.add_trace(go.Scatter(x=x_axis, y=ccs_profile[start_hour:end_hour], name='CCS Gas Gen', stackgroup='one', line=dict(color='brown'), fill='tonexty'))
    fig.add_trace(go.Scatter(x=x_axis, y=geo_profile[start_hour:end_hour], name='Geothermal Gen', stackgroup='one', line=dict(color='red'), fill='tonexty'))
    fig.add_trace(go.Scatter(x=x_axis, y=nuc_profile[start_hour:end_hour], name='Nuclear Gen', stackgroup='one', line=dict(color='purple'), fill='tonexty'))
    fig.add_trace(go.Scatter(x=x_axis, y=batt_discharge[start_hour:end_hour], name='Battery Discharge', stackgroup='one', line=dict(color='#1f77b4'), fill='tonexty'))

    fig.update_layout(
        title=f"Load vs. Matched Generation {title_suffix}", 
        xaxis_title="Date / Time", 
        yaxis_title="Power (MW)", 
        template=chart_template,
        paper_bgcolor=chart_bg,
        plot_bgcolor=chart_bg,
        font=dict(color=chart_font_color),
        xaxis=dict(
            tickformat="%b %d<br>%H:%M" if view_mode == "Select Week" else "%b"
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Monthly Analysis")
    
    # Group by month
    # Create a simple dataframe for grouping
    df_hourly = pd.DataFrame({
        'Load': total_load_profile,
        'Generation': total_gen_profile,
        'Battery': batt_discharge,
        'Matched': matched_profile
    })
    
    df_hourly['Month'] = pd.date_range(start='2024-01-01', periods=8760, freq='h').month
    monthly_stats = df_hourly.groupby('Month').sum()
    
    # Map month numbers to names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_stats.index = month_names
    
    # Calculate percentages for labels
    monthly_stats['Matched_Pct'] = (monthly_stats['Matched'] / monthly_stats['Load']).apply(lambda x: f"{x:.0%}")
    
    fig_bar = go.Figure()
    
    # 1. Total Generation (Background)
    fig_bar.add_trace(go.Bar(x=monthly_stats.index, y=monthly_stats['Generation'], name='Total Clean Energy', marker_color='#2ca02c', opacity=0.6)) # Standard Green
    
    # 2. Matched Energy (Middle)
    fig_bar.add_trace(go.Bar(
        x=monthly_stats.index, 
        y=monthly_stats['Matched'], 
        name='Hourly Matched Clean Energy', 
        marker_color='#FFA500'
    )) # Orange without labels
    
    # 3. Battery (Foreground - On top of Matched)
    fig_bar.add_trace(go.Bar(x=monthly_stats.index, y=monthly_stats['Battery'], name='Battery Discharge', marker_color='#1f77b4')) # Standard blue
    
    # 4. Load (Line - Top)
    fig_bar.add_trace(go.Scatter(x=monthly_stats.index, y=monthly_stats['Load'], name='Load', mode='lines', line=dict(color='red', width=3)))

    # 5. Percentage Labels (Absolute Top Layer)
    fig_bar.add_trace(go.Scatter(
        x=monthly_stats.index, 
        y=monthly_stats['Matched'], 
        mode='text',
        name='Matched %',
        text=monthly_stats['Matched_Pct'],
        textposition='top center',
        textfont=dict(
            family="Arial Black",
            size=14,
            color="white"
        ),
        showlegend=False
    ))
    
    fig_bar.update_layout(
        title="Monthly Energy Totals", 
        xaxis_title="Month", 
        yaxis_title="Energy (MWh)", 
        barmode='overlay', 
        template=chart_template,
        paper_bgcolor=chart_bg,
        plot_bgcolor=chart_bg,
        font=dict(color=chart_font_color),
        legend=dict(traceorder='reversed')
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Heatmap Analysis ---
    st.subheader("Heatmap Analysis")
    # Prepare matrix (Hour of Day x Day of Year)
    # 8760 hours -> 365 days x 24 hours
    
    days = 365
    hours_per_day = 24
    
    # CFE Heatmap Logic
    # 1 if Matched >= Load (Green), 0 if Deficit (Red) ?
    # Let's show "Deficit Magnitude" or just "Matched %"
    # Actually, binary "Green/Red" is often most useful for 24/7.
    # Let's do: Matched Energy / Load (capped at 1.0)
    
    # Reshape array
    # Data must be 24 rows (hours) x 365 cols (days) for typical heatmap
    # Slice first 8760
    
    # Safety Check for array handling
    clean_load = total_load_profile.values[:8760]
    clean_matched = matched_profile.values[:8760]
    
    # Calculate % met
    # Avoid div by zero
    percent_met = np.divide(clean_matched, clean_load, out=np.zeros_like(clean_matched), where=clean_load!=0)
    
    z_data = percent_met.reshape((365, 24)).T # Transpose to get 24 rows, 365 cols
    
    # Generate labels
    dates_x = pd.date_range(start='2024-01-01', periods=365, freq='D')
    times_y = [datetime.time(h).strftime('%I %p') for h in range(24)] # "12 AM", "01 AM"...
    # Cleanup time labels to remove leading zeros if preferred, e.g. "1 AM"
    times_y = [t.lstrip('0') for t in times_y]

    fig_heat = go.Figure(data=go.Heatmap(
        z=z_data,
        x=dates_x,
        y=times_y,
        colorscale='RdYlGn', # Red to Green
        hovertemplate='Date: %{x|%b %d}<br>Time: %{y}<br>Matched: %{z:.1%}<extra></extra>'
    ))
    
    fig_heat.update_layout(
        title="24/7 Matching Heatmap", 
        xaxis_title="Date", 
        yaxis_title="Time of Day", 
        template=chart_template,
        paper_bgcolor=chart_bg,
        plot_bgcolor=chart_bg,
        font=dict(color=chart_font_color)
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # --- PPA vs Capture Value Analysis (Last Chart) ---
    st.markdown("---")
    st.subheader("Economic Analysis")
    st.markdown("PPA prices vs market capture values for each technology")
    
    # Calculate capture value for each technology
    market_price_profile = generate_dummy_price_profile(market_price) * price_scaler
    
    tech_data = []
    tech_capacities = {
        'Solar': solar_capacity,
        'Wind': wind_capacity,
        'CCS Gas': ccs_capacity,
        'Geothermal': geo_capacity,
        'Nuclear': nuc_capacity,
        'Battery': batt_capacity
    }
    
    for tech, capacity in tech_capacities.items():
        if capacity > 0:
            # Get generation profile for this tech
            if tech in tech_profiles:
                tech_gen = tech_profiles[tech]
                # Calculate capture value (time-weighted average market price)
                capture_value = (tech_gen * market_price_profile).sum() / tech_gen.sum() if tech_gen.sum() > 0 else 0
            else:
                capture_value = 0
            
            # Get PPA price
            ppa_price = tech_prices.get(tech, 0)
            
            tech_data.append({
                'Technology': tech,
                'PPA Price': ppa_price,
                'Capture Value': capture_value,
                'Spread': capture_value - ppa_price
            })
    
    if tech_data:
        # Create grouped bar chart
        tech_df = pd.DataFrame(tech_data)
        
        fig_ppa = go.Figure()
        
        fig_ppa.add_trace(go.Bar(
            name='PPA Price',
            x=tech_df['Technology'],
            y=tech_df['PPA Price'],
            marker_color='#3498db',  # Blue instead of red
            text=tech_df['PPA Price'].round(2),
            textposition='outside',
            texttemplate='$%{text:.2f}',
            textfont=dict(size=14, color='white')
        ))
        
        fig_ppa.add_trace(go.Bar(
            name='Capture Value (2024 Base)',
            x=tech_df['Technology'],
            y=tech_df['Capture Value'],
            marker_color='#f39c12',  # Orange instead of green
            text=tech_df['Capture Value'].round(2),
            textposition='outside',
            texttemplate='$%{text:.2f}',
            textfont=dict(size=14, color='white')
        ))
        
        fig_ppa.update_layout(
            barmode='group',
            xaxis_title='Technology',
            yaxis_title='Price ($/MWh)',
            legend=dict(x=0.01, y=0.99),
            height=500,  # Increased from 400 for more space
            hovermode='x unified',
            template=chart_template,
            paper_bgcolor=chart_bg,
            plot_bgcolor=chart_bg,
            font=dict(color=chart_font_color),
            margin=dict(t=50, b=50, l=50, r=50)  # Add margins for labels
        )
        
        st.plotly_chart(fig_ppa, use_container_width=True)
        
        # Show spread analysis
        st.markdown("**Value Spread** (Capture Value - PPA Price)")
        spread_cols = st.columns(len(tech_data))
        for idx, row in enumerate(tech_data):
            with spread_cols[idx]:
                spread_val = row['Spread']
                st.metric(
                    row['Technology'],
                    f"${spread_val:.2f}/MWh",
                    delta=f"{'+' if spread_val >= 0 else ''}{spread_val:.2f}"
                )
    else:
        st.info("Add generation capacity to see PPA vs Capture Value comparison")

    # --- Battery Waterfall (Moved to Bottom) ---
    if enable_battery and batt_capacity > 0:
        st.markdown("---")
        st.subheader("Battery Financial Settlement")
        st.markdown("Breakdown of battery revenue components and penalties.")
        
        # Waterfall Chart
        fig_batt = go.Figure(go.Waterfall(
            name = "Settlement", orientation = "v",
            measure = ["relative", "relative", "relative", "total"],
            x = ["Capacity Payment", "VOM Payment", "RTE Penalty", "Net Invoice"],
            textposition = "outside",
            text = [f"${batt_financials['capacity_payment']/1000:.0f}k", 
                    f"${batt_financials['vom_payment']/1000:.0f}k", 
                    f"-${batt_financials['rte_penalty']/1000:.0f}k", 
                    f"${batt_financials['net_invoice']/1000:.0f}k"],
            y = [batt_financials['capacity_payment'], 
                 batt_financials['vom_payment'], 
                 -batt_financials['rte_penalty'], 
                 0],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))

        fig_batt.update_layout(
                title = "Battery Settlement Waterfall",
                showlegend = False,
                template=chart_template,
                paper_bgcolor=chart_bg,
                plot_bgcolor=chart_bg,
                font=dict(color=chart_font_color),
                height=500, # Consistent height
        )

        st.plotly_chart(fig_batt, use_container_width=True)

        # --- Pro Forma Table ---
        st.markdown("#### 📊 Battery Pro Forma")
        
        # Calculate Market Value items
        # Charging Cost = Sum(Charge_MW * Price)
        total_charge_cost = (batt_charge * market_price_profile_series).sum()
        
        # Revenue (Discharge Value) = Sum(Discharge_MW * Price)
        total_discharge_value = (batt_discharge * market_price_profile_series).sum()
        
        # Net Benefit
        total_expense = batt_financials['capacity_payment'] + batt_financials['vom_payment'] + total_charge_cost - batt_financials['rte_penalty']
        net_benefit = total_discharge_value - total_expense
        
        pro_forma_data = [
            {"Category": "Revenue", "Item": "Energy Arbitrage (Market Value)", "Amount": total_discharge_value, "Notes": "Value of energy sent to grid"},
            {"Category": "Expense", "Item": "Capacity Payment (Lease)", "Amount": -batt_financials['capacity_payment'], "Notes": "Fixed cost for capacity"},
            {"Category": "Expense", "Item": "Charging Cost", "Amount": -total_charge_cost, "Notes": "Cost of energy from grid/gen"},
            {"Category": "Expense", "Item": "VOM Charges", "Amount": -batt_financials['vom_payment'], "Notes": "Variable usage fee"},
            {"Category": "Contra-Expense", "Item": "Performance Penalties", "Amount": batt_financials['rte_penalty'], "Notes": "Credit for underperformance"},
            {"Category": "Net", "Item": "NET BENEFIT (Profit/Loss)", "Amount": net_benefit, "Notes": "Total Economic Value"}
        ]
        
        pf_df = pd.DataFrame(pro_forma_data)
        
        # Formatting
        st.dataframe(
            pf_df.style.format({"Amount": "${:,.0f}"}), 
            hide_index=True, 
            use_container_width=True
        )

    # --- Data Export ---
    st.subheader("Export Results")
    
    # Create Datetime Index for 2024 (Leap year handling if needed, but standard 8760 usually implies non-leap or truncated)
    # Using 2024 implies leap year (8784 hours), but our arrays are 8760. 
    # Let's use 2023 to be safe for 8760, or just truncate 2024.
    # Actually, let's just use a generic range or 2023 (not leap).
    datetime_index = pd.date_range(start='2024-01-01', periods=8760, freq='h')

    results_df = pd.DataFrame({
        'Datetime': datetime_index,
        'Load_MW': total_load_profile,
        'Matched_MW': matched_profile,
        'Solar_MW': solar_profile,
        'Wind_MW': wind_profile,
        'Geothermal_Gen_MW': geo_profile,
        'Nuclear_Gen_MW': nuc_profile,
        'CCS_Gas_Gen_MW': ccs_profile,
        'Total_Raw_Gen_MW': total_gen_profile,
        'Battery_Discharge_MW': batt_discharge,
        'Battery_State_of_Charge_MWh': batt_soc,
        'Grid_Deficit_MW': deficit,
        'Surplus_MW': surplus
    })

    # --- Financial Columns for CSV ---
    # 1. Market Price (Hourly)
    market_price_profile = generate_dummy_price_profile(market_price)
    results_df['Market_Capture_Price_$/MWh'] = market_price_profile
    
    # 2. Technology PPA Prices (Constant)
    results_df['Solar_PPA_Price'] = solar_price
    results_df['Wind_PPA_Price'] = wind_price
    results_df['Geothermal_PPA_Price'] = geo_price
    results_df['Nuclear_PPA_Price'] = nuc_price
    results_df['CCS_Gas_PPA_Price'] = ccs_price
    
    # Recalculate effective battery price for CSV if local variable not available in scope
    # (Though it should be available since defined above in main script flow)
    # Using 'effective_batt_price_mwh' calculated earlier 
    results_df['Battery_Adder_Price'] = effective_batt_price_mwh
    
    # 3. Hourly Blended PPA Price
    # Cost = Sum(Gen_i * Price_i)
    # Price = Cost / TotalGen
    
    # Calculate Total Generation Cost per Hour
    # (Using the profiles already in the DF for consistency)
    hourly_gen_cost = (
        results_df['Solar_MW'] * solar_price +
        results_df['Wind_MW'] * wind_price +
        results_df['Geothermal_Gen_MW'] * geo_price +
        results_df['Nuclear_Gen_MW'] * nuc_price +
        results_df['CCS_Gas_Gen_MW'] * ccs_price + 
        # Use EFFECTIVE battery price (converted to energy)
        results_df['Battery_Discharge_MW'] * effective_batt_price_mwh
    )
    
    # Total Energy Serving Load (Matched + Surplus? Or just Total Generated?)
    # "Blended PPA Price" usually means the price of the energy produced.
    total_production = results_df['Total_Raw_Gen_MW'] + results_df['Battery_Discharge_MW']
    
    # Avoid zero division
    results_df['Hourly_Blended_PPA_Price_$/MWh'] = hourly_gen_cost / total_production
    results_df['Hourly_Blended_PPA_Price_$/MWh'] = results_df['Hourly_Blended_PPA_Price_$/MWh'].fillna(0.0) # Handle hours with 0 gen
    
    csv = results_df.to_csv(index=False).encode('utf-8')

    # Create ZIP buffer
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        # Add CSV results
        zf.writestr("simulation_results.csv", csv)
        
        # Create Scenario Configuration
        scenario_config = {
            "region": "ERCOT North",
            "solar_capacity": solar_capacity,
            "wind_capacity": wind_capacity,
            "geo_capacity": geo_capacity,
            "nuc_capacity": nuc_capacity,
            "batt_capacity": batt_capacity,
            "batt_duration": batt_duration,
            "solar_price": solar_price,
            "wind_price": wind_price,
            "ccs_price": ccs_price,
            "geo_price": geo_price,
            "nuc_price": nuc_price,
            "batt_base_rate": batt_base_rate,
            "batt_guar_avail": batt_guar_avail,
            "batt_guar_rte": batt_guar_rte,
            "batt_vom": batt_vom,
            "market_price": market_price,
            "rec_price": rec_price,
            # Extract participants from session state
            "participants": st.session_state.participants,
            # Store exclusions
            "excluded_techs": st.session_state.get('excluded_techs', [])
        }
        
        # Add JSON config
        json_str = json.dumps(scenario_config, indent=4)
        zf.writestr("scenario_config.json", json_str)
        
    st.download_button(
        label="Download Results & Scenario (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="simulation_results.zip",
        mime="application/zip"
    )
    
    with st.expander("Preview Data"):
        st.dataframe(results_df, use_container_width=True)
    
