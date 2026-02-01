import streamlit as st # Fresh build trigger v2
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
    generate_load_factor_profile,
    calculate_cfe_score,
    simulate_battery_storage,
    recommend_portfolio,
    calculate_portfolio_metrics,
    calculate_financials,
    process_uploaded_profile,
    get_market_price_profile_v2,
    generate_pdf_report,
    calculate_battery_financials,
    calculate_buyer_pl,
    calculate_proxy_battery_revenue
)
from excel_reporter import generate_excel_report
import project_matcher


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for NumPy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


st.set_page_config(page_title="ERCOT North Aggregation", layout="wide")

# Global Chart Settings
chart_template = 'plotly_white'




st.title("ERCOT North Renewable Energy Aggregation")
st.markdown("Aggregate load participants and optimize for 24/7 clean energy matching.")

# Session state to store participants
if 'participants' not in st.session_state:
    st.session_state.participants = []

# --- Global Settings (Sidebar - Load Scenario) ---
# --- Global Settings (Sidebar - Load Scenario) ---
# Callback to load demo
def load_demo_scenario():
    # Clear existing
    st.session_state.participants = []
    
    # Add Demo Data Center
    st.session_state.participants.append({
        "name": "Hyperscale Graph DC",
        "type": "Data Center",
        "load": 250000 # 250 GWh/yr ~ 28 MW avg
    })
    
    # Set Portfolio for ~95% CFE
    st.session_state.solar_input = 85.0
    st.session_state.wind_input = 60.0
    st.session_state.ccs_input = 15.0
    st.session_state.geo_input = 0.0
    st.session_state.nuc_input = 0.0
    st.session_state.batt_input = 30.0
    st.session_state.batt_duration_input = 4.0
    
    st.session_state.market_year_input = 2024
    st.session_state.portfolio_recommended = True # Trigger success message
    st.toast("⚡ Instant Demo Loaded! (95% CFE Target)")

# Callback to handle loading
def load_scenario():
    uploaded_file = st.session_state.get('uploaded_scenario_tab')
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

            # DEBUG: Uncomment to see what was parsed
            # st.sidebar.write("DEBUG: Parsed Config:", config)

            # Apply to Session State
            # 1. Participants
            if 'participants' in config:
                st.session_state.participants = config['participants']
            if 'load_gen_sites' in config:
                st.session_state.load_gen_sites = config['load_gen_sites']
            if 'external_profiles' in config:
                # Convert list back to pd.Series
                st.session_state.external_profiles = {name: pd.Series(prof) for name, prof in config['external_profiles'].items()}
            
            # 2. Generation Capacities
            if 'solar_capacity' in config: st.session_state.solar_input = float(config['solar_capacity'])
            if 'wind_capacity' in config: st.session_state.wind_input = float(config['wind_capacity'])
            if 'ccs_capacity' in config: st.session_state.ccs_input = float(config['ccs_capacity'])
            if 'geo_capacity' in config: st.session_state.geo_input = float(config['geo_capacity'])
            if 'nuc_capacity' in config: st.session_state.nuc_input = float(config['nuc_capacity'])
            if 'batt_capacity' in config: st.session_state.batt_input = float(config['batt_capacity'])
            if 'batt_duration' in config: st.session_state.batt_duration_input = max(0.5, float(config['batt_duration']))
            
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
            
            # CVTA Support
            # Map legacy or new keys to CVTA session state controls
            if 'batt_base_rate' in config: st.session_state.cvta_fixed = float(config['batt_base_rate']) # Map legacy
            if 'cvta_fixed_price' in config: st.session_state.cvta_fixed = float(config['cvta_fixed_price'])
            
            if 'batt_guar_rte' in config: st.session_state.cvta_rte = float(config['batt_guar_rte'])
            if 'cvta_rte' in config: st.session_state.cvta_rte = float(config['cvta_rte'])
            
            if 'batt_vom' in config: st.session_state.cvta_vom = float(config['batt_vom'])
            if 'cvta_vom' in config: st.session_state.cvta_vom = float(config['cvta_vom'])

            if 'market_price' in config: st.session_state.market_input = float(config['market_price'])
            if 'rec_price' in config: st.session_state.rec_input = float(config['rec_price'])
            
            # 4. Exclusions
            if 'excluded_techs' in config: st.session_state.excluded_techs_input = config['excluded_techs']
            
            # 5. Market Logic
            if 'market_year' in config:
                m_yr_val = config['market_year']
                # Handle both int and string "Average"
                st.session_state.market_year_input = str(m_yr_val) if m_yr_val == "Average" else int(m_yr_val)
            if 'price_scaler' in config: st.session_state.price_scaler_input = float(config['price_scaler'])
            if 'ppa_price_scaler' in config: st.session_state.ppa_scaler_input = float(config['ppa_price_scaler'])
            
            # 6. Custom Profiles (Large Arrays)
            # Restore Solar
            if 'custom_solar_profile' in config:
                st.session_state['custom_solar_profile'] = pd.Series(config['custom_solar_profile'])
            
            # Restore Wind
            if 'custom_wind_profile' in config:
                st.session_state['custom_wind_profile'] = pd.Series(config['custom_wind_profile'])
                
            # Restore Battery Market Prices
            if 'custom_battery_prices' in config:
                # Expecting list of dicts or similar structure, convert back to DataFrame
                # We saved it as: df.reset_index().to_dict('records')
                # But 'records' loses index. We need to handle datetime index.
                # Actually, simple list of prices + known start date?
                # Let's check how we save it.
                # If we assume standard year, we can just save the 'Price' column as list.
                # But 'shared_market_prices' is a DataFrame with 'Price' and Datetime Index.
                
                price_data = config['custom_battery_prices']
                # If it's a list (just prices), reconstruct index
                if isinstance(price_data, list):
                     # Reconstruct simple index for current year
                     # We might need to know the year. Use market_year.
                     year_val = st.session_state.get('market_year_input', 2024)
                     year_for_dates = 2024 if year_val == 'Average' else year_val
                     dates = pd.date_range(start=f'{year_for_dates}-01-01', periods=len(price_data), freq='h')
                     st.session_state['shared_market_prices'] = pd.DataFrame({'Price': price_data}, index=dates)

            st.toast("Scenario Loaded Successfully!")
            st.toast(f"Solar: {st.session_state.get('solar_input', 0)} MW | Wind: {st.session_state.get('wind_input', 0)} MW")
            
        except Exception as e:
            st.error(f"Error parsing scenario: {e}")

# MUST be defined BEFORE the widgets that use these session state values are instantiated.
with st.sidebar:
    st.markdown("### 0. Load Scenario")
    st.button("⚡ Instant Demo (90% CFE)", on_click=load_demo_scenario, help="Load a sample Hyperscale Data Center scenario", type="primary")


    st.markdown("---")


import random

# --- Random Scenario Generator ---
def generate_random_scenario():
    # 1. Clear State
    st.session_state.participants = []
    
    # 2. Random Load
    industries = ["Data Center", "Manufacturing", "Office Campus", "Green Hydrogen"]
    sizes = [50000, 250000, 500000, 1000000] # GWh
    
    ind = random.choice(industries)
    size = random.choice(sizes)
    
    st.session_state.participants.append({
        "name": f"Random {ind} Project",
        "type": ind if ind != "Green Hydrogen" else "Data Center", # Partial mapping
        "load": size
    })
    
    # 3. Optimize Portfolio for 90% CFE (Like Instant Demo)
    # Generate temporary load profile for optimization
    temp_load = pd.Series(0.0, index=range(8760))
    for p in st.session_state.participants:
        temp_load += generate_dummy_load_profile(p['load'], p['type'])
    
    # Run Optimizer
    rec = recommend_portfolio(temp_load, target_cfe=0.90, excluded_techs=[])
    
    st.session_state.solar_input = rec['Solar']
    st.session_state.wind_input = rec['Wind']
    st.session_state.ccs_input = rec['CCS Gas']
    st.session_state.geo_input = rec['Geothermal']
    st.session_state.nuc_input = rec['Nuclear']
    st.session_state.batt_input = rec['Battery_MW']
    st.session_state.batt_duration_input = rec['Battery_Hours']
        
    st.toast(f"🎲 Generated: {ind} (Optimized for 90% CFE)")
    st.toast("ℹ️ Go to 'Generation Portfolio' tab to tweak settings.")



# --- Executive Summary Container (Placeholder) ---
# Populated at the end of the run
exec_summary_container = st.container()

# --- Configuration Section (Top) ---
tab_guide, tab_load, tab_gen, tab_fin, tab_offtake, tab_scenario, tab_dl = st.tabs(["1\\. Start Here", "2\\. Load Setup", "3\\. Generation Profile", "4\\. Financial Analysis", "5\\. Battery Financials", "6\\. Scenario Manager", "7\\. Download Results"])


# --- Tab 1: User Guide (Moved to Top) ---
with tab_guide:
    st.header("1. Start Here")


    st.markdown("## 📘 User Guide & Methodology")
    
    st.markdown("### 🚀 How to Use This Tool")
    st.markdown("""
    **Step 1: Define Load Profile (Tab 2)**

    Choose how to build your hourly load (8760 hours):
    
    *   **Portfolio Generator**
        *   Build a bottom-up synthetic load by adding individual sites (Data Centers, Offices, Manufacturing).
        *   Configure operating hours, seasonality, and load factors for each property.
        *   **🎲 Randomize Portfolio**: Instantly generate a test portfolio of 3-6 varied sites.
    *   **Upload 8760 Profile (BETA)**
        *   Upload one or more Excel/CSV files containing hourly demand data.
        *   The tool aggregates multiple files into a single portfolio, tracking each as a distinct property.

    **Output:** A consolidated hourly load profile used across all subsequent analyses.

    **Step 2: Design Generation Portfolio (Tab 3)**

    Configure clean energy supply (Optimized for **North Texas** resources):
    *   **Solar**: Modeled for Graham, TX (Solar-rich region).
    *   **Wind**: Modeled for Wichita Falls, TX (High-wind region using modern turbine curves).

    *   **Set Capacities (MW) for:**
        *   Solar
        *   Wind
        *   Nuclear
        *   Geothermal
        *   CCS Gas
        *   Battery Storage: Power (MW) and Duration (hours)
    *   **✨ Smart Fill (Optional)**
        *   Automatically recommends a portfolio designed to achieve a >95% Carbon-Free Energy (CFE) target.
        *   Lock technologies you want to keep fixed while the optimizer adjusts the remaining resources.

    **Output:** Hourly clean generation and storage dispatch aligned to your load profile.

    **Step 3: Analyze Financials (Tab 4)**

    **Inputs**
    *   **PPA Prices:** Enter fixed contract prices ($/MWh) for each generation technology.
    *   **Market Price Data:** Select a historical year (2020-2024), 2025 Forecast, or 'Average' composite for 2020-2025. **All prices are based on ERCOT HB_NORTH data.**

    **Key Metrics**
    *   PPA Costs
    *   Market Value (Capture)
    *   Net Settlement (PPA vs. Market)
    *   Exports

    **Download full simulation outputs:**
    *   Hourly CSVs
    *   JSON configuration files
    *   Generate a professional PDF report for internal or external use.

    **Step 4: Battery Financials (Tab 5)**

    **CVTA Model (Corporate Virtual Tolling Agreement)**
    
    Configure battery monetization terms:
    *   Fixed Capacity Payment
    *   Market Revenue Sharing

    **Outputs:** Battery-specific revenue, cost, and net value contributions.

    **Step 5: Scenario Manager (Tab 6)**

    *   **Save & Restore**: Download your full configuration as a JSON file and restore it later.
    *   **Capture & Compare**: Capture metrics from the current session for side-by-side comparison.

    **Step 6: Download Results (Tab 7)**

    *   **Export Results**: Export hourly CSVs, JSON configurations, and professional Excel or PDF reports.
    """)
    
    st.markdown("---")
    st.markdown("### 🧮 Methodology")

    st.markdown("#### 1. Operational Metrics")
    
    st.markdown("**Core Volume Metrics**")
    st.markdown("- **Total Electricity Usage:** The sum of the hourly load demand across the entire year (8760 hours).")
    st.markdown("- **Clean Energy Generation:** The total annual energy produced by all renewable sources (Solar, Wind, Geothermal, Nuclear) in the portfolio.")
    st.markdown("- **Annual Clean Energy / Annual Load:** The ratio of Total Clean Generation to Total Load. This represents the raw volume of clean energy relative to demand, strictly on an annual basis (ignoring hourly timing).")
    st.latex(r"\text{Annual Ratio} = \frac{\sum \text{Generation}}{\sum \text{Load}}")

    st.markdown("**Matching & Performance Metrics**")
    st.markdown("- **CFE Score (24/7):** The percentage of hourly load that is matched by clean generation in the exact same hour.")
    st.latex(r"CFE = \frac{\sum_{t=1}^{8760} \min(\text{Gen}_t, \text{Load}_t)}{\sum \text{Total Load}}")
    
    st.markdown("- **Loss of Green Hours (LoGH):** The percentage of hours in the year where the clean generation is insufficient to meet the load 100%.")
    st.latex(r"LoGH = \frac{\text{Count}(\text{Hours where Gen} < \text{Load})}{8760}")

    st.markdown("- **MW Match Productivity:** A measure of portfolio efficiency, calculated as the total matched MWh divided by the total installed capacity.")
    st.latex(r"Productivity = \frac{\sum \text{Matched Generation (MWh)}}{\text{Total Installed Capacity (MW)}}")

    st.markdown("**Grid Interaction**")
    st.markdown("- **Grid Consumption:** The total amount of energy drawn from the grid to meet load when clean generation is insufficient (Deficit).")
    st.latex(r"\text{Deficit} = \sum \max(0, \text{Load}_t - \text{Gen}_t)")

    st.markdown("- **Excess Generation:** The gross amount of clean energy generated in excess of the load (Surplus). This is energy available for battery charging or export.")
    st.latex(r"\text{Surplus} = \sum \max(0, \text{Gen}_t - \text{Load}_t)")
    
    st.markdown("- **Battery Discharge:** Total energy discharged by the battery system to meet load or sell to the market.")

    st.markdown("#### 2. Financial Settlement (Fixed-Volume PPA)")
    st.markdown("""
    For each technology, the Buyer pays the PPA Price and receives the Market (Capture) Value for the generated energy.
    """)
    st.latex(r"\text{Cost}_{PPA} = \sum (\text{Gen}_{t} \times \text{Price}_{PPA})")
    st.latex(r"\text{Value}_{Market} = \sum (\text{Gen}_{t} \times \text{Price}_{Market, t})")
    st.latex(r"\text{Net Settlement} = \text{Value}_{Market} - \text{Cost}_{PPA}")
    
    st.markdown("**Weighted Average PPA Price:**")
    st.latex(r"\text{Avg Price}_{PPA} = \frac{\sum \text{Total PPA Cost}}{\sum \text{Total Matched Generation (MWh)}}")
    
    st.markdown("**Excess REC Value:**")
    st.latex(r"\text{Excess Value} = \sum (\text{Surplus Generation}_t) \times \text{REC Price}")

    st.markdown("#### 3. Battery Proxy Model (CVTA)")
    st.markdown("""
    The Corporate Virtual Tolling Agreement (CVTA) is a financial swap for battery storage.
    
    **Fixed Leg (Buyer Pays):**
    """)
    st.latex(r"\text{Fixed Cost} = \text{Capacity (MW)} \times \text{Fixed Rate (\$/MW-mo)} \times 12")
    
    st.markdown("""
    **Floating Leg (Buyer Receives):**
    The model simulates "perfect foresight" arbitrage to maximize revenue against historical prices.
    """)
    st.latex(r"\text{Revenue} = \sum (\text{Discharge}_t \times \text{Price}_t) - \sum (\text{Charge}_t \times \text{Price}_t)")
    st.latex(r"\text{Net Cost} = \text{Fixed Cost} - \text{Revenue}")
    
    st.markdown("**Battery Efficiency (RTE):**")
    st.markdown("Energy is lost during charging based on the Round Trip Efficiency (RTE).")
    st.latex(r"\text{Energy Stored} = \text{Energy Charged} \times \text{RTE \%}")
    
    st.markdown("**Battery Adder Price (Effective Net Cost):**")
    st.markdown("Represents the effective premium paid per MWh of battery energy dispatched.")
    st.latex(r"\text{Adder Price} = \frac{\text{Net Cost}}{\text{Total Discharged Energy (MWh)}}")
    
    st.markdown("#### 4. Synthetic Market Price Model (Duck Curve)")
    st.markdown("""
    When historical market data is not available, the tool generates a synthetic "Duck Curve" price addapted to the average price input.
    """)
    st.latex(r"\text{Price}_t = \text{Base}_t \times \text{Seasonal Factor}_t + \text{Noise}")
    st.markdown("""
    - **Base Shape**: Sinusoidal with a trough at midday (solar cannibalization) and peak at evening (17:00-21:00).
    - **Seasonal Factor**: Higher prices in Summer (`cos` function peaking around day 172).
    """)

    st.markdown("#### 5. Data Sources & Locations")
    st.markdown("""
    **Resource Modeling Locations (ERCOT North Region):**
    -   **Solar Profile**: Graham, TX (33.1070° N, 98.5895° W).
    -   **Wind Profile**: Wichita Falls, TX (33.9137° N, 98.4934° W).
    -   **Market Prices**: ERCOT North Hub (HB_NORTH) Real-Time Market data.
    """)
# --- Tab 2: Load Setup ---
with tab_load:
    st.header("2. Load Setup")
    st.markdown("Configure your load profile by either building a multi-site portfolio or uploading a pre-made 8760-hour profile.")
    
    # Mode Selection: Generator vs Upload
    load_entry_mode = st.radio("Select Load Entry Method:", ["Portfolio Generator", "Upload 8760 (CSV/Excel)"], horizontal=True, key="load_entry_mode")
    
    if load_entry_mode == "Portfolio Generator":
        st.info("Synthetic Portfolio Generator: Build a profile from multiple sites (DC, Mfg, Office) using load factor logic.")
        # --- Generator Logic ported from Tab 8 ---
        # Initialize session state for sites
        if 'load_gen_sites' not in st.session_state:
            st.session_state.load_gen_sites = []

        # Migration: Ensure all sites are stored in kWh for input consistency
        if "load_gen_sites" in st.session_state and st.session_state.load_gen_sites:
            for site in st.session_state.load_gen_sites:
                if "annual_mwh" in site:
                    val = site.pop("annual_mwh")
                    site["annual_kwh"] = val * 1000.0 if val < 50000 else val

        def load_site_data():
            selection = st.session_state.edit_site_select
            if selection != "-- New Site --":
                site_d = next((s for s in st.session_state.load_gen_sites if s['name'] == selection), None)
                if site_d:
                    st.session_state.site_name_input = site_d['name']
                    st.session_state.category_input = site_d['category']
                    st.session_state.annual_kwh_input = site_d.get('annual_kwh', 6000000)
                    st.session_state.mon = site_d['hours_per_day'][0]
                    st.session_state.tue = site_d['hours_per_day'][1]
                    st.session_state.wed = site_d['hours_per_day'][2]
                    st.session_state.thu = site_d['hours_per_day'][3]
                    st.session_state.fri = site_d['hours_per_day'][4]
                    st.session_state.sat = site_d['hours_per_day'][5]
                    st.session_state.sun = site_d['hours_per_day'][6]
                    st.session_state.sh_wd = site_d.get('start_hour_weekday', 8)
                    st.session_state.sh_we = site_d.get('start_hour_weekend', 8)
                    st.session_state.weekend_scaler = site_d.get('weekend_scaler', 1.0)
                    st.session_state.smult_input = site_d.get('summer_multiplier', 1.0)
                    st.session_state.wmult_input = site_d.get('winter_multiplier', 1.0)
                    st.session_state.last_loaded_site = selection
            else:
                st.session_state.last_loaded_site = None
                st.session_state.site_name_input = ""
                st.session_state.category_input = "DC"
                for day in ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']: st.session_state[day] = 24
                st.session_state.sh_wd = 0; st.session_state.sh_we = 0; st.session_state.annual_kwh_input = 6000000
                st.session_state.weekend_scaler = 1.0; st.session_state.smult_input = 1.0; st.session_state.wmult_input = 1.0

        def load_category_defaults():
            cat = st.session_state.category_input
            if cat == "DC":
                for day in ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']: st.session_state[day] = 24
                st.session_state.sh_wd = 0; st.session_state.sh_we = 0
            elif cat == "MFG":
                for day in ['mon', 'tue', 'wed', 'thu', 'fri']: st.session_state[day] = 18
                st.session_state.sat = 0; st.session_state.sun = 0; st.session_state.sh_wd = 6
            elif cat == "Office":
                for day in ['mon', 'tue', 'wed', 'thu', 'fri']: st.session_state[day] = 10
                st.session_state.sat = 0; st.session_state.sun = 0; st.session_state.sh_wd = 8
        def randomize_portfolio():
            import random
            num_sites = random.randint(3, 6)
            st.session_state.load_gen_sites = []
            categories = ["DC", "MFG", "Office", "Other"]
            for i in range(num_sites):
                cat = random.choice(categories)
                name = f"{'Data Center' if cat=='DC' else 'Factory' if cat=='MFG' else 'Office' if cat=='Office' else 'Commercial'} {i+1}"
                kwh = random.randint(5, 50) * 1_000_000
                
                if cat == "DC":
                    hrs = [24]*7; sh_wd = 0; sh_we = 0
                elif cat == "MFG":
                    hrs = [18, 18, 18, 18, 18, 0, 0]; sh_wd = 6; sh_we = 0
                elif cat == "Office":
                    hrs = [10, 10, 10, 10, 10, 0, 0]; sh_wd = 8; sh_we = 0
                else: 
                    hrs = [12]*7; sh_wd = 6; sh_we = 6
                
                st.session_state.load_gen_sites.append({
                    "name": name, "category": cat, "annual_kwh": kwh, 
                    "hours_per_day": hrs, "start_hour_weekday": sh_wd, "start_hour_weekend": sh_we,
                    "summer_multiplier": round(random.uniform(0.9, 1.3), 2),
                    "winter_multiplier": round(random.uniform(0.9, 1.3), 2),
                    "weekend_scaler": round(random.uniform(0.7, 1.0), 2)
                })

        # --- Site Entry Form ---
        site_options = ["-- New Site --"] + ([s['name'] for s in st.session_state.load_gen_sites] if st.session_state.load_gen_sites else [])
        edit_site_selection = st.selectbox("Edit Existing Site", site_options, key="edit_site_select", on_change=load_site_data)
        
        is_edit = edit_site_selection != "-- New Site --"
        current_site_is_ext = False
        if is_edit:
            idx_chk = next((i for i, s in enumerate(st.session_state.load_gen_sites) if s['name'] == edit_site_selection), -1)
            if idx_chk != -1: current_site_is_ext = st.session_state.load_gen_sites[idx_chk].get('is_external', False)
        
        st.subheader("Add / Update Site")
        if current_site_is_ext:
            st.info("💡 This is an **Uploaded Property**. Operational parameters are fixed to the source file data. You can still update the Name and Category.")
            
        site_name = st.text_input("Site Name", placeholder="e.g. Data Center 1", key="site_name_input")
        site_category = st.selectbox("Category", ["DC", "MFG", "Office", "Other", "External / Uploaded"], format_func=lambda x: {"DC":"Data Center", "MFG":"Manufacturing", "Office":"Office", "Other":"Other (12h/7d)", "External / Uploaded":"External / Uploaded"}.get(x,x), key="category_input", on_change=load_category_defaults)
        annual_kwh = st.number_input("Annual kWh", min_value=1000, value=6000000, step=100000, format="%d", key="annual_kwh_input", disabled=current_site_is_ext)
        
        st.markdown("**Hours of Operation per Day (0-24)**")
        sync_col1, sync_col2 = st.columns([2, 1])
        sync_col1, sync_col2 = st.columns([2, 1])
        with sync_col1: sync_hours = st.number_input("Set all days to:", min_value=0, max_value=24, value=24, step=1, key="sync_hrs", disabled=current_site_is_ext)
        with sync_col2: 
            st.write(""); 
            if st.button("Apply", disabled=current_site_is_ext):
                for day in ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']: st.session_state[day] = sync_hours
                st.rerun()
        
        wd_cl1, wd_cl2, wd_cl3 = st.columns(3)
        with wd_cl1: mon_hrs = st.number_input("Mon", 0, 24, key="mon", disabled=current_site_is_ext); tue_hrs = st.number_input("Tue", 0, 24, key="tue", disabled=current_site_is_ext)
        with wd_cl2: wed_hrs = st.number_input("Wed", 0, 24, key="wed", disabled=current_site_is_ext); thu_hrs = st.number_input("Thu", 0, 24, key="thu", disabled=current_site_is_ext)
        with wd_cl3: fri_hrs = st.number_input("Fri", 0, 24, key="fri", disabled=current_site_is_ext)
        
        we_cl1, we_cl2 = st.columns(2)
        with we_cl1: sat_hrs = st.number_input("Sat", 0, 24, key="sat", disabled=current_site_is_ext)
        with we_cl2: sun_hrs = st.number_input("Sun", 0, 24, key="sun", disabled=current_site_is_ext)
        
        treat_holidays_as_weekends = st.checkbox("Treat US Federal Holidays as Weekends", False, key="treat_holidays_input", disabled=current_site_is_ext)
        sh_cl1, sh_cl2 = st.columns(2)
        with sh_cl1: start_hour_weekday = st.number_input("Weekday Start", 0, 23, key="sh_wd", disabled=current_site_is_ext)
        with sh_cl2: start_hour_weekend = st.number_input("Weekend Start", 0, 23, key="sh_we", disabled=current_site_is_ext)
        
        st.markdown("### Load Factor & Seasonality")
        lf1, lf2, lf3 = st.columns(3)
        with lf1: baseline_lf_val = st.number_input("Baseline LF", 0.0, 1.0, 0.20, 0.01, key="baseline_lf_input", disabled=current_site_is_ext)
        with lf2: ramp_lf_val = st.number_input("Ramp LF", 0.0, 1.0, 0.40, 0.01, key="ramp_lf_input", disabled=current_site_is_ext)
        with lf3: weekend_scaler_val = st.number_input("Weekend Scaler", 0.0, 1.0, 1.0, 0.1, key="weekend_scaler", disabled=current_site_is_ext)
        
        sea1, sea2 = st.columns(2)
        with sea1: summer_mult_val = st.number_input("Summer Peak (Jun-Sep)", 0.5, 2.0, 1.0, 0.1, key="smult_input", disabled=current_site_is_ext)
        with sea2: winter_mult_val = st.number_input("Winter Peak (Dec-Feb)", 0.5, 2.0, 1.0, 0.1, key="wmult_input", disabled=current_site_is_ext)
        
        is_edit = edit_site_selection != "-- New Site --"
        b_col1, b_col2 = st.columns([1, 1])
        with b_col1:
            if st.button("Update Site" if is_edit else "Add Site", type="primary", use_container_width=True):
                site_data = {"name": site_name or f"Site {len(st.session_state.load_gen_sites)+1}", "category": site_category, "annual_kwh": annual_kwh, "hours_per_day": [mon_hrs, tue_hrs, wed_hrs, thu_hrs, fri_hrs, sat_hrs, sun_hrs], "start_hour_weekday": start_hour_weekday, "start_hour_weekend": start_hour_weekend, "summer_multiplier": summer_mult_val, "winter_multiplier": winter_mult_val, "weekend_scaler": weekend_scaler_val}
                if is_edit:
                    idx = next((i for i, s in enumerate(st.session_state.load_gen_sites) if s['name'] == edit_site_selection), -1)
                    if idx != -1:
                        # Preserve 'is_external' flag and other external metadata if it exists
                        old_site = st.session_state.load_gen_sites[idx]
                        if old_site.get('is_external'):
                            site_data['is_external'] = True
                            # If name changed, migrate the profile in external_profiles
                            if site_data['name'] != old_site['name'] and 'external_profiles' in st.session_state:
                                profile = st.session_state.external_profiles.pop(old_site['name'], None)
                                if profile is not None:
                                    st.session_state.external_profiles[site_data['name']] = profile
                        st.session_state.load_gen_sites[idx] = site_data
                else: 
                    st.session_state.load_gen_sites.append(site_data)
                st.session_state.last_loaded_site = None; st.rerun()
        with b_col2:
            if st.button("🎲 Randomize Portfolio", use_container_width=True, help="Generate 3-6 random sites for testing"):
                randomize_portfolio(); st.rerun()

        if st.session_state.load_gen_sites:
            st.markdown("---")
            st.subheader("Current Portfolio")
            
            # Visual List with Deletion
            for i, site in enumerate(st.session_state.load_gen_sites):
                c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
                with c1: st.markdown(f"**{site['name']}**")
                with c2: st.caption(site['category'])
                with c3: st.caption(f"{site['annual_kwh']:,.0f} kWh")
                with c4:
                    if st.button("🗑️", key=f"del_site_{i}", help=f"Remove {site['name']}"):
                        # Clean up external profile if exists
                        if site.get('is_external') and 'external_profiles' in st.session_state:
                            st.session_state.external_profiles.pop(site['name'], None)
                        st.session_state.load_gen_sites.pop(i)
                        st.rerun()

            with st.popover("🗑️ Clear All Sites"):
                if st.button("Confirm: Wipe Entire Portfolio", type="primary"): 
                    st.session_state.load_gen_sites = []
                    if "custom_aggregated_profile" in st.session_state:
                         del st.session_state["custom_aggregated_profile"]
                    if "external_profiles" in st.session_state:
                         st.session_state.external_profiles = {}
                    st.rerun()
            
            # --- Generate Combined Profile ---
            all_profiles = []
            total_kw = None
            for s in st.session_state.load_gen_sites:
                if s.get('is_external') and 'external_profiles' in st.session_state and s['name'] in st.session_state.external_profiles:
                    # Retrieve pre-loaded profile
                    ext_profile = st.session_state.external_profiles[s['name']]
                    # Convert to DataFrame with 'kW' column for downstream compatibility
                    p_df = pd.DataFrame({'Datetime': pd.date_range(start='2024-01-01', periods=8760, freq='h'), 'kW': ext_profile.values})
                else:
                    # Standard synthetic generation
                    p_df = generate_load_factor_profile(annual_kwh=s['annual_kwh'], hours_per_day=s['hours_per_day'], start_hour_weekday=s.get('start_hour_weekday', 8), start_hour_weekend=s.get('start_hour_weekend', 8), year=2024, baseline_lf=baseline_lf_val, ramp_lf=ramp_lf_val, treat_holidays_as_weekends=treat_holidays_as_weekends, summer_multiplier=s.get('summer_multiplier', 1.0), winter_multiplier=s.get('winter_multiplier', 1.0), weekend_scaler=s.get('weekend_scaler', 1.0))
                
                p_df.attrs['site_name'] = s['name']
                all_profiles.append(p_df)
                total_kw = p_df['kW'].copy() if total_kw is None else total_kw + p_df['kW']
            
            combined_mw = total_kw / 1000.0
            st.session_state["custom_aggregated_profile"] = combined_mw
            
            m1, m2 = st.columns(2)
            m1.metric("Total Load", f"{combined_mw.sum():,.0f} MWh")
            m2.metric("Peak Load", f"{combined_mw.max():,.2f} MW")
            st.line_chart(combined_mw)
            
            # CSV Download: Only export the 'kW' column for each site, renamed to the site name
            site_series = []
            for p in all_profiles:
                s_name = p.attrs.get('site_name', 'Site')
                # Extract only the kW column and rename it
                s_series = p.set_index('Datetime')['kW'].copy()
                s_series.name = s_name
                site_series.append(s_series)
            
            download_df = pd.concat(site_series, axis=1)
            # Add Unit Suffixes to Download Columns
            download_df.columns = [f"{c} (kW)" for c in download_df.columns]
            download_df['TOTAL_MW'] = total_kw / 1000.0
            csv_str = download_df.to_csv().encode('utf-8')
            st.download_button("📥 Download 8760 CSV", csv_str, "portfolio_load.csv", "text/csv")
    
    else:
        # --- Upload 8760 Logic ---
        st.subheader("Upload 8760 Profile (BETA UNDER REVIEW)")
        st.info("Upload one or more hourly (8760-hour) load profiles. Each file will be treated as a distinct property.")
        uploaded_files = st.file_uploader("Choose CSV or Excel files", type=["csv", "xlsx", "xls"], key="8760_uploader", accept_multiple_files=True)
        
        if uploaded_files:
            try:
                # Initialize DataFrame for property profiles
                m_year = st.session_state.get('market_year_input', 2024)
                year_for_dates = 2024 if m_year == 'Average' else m_year
                dates = pd.date_range(start=f'{year_for_dates}-01-01', periods=8760, freq='h')
                
                property_profiles_df = pd.DataFrame({'Datetime': dates})
                total_profile = pd.Series(0.0, index=range(8760))
                file_details = []
                
                for uploaded_file in uploaded_files:
                    if uploaded_file.name.endswith('.csv'): df_up = pd.read_csv(uploaded_file)
                    else: df_up = pd.read_excel(uploaded_file)
                    
                    if len(df_up) < 8760: 
                        st.error(f"File '{uploaded_file.name}' must have 8760 rows. Found {len(df_up)}.")
                        continue
                        
                    # Robust Column Detection: Skip datetime/index columns
                    potential_cols = [c for c in df_up.columns if not any(k in str(c).lower() for k in ['date', 'time', 'timestamp', 'index', 'interval', 'hour'])]
                    
                    if not potential_cols:
                        st.error(f"No numeric data columns found in '{uploaded_file.name}'.")
                        continue
                    
                    # Identify numeric columns only
                    numeric_cols = []
                    for col in potential_cols:
                        try:
                            pd.to_numeric(df_up[col], errors='raise')
                            numeric_cols.append(col)
                        except:
                            continue
                            
                    if not numeric_cols:
                        st.error(f"No valid numeric load columns found in '{uploaded_file.name}'.")
                        continue

                    # Multi-Property Detection: If multiple numeric columns, check for an aggregate
                    total_keywords = ['total', 'aggregate', 'load_mw', 'load_kw']
                    total_col = next((c for c in numeric_cols if any(k in str(c).lower() for k in total_keywords)), None)
                    
                    # System columns to ignore during "explosion" (app-generated results or generator internals)
                    system_cols = [
                        'matched_mw', 'solar_mw', 'wind_mw', 'geothermal_gen_mw', 'nuclear_gen_mw', 
                        'ccs_gas_gen_mw', 'total_raw_gen_mw', 'battery_discharge_mw', 
                        'battery_state_of_charge_mwh', 'grid_deficit_mw', 'surplus_mw', 
                        'market_price_$/mwh', 'total_mw', 'total_kw', 'lf', 'kw', 'hour', 'month', 'day_type'
                    ]
                    
                    # Columns to treat as properties
                    prop_cols = []
                    if len(numeric_cols) > 1:
                        # Filter out system columns and the total column
                        # Use split('.') to handle pandas renamed duplicates like LF.1, LF.2
                        for c in numeric_cols:
                            c_base = str(c).lower().split('.')[0]
                            if c_base not in system_cols and c != total_col:
                                prop_cols.append(c)
                        
                        # If we filtered everything out (e.g. it was just system cols), 
                        # fall back to the total column if it exists
                        if not prop_cols and total_col:
                            prop_cols = [total_col]
                        elif not prop_cols:
                            # If no total and no non-system cols, just take the first one
                            prop_cols = [numeric_cols[0]]
                    else:
                        # Just one column
                        prop_cols = numeric_cols

                    for l_col in prop_cols:
                        raw_profile = pd.to_numeric(df_up[l_col], errors='coerce').fillna(0).head(8760)
                        
                        # Unit Detection Logic
                        l_col_str = str(l_col).lower()
                        if 'mw' in l_col_str or 'mwh' in l_col_str:
                            profile = raw_profile * 1000.0  # Convert MW to kW
                        elif 'kw' in l_col_str or 'kwh' in l_col_str:
                            profile = raw_profile  # Already in kW
                        elif raw_profile.max() < 1000:
                            # Heuristic: If values are small and unlabeled, assume MW
                            profile = raw_profile * 1000.0
                        else:
                            # Heuristic: Assume kW if values are large
                            profile = raw_profile
                        
                        # Store property
                        prop_label = f"{uploaded_file.name}: {l_col}" if len(prop_cols) > 1 else uploaded_file.name
                        
                        property_profiles_df[prop_label] = profile.values
                        total_profile += profile.values
                        
                        file_details.append({
                            "Property": prop_label,
                            "Total (kWh)": f"{profile.sum():,.0f}",
                            "Peak (kW)": f"{profile.max():,.2f}"
                        })
                
                else:
                    # Register Properties in Portfolio Generator
                    if 'external_profiles' not in st.session_state:
                         st.session_state.external_profiles = {}
                    
                    for i, detail in enumerate(file_details):
                        prop_name = detail["Property"]
                        # Check if site already exists to avoid duplicates
                        if not any(s['name'] == prop_name for s in st.session_state.load_gen_sites):
                            st.session_state.load_gen_sites.append({
                                'name': prop_name,
                                'category': 'External / Uploaded',
                                'annual_kwh': float(detail["Total (kWh)"].replace(',', '')),
                                'hours_per_day': [24]*7, # Placeholder for UI
                                'is_external': True
                            })
                        
                        # Store the actual profile
                        st.session_state.external_profiles[prop_name] = pd.Series(property_profiles_df[prop_name].values)

                    st.success(f"✅ Registered {len(file_details)} property(ies) in Portfolio Generator")
                    # No longer need st.session_state["custom_aggregated_profile"] here 
                    # as it will be re-calculated from load_gen_sites in the main loop
                    # but we keep it for immediate UI feedback in this tab if needed.
                    st.session_state["custom_aggregated_profile"] = total_profile
                    
                    # Detailed Metrics Breakdown
                    st.markdown("### Property Breakdown")
                    breakdown_cols = st.columns(min(len(file_details), 4))
                    for i, detail in enumerate(file_details):
                        with breakdown_cols[i % 4]:
                            st.metric(detail["Property"], f"{float(detail['Total (kWh)'].replace(',',''))/1000:,.1f} MWh", f"Peak: {float(detail['Peak (kW)'].replace(',',''))/1000:,.2f} MW")

                    # Visualization
                    st.markdown("### Load Profile Visualization")
                    fig_multi = go.Figure()
                    
                    # Add trace for each property
                    for detail in file_details:
                        prop_name = detail["Property"]
                        fig_multi.add_trace(go.Scatter(
                            x=property_profiles_df['Datetime'],
                            y=property_profiles_df[prop_name],
                            mode='lines',
                            name=prop_name,
                            line=dict(width=1.5)
                        ))
                    
                    # Add Aggregate Total trace
                    fig_multi.add_trace(go.Scatter(
                        x=property_profiles_df['Datetime'],
                        y=total_profile,
                        mode='lines',
                        name='TOTAL',
                        line=dict(width=3, color='Navy', dash='dot')
                    ))
                    
                    fig_multi.update_layout(
                        xaxis_title="Time",
                        yaxis_title="Load (kW)",
                        template=chart_template,
                        hovermode='x unified',
                        height=500
                    )
                    st.plotly_chart(fig_multi, use_container_width=True)
                    
                    # Prepare export CSV with all columns
                    export_df = property_profiles_df.copy()
                    # Add unit suffixes to property columns (excluding Datetime)
                    export_df.columns = [f"{c} (kW)" if c != 'Datetime' else c for c in export_df.columns]
                    export_df['TOTAL_MW'] = total_profile / 1000.0
                    csv_export = export_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Property Breakdown (8760 CSV)", csv_export, "load_breakdown_8760.csv", "text/csv")
                    
            except Exception as e: st.error(f"Error processing files: {e}")
        else:
            # Clear profile if uploader is empty and we're in upload mode
            if "custom_aggregated_profile" in st.session_state and not st.session_state.load_gen_sites:
                 del st.session_state["custom_aggregated_profile"]
# --- Tab 2: Generation Portfolio ---
with tab_gen:
    st.header("3. Generation Profile")
    # Define callback for clearing portfolio
    def clear_portfolio():
        st.session_state.solar_input = 0.0
        st.session_state.wind_input = 0.0
        st.session_state.ccs_input = 0.0
        st.session_state.geo_input = 0.0
        st.session_state.nuc_input = 0.0
        st.session_state.batt_input = 0.0
        st.session_state.batt_duration_input = 2.0
        st.session_state.matched_projects = {}
        st.session_state.portfolio_recommended = False

    # Define callback for recommendation
    def apply_recommendation():
        # Calculate total load from participants OR custom profile
        temp_load = None
        
        if "custom_aggregated_profile" in st.session_state:
             # Convert MW back to MWh/h (same thing) but verify unit?
             # app.py generally assumes MW for profile. recommend_portfolio takes profile in MW.
             # custom_aggregated_profile is stored as MW (Total_kW / 1000).
             temp_load = st.session_state["custom_aggregated_profile"]
             
        elif st.session_state.participants:
            temp_load = pd.Series(0.0, index=range(8760))
            for p in st.session_state.participants:
                temp_load += generate_dummy_load_profile(p['load'], p['type'])
        
        if temp_load is not None and temp_load.sum() > 0:

            # Smart Fill: Always use existing values to build around them
            existing_capacities = {
                'Solar': st.session_state.get('solar_input', 0.0),
                'Wind': st.session_state.get('wind_input', 0.0),
                'CCS Gas': st.session_state.get('ccs_input', 0.0),
                'Geothermal': st.session_state.get('geo_input', 0.0),
                'Nuclear': st.session_state.get('nuc_input', 0.0),
                'Battery_MW': st.session_state.get('batt_input', 0.0)
            }
            
            # Pass excluded techs from session state (widget key='excluded_techs_input')
            marker_yr = st.session_state.get('market_year_input', 2024)
            use_syn = st.session_state.get('use_synthetic_input', False)
            
            rec = recommend_portfolio(
                temp_load, 
                target_cfe=1.0, 
                excluded_techs=st.session_state.get('excluded_techs_input', []),
                existing_capacities=existing_capacities,
                fixed_techs=st.session_state.get('fixed_techs_input', []),
                year=marker_yr,
                use_synthetic=use_syn
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
            st.session_state.portfolio_error = "No load profile found. Please configure load in '2. Load Setup'."

    col_gen_1, col_gen_2 = st.columns([1, 1])
    
    with col_gen_1:
        st.markdown("#### Capacities")
        

        st.markdown("---")
        
        # Input Widgets (Keys mapped to session state)
        solar_capacity = st.number_input("Solar Capacity (MW)", min_value=0.0, step=50.0, key='solar_input')
        wind_capacity = st.number_input("Wind Capacity (MW)", min_value=0.0, step=50.0, key='wind_input')
        geo_capacity = st.number_input("Geothermal Capacity (MW)", min_value=0.0, step=50.0, key='geo_input')
        nuc_capacity = st.number_input("Nuclear Capacity (MW)", min_value=0.0, step=50.0, key='nuc_input')
        ccs_capacity = st.number_input("CCS Gas Capacity (MW)", min_value=0.0, step=50.0, key='ccs_input')
        
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

        # Battery Sizing (Physical)
        # Battery Sizing (Physical)
        st.markdown("**Battery Storage**")
        
        # Initialize session state to avoid "created with default value" warning
        if 'batt_input' not in st.session_state:
            st.session_state.batt_input = 0.0
        batt_capacity = st.number_input("Battery Power (MW)", min_value=0.0, step=50.0, key='batt_input')

        if 'batt_duration_input' not in st.session_state:
            st.session_state.batt_duration_input = 2.0
        elif st.session_state.batt_duration_input < 0.5:
            st.session_state.batt_duration_input = 0.5
            
        batt_duration = st.number_input("Battery Duration (Hours)", min_value=0.5, step=0.5, key='batt_duration_input')


    with col_gen_2:
        st.markdown("#### Portfolio Recommendation")
        
        st.markdown("---")
        
        # Exclude Tech multiselect
        # Exclude Tech multiselect
        excluded_techs = st.multiselect(
            "Exclude Technologies from Recommendation",
            ['Solar', 'Wind', 'CCS Gas', 'Geothermal', 'Nuclear', 'Battery'],
            key='excluded_techs_input'
        )
        
        # Lock Tech multiselect (Smart Fill)
        fixed_techs = st.multiselect(
            "Lock Technologies (Support Smart Fill)",
            ['Solar', 'Wind', 'CCS Gas', 'Geothermal', 'Nuclear', 'Battery'],
            help="Select technologies to keep fixed at their current values. The solver will adjust the others.",
            key='fixed_techs_input'
        )

        col_btn, col_chk = st.columns([1, 1])
        col_btn.button("✨ Smart Fill / Recommend", on_click=apply_recommendation)
        
        # Show success/error messages after rerun
        if st.session_state.get('portfolio_recommended', False):
            st.success("Portfolio Recommended!")
            st.session_state.portfolio_recommended = False # Reset flag
        if st.session_state.get('portfolio_error', None):
            st.warning(st.session_state.portfolio_error)
            st.session_state.portfolio_error = None # Reset error

        # Clear Portfolio Button (Moved from Left)
        st.button("🗑️ Clear Portfolio (Reset to 0)", on_click=clear_portfolio)
        
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
                                'Hub': proj.get('hub', 'North'),
                                'County': proj['county'],
                                'Coordinates': f"{proj.get('lat', 0):.3f}, {proj.get('lon', 0):.3f}" if proj.get('lat') else "-",
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



    



# --- Tab 4: Financials ---
with tab_fin:
    st.header("4. Financial Analysis")
    st.caption("Market: ERCOT North Hub (HB_NORTH)")
    st.markdown("#### PPA Prices ($/MWh)")

    
    # 1. Scaler First
    col_sc_1, col_sc_2 = st.columns([1, 2])
    ppa_price_scaler = col_sc_1.number_input("PPA Price Scaler", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key='ppa_scaler_input', help="Multiplier for all PPA Prices")
    
    c_fin_1, c_fin_2, c_fin_3 = st.columns(3)
    with c_fin_1:
        solar_price = st.number_input("Solar PPA Price", min_value=0.0, value=50.0, step=0.5, key='solar_price_input', help="2025 Market Est: $45.00 - $52.00")
        st.markdown(f"**Scaled PPA Price: ${solar_price * ppa_price_scaler:.2f}**")
        
        wind_price = st.number_input("Wind PPA Price", min_value=0.0, value=50.0, step=0.5, key='wind_price_input', help="2025 Market Est: $40.00 - $45.00")
        st.markdown(f"**Scaled PPA Price: ${wind_price * ppa_price_scaler:.2f}**")
        
    with c_fin_2:
        ccs_price = st.number_input("CCS Gas PPA Price", min_value=0.0, value=65.0, step=1.0, key='ccs_price_input', help="2025 Market Est: $55.00 - $75.00 (w/ 45Q)")
        st.markdown(f"**Scaled PPA Price: ${ccs_price * ppa_price_scaler:.2f}**")
        
        geo_price = st.number_input("Geothermal PPA Price", min_value=0.0, value=77.5, step=0.5, key='geo_price_input', help="2025 Market Est: $70.00 - $85.00")
        st.markdown(f"**Scaled PPA Price: ${geo_price * ppa_price_scaler:.2f}**")
        
    with c_fin_3:
        nuc_price = st.number_input("Nuclear PPA Price", min_value=0.0, value=95.0, step=1.0, key='nuc_price_input', help="2025 Market Est: $90.00 - $100.00")
        st.markdown(f"**Scaled PPA Price: ${nuc_price * ppa_price_scaler:.2f}**")

    # Calculate Effective PPA Prices (Global Vars)
    solar_price_eff = solar_price * ppa_price_scaler
    wind_price_eff = wind_price * ppa_price_scaler
    ccs_price_eff = ccs_price * ppa_price_scaler
    geo_price_eff = geo_price * ppa_price_scaler
    nuc_price_eff = nuc_price * ppa_price_scaler




    st.markdown("---")
    st.markdown("#### Market Assumptions")
    c_mkt_1, c_mkt_2, c_mkt_3, c_mkt_4 = st.columns(4)
    
    # Market Price Year Selection
    # Add 'Average' option for multi-year composite
    year_options = [2025, 2024, 2023, 2022, 2021, 2020, "Average"]
    market_year = c_mkt_1.selectbox("Market Year", year_options, help="Select historical price year or 'Average' for 2020-2025 composite", key='market_year_input')
    
    # UI Check for data availability
    import os
    current_dir = os.path.dirname(__file__)
    
    if market_year == "Average":
         c_mkt_1.success(f"Composite: 2020-2025 ✅")
    else:
        file_path = os.path.join(current_dir, f'ercot_rtm_{market_year}.parquet')
        if os.path.exists(file_path):
             c_mkt_1.success(f"Loaded {market_year} Data ✅")
        else:
             c_mkt_1.warning(f"Missing Data (Using Synthetic)")
    
    # Weather Source Toggle
    use_synthetic_weather = c_mkt_1.checkbox("Force TMY Weather", value=False, help="If checked, forces use of Typical Meteorological Year (TMY) data even if Actuals are available.", key='use_synthetic_input')

    # Get base average from actual data
    _, base_market_avg = get_market_price_profile_v2(32.0, return_base_avg=True, year=market_year)
    
    market_price = base_market_avg
    c_mkt_1.metric(
        f"Base Avg ({market_year})", 
        f"${base_market_avg:.2f}",
        help=f"Actual average from {market_year} ERCOT HB_NORTH data"
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
    
    
    # Display Historical Averages
    try:
        if os.path.exists("ercot_rec_prices_est_2020_2024.csv"):
            rec_df = pd.read_csv("ercot_rec_prices_est_2020_2024.csv")
            rec_df['Date'] = pd.to_datetime(rec_df['Date'])
            rec_avgs = rec_df.groupby(rec_df['Date'].dt.year)['Price_USD_MWh'].mean()
            
            c_mkt_4.markdown("---")
            c_mkt_4.caption("**Hist. Avg ($/MWh)**", help="⚠️ Data Source: Representative estimates based on public market trends (e.g. NREL/market reports). This is NOT official exchange data.")
            for yr, pr in rec_avgs.items():
                c_mkt_4.caption(f"{yr}: **${pr:.2f}**")
    except Exception:
        pass
    
    # Download Market Price Data
    st.markdown("---")
    st.markdown("##### 📥 Download Market Price Data")

    # --- Price Preview Chart ---
    # --- Price Preview Chart ---
    with st.expander("📈 View Price Preview", expanded=True):
        view_mode = st.radio("View By", ["Monthly", "Comparison (Annual)"], horizontal=True, label_visibility="collapsed")
        
        fig_preview = go.Figure()

        if view_mode == "Monthly":
            # 1. Monthly View (Selected Year)
            # Generate hourly profile for Market Price
            preview_selected = get_market_price_profile_v2(market_price, year=market_year) * price_scaler
            
            # Create DF to resample
            df_preview = pd.DataFrame({'Price': preview_selected.values}, index=pd.date_range('2024-01-01', periods=8760, freq='h'))
            monthly_avg = df_preview.resample('ME')['Price'].mean()
            
            fig_preview.add_trace(go.Bar(
                x=monthly_avg.index.strftime('%b'),
                y=monthly_avg.values,
                name=f"{market_year} Market",
                marker_color='#1f77b4'
            ))
            
            # --- Calculate Blended PPA Price (Weighted Average) ---
            # Try to get capacities from session state or LOCALS (which is safer if defined in same run)
            try:
                # Capacity Lookup Helper
                def get_cap(key, var_name):
                    # Check GLOBAL scope for the variable (since we are inside a function/block)
                    if var_name in globals():
                        return globals()[var_name]
                    # Check Session State
                    return st.session_state.get(key, 0.0)

                c_sol = get_cap('solar_input', 'solar_capacity')
                c_win = get_cap('wind_input', 'wind_capacity')
                c_geo = get_cap('geo_input', 'geo_capacity')
                c_nuc = get_cap('nuc_input', 'nuc_capacity')
                c_ccs = get_cap('ccs_input', 'ccs_capacity')
                
                # Debug Info
                # st.write(f"Debug: Sol={c_sol}, Win={c_win}")

                # Re-generate profiles locally for preview
                # Note: These use 2024 shape by default in utils, which matches our analysis year approx
                p_sol = generate_dummy_generation_profile(c_sol, 'Solar', use_synthetic=use_synthetic_weather, year=market_year)
                p_win = generate_dummy_generation_profile(c_win, 'Wind', use_synthetic=use_synthetic_weather, year=market_year)
                p_geo = generate_dummy_generation_profile(c_geo, 'Geothermal', use_synthetic=use_synthetic_weather, year=market_year)
                p_nuc = generate_dummy_generation_profile(c_nuc, 'Nuclear', use_synthetic=use_synthetic_weather, year=market_year)
                p_ccs = generate_dummy_generation_profile(c_ccs, 'CCS Gas', use_synthetic=use_synthetic_weather, year=market_year)
                
                # Prices (Defined locally in Tab 4 just above)
                # Use default fallback if variable not bound yet (though it should be)
                pr_sol = solar_price_eff if 'solar_price_eff' in locals() else 0.0
                pr_win = wind_price_eff if 'wind_price_eff' in locals() else 0.0
                pr_geo = geo_price_eff if 'geo_price_eff' in locals() else 0.0
                pr_nuc = nuc_price_eff if 'nuc_price_eff' in locals() else 0.0
                pr_ccs = ccs_price_eff if 'ccs_price_eff' in locals() else 0.0
                
                # Hourly Totals
                total_gen_hourly = p_sol + p_win + p_geo + p_nuc + p_ccs
                total_cost_hourly = (p_sol * pr_sol) + (p_win * pr_win) + (p_geo * pr_geo) + (p_nuc * pr_nuc) + (p_ccs * pr_ccs)
                
                # Monthly Weighted Average
                # Resample Sums first
                df_ppa = pd.DataFrame({
                    'Cost': total_cost_hourly,
                    'Gen': total_gen_hourly
                }, index=pd.date_range('2024-01-01', periods=8760, freq='h'))
                
                df_ppa_monthly = df_ppa.resample('ME').sum()
                
                # Avoid div by zero
                monthly_blended_ppa = np.divide(
                    df_ppa_monthly['Cost'].values, 
                    df_ppa_monthly['Gen'].values, 
                    out=np.zeros_like(df_ppa_monthly['Cost'].values), 
                    where=df_ppa_monthly['Gen'].values!=0
                )
                
                # Add Line Trace
                # Only show if there is generation
                if df_ppa_monthly['Gen'].sum() > 0:
                    fig_preview.add_trace(go.Scatter(
                        x=monthly_avg.index.strftime('%b'),
                        y=monthly_blended_ppa,
                        name="Average Capture Price",
                        mode='lines+markers',
                        line=dict(color='#2ca02c', width=3, dash='dot'), # Green dotted
                        text=[f"${x:.2f}" for x in monthly_blended_ppa],
                        textposition="top center"
                    ))
            
            except Exception as e:
                # st.caption(f"⚠️ Could not calc Pricing: {str(e)}")
                pass

            fig_preview.update_layout(title=f"Monthly Average Prices ({market_year})")

        else:
            # 2. Annual Comparison
            years = [2020, 2021, 2022, 2023, 2024, 2025, "Average"]
            annual_data = []
            colors = []
            
            for y in years:
                # Use base function, apply current scaler to ALL for fair comparison? 
                # Or just show raw historical? Usually people want to see raw historical vs their scaled scenario.
                # Let's show raw historical, but highlight the current SCENARIO year with the scaler applied.
                
                # Actually, simplest is just show averages of the profiles.
                prof, base_avg = get_market_price_profile_v2(market_price, return_base_avg=True, year=y, scale_hist=False)
                
                val = base_avg
                color = 'lightgrey'
                
                # Highlight selected year
                if str(y) == str(market_year):
                    color = '#1f77b4' 

                annual_data.append({'Year': str(y), 'Price': val, 'Color': color})

            df_annual = pd.DataFrame(annual_data)
            
            fig_preview.add_trace(go.Bar(
                x=df_annual['Year'],
                y=df_annual['Price'],
                marker_color=df_annual['Color'],
                text=df_annual['Price'],
                texttemplate='$%{text:.2f}',
                textposition='auto'
            ))
            
            fig_preview.update_layout(title="Annual Comparison (Selected vs Historical Raw)")

        fig_preview.update_layout(
            yaxis_title="Price ($/MWh)",
            template=chart_template,
            height=350,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_preview, use_container_width=True)

        # Tabular Preview (Always useful to have raw hourly)
        st.markdown("**Hourly Data Preview (CSV)**")
        # Regenerate for table context
        preview_selected = get_market_price_profile_v2(market_price, year=market_year) * price_scaler
        
        data_dict = {
            'Datetime': pd.date_range(start='2024-01-01', periods=8760, freq='h'),
            f'Price_{market_year}': preview_selected
        }
        
        preview_df = pd.DataFrame(data_dict)
        
        # Use columns to restrict width (half width)
        col_tbl, _ = st.columns([1, 1])
        with col_tbl:
            st.dataframe(preview_df.head(100), use_container_width=True, height=200, hide_index=True)
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        # Single Year Download
        download_price_profile = get_market_price_profile_v2(market_price, year=market_year) * price_scaler
        
        # Use correct year for Datetime (Default to 2024 for 'Average')
        ts_year = 2024 if str(market_year) == 'Average' else market_year
        price_dates = pd.date_range(start=f'{ts_year}-01-01', periods=8760, freq='h')
        
        price_df = pd.DataFrame({
            'Datetime': price_dates,
            'Price_USD_MWh': download_price_profile.values
        })
        
        price_csv = price_df.to_csv(index=False)
        
        st.download_button(
            label=f"📊 Download {market_year} Prices",
            data=price_csv,
            file_name=f"ercot_prices_{market_year}_scaled_{price_scaler}x.csv",
            mime="text/csv",
            help=f"Hourly market prices for {market_year} (Base: ${base_market_avg:.2f}, Scaler: {price_scaler}x)",
            use_container_width=True
        )
    
    with col_dl2:
        # All Years Download (ZIP)
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            all_years = [2020, 2021, 2022, 2023, 2024, 2025, "Average"]
            
            for yr in all_years:
                yr_profile = get_market_price_profile_v2(30.0, year=yr) * price_scaler
                
                # Dynamic Year
                ts_yr = 2024 if str(yr) == 'Average' else yr
                yr_dates = pd.date_range(start=f'{ts_yr}-01-01', periods=8760, freq='h')
                
                yr_df = pd.DataFrame({
                    'Datetime': yr_dates,
                    'Price_USD_MWh': yr_profile.values
                })
                
                yr_csv = yr_df.to_csv(index=False)
                zf.writestr(f"ercot_prices_{yr}.csv", yr_csv)
        
        st.download_button(
            label="📦 Download All Years (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"ercot_prices_all_years_scaled_{price_scaler}x.zip",
            mime="application/zip",
            help=f"Includes 2020-2025 and Average composite (Scaler: {price_scaler}x)",
            use_container_width=True
        )


# --- Tab 4: Battery Financials (CVTA) ---
with tab_offtake:
    st.header("5. Battery Financials")
    st.markdown("#### 🔋 Corporate Virtual Tolling Agreement (CVTA)")
    st.caption("Financial Battery PPA | Proxy Battery Model")
    
    with st.popover("📖 How it Works: CVTA Methodology"):
        st.markdown("""
        ### Corporate Virtual Tolling Agreement (CVTA)
        The CVTA is a financial swap that allows a Corporate Buyer to monetize battery storage without physical operation. 
        
        #### ⚡ Dual Dispatch Reality
        The model runs **two separate hourly simulations** for the battery to reflect its dual value:
        1. **Physical Dispatch (CFE Tracking)**: In the summary charts, the battery is dispatched to **maximize matching**. It charges from renewable surplus and discharges to cover load deficits.
        2. **Financial Dispatch (Settlement)**: In this Financials tab, the battery is dispatched to **maximize market revenue** (Arbitrage). It charges during low-price hours and discharges during high-price hours.
        
        > [!NOTE]
        > This "Perfect Foresight" arbitrage is a standard proxy for a financial PPA, where the buyer receives the maximum possible market value regardless of physical matching.

        #### 1. The Fixed Leg (Buyer Pays)
        The Buyer pays a monthly **Fixed Capacity Price** ($/MW-mo) to the developer.
        * **Calculation:** `Fixed Cost = Capacity (MW) * Fixed Price ($/MW-mo) * 12`

        #### 2. The Floating Leg (Buyer Receives)
        The Buyer receives the **Net Market Revenue** generated by the financial dispatch.
        * **Efficiency**: Accounts for **Round Trip Efficiency (RTE)**—energy is lost during the cycle.
        * **O&M**: Deducts **Variable O&M** ($/MWh) from the gross market revenue.

        #### 3. Net Settlement
        The final "Net Cost" is the difference between the Fixed Payment and the Market Revenue.
        * **Formula**: `Net Settlement = Market Revenue - Fixed Payment`
        """)
    
    col_cvta_inputs, col_cvta_charts = st.columns([1, 2])
    
    with col_cvta_inputs:
        st.markdown("##### 1. Linked Battery Specs")
        
        # Link to Generation Portfolio Inputs
        # Get values from session state or variables (batt_capacity is available in scope)
        # Fallback to session state if standard run
        
        # Using st.session_state is safest if variable scope is tricky, but batt_capacity is in scope.
        # Let's use the variable 'batt_capacity' and 'batt_duration' defined in Tab 2.
        
        cvta_cap = batt_capacity if 'batt_capacity' in locals() else 0.0
        cvta_dur = batt_duration if 'batt_duration' in locals() else 0.0
        
        if cvta_cap > 0:
            st.info(f"**Linked Portfolio Battery:**\n\n⚡ **{cvta_cap:,.0f} MW**\n\n⏳ **{cvta_dur:.1f} Hours**")

        else:
            st.warning("⚠️ **No Battery Configured**\n\nGo to '2. Generation Portfolio' to size the battery.")

        cvta_rte = st.number_input("Round Trip Efficiency (%)", value=85.0, step=1.0, key='cvta_rte')
        cvta_vom = st.number_input("Variable O&M ($/MWh)", value=2.0, step=0.1, key='cvta_vom')
        
        st.markdown("---")
        st.markdown("##### 2. Contract Terms")
        cvta_fixed_price = st.number_input("Fixed Capacity Price ($/MW-mo)", value=10000.0, step=250.0, help="Monthly fixed payment from Corporate to Developer per MW. 2025 Est: $8,000 - $12,000", key='cvta_fixed')
        
        st.markdown("---")
        st.markdown("##### 3. Market Data")

        # Linked to Financial Analysis Market Year
        cvta_year = market_year
        st.info(f"Using Market Year: **{cvta_year}** (See '3. Financial Analysis')")
        
        uploaded_lmp = st.file_uploader("Upload Hourly LMP CSV (Columns: Datetime, Price)", type=['csv'])
        
        # Data Loading
        df_prices = None
        if uploaded_lmp:
            try:
                df_prices = pd.read_csv(uploaded_lmp)
                # Try to parse datetime
                if 'Datetime' in df_prices.columns:
                    df_prices['Datetime'] = pd.to_datetime(df_prices['Datetime'])
                    df_prices.set_index('Datetime', inplace=True)
                elif 'Time' in df_prices.columns:
                    df_prices['Time'] = pd.to_datetime(df_prices['Time'])
                    df_prices.set_index('Time', inplace=True)
                else:
                    st.error("CSV must have 'Datetime' or 'Time' column.")
                    df_prices = None
            except Exception as e:
                st.error(f"Error parsing file: {e}")
        
        
        if df_prices is None:
            # Default to Auto-Load ERCOT Data (Year Based)
            try:
                # Apply Global Scaler if available
                scaler = st.session_state.get('price_scaler_input', 1.0)
                
                price_series = get_market_price_profile_v2(30.0, year=cvta_year) * scaler
                # Create date range for the specific year (use 2024 if Average)
                cvta_year_for_dates = 2024 if cvta_year == 'Average' else cvta_year
                dates = pd.date_range(start=f'{cvta_year_for_dates}-01-01', periods=len(price_series), freq='h')
                df_prices = pd.DataFrame({'Price': price_series.values}, index=dates)
                
                if scaler != 1.0:
                    st.success(f"✅ Using Default Data: **ERCOT HB_NORTH {cvta_year}** (Scaled x{scaler})")
                else:
                    st.success(f"✅ Using Default Data: **ERCOT HB_NORTH {cvta_year}**")
                with st.expander("View Data Preview"):
                     st.dataframe(df_prices.head(24), use_container_width=True)
                     
            except Exception as e:
                st.error(f"Failed to load default data: {e}")
        
        # STORE DATA FOR GLOBAL USE
        if df_prices is not None:
            st.session_state['shared_market_prices'] = df_prices
    
    with col_cvta_charts:
        if df_prices is not None:
            # --- RUN MODEL ---
            # 1. Fixed Leg (Debit)
            # Cap MW * Price/MW-mo * 12 months (Annualized for comparison, but we do monthly)
            # Updated: Removed * 1000 factor as input is now $/MW-mo
            monthly_fixed_cost = cvta_cap * cvta_fixed_price
            
            # 2. Floating Leg (Credit) - Proxy Dispatch
            daily_results = calculate_proxy_battery_revenue(df_prices, cvta_cap, cvta_dur, cvta_rte, cvta_vom)
            
            if not daily_results.empty:
                # Aggregate Monthly
                monthly_results = daily_results.resample('ME').sum()
                monthly_results['Fixed_Payment'] = monthly_fixed_cost
                monthly_results['Net_Settlement'] = monthly_fixed_cost - monthly_results['Net_Revenue'] 
                # Net Settlement > 0: Corporate PAYS (Fixed > Floating) -> Cost
                # Net Settlement < 0: Corporate RECEIVES (Floating > Fixed) -> Gain
                
                monthly_results['Month'] = monthly_results.index.strftime('%b')
                
                # --- VIZ 1: Monthly Settlement Bars ---
                # Stacked Bar: Fixed (Cost) vs Floating (Revenue)
                # Actually standard way to show this is "Net Cost" bar
                fig_settlement = go.Figure()
                
                fig_settlement.add_trace(go.Bar(
                    x=monthly_results['Month'], 
                    y=monthly_results['Fixed_Payment'],
                    name='Fixed Payment (Cost)',
                    marker_color='#d62728' # Red
                ))
                
                fig_settlement.add_trace(go.Bar(
                    x=monthly_results['Month'], 
                    y=monthly_results['Net_Revenue'], # This is the credit back
                    name='Market Revenue (Credit)',
                    marker_color='#2ca02c' # Green
                ))
                
                fig_settlement.update_layout(
                    title='Monthly Settlement: Fixed Payment vs. Market Revenue',
                    barmode='group',
                    yaxis_title='Value ($)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template=chart_template
                )
                st.plotly_chart(fig_settlement, use_container_width=True)
                
                # --- VIZ 2: Cumulative Cash Flow ---
                monthly_results['Cumulative_Cash_Flow'] = -monthly_results['Net_Settlement'].cumsum()
                
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=monthly_results['Month'], 
                    y=monthly_results['Cumulative_Cash_Flow'],
                    mode='lines+markers',
                    name='Cumulative Cash Flow',
                    line=dict(width=3, color='#F63366'),
                    fill='tozeroy'
                ))
                fig_cum.update_layout(
                    title='Cumulative Net Cash Flow (Offtaker Perspective)',
                    yaxis_title='Cumulative $ (Negative = Cost)',
                    template=chart_template
                )
                st.plotly_chart(fig_cum, use_container_width=True)
                
                # --- VIZ 3: Arbitrage Spread Heatmap/Scatter ---
                # daily_results has 'Daily_Spread'
                fig_spread = go.Figure()
                fig_spread.add_trace(go.Scatter(
                    x=daily_results.index,
                    y=daily_results['Daily_Spread'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=daily_results['Daily_Spread'], # Set color equal to value
                        colorscale='Plasma', 
                        showscale=True
                    ),
                    name='Daily Spread'
                ))
                fig_spread.update_layout(
                    title='Daily Price Volatility (Avg High - Avg Low)',
                    yaxis_title='Spread ($/MWh)',
                    template=chart_template
                )
                st.plotly_chart(fig_spread, use_container_width=True)
                
                # Summary Metrics (Top of Col 2)
                total_fixed = monthly_results['Fixed_Payment'].sum()
                total_floating = monthly_results['Net_Revenue'].sum()
                net_outcome = monthly_results['Net_Settlement'].sum()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Annual Fixed Pmt", f"${total_fixed/1e6:.2f}M")
                m2.metric("Annual Market Rev", f"${total_floating/1e6:.2f}M")
                m3.metric("Net Cash Flow", f"${-net_outcome/1e6:.2f}M", 
                          delta=f"{-net_outcome/1e6:.2f}M", delta_color="normal") # Normal: Positive (Green) is Good, Negative (Red) is Cost
            else:
                st.warning("No valid daily dispatch results. Check data or params.")
        else:
             st.info("👈 Upload data or generates test data to see results.")


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

# Initialize variables for global availability (empty states)
total_load_profile = pd.Series(0.0, index=range(8760))
total_annual_load = 0.0
metrics = {}
fin_metrics = {}
pdf_bytes = None
excel_data = None
csv = None
zip_buffer = io.BytesIO()
json_str_full = "{}"
json_str_ai = "{}"

# 1. Calculate Aggregated Load
active_scenario = st.session_state.get("custom_aggregated_profile") is not None or st.session_state.get("participants", [])

if active_scenario:
    # Aggregate Load
    if "custom_aggregated_profile" in st.session_state:
        # Priority 1: Custom Profile from Generator or Upload
        total_load_profile = st.session_state["custom_aggregated_profile"]
    elif st.session_state.get("participants"):
        # Logic for legacy participants
        total_load_profile = pd.Series(0.0, index=range(8760))
        for p in st.session_state.participants:
            # Handle both old 'type/load' and new generator 'category/annual_kwh' if needed
            # but usually participants had type/load
            p_type = p.get('type', p.get('category', 'Flat'))
            p_load = p.get('load', p.get('annual_kwh', 0) / 1000.0) # Assume MWh
            profile = generate_dummy_load_profile(p_load, p_type)
            total_load_profile += profile
        
    total_annual_load = total_load_profile.sum()
    
    # 2. Calculate Generation
    
    # Solar
    if uploaded_solar_file:
        solar_unit_profile = process_uploaded_profile(uploaded_solar_file, keywords=['solar', 'pv', 'generation', 'output'])
        if solar_unit_profile is not None:
             st.session_state['custom_solar_profile'] = solar_unit_profile # Save for persistence
             # Scale by capacity input
             solar_profile = solar_unit_profile * solar_capacity
        else:
             st.error("Error parsing Solar file.")
             st.stop()
    elif 'custom_solar_profile' in st.session_state:
         # Use restored custom profile
         solar_profile = st.session_state['custom_solar_profile'] * solar_capacity
    else:
        solar_profile = generate_dummy_generation_profile(solar_capacity, 'Solar', use_synthetic=use_synthetic_weather, year=market_year)

    # Wind
    if uploaded_wind_file:
        wind_unit_profile = process_uploaded_profile(uploaded_wind_file, keywords=['wind', 'turbine', 'generation', 'output'])
        if wind_unit_profile is not None:
             st.session_state['custom_wind_profile'] = wind_unit_profile # Save for persistence
             wind_profile = wind_unit_profile * wind_capacity
        else:
             st.error("Error parsing Wind file.")
             st.stop()
    elif 'custom_wind_profile' in st.session_state:
         # Use restored custom profile
         wind_profile = st.session_state['custom_wind_profile'] * wind_capacity
    else:
        wind_profile = generate_dummy_generation_profile(wind_capacity, 'Wind', use_synthetic=use_synthetic_weather, year=market_year)

    # Geothermal / Nuclear (still dummy for now, rare to resize profile shape)
    ccs_profile = generate_dummy_generation_profile(ccs_capacity, 'CCS Gas', use_synthetic=use_synthetic_weather, year=market_year)
    geo_profile = generate_dummy_generation_profile(geo_capacity, 'Geothermal', use_synthetic=use_synthetic_weather, year=market_year)
    nuc_profile = generate_dummy_generation_profile(nuc_capacity, 'Nuclear', use_synthetic=use_synthetic_weather, year=market_year)
    
    total_gen_profile = solar_profile + wind_profile + ccs_profile + geo_profile + nuc_profile
    
    # 3. Calculate Surplus/Deficit BEFORE Battery
    surplus = (total_gen_profile - total_load_profile).clip(lower=0)
    deficit = (total_load_profile - total_gen_profile).clip(lower=0)
    
    # 4. Simulate Battery
    
    # Generate Availability Profile (for Outage Simulation)
    availability_profile = pd.Series(batt_capacity, index=range(8760))
    simulate_outages = False # Default to False as tab was removed
    if simulate_outages and batt_capacity > 0:
        # Create random outages (~2% of year = ~175 hours)
        # Use a fixed seed for reproducibility of the "random" outages in this session
        rng_outage = np.random.default_rng(42)
        outage_mask = rng_outage.random(8760) < 0.02 # 2% probability
        availability_profile[outage_mask] = 0.0 # Full outage for that hour
    
    batt_discharge, batt_soc, batt_charge = simulate_battery_storage(surplus, deficit, batt_capacity, batt_duration, availability_profile)
    
    # 5. Final Matching
    total_gen_with_battery = total_gen_profile + batt_discharge
    cfe_score, matched_profile = calculate_cfe_score(total_load_profile, total_gen_with_battery)
    
    # Calculate detailed metrics
    total_gen_capacity = solar_capacity + wind_capacity + ccs_capacity + geo_capacity + nuc_capacity
    metrics = calculate_portfolio_metrics(total_load_profile, matched_profile, total_gen_capacity)
    metrics['clean_gen_mwh'] = total_gen_profile.sum()
    metrics['total_load_mwh'] = total_load_profile.sum() # Ensure this is also available in metrics if needed, or stick to scenario_config consistency
    
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

    
    # Calculate Effective Battery Price ($/MWh) from Capacity Payment ($/kW-mo)
    # Input: batt_price ($/kW-mo)
    # Annual Cost per MW = batt_price * 1000 kW/MW * 12 months
    # Annual Cost = (batt_price * 12000) * batt_capacity
    # Effective $/MWh = Annual Cost / Total Annual Discharge MWh
    
    total_discharge_mwh = batt_discharge.sum()
    
    # Helper to generate market price series for the financial calc
    market_price_profile_series = get_market_price_profile_v2(market_price, year=market_year) * price_scaler
    
    batt_ops_data = {
        'available_mw_profile': availability_profile,
        'discharge_mwh_profile': batt_discharge,
        'charge_mwh_profile': batt_charge,
        'market_price_profile': market_price_profile_series
    }

    if batt_capacity > 0:
        # Calculate Battery Financials Detailed
        if 'cvta_fixed_price' not in locals(): cvta_fixed_price = 12000.0
        if 'cvta_rte' not in locals(): cvta_rte = 85.0
        if 'cvta_vom' not in locals(): cvta_vom = 2.0
            
        # CVTA Logic Alignment
        # 1. Run Proxy Dispatch (Financial)
        
        # Use SHARED Market Data from CVTA Tab if available to ensure match
        if 'shared_market_prices' in st.session_state:
            df_proxy_input = st.session_state['shared_market_prices']
        else:
            # Fallback (Should typically not happen if app renders top-down)
            # Use 2024 for date generation if Average is selected
            year_for_dates = 2024 if market_year == 'Average' else market_year
            dates = pd.date_range(start=f'{year_for_dates}-01-01', periods=8760, freq='h')
            df_proxy_input = pd.DataFrame({'Price': market_price_profile_series.values}, index=dates)
        
        cvta_daily_results = calculate_proxy_battery_revenue(df_proxy_input, batt_capacity, batt_duration, cvta_rte, cvta_vom)
        
        if cvta_daily_results is not None:
            annual_market_revenue = cvta_daily_results['Net_Revenue'].sum()
            annual_fixed_payment = batt_capacity * cvta_fixed_price * 12 # Monthly * 12
            
            # Net Invoice (Cost to Offtaker) = Fixed Payment - Market Revenue
            net_invoice = annual_fixed_payment - annual_market_revenue
            
            batt_financials = {
                'net_invoice': net_invoice,
                'capacity_payment': annual_fixed_payment,
                'vom_payment': 0.0, # Implicit in net revenue for proxy
                'rte_penalty': 0.0, # Not explicit in this deal structure
                'actual_availability': 1.0,
                'actual_rte': cvta_rte / 100.0,
                # Store extra metadata for global table override
                # Proxy model discharges full duration every day of the results
                'financial_mwh': len(cvta_daily_results) * batt_capacity * batt_duration,
                'market_revenue': annual_market_revenue
            }
        else:
             batt_financials = {
                'net_invoice': 0.0, 'capacity_payment': 0.0, 
                'vom_payment': 0.0, 'rte_penalty': 0.0,
                'actual_availability': 1.0, 'actual_rte': 0.0,
                'financial_mwh': 0.0, 'market_revenue': 0.0
            }
        
        # Effective Price for Global Financials logic (Net Cost / MWh)
        # Avoid div by zero
        effective_batt_price_mwh = batt_financials['net_invoice'] / total_discharge_mwh if total_discharge_mwh > 0 else 0.0
    else:
        # No Battery
        batt_financials = {
            'net_invoice': 0.0, 'capacity_payment': 0.0, 
            'vom_payment': 0.0, 'rte_penalty': 0.0,
            'actual_availability': 1.0, 'actual_rte': 0.0,
            'financial_mwh': 0.0, 'market_revenue': 0.0
        }
        effective_batt_price_mwh = 0.0
        
    tech_prices = {
        'Solar': solar_price_eff,
        'Wind': wind_price_eff,
        'CCS Gas': ccs_price_eff,
        'Geothermal': geo_price_eff,
        'Nuclear': nuc_price_eff,
        'Battery': effective_batt_price_mwh # Use calculated effective price
    }
    
    fin_metrics = calculate_financials(matched_profile, deficit, tech_profiles, tech_prices, market_price, rec_price, price_scaler, year=market_year)
    
    # --- GLOBAL TABLE OVERRIDE FOR BATTERY (Ensure Match with CVTA) ---
    if 'Battery' in fin_metrics['tech_details'] and batt_capacity > 0:
        # Override the Physical Dispatch based numbers with Financial Dispatch Numbers from CVTA
        # Net Settlement (Value - Cost) should equal -Net Invoice (Revenue - Fixed)
        # Note: Global Table Logic is: Settlement = Market Value - PPA Cost.
        # So we map: PPA Cost -> Fixed Payment, Market Value -> Financial Revenue.
        
        f_mwh = batt_financials.get('financial_mwh', 0.0)
        f_rev = batt_financials.get('market_revenue', 0.0)
        f_cost = batt_financials.get('capacity_payment', 0.0) # Fixed Payment
        
        fin_metrics['tech_details']['Battery'] = {
            'Matched_MWh': f_mwh,
            'PPA_Price': (f_cost / f_mwh) if f_mwh > 0 else 0.0, # Implied Price
            'Total_Cost': f_cost, # Fixed Payment
            'Market_Value': f_rev, # Financial Revenue
            'Settlement': f_rev - f_cost # Net Settlement (should be neg of Net Invoice)
        }
        
        # Recalculate Global Metrics to reflect override
        new_total_ppa_cost = sum(d['Total_Cost'] for d in fin_metrics['tech_details'].values())
        new_market_value_matched = sum(d['Market_Value'] for d in fin_metrics['tech_details'].values())
        
        fin_metrics['settlement_value'] = new_market_value_matched - new_total_ppa_cost
        fin_metrics['net_cost'] = (total_annual_load - matched_profile.sum()) * (market_price * price_scaler) + new_total_ppa_cost + fin_metrics['rec_cost']
        fin_metrics['weighted_ppa_price'] = new_total_ppa_cost / matched_profile.sum() if matched_profile.sum() > 0 else 0.0
    
    # --- Capture Detailed Financials for Scenario Comparison ---
    # Recalculate or grab final totals to ensure consistency (works for both if/else cases)
    final_ppa_cost = sum(d['Total_Cost'] for d in fin_metrics['tech_details'].values())
    final_market_revenue = sum(d['Market_Value'] for d in fin_metrics['tech_details'].values())
    
    st.session_state.cfe_score = cfe_score
    st.session_state.avg_ppa_price = fin_metrics['weighted_ppa_price']
    st.session_state.net_settlement = fin_metrics['settlement_value']
    st.session_state.total_cost = fin_metrics['net_cost']
    st.session_state.total_load_mwh = total_annual_load
    
    # Detailed Financials
    st.session_state.market_revenue = final_market_revenue
    st.session_state.gross_ppa_cost = final_ppa_cost
    st.session_state.fixed_costs = fin_metrics['net_cost'] - fin_metrics['settlement_value'] # Proxy? No.
    # Total Cost = Deficit Cost + PPA Cost + REC Cost.
    # Net Settlement = Market Revenue - PPA Cost.
    # This is getting confusing. Let's just store the explicit components.
    st.session_state.rec_cost = fin_metrics['rec_cost']
    st.session_state.deficit_cost = fin_metrics['net_cost'] - final_ppa_cost - fin_metrics['rec_cost']
    
    # --- Populate Executive Summary Container ---
    with exec_summary_container:
        st.divider()
        st.caption("EXECUTIVE SUMMARY (Live Results | Region: ERCOT North)")
        ec1, ec2, ec3 = st.columns(3)
        
        # 1. CFE Score
        safe_cfe = metrics['cfe_score'] # Local variable from above
        cfe_delta = safe_cfe - 0.90 # Compare vs 90% benchmark
        ec1.metric(
            "CFE Score (24/7)", 
            f"{safe_cfe:.1%}", 
            delta=f"{cfe_delta:.1%} vs 90% Target", 
            help="Carbon-Free Energy Score: The % of your hourly load matched by clean generation."
        )
        
        # 2. Net Settlement
        net_set = fin_metrics['settlement_value']
        ec2.metric(
            "Est. Annual Benefit/Cost", 
            f"${net_set/1e6:,.2f}M", 
            delta="Net Benefit" if net_set>=0 else "Net Cost",
            delta_color="normal" if net_set>=0 else "inverse",
            help="Total Market Value of Generation minus Fixed PPA Costs."
        )
        
        # 3. Dynamic Insight
        insight = "✅ Portfolio performing well."
        if safe_cfe < 0.80:
            insight = "⚠️ **Low CFE**: Consider increasing Firm Generation (Geothermal/Nuclear) or Battery Duration."
        elif safe_cfe < 0.90:
            insight = "📈 **Optimization Opportunity**: A 4-hour battery could help reach >90% CFE."
        elif net_set < 0 and abs(net_set) > 5000000:
            insight = "💰 **High Cost**: Review PPA Strike Prices or reduce expensive firm capacity."
            
        if "✅" in insight:
            ec3.success(insight)
        elif "📈" in insight:
            ec3.info(insight)
        else:
            ec3.warning(insight)
        
    # --- Dashboard moved to Tabs ---
    
    # --- Tab 2: Generation Portfolio (Results) ---
    with tab_gen:
        st.markdown("---")
        st.markdown("#### Operational Analysis")
        
        # Metrics - Row 1
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Electricity Usage", f"{total_annual_load:,.0f} MWh")
        col2.metric("Clean Energy Generation", f"{total_gen_profile.sum():,.0f} MWh", help="Total renewable generation + nuclear")
        col3.metric("CFE Score (24/7)", f"{metrics['cfe_score']:.1%}", help="Percentage of total load met by Carbon Free Energy generation in the same hour")
        
        annual_clean_ratio = (total_gen_profile.sum()) / total_annual_load if total_annual_load > 0 else 0
        col4.metric("Annual Clean Energy / Annual Load", f"{annual_clean_ratio:.1%}", help="Ratio of Total Clean Generation to Total Load")
        
        col5.metric("Battery Discharge", f"{batt_discharge.sum():,.0f} MWh")
        
        # Metrics - Row 2
        col6, col7, col8, col9 = st.columns(4)
        col6.metric("MW Match Productivity", f"{metrics['productivity']:,.0f} MWh/MW", help="MWh of Clean Energy Matched per MW of Installed Capacity")
        col7.metric("Loss of Green Hours", f"{metrics['logh']:.1%}", help="% of hours where load is not fully matched by clean energy")
        col8.metric("Grid Consumption", f"{metrics['grid_consumption']:,.0f} MWh", help="Total energy drawn from grid (deficit)")
        col9.metric("Excess Generation", f"{surplus.sum():,.0f} MWh", help="Gross overgeneration before battery charging")

        # Charts
        st.markdown("---")
        st.markdown("#### Hourly Energy Balance (Sample Week)")
        
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
        
        # Calculate Total Supply if not already (Gen + Battery)
        if 'total_gen_with_battery' not in locals():
             total_gen_with_battery = total_gen_profile + batt_discharge
        # Stacked generation profiles
        # Stacked generation profiles - Logic: Baseload first, then Battery, then VRE
        # This stacking order helps visualize how load is met
        fig.add_trace(go.Scatter(x=x_axis, y=nuc_profile[start_hour:end_hour], name='Nuclear Gen', stackgroup='one', line=dict(color='purple'), fill='tonexty'))
        fig.add_trace(go.Scatter(x=x_axis, y=geo_profile[start_hour:end_hour], name='Geothermal Gen', stackgroup='one', line=dict(color='#e6550d'), fill='tonexty'))
        fig.add_trace(go.Scatter(x=x_axis, y=ccs_profile[start_hour:end_hour], name='CCS Gas Gen', stackgroup='one', line=dict(color='brown'), fill='tonexty'))
        
        # Put Battery in the middle (filling gaps)
        fig.add_trace(go.Scatter(x=x_axis, y=batt_discharge[start_hour:end_hour], name='Battery Discharge', stackgroup='one', line=dict(color='#1f77b4'), fill='tonexty'))
        
        # VRE Top
        fig.add_trace(go.Scatter(x=x_axis, y=wind_profile[start_hour:end_hour], name='Wind Gen', stackgroup='one', line=dict(color='lightblue'), fill='tonexty'))
        fig.add_trace(go.Scatter(x=x_axis, y=solar_profile[start_hour:end_hour], name='Solar Gen', stackgroup='one', line=dict(color='gold'), fill='tonexty'))

        # Add Line Plots LAST (to stay on top)
        # 1. Total Supply
        fig.add_trace(go.Scatter(x=x_axis, y=total_gen_with_battery[start_hour:end_hour],
                                 mode='lines', name='Total Supply (Gen+Batt)', line=dict(color='#2ca02c', width=2)))
                                 
        # 2. Aggregated Load (The most important line)
        fig.add_trace(go.Scatter(x=x_axis, y=total_load_profile[start_hour:end_hour],
                                 mode='lines', name='Aggregated Load', line=dict(color='red', width=2)))

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
            ),
            hoverlabel=dict(bgcolor="#333333", font_size=12, font_family="Arial", font=dict(color="white"))
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ℹ️ **How to read this:** The Red line is your usage. Stacked colors are detailed generation. Any gap between the stack and the red line is a **Deficit** (Grid Power).")
        
        st.subheader("Monthly Analysis")
        
        # Group by month
        # Create a simple dataframe for grouping
        df_hourly = pd.DataFrame({
            'Load': total_load_profile,
            'Generation': total_gen_profile,
            'Battery': batt_discharge,
            'Total_Supply': total_gen_profile + batt_discharge,
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
        
        # 1. Total Supply (Gen + Battery) (Background)
        fig_bar.add_trace(go.Bar(x=monthly_stats.index, y=monthly_stats['Total_Supply'], name='Total Supply (Gen+Batt)', marker_color='#2ca02c', opacity=0.6)) # Standard Green
        
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
            legend=dict(traceorder='reversed'),
            hoverlabel=dict(bgcolor="#333333", font_size=12, font_family="Arial", font=dict(color="white"))
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
            zmin=0,
            zmax=1,
            hovertemplate='Date: %{x|%b %d}<br>Time: %{y}<br>Matched: %{z:.1%}<extra></extra>'
        ))
        
        fig_heat.update_layout(
            title="24/7 Matching Heatmap", 
            xaxis_title="Date", 
            yaxis_title="Time of Day", 
            template=chart_template,
            paper_bgcolor=chart_bg,
            plot_bgcolor=chart_bg,
            font=dict(color=chart_font_color),
            hoverlabel=dict(bgcolor="#333333", font_size=12, font_family="Arial", font=dict(color="white"))
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("ℹ️ **Heatmap Guide:** Green areas = 100% Carbon Free. Red areas = Relying on Grid. Look for 'Red Bands' (e.g., Summer Evenings) to guide your battery strategy.")
    
    # --- PPA vs Capture Value Analysis (Last Chart) ---
    with tab_fin:
        st.markdown("---")
        st.markdown("#### Financial Analysis Result")
        
        # Metrics - Row 3 (Financials)
        st.subheader("Financial Overview")
        col9, col10, col11, col12, col13 = st.columns(5)
        col9.metric("Annual PPA Settlement Value", f"${fin_metrics['settlement_value']:,.0f}", help="Annual Revenue (or Cost) from PPA Settlement: (Market - Strike) * Matched Vol")
        col10.metric("Weighted Avg PPA Price", f"${fin_metrics['weighted_ppa_price']:.2f}/MWh", help="Average cost of matched energy based on technology mix")
        col11.metric(f"Capture Value ({market_year} Base)", f"${fin_metrics['weighted_market_price']:.2f}/MWh", help=f"Average market value of matched energy ({market_year} ERCOT prices × scaler)")
        col12.metric("REC Value", f"${fin_metrics['rec_cost']:,.0f}", help="Value of RECs (Matched)")
        
        # Calculate Value of Excess RECs
        excess_rec_value = surplus.sum() * rec_price
        fin_metrics['excess_rec_value'] = excess_rec_value # Add to metrics for PDF
        col13.metric("Excess REC Value", f"${excess_rec_value:,.0f}", help="Potential value of RECs from excess generation")

        st.markdown("---")
        st.subheader("Economic Analysis")
        st.markdown("PPA prices vs market capture values for each technology")
        
        # Calculate capture value for each technology
        market_price_profile = get_market_price_profile_v2(market_price, year=market_year) * price_scaler
        
        tech_data = []
        if 'tech_details' in fin_metrics:
            for tech, details in fin_metrics['tech_details'].items():
                if details['Matched_MWh'] > 0:
                    tech_data.append({
                        'Technology': tech,
                        'PPA Price': details['PPA_Price'],
                        'Capture Value': details['Market_Value'] / details['Matched_MWh'],
                        'Spread': (details['Market_Value'] / details['Matched_MWh']) - details['PPA_Price']
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
                name=f'Capture Value ({market_year} Base)',
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
                yaxis_title='Exchanged Price ($/MWh)',
                legend=dict(x=0.01, y=0.99),
                height=500,  # Increased from 400 for more space
                hovermode='x unified',
                template=chart_template,
                paper_bgcolor=chart_bg,
                plot_bgcolor=chart_bg,
                font=dict(color=chart_font_color),
                margin=dict(t=50, b=50, l=50, r=50),  # Add margins for labels
                yaxis=dict(tickprefix="$"),
                hoverlabel=dict(bgcolor="#333333", font_size=12, font_family="Arial", font=dict(color="white"))
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

        # --- Settlement by Technology ---
        if 'tech_details' in fin_metrics:
            st.markdown("---")
            st.subheader("Settlement by Technology")
            st.markdown("Detailed breakdown of costs and market value by generation source.")
            
            # Convert to DataFrame
            settlement_data = []
            for tech, details in fin_metrics['tech_details'].items():
                if details['Matched_MWh'] > 0:
                    settlement_data.append({
                        'Technology': tech,
                        'Generation (MWh)': details['Matched_MWh'],
                        'PPA Price ($/MWh)': details['PPA_Price'],
                        'PPA Cost ($)': details['Total_Cost'],
                        'Market Value ($)': details['Market_Value'],
                        'Net Settlement ($)': details['Settlement'],
                        'Value ($/MWh)': details['Market_Value'] / details['Matched_MWh'] if details['Matched_MWh'] > 0 else 0
                    })
            
            if settlement_data:
                df_settlement = pd.DataFrame(settlement_data)
                
                # Calculate Totals
                total_gen = df_settlement['Generation (MWh)'].sum()
                total_cost = df_settlement['PPA Cost ($)'].sum()
                total_value = df_settlement['Market Value ($)'].sum()
                total_settlement_val = df_settlement['Net Settlement ($)'].sum()
                
                # Weighted Averages
                avg_ppa_price = total_cost / total_gen if total_gen > 0 else 0
                avg_val_price = total_value / total_gen if total_gen > 0 else 0
                
                # Append Total Row using pd.concat
                total_row = pd.DataFrame([{
                    'Technology': '**Total**',
                    'Generation (MWh)': total_gen,
                    'PPA Price ($/MWh)': avg_ppa_price,
                    'PPA Cost ($)': total_cost,
                    'Market Value ($)': total_value,
                    'Net Settlement ($)': total_settlement_val,
                    'Value ($/MWh)': avg_val_price
                }])
                
                df_settlement = pd.concat([df_settlement, total_row], ignore_index=True)
                
                # Format columns
                st.dataframe(
                    df_settlement.style.format({
                        'Generation (MWh)': '{:,.0f}',
                        'PPA Price ($/MWh)': '${:,.2f}',
                        'PPA Cost ($)': '${:,.0f}',
                        'Market Value ($)': '${:,.0f}',
                        'Net Settlement ($)': '${:,.0f}',
                        'Value ($/MWh)': '${:,.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Simple Bar Chart for Settlement (Exclude Total Row)
                fig_set = go.Figure()
                
                # Slice to exclude the last row (Total)
                df_chart = df_settlement.iloc[:-1]
                
                colors = ['#2ca02c' if x >= 0 else '#d62728' for x in df_chart['Net Settlement ($)']]
                
                fig_set.add_trace(go.Bar(
                    x=df_chart['Technology'],
                    y=df_chart['Net Settlement ($)'],
                    marker_color=colors,
                    text=df_chart['Net Settlement ($)'],
                    texttemplate='$%{text:,.0f}',
                    textposition='auto'
                ))
                
                fig_set.update_layout(
                    title="Net Financial Settlement by Tech (Value - Cost)",
                    yaxis_title="Net Value ($)",
                    template=chart_template,
                    height=400,
                     paper_bgcolor=chart_bg,
                    plot_bgcolor=chart_bg,
                    font=dict(color=chart_font_color),
                    hoverlabel=dict(bgcolor="#333333", font_size=12, font_family="Arial", font=dict(color="white"))
                )
                
                st.plotly_chart(fig_set, use_container_width=True)

        # --- Historical Sensitivity Analysis ---
        st.markdown("---")
        with st.expander("📊 Historical Sensitivity (2020-2025)", expanded=False):
            st.markdown("Run your current portfolio settings against historical ERCOT North Hub prices.")
            
            if st.button("Run Multi-Year Analysis"):
                sensitivity_results = []
                years_to_test = [2020, 2021, 2022, 2023, 2024, 2025]
                
                progress_bar = st.progress(0)
                
                for i, year in enumerate(years_to_test):
                    if str(year) == str(market_year) and 'market_price_profile_series' in locals():
                         # Exact Match: Use exact price profile from main run (respects user "Market Price" input)
                         hist_price_series = market_price_profile_series
                    else:
                         # 1. Load Data using robust utility (skip scaling to get actual historical prices, then apply manual scaler)
                         # Apply User's Price Scaler to historical data to simulate "Current Conditions applied to History"
                         scaler = st.session_state.get('price_scaler_input', 1.0)
                         hist_prices = get_market_price_profile_v2(30.0, year=year, scale_hist=False) * scaler
                         
                         # Ensure series length matches profile (truncate or pad if necessary, though utils handles it)
                         h_vals = hist_prices.values
                         if len(h_vals) > 8760: h_vals = h_vals[:8760]
                         if len(h_vals) < 8760:
                             h_vals = np.pad(h_vals, (0, 8760-len(h_vals)), 'edge')
                         
                         hist_price_series = pd.Series(h_vals)
                    
                    # 2. Calculate Portfolio Settlement (Techs)
                    # Net Settlement = Market Revenue - PPA Cost
                    
                    net_solar = 0.0
                    net_wind = 0.0
                    net_firm = 0.0
                    net_battery = 0.0
                    
                    if solar_capacity > 0:
                        if str(year) == str(market_year) and 'solar_profile' in locals():
                            # Exact Match: Use exact profile from main run
                            sens_solar_profile = solar_profile 
                        elif 'custom_solar_profile' in st.session_state:
                             # Consistency Fix: Use Custom Profile if available (Static logic)
                             sens_solar_profile = st.session_state['custom_solar_profile'] * solar_capacity
                        else:
                             # Regenerate profile for this specific historical year (weather varies)
                             sens_solar_profile = generate_dummy_generation_profile(solar_capacity, 'Solar', use_synthetic=use_synthetic_weather, year=year)
                        
                        rev = np.sum(sens_solar_profile.values * hist_price_series.values)
                        cost = np.sum(sens_solar_profile.values * solar_price_eff)
                        net_solar = (rev - cost)
                        
                    # Wind
                    if wind_capacity > 0:
                        if str(year) == str(market_year) and 'wind_profile' in locals():
                            sens_wind_profile = wind_profile
                        elif 'custom_wind_profile' in st.session_state:
                             sens_wind_profile = st.session_state['custom_wind_profile'] * wind_capacity
                        else:
                            # Regenerate profile for this specific historical year
                            sens_wind_profile = generate_dummy_generation_profile(wind_capacity, 'Wind', use_synthetic=use_synthetic_weather, year=year)
                        
                        rev = np.sum(sens_wind_profile.values * hist_price_series.values)
                        cost = np.sum(sens_wind_profile.values * wind_price_eff)
                        net_wind = (rev - cost)
                        
                    # Firm (CCS/Geo/Nuc)
                    firm_specs = [
                        (ccs_capacity, ccs_price_eff, 'CCS Gas', ccs_profile if 'ccs_profile' in locals() else None), 
                        (geo_capacity, geo_price_eff, 'Geothermal', geo_profile if 'geo_profile' in locals() else None), 
                        (nuc_capacity, nuc_price_eff, 'Nuclear', nuc_profile if 'nuc_profile' in locals() else None)
                    ]
                    
                    for cap, price, name, main_prof in firm_specs:
                        if cap > 0:
                             if str(year) == str(market_year) and main_prof is not None:
                                 prof = main_prof
                             else:
                                 # Use standard profile generator for consistency
                                 prof = generate_dummy_generation_profile(cap, name, use_synthetic=use_synthetic_weather, year=year)
                                 
                             rev = np.sum(prof.values * hist_price_series.values)
                             cost = np.sum(prof.values * price)
                             net_firm += (rev - cost)

                    # 3. Battery Financials (CVTA)
                    if batt_capacity > 0:
                        # Construct DF for function - MUST match main analysis data source
                        if str(year) == str(market_year) and 'shared_market_prices' in st.session_state:
                            # Exact Match: Use the same price data source as main battery calculation
                            df_p = st.session_state['shared_market_prices']
                        else:
                            # Historical year: construct price DF with correct calendar
                            loop_year_for_dates = 2024 if str(year) == 'Average' else int(year)
                            ts = pd.date_range(start=f'{loop_year_for_dates}-01-01', periods=8760, freq='h')
                            df_p = pd.DataFrame({'Price': hist_price_series.values}, index=ts)
                        
                        # Use Session State values if available, otherwise defaults
                        c_rte = st.session_state.get('cvta_rte', 85.0)
                        c_vom = st.session_state.get('cvta_vom', 2.0)
                        c_fixed = st.session_state.get('cvta_fixed', 12000.0)
                        
                        dr = calculate_proxy_battery_revenue(df_p, batt_capacity, batt_duration, c_rte, c_vom)
                        if dr is not None:
                            mkt_rev = dr['Net_Revenue'].sum()
                            fixed_pymt = batt_capacity * c_fixed * 12
                            # Net Settlement = Market Revenue - Fixed Payment
                            net_battery = mkt_rev - fixed_pymt
                    
                    total_settlement = net_solar + net_wind + net_firm + net_battery
                    
                    # --- CONSISTENCY CHECK ---
                    if str(year) == str(market_year):
                         # Get main result from session state
                         main_result = st.session_state.get('net_settlement', 0.0)
                         diff = abs(total_settlement - main_result)
                         if diff < 1000: # Allow small float error ($1000 tolerance on millions)
                             st.toast(f"✅ Year {year} Re-verification Passed! (Diff: ${diff:.0f})", icon="✅")
                         else:
                             st.toast(f"⚠️ Verification Mismatch for {year}! Main: ${main_result:,.0f} vs Loop: ${total_settlement:,.0f}", icon="⚠️")
                    
                    total_settlement = net_solar + net_wind + net_firm + net_battery
                    
                    sensitivity_results.append({
                        'Year': str(year),
                        'Total Net Settlement': total_settlement,
                        'Solar': net_solar,
                        'Wind': net_wind,
                        'Firm': net_firm,
                        'Battery': net_battery
                    })
                    
                    progress_bar.progress((i + 1) / len(years_to_test))
                
                # Render Results
                s_df = pd.DataFrame(sensitivity_results)
                # Calculate Cumulative
                s_df['Cumulative Net Settlement'] = s_df['Total Net Settlement'].cumsum()
                
                st.write("### Multi-Year Results")
                
                # 1. Chart (Top)
                fig_sens = go.Figure()
                
                # Grouped Bars
                if s_df['Solar'].abs().sum() > 0:
                    fig_sens.add_trace(go.Bar(name='Solar', x=s_df['Year'], y=s_df['Solar'], marker_color='#FFA500', hovertemplate='$%{y:,.0f}'))
                if s_df['Wind'].abs().sum() > 0:
                    fig_sens.add_trace(go.Bar(name='Wind', x=s_df['Year'], y=s_df['Wind'], marker_color='#1f77b4', hovertemplate='$%{y:,.0f}'))
                if s_df['Firm'].abs().sum() > 0:
                    fig_sens.add_trace(go.Bar(name='Firm', x=s_df['Year'], y=s_df['Firm'], marker_color='grey', hovertemplate='$%{y:,.0f}'))
                if s_df['Battery'].abs().sum() > 0:
                    fig_sens.add_trace(go.Bar(name='Battery', x=s_df['Year'], y=s_df['Battery'], marker_color='#2ca02c', hovertemplate='$%{y:,.0f}'))
                
                # Cumulative Net Total Line
                fig_sens.add_trace(go.Scatter(
                    name='Cumulative Net Total', 
                    x=s_df['Year'], 
                    y=s_df['Cumulative Net Settlement'], 
                    mode='lines+markers+text',
                    text=s_df['Cumulative Net Settlement'],
                    texttemplate='$%{text:,.2s}',
                    textposition='top center',
                    line=dict(color='#9467bd', width=3), # Purple
                    hovertemplate='$%{y:,.0f}'
                ))
                
                fig_sens.update_layout(
                    title="Portfolio Financial Performance by Source (2020-2025)",
                    yaxis_title="Net Settlement ($)",
                    barmode='group', # Grouped bars next to each other
                    template=chart_template,
                    paper_bgcolor=chart_bg,
                    plot_bgcolor=chart_bg,
                    font=dict(color=chart_font_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_sens, use_container_width=True)

                # 2. Table (Bottom)
                # Format table
                display_cols = ['Year', 'Solar', 'Wind', 'Firm', 'Battery', 'Total Net Settlement', 'Cumulative Net Settlement']
                # Filter out columns that are all 0 (except totals)
                cols_to_show = ['Year']
                for c in ['Solar', 'Wind', 'Firm', 'Battery']:
                    if s_df[c].sum() != 0:
                        cols_to_show.append(c)
                cols_to_show.append('Total Net Settlement')
                cols_to_show.append('Cumulative Net Settlement')
                        
                st.dataframe(s_df[cols_to_show].style.format({
                    'Solar': '${:,.0f}',
                    'Wind': '${:,.0f}',
                    'Firm': '${:,.0f}',
                    'Battery': '${:,.0f}',
                    'Total Net Settlement': '${:,.0f}',
                    'Cumulative Net Settlement': '${:,.0f}'
                }), use_container_width=True)




            st.subheader("Financial Analysis Results")



    # --- Data Export ---

    
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

    # Add Individual Participant Loads
    if st.session_state.participants:
        for p in st.session_state.participants:
            # Generate profile for this participant
            if p.get('is_external') and 'external_profiles' in st.session_state and p['name'] in st.session_state.external_profiles:
                # Convert kW to MW for analysis results
                p_profile = st.session_state.external_profiles[p['name']].values / 1000.0
            else:
                # Re-generate synthetic if not external
                # Re-generate to ensure consistency (using same seed/logic in utils)
                p_profile = generate_dummy_load_profile(p['load'], p['type'])
            
            # Add to DF with formatted name
            col_name = f"{p['name']} ({p['type']})_MW"
            results_df[col_name] = p_profile

    # --- Financial Columns for CSV ---
    # --- Financial Columns for CSV / Excel ---
    # 1. Market Price (Hourly) - Current Scenario
    # Apply the price scaler to match Financial Analysis
    market_price_profile = get_market_price_profile_v2(market_price, year=market_year) * price_scaler
    results_df['Market_Price_$/MWh'] = market_price_profile

    # 1b. Market Prices (All Years) - For Excel Analysis
    all_years_hist = [2020, 2021, 2022, 2023, 2024, 2025, "Average"]
    for yr_hist in all_years_hist:
        # Get raw profile for that year, scaled by the scenario's price scaler
        # We use the scenario 'market_price' input as base if year='Average', 
        # or just the historical shape scaled.
        # Use raw historical shape (scale_hist=False) so we don't normalize to 30.0.
        # Then apply the price scaler.
        yr_profile = get_market_price_profile_v2(0.0, year=yr_hist, scale_hist=False) * price_scaler
        
        # Column Name
        col_lbl = f"Price_{yr_hist} ($/MWh)"
        results_df[col_lbl] = yr_profile
    
    # 2. Technology PPA Prices (Constant)
    results_df['PPA_Price_Solar'] = solar_price_eff
    results_df['PPA_Price_Wind'] = wind_price_eff
    results_df['PPA_Price_Geothermal'] = geo_price_eff
    results_df['PPA_Price_Nuclear'] = nuc_price_eff
    results_df['PPA_Price_CCS'] = ccs_price_eff
    results_df['PPA_Price_Battery'] = effective_batt_price_mwh

    # 3. Hourly Settlement by Technology
    # Settlement = Generation * (Market Price - PPA Price)
    
    # Solar
    results_df['Settlement_Solar_($)'] = solar_profile * (market_price_profile - solar_price_eff)
    
    # Wind
    results_df['Settlement_Wind_($)'] = wind_profile * (market_price_profile - wind_price_eff)
    
    # Geothermal
    results_df['Settlement_Geothermal_($)'] = geo_profile * (market_price_profile - geo_price_eff)
    
    # Nuclear
    results_df['Settlement_Nuclear_($)'] = nuc_profile * (market_price_profile - nuc_price_eff)
    
    # CCS
    results_df['Settlement_CCS_($)'] = ccs_profile * (market_price_profile - ccs_price_eff)
    
    # Battery (Discharge)
    results_df['Settlement_Battery_($)'] = batt_discharge * (market_price_profile - effective_batt_price_mwh)
    
    # Total Hourly Settlement
    results_df['Total_Hourly_Settlement_($)'] = (
        results_df['Settlement_Solar_($)'] +
        results_df['Settlement_Wind_($)'] +
        results_df['Settlement_Geothermal_($)'] +
        results_df['Settlement_Nuclear_($)'] +
        results_df['Settlement_CCS_($)'] +
        results_df['Settlement_Battery_($)']
    )
    
    # Recalculate effective battery price for CSV if local variable not available in scope
    # (Though it should be available since defined above in main script flow)
    # Using 'effective_batt_price_mwh' calculated earlier 
    results_df['PPA_Price_Battery'] = effective_batt_price_mwh
    
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

    # --- Prepare Artifacts for Export ---
    
    # 1. Scenario Configuration (Full)
    export_config = {
        "region": "ERCOT North",
        "total_load_mwh": total_annual_load,
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
        "batt_base_rate": cvta_fixed_price if 'cvta_fixed_price' in locals() else 12000.0,
        "batt_guar_avail": 0.98,
        "batt_guar_rte": cvta_rte if 'cvta_rte' in locals() else 85.0,
        "batt_vom": cvta_vom if 'cvta_vom' in locals() else 2.0,
        "market_price": market_price,
        "rec_price": rec_price,
        "participants": st.session_state.participants,
        "load_gen_sites": st.session_state.get('load_gen_sites', []),
        "excluded_techs": st.session_state.get('excluded_techs', [])
    }
    # Add optional large arrays
    if 'custom_solar_profile' in st.session_state:
        export_config['custom_solar_profile'] = st.session_state['custom_solar_profile'].tolist()
    if 'custom_wind_profile' in st.session_state:
        export_config['custom_wind_profile'] = st.session_state['custom_wind_profile'].tolist()
    if 'shared_market_prices' in st.session_state:
        df_prices = st.session_state['shared_market_prices']
        if 'Price' in df_prices.columns:
            export_config['custom_battery_prices'] = df_prices['Price'].tolist()
            
    json_str_full = json.dumps(export_config, indent=4, cls=NumpyEncoder)

    # 2. AI-Optimized Configuration
    ai_config = export_config.copy()
    ai_config.pop('custom_solar_profile', None)
    ai_config.pop('custom_wind_profile', None)
    ai_config.pop('custom_battery_prices', None)
    ai_config['region'] = "ERCOT North"
    ai_config['total_load_mwh'] = total_annual_load
    
    json_str_ai = json.dumps(ai_config, indent=4, cls=NumpyEncoder)

    # 3. PDF Report
    figures = {}
    if 'fig' in locals(): figures["Hourly Energy Balance"] = fig
    if 'fig_bar' in locals(): figures["Monthly Analysis"] = fig_bar
    if 'fig_heat' in locals(): figures["24/7 Heatmap"] = fig_heat
    if 'fig_ppa' in locals(): figures["PPA Price vs Capture Value"] = fig_ppa
    if 'fig_set' in locals(): figures["Net Settlement by Tech"] = fig_set
        
    # Pass matched projects from session state
    matched_projects = st.session_state.get('matched_projects', {})
    pdf_bytes = generate_pdf_report(metrics, export_config, fin_metrics, figures=figures, matched_projects=matched_projects)

    # 4. Create ZIP Bundle (Updated to include Excel)
    zip_buffer = io.BytesIO()
    
    # 5. Create Excel Report
    excel_buffer = io.BytesIO()
    generate_excel_report(excel_buffer, results_df, export_config, fin_metrics, monthly_stats=monthly_stats)
    excel_data = excel_buffer.getvalue()
    
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("simulation_results.csv", csv)
        zf.writestr("Portfolio_Report.pdf", pdf_bytes)
        zf.writestr("Interactive_Report.xlsx", excel_data)

    # Remove the buttons from inside the tab
    
# --- Global Configuration Generation (Runs every time) ---
# Ensure export_config is always available (re-construct if needed)
if 'export_config' not in locals():
    # Helper for safe retrieval
    def safe_get(var, default): return locals().get(var, st.session_state.get(var, default))
    
    export_config = {
        "region": "ERCOT North",
        "total_load_mwh": locals().get('total_annual_load', 0.0),
        "solar_capacity": st.session_state.get('solar_input', 0.0),
        "wind_capacity": st.session_state.get('wind_input', 0.0),
        "geo_capacity": st.session_state.get('geo_input', 0.0),
        "nuc_capacity": st.session_state.get('nuc_input', 0.0),
        "batt_capacity": st.session_state.get('batt_input', 0.0),
        "batt_duration": st.session_state.get('batt_duration_input', 2.0),
        "solar_price": st.session_state.get('solar_price_input', 35.0),
        "wind_price": st.session_state.get('wind_price_input', 25.0),
        "ccs_price": st.session_state.get('ccs_price_input', 85.0),
        "geo_price": st.session_state.get('geo_price_input', 75.0),
        "nuc_price": st.session_state.get('nuc_price_input', 90.0),
        # Battery Financials (defaults if not calculated)
        "batt_base_rate": locals().get('cvta_fixed_price', 12000.0),
        "batt_guar_avail": 0.98,
        "batt_guar_rte": locals().get('cvta_rte', 85.0),
        "batt_vom": locals().get('cvta_vom', 2.0),
        "market_price": st.session_state.get('market_input', 30.0),
        "rec_price": st.session_state.get('rec_input', 4.0),
        "participants": st.session_state.participants,
        "load_gen_sites": st.session_state.get('load_gen_sites', []),
        "external_profiles": {name: prof.tolist() for name, prof in st.session_state.get('external_profiles', {}).items()},
        "excluded_techs": st.session_state.get('excluded_techs', [])
    }
    # Add optional large arrays
    if 'custom_solar_profile' in st.session_state:
        export_config['custom_solar_profile'] = st.session_state['custom_solar_profile'].tolist()
    if 'custom_wind_profile' in st.session_state:
        export_config['custom_wind_profile'] = st.session_state['custom_wind_profile'].tolist()
    if 'shared_market_prices' in st.session_state:
        df_prices = st.session_state['shared_market_prices']
        if 'Price' in df_prices.columns:
            export_config['custom_battery_prices'] = df_prices['Price'].tolist()

json_str_full = json.dumps(export_config, indent=4, cls=NumpyEncoder)

# AI Optimized Config
ai_config = export_config.copy()
ai_config.pop('custom_solar_profile', None)
ai_config.pop('custom_wind_profile', None)
ai_config.pop('custom_battery_prices', None)
json_str_ai = json.dumps(ai_config, indent=4, cls=NumpyEncoder)

# --- Tab 6: Scenario Manager (Global) ---
with tab_scenario:
    st.header("6. Scenario Manager")
    st.caption("Save your current configuration to a JSON file or load a previously saved scenario.")

    st.subheader("📤 Save Scenario")
    st.markdown("Download your current configuration as a JSON file.")

    st.download_button(
        label="💾 Download Configuration (JSON)",
        data=json_str_full,
        file_name="scenario_config.json",
        mime="application/json",
        use_container_width=False,
        type="primary"
    )
    
    st.markdown("---")
    
    st.subheader("📥 Load Scenario")
    st.markdown("Upload a `scenario_config.json` file to restore settings.")
    uploaded_scen = st.file_uploader(
        "Select JSON File", 
        type=['json', 'txt'], 
        key='uploaded_scenario_tab', 
        on_change=load_scenario
    )
    if uploaded_scen:
        st.success("Scenario loaded successfully!")

        st.markdown("---")
        st.subheader("📸 Scenario Comparison Capture")
        st.markdown("Capture current configuration and results for side-by-side comparison.")
        
        cap_name = st.text_input("Scenario Name", f"Scenario {len(st.session_state.get('comparison_scenarios', {})) + 1}")
        
        if st.button("Capture for Comparison"):
            if 'comparison_scenarios' not in st.session_state:
                st.session_state.comparison_scenarios = {}
            
            # Use safe accesses for metrics since calculation might not have run
            current_metrics = {
                'total_load_mwh': st.session_state.get('total_load_mwh', 0),
                'cfe_score': st.session_state.get('cfe_score', 0),
                'avg_ppa_price': st.session_state.get('avg_ppa_price', 0),
                'net_settlement': st.session_state.get('net_settlement', 0),
                'total_cost': st.session_state.get('total_cost', 0),
                # Detailed Financials
                'market_revenue': st.session_state.get('market_revenue', 0),
                'gross_ppa_cost': st.session_state.get('gross_ppa_cost', 0),
                'rec_cost': st.session_state.get('rec_cost', 0),
                'deficit_cost': st.session_state.get('deficit_cost', 0)
            }
            
            current_caps = {
                'solar': st.session_state.get('solar_input', 0),
                'wind': st.session_state.get('wind_input', 0),
                'firm': (st.session_state.get('geo_input', 0) + 
                         st.session_state.get('nuc_input', 0) + 
                         st.session_state.get('ccs_input', 0)),
                'batt_mw': st.session_state.get('batt_input', 0)
            }
            
            st.session_state.comparison_scenarios[cap_name] = {
                'metrics': current_metrics,
                'caps': current_caps
            }
            st.success(f"Scenario '{cap_name}' captured!")
            st.toast(f"Captured {cap_name}")
            st.rerun()

    # --- Scenario Comparison Display (Logic moved from disabled block) ---
    if 'comparison_scenarios' in st.session_state and st.session_state.comparison_scenarios:
        st.markdown("---")
        st.subheader("📋 Scenario Comparison")
        
        # Prepare data for comparison
        comp_data = []
        for name, data in st.session_state.comparison_scenarios.items():
            metrics_vals = data['metrics']
            caps_vals = data['caps']
            comp_data.append({
                "Scenario": name,
                "CFE Score": metrics_vals.get('cfe_score', 0) * 100,
                "Total Load (GWh)": metrics_vals.get('total_load_mwh', 0) / 1000,
                "Avg PPA Price ($/MWh)": metrics_vals.get('avg_ppa_price', 0),
                "Net Settlement ($M)": metrics_vals.get('net_settlement', 0) / 1e6,
                "Total Cost ($M)": metrics_vals.get('total_cost', 0) / 1e6,
                "Market Revenue ($M)": metrics_vals.get('market_revenue', 0) / 1e6,
                "PPA Cost ($M)": metrics_vals.get('gross_ppa_cost', 0) / 1e6,
                "REC Cost ($M)": metrics_vals.get('rec_cost', 0) / 1e6,
                "Deficit Cost ($M)": metrics_vals.get('deficit_cost', 0) / 1e6,
                "Solar (MW)": caps_vals.get('solar', 0),
                "Wind (MW)": caps_vals.get('wind', 0),
                "Firm (MW)": caps_vals.get('firm', 0),
                "Battery (MW)": caps_vals.get('batt_mw', 0)
            })
        
        comp_df = pd.DataFrame(comp_data)
        
        if not comp_df.empty:
            # Visual Trade-Offs
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("**Efficiency Frontier: CFE vs. Cost**")
                fig_scatter = go.Figure()
                for i, row in comp_df.iterrows():
                    fig_scatter.add_trace(go.Scatter(
                        x=[row['CFE Score']], y=[row['Total Cost ($M)']],
                        mode='markers+text', text=[row['Scenario']],
                        textposition="top center", marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
                        name=row['Scenario']
                    ))
                fig_scatter.update_layout(xaxis_title="CFE Score (%)", yaxis_title="Total Annual Cost ($M)", template=chart_template, showlegend=False, height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with c2:
                st.markdown("**Portfolio Capacity Mix (MW)**")
                cap_cols = ['Solar (MW)', 'Wind (MW)', 'Firm (MW)', 'Battery (MW)']
                fig_mix = go.Figure()
                for col in cap_cols:
                    fig_mix.add_trace(go.Bar(name=col.replace(" (MW)", ""), x=comp_df['Scenario'], y=comp_df[col]))
                fig_mix.update_layout(barmode='stack', template=chart_template, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=400)
                st.plotly_chart(fig_mix, use_container_width=True)

            # Styled Table
            format_dict_styled = {
                "CFE Score": "{:.1f}%", "Total Load (GWh)": "{:,.1f}", "Avg PPA Price ($/MWh)": "${:.2f}",
                "Net Settlement ($M)": "${:,.1f}", "Total Cost ($M)": "${:,.1f}", "Market Revenue ($M)": "${:,.1f}",
                "PPA Cost ($M)": "${:,.1f}", "REC Cost ($M)": "${:,.1f}", "Deficit Cost ($M)": "${:,.1f}",
                "Solar (MW)": "{:,.0f}", "Wind (MW)": "{:,.0f}", "Firm (MW)": "{:,.0f}", "Battery (MW)": "{:,.0f}"
            }
            styled_df_comp = comp_df.style.format(format_dict_styled).background_gradient(subset=["CFE Score", "Net Settlement ($M)"], cmap="RdYlGn").background_gradient(subset=["Total Cost ($M)"], cmap="RdYlGn_r")
            st.dataframe(styled_df_comp, use_container_width=True, hide_index=True)
            
            if st.button("🗑️ Clear Comparison Scenarios"):
                st.session_state.comparison_scenarios = {}
                st.rerun()

# --- Tab 7: Download Results Buttons ---
with tab_dl:
    st.header("7. Download Results")
    st.markdown("Export your configuration and analysis reports.")
    
    is_results_ready = active_scenario # Use the flag we defined earlier
    if not is_results_ready:
        st.warning("⚠️ **No Results Generated Yet.** Please add participants (Tab 1) or load a scenario to generate reports.")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.subheader("📄 Reports & Data")
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes if pdf_bytes else b"",
            file_name="Portfolio_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
            disabled=(pdf_bytes is None)
        )
        st.download_button(
            label="📊 Download Excel Report",
            data=excel_data if excel_data else b"",
            file_name="Interactive_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            disabled=(excel_data is None)
        )
        st.download_button(
            label="📊 Download Results CSV",
            data=csv if csv else "",
            file_name="simulation_results.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=(csv is None)
        )
        # Handle zip buffer empty check
        zip_val = zip_buffer.getvalue()
        st.download_button(
            label="📦 Download All Files (ZIP)",
            data=zip_val if len(zip_val) > 0 else b"",
            file_name="full_simulation_package.zip",
            mime="application/zip",
            use_container_width=True,
            help="Includes: Portfolio Report (PDF), Interactive Report (Excel), and Results CSV.",
            disabled=(len(zip_val) == 0)
        )

