import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from utils import (
    generate_dummy_load_profile, 
    generate_dummy_generation_profile, 
    calculate_cfe_score, 
    simulate_battery_storage, 
    recommend_portfolio, 
    calculate_portfolio_metrics, 
    calculate_financials, 
    process_uploaded_profile
)

st.set_page_config(page_title="ERCOT North Aggregation", layout="wide")

st.title("ERCOT North Renewable Energy Aggregation")
st.markdown("Aggregate load participants and optimize for 24/7 clean energy matching.")

# Session state to store participants
if 'participants' not in st.session_state:
    st.session_state.participants = []

# Check session state for uploaded file (widget moved to bottom)
uploaded_load_file = st.session_state.get('uploaded_load_file')

# Only show participant builder if no file is uploaded
if not uploaded_load_file:
    st.sidebar.markdown("---")
    st.sidebar.header("1. Load Participants")
    
    with st.sidebar.form("add_participant"):
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

# Display current participants
if st.session_state.participants:
    st.sidebar.subheader("Current Participants")
    p_df = pd.DataFrame(st.session_state.participants)
    if not uploaded_load_file:
         st.sidebar.dataframe(p_df, hide_index=True)
    
    if st.sidebar.button("Clear Participants"):
        st.session_state.participants = []
        st.rerun()


# Dark Mode Toggle
st.sidebar.markdown("---")
dark_mode = st.sidebar.toggle("Dark Mode", value=False)

if dark_mode:
    # Custom CSS for Dark Mode
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stSidebar {
            background-color: #262730;
        }
        /* Update user inputs to match dark theme better */
        .stTextInput, .stNumberInput, .stSelectbox {
            color: #FAFAFA;
        }
        </style>
        """, unsafe_allow_html=True)
    chart_template = 'plotly_dark'
else:
    chart_template = 'plotly'


st.sidebar.header("2. Renewable Projects")

# Recommendation Button
if st.sidebar.button("✨ Recommend Portfolio"):
    # Calculate total load first
    temp_load = pd.Series(0.0, index=range(8760))
    for p in st.session_state.participants:
        temp_load += generate_dummy_load_profile(p['load'], p['type'])
        
    if temp_load.sum() > 0:
        rec = recommend_portfolio(temp_load)
        
        # Update session state KEYS for widgets to trigger UI update
        st.session_state.solar_input = rec['Solar']
        st.session_state.wind_input = rec['Wind']
        st.session_state.geo_input = rec['Geothermal']
        st.session_state.nuc_input = rec['Nuclear']
        st.session_state.batt_input = rec['Battery_MW']
        st.session_state.batt_duration_input = rec['Battery_Hours']
        
        # Also update the backing values (though widgets drive the next run)
        st.session_state.solar_cap = rec['Solar']
        st.session_state.wind_cap = rec['Wind']
        st.session_state.geo_cap = rec['Geothermal']
        st.session_state.nuc_cap = rec['Nuclear']
        st.session_state.batt_cap = rec['Battery_MW']
        
        st.success("Portfolio Recommended!")
        # Rerun to update widgets immediately
        st.rerun()
    else:
        st.warning("Add participants first!")

# Initialize session state for inputs if not present
if 'solar_cap' not in st.session_state: st.session_state.solar_cap = 10.0
if 'wind_cap' not in st.session_state: st.session_state.wind_cap = 5.0
if 'geo_cap' not in st.session_state: st.session_state.geo_cap = 0.0
if 'nuc_cap' not in st.session_state: st.session_state.nuc_cap = 0.0
if 'batt_cap' not in st.session_state: st.session_state.batt_cap = 0.0

# Initialize widget keys if not present
if 'solar_input' not in st.session_state: st.session_state.solar_input = st.session_state.solar_cap
if 'wind_input' not in st.session_state: st.session_state.wind_input = st.session_state.wind_cap
if 'geo_input' not in st.session_state: st.session_state.geo_input = st.session_state.geo_cap
if 'nuc_input' not in st.session_state: st.session_state.nuc_input = st.session_state.nuc_cap
if 'batt_input' not in st.session_state: st.session_state.batt_input = st.session_state.batt_cap
if 'batt_duration_input' not in st.session_state: st.session_state.batt_duration_input = 4.0



solar_capacity = st.sidebar.number_input("Solar Capacity (MW)", min_value=0.0, step=1.0, key='solar_input')
wind_capacity = st.sidebar.number_input("Wind Capacity (MW)", min_value=0.0, step=1.0, key='wind_input')
geo_capacity = st.sidebar.number_input("Geothermal Capacity (MW)", min_value=0.0, step=1.0, key='geo_input')
nuc_capacity = st.sidebar.number_input("Nuclear Capacity (MW)", min_value=0.0, step=1.0, key='nuc_input')
batt_capacity = st.sidebar.number_input("Battery Power (MW)", min_value=0.0, step=1.0, key='batt_input')
batt_duration = st.sidebar.number_input("Battery Duration (Hours)", min_value=0.5, step=0.5, key='batt_duration_input')

# Update session state from inputs (in case user manually changes them after recommendation)
st.session_state.solar_cap = solar_capacity
st.session_state.wind_cap = wind_capacity
st.session_state.geo_cap = geo_capacity
st.session_state.nuc_cap = nuc_capacity
st.session_state.batt_cap = batt_capacity


# --- Generation Profiles ---
with st.sidebar.expander("Custom Generation Profiles"):
    st.markdown("Upload **Unit Profiles** (MW output per 1 MW capacity). The app will scale these by the capacity sliders above.")
    uploaded_solar_file = st.file_uploader("Upload Solar Profile (CSV)", type=['csv'])
    uploaded_wind_file = st.file_uploader("Upload Wind Profile (CSV)", type=['csv'])

st.sidebar.header("3. Financial Assumptions")
strike_price = st.sidebar.number_input("PPA Strike Price ($/MWh)", min_value=0.0, value=30.0, step=1.0)
market_price = st.sidebar.number_input("Avg Market Price ($/MWh)", min_value=0.0, value=35.0, step=1.0)
rec_price = st.sidebar.number_input("REC Price ($/MWh)", min_value=0.0, value=8.0, step=0.5)

# --- Upload Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("Upload Load Profile")
st.sidebar.file_uploader("Upload CSV (Hourly load in MW)", type=['csv'], key='uploaded_load_file')

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
        solar_profile = generate_dummy_generation_profile(solar_capacity, 'Solar')

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
    geo_profile = generate_dummy_generation_profile(geo_capacity, 'Geothermal')
    nuc_profile = generate_dummy_generation_profile(nuc_capacity, 'Nuclear')
    
    total_gen_profile = solar_profile + wind_profile + geo_profile + nuc_profile
    
    # 3. Calculate Surplus/Deficit BEFORE Battery
    surplus = (total_gen_profile - total_load_profile).clip(lower=0)
    deficit = (total_load_profile - total_gen_profile).clip(lower=0)
    
    # 4. Simulate Battery
    batt_discharge, batt_soc = simulate_battery_storage(surplus, deficit, batt_capacity, batt_duration)
    
    # 5. Final Matching
    total_gen_with_battery = total_gen_profile + batt_discharge
    cfe_score, matched_profile = calculate_cfe_score(total_load_profile, total_gen_with_battery)
    
    # Calculate detailed metrics
    total_gen_capacity = solar_capacity + wind_capacity + geo_capacity + nuc_capacity
    metrics = calculate_portfolio_metrics(total_load_profile, matched_profile, total_gen_capacity)
    
    # Financials
    fin_metrics = calculate_financials(matched_profile, deficit, strike_price, market_price, rec_price)
    
    # --- Dashboard ---
    
    # Metrics - Row 1
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Annual Load", f"{total_annual_load:,.0f} MWh")
    col2.metric("Total Generation", f"{total_gen_profile.sum():,.0f} MWh")
    col3.metric("CFE Score (24/7)", f"{metrics['cfe_score']:.1%}")
    col4.metric("Battery Discharge", f"{batt_discharge.sum():,.0f} MWh")
    
    # Metrics - Row 2
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("MW Match Productivity", f"{metrics['productivity']:,.0f} MWh/MW", help="MWh of Clean Energy Matched per MW of Installed Capacity")
    col6.metric("Loss of Green Hours", f"{metrics['logh']:.1%}", help="% of hours where load is not fully matched by clean energy")
    col7.metric("Grid Consumption", f"{metrics['grid_consumption']:,.0f} MWh", help="Total energy drawn from grid (deficit)")
    col8.metric("Curtailment / Overgen", f"{surplus.sum():,.0f} MWh", help="Gross overgeneration before battery charging")
    
    # Metrics - Row 3 (Financials)
    st.subheader("Financial Overview")
    col9, col10, col11 = st.columns(3)
    col9.metric("PPA Settlement Value", f"${fin_metrics['settlement_value']:,.0f}", help="Revenue (or Cost) from PPA Settlement: (Market - Strike) * Matched Vol")
    col10.metric("REC Cost", f"${fin_metrics['rec_cost']:,.0f}", help="Cost of RECs for matched energy")
    col11.metric("Net Energy Cost", f"${fin_metrics['avg_cost_per_mwh']:.2f} / MWh", help="Total Net Cost / Total Load")
    
    # Charts
    st.subheader("Hourly Energy Balance")
    
    # Use columns to make the slider more compact
    col_slider, col_space = st.columns([1, 3])
    with col_slider:
        zoom_level = st.select_slider("Zoom Level", options=["Sample Week", "Full Year"])
    
    if zoom_level == "Sample Week":
        # Slice for a sample week (e.g., first week of June ~ hour 3600)
        start_hour = 3600
        end_hour = 3600 + 168
        title_suffix = "(Summer Week)"
    else:
        start_hour = 0
        end_hour = 8760
        title_suffix = "(Full Year)"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(start_hour, end_hour)), y=total_load_profile[start_hour:end_hour],
                             mode='lines', name='Aggregated Load', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=list(range(start_hour, end_hour)), y=matched_profile[start_hour:end_hour],
                             mode='lines', name='Matched Energy', fill='tozeroy', line=dict(color='#006400', width=0)))
    fig.add_trace(go.Scatter(x=list(range(start_hour, end_hour)), y=total_gen_profile[start_hour:end_hour],
                             mode='lines', name='Renewable Gen', line=dict(color='#2ca02c', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=list(range(start_hour, end_hour)), y=batt_discharge[start_hour:end_hour],
                             mode='lines', name='Battery Discharge', line=dict(color='#1f77b4', width=1)))
    
    fig.update_layout(title=f"Load vs. Matched Generation {title_suffix}", xaxis_title="Hour of Year", yaxis_title="Power (MW)", template=chart_template)
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
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=monthly_stats.index, y=monthly_stats['Load'], name='Load', marker_color='red'))
    fig_bar.add_trace(go.Bar(x=monthly_stats.index, y=monthly_stats['Matched'], name='Matched Energy', marker_color='#006400')) # Dark Green
    fig_bar.add_trace(go.Bar(x=monthly_stats.index, y=monthly_stats['Generation'], name='Renewable Gen', marker_color='#2ca02c', opacity=0.6)) # Standard Green
    fig_bar.add_trace(go.Bar(x=monthly_stats.index, y=monthly_stats['Battery'], name='Battery Discharge', marker_color='#1f77b4')) # Standard blue
    
    fig_bar.update_layout(title="Monthly Energy Totals", xaxis_title="Month", yaxis_title="Energy (MWh)", barmode='overlay', template=chart_template)
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
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=z_data,
        x=list(range(1, 366)),
        y=list(range(24)),
        colorscale='RdYlGn', # Red to Green
        zmin=0, zmax=1,
        colorbar=dict(title="Matched %")
    ))
    
    fig_heat.update_layout(
        title="24/7 Matching Heatmap (Hour vs Day)",
        xaxis_title="Day of Year",
        yaxis_title="Hour of Day",
        height=400,
        template=chart_template
    )
    st.plotly_chart(fig_heat, use_container_width=True)

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
        'Geothermal_MW': geo_profile,
        'Nuclear_MW': nuc_profile,
        'Battery_Discharge_MW': batt_discharge,
        'Battery_SoC_MWh': batt_soc,
        'Grid_Deficit_MW': deficit,
        'Surplus_MW': surplus
    })
    
    csv = results_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Hourly Results (CSV)",
        data=csv,
        file_name='simulation_results_8760.csv',
        mime='text/csv',
    )

