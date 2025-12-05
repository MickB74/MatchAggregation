import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import generate_dummy_load_profile, generate_dummy_generation_profile, calculate_cfe_score, simulate_battery_storage, recommend_portfolio, calculate_portfolio_metrics

st.set_page_config(page_title="ERCOT North Aggregation", layout="wide")

st.title("ERCOT North Renewable Energy Aggregation")
st.markdown("Aggregate load participants and optimize for 24/7 clean energy matching.")

# --- Sidebar: Configuration ---
st.sidebar.header("1. Load Participants")

# Session state to store participants
if 'participants' not in st.session_state:
    st.session_state.participants = []

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
    st.sidebar.dataframe(p_df, hide_index=True)
    
    if st.sidebar.button("Clear Participants"):
        st.session_state.participants = []
        st.rerun()

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

solar_capacity = st.sidebar.number_input("Solar Capacity (MW)", min_value=0.0, value=st.session_state.solar_cap, step=1.0, key='solar_input')
wind_capacity = st.sidebar.number_input("Wind Capacity (MW)", min_value=0.0, value=st.session_state.wind_cap, step=1.0, key='wind_input')
geo_capacity = st.sidebar.number_input("Geothermal Capacity (MW)", min_value=0.0, value=st.session_state.geo_cap, step=1.0, key='geo_input')
nuc_capacity = st.sidebar.number_input("Nuclear Capacity (MW)", min_value=0.0, value=st.session_state.nuc_cap, step=1.0, key='nuc_input')
batt_capacity = st.sidebar.number_input("Battery Power (MW)", min_value=0.0, value=st.session_state.batt_cap, step=1.0, key='batt_input')
batt_duration = st.sidebar.number_input("Battery Duration (Hours)", min_value=0.5, value=4.0, step=0.5, key='batt_duration_input')

# Update session state from inputs (in case user manually changes them after recommendation)
st.session_state.solar_cap = solar_capacity
st.session_state.wind_cap = wind_capacity
st.session_state.geo_cap = geo_capacity
st.session_state.nuc_cap = nuc_capacity
st.session_state.batt_cap = batt_capacity

# --- Main Content ---

# 1. Calculate Aggregated Load
if not st.session_state.participants:
    st.info("Please add load participants in the sidebar to begin.")
else:
    # Aggregate Load
    total_load_profile = pd.Series(0.0, index=range(8760))
    
    for p in st.session_state.participants:
        profile = generate_dummy_load_profile(p['load'], p['type'])
        total_load_profile += profile
        
    total_annual_load = total_load_profile.sum()
    
    # 2. Calculate Generation
    solar_profile = generate_dummy_generation_profile(solar_capacity, 'Solar')
    wind_profile = generate_dummy_generation_profile(wind_capacity, 'Wind')
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
    
    # Charts
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
                             mode='lines', name='Aggregated Load', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=list(range(start_hour, end_hour)), y=matched_profile[start_hour:end_hour],
                             mode='lines', name='Matched Energy', fill='tozeroy', line=dict(color='green', width=0)))
    fig.add_trace(go.Scatter(x=list(range(start_hour, end_hour)), y=total_gen_profile[start_hour:end_hour],
                             mode='lines', name='Renewable Gen', line=dict(color='orange', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=list(range(start_hour, end_hour)), y=batt_discharge[start_hour:end_hour],
                             mode='lines', name='Battery Discharge', line=dict(color='blue', width=1)))
    
    fig.update_layout(title=f"Load vs. Matched Generation {title_suffix}", xaxis_title="Hour of Year", yaxis_title="Power (MW)")
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
    fig_bar.add_trace(go.Bar(x=monthly_stats.index, y=monthly_stats['Load'], name='Load', marker_color='black'))
    fig_bar.add_trace(go.Bar(x=monthly_stats.index, y=monthly_stats['Matched'], name='Matched Energy', marker_color='green'))
    fig_bar.add_trace(go.Bar(x=monthly_stats.index, y=monthly_stats['Generation'], name='Renewable Gen', marker_color='orange', opacity=0.5))
    fig_bar.add_trace(go.Bar(x=monthly_stats.index, y=monthly_stats['Battery'], name='Battery Discharge', marker_color='blue'))
    
    fig_bar.update_layout(title="Monthly Energy Totals", xaxis_title="Month", yaxis_title="Energy (MWh)", barmode='overlay')
    st.plotly_chart(fig_bar, use_container_width=True)

