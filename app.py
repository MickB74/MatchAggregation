import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import datetime
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

        with col_load_2:
            st.markdown("#### Current Participants")
            if st.session_state.participants:
                p_df = pd.DataFrame(st.session_state.participants)
                st.dataframe(p_df, hide_index=True, use_container_width=True)
            else:
                st.info("No participants added yet.")
                
            st.markdown("---")
            st.markdown("#### Or Upload Aggregate Load Profile")
            uploaded_load_file = st.file_uploader("Upload CSV (Hourly load in MW)", type=['csv'], key='uploaded_load_file')


    # --- Tab 2: Generation Portfolio ---
    with tab_gen:
        col_gen_1, col_gen_2 = st.columns(2)
        
        with col_gen_1:
            st.markdown("#### Capacities")
            # Solar
            use_synthetic_solar = st.checkbox("Use Synthetic Solar Profile (Ignore CSV)", value=False)
            solar_capacity = st.number_input("Solar Capacity (MW)", min_value=0.0, step=1.0, key='solar_input')
            
            wind_capacity = st.number_input("Wind Capacity (MW)", min_value=0.0, step=1.0, key='wind_input')
            geo_capacity = st.number_input("Geothermal Capacity (MW)", min_value=0.0, step=1.0, key='geo_input')
            nuc_capacity = st.number_input("Nuclear Capacity (MW)", min_value=0.0, step=1.0, key='nuc_input')

        with col_gen_2:
            st.markdown("#### Storage & Recommendation")
            enable_battery = st.checkbox("Enable Battery Storage", value=True)
            batt_capacity = st.number_input("Battery Power (MW)", min_value=0.0, step=1.0, key='batt_input', disabled=not enable_battery)
            batt_duration = st.number_input("Battery Duration (Hours)", min_value=0.5, step=0.5, key='batt_duration_input', disabled=not enable_battery)
            
            st.markdown("---")
            
            # Exclude Tech multiselect
            excluded_techs = st.multiselect(
                "Exclude Technologies from Recommendation",
                ['Solar', 'Wind', 'Geothermal', 'Nuclear', 'Battery'],
                default=[],
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
                        # Pass excluded techs from session state (widget key='excluded_techs_input')
                        rec = recommend_portfolio(temp_load, excluded_techs=st.session_state.excluded_techs_input)
                        st.session_state.solar_input = rec['Solar']
                        st.session_state.wind_input = rec['Wind']
                        st.session_state.geo_input = rec['Geothermal']
                        st.session_state.nuc_input = rec['Nuclear']
                        st.session_state.batt_input = rec['Battery_MW']
                        st.session_state.batt_duration_input = rec['Battery_Hours']
                        st.session_state.portfolio_recommended = True
                    else:
                        st.session_state.portfolio_error = "Participant load is zero."
                else:
                    st.session_state.portfolio_error = "Add participants first."

            st.button("✨ Recommend Portfolio", on_click=apply_recommendation)
            
            # Show success/error messages after rerun
            if st.session_state.get('portfolio_recommended', False):
                st.success("Portfolio Recommended!")
                st.session_state.portfolio_recommended = False # Reset flag
            if st.session_state.get('portfolio_error', None):
                st.warning(st.session_state.portfolio_error)
                st.session_state.portfolio_error = None # Reset error

        st.markdown("#### Custom Profiles (Upload Unit Profiles)")
        c_prof_1, c_prof_2 = st.columns(2)
        uploaded_solar_file = c_prof_1.file_uploader("Upload Solar Profile (CSV)", type=['csv'])
        uploaded_wind_file = c_prof_2.file_uploader("Upload Wind Profile (CSV)", type=['csv'])


    # --- Tab 3: Financials ---
    with tab_fin:
        c_fin_1, c_fin_2, c_fin_3 = st.columns(3)
        strike_price = c_fin_1.number_input("PPA Strike Price ($/MWh)", min_value=0.0, value=30.0, step=1.0)
        market_price = c_fin_2.number_input("Avg Market Price ($/MWh)", min_value=0.0, value=35.0, step=1.0)
        rec_price = c_fin_3.number_input("REC Price ($/MWh)", min_value=0.0, value=8.0, step=0.5)


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
    geo_profile = generate_dummy_generation_profile(geo_capacity, 'Geothermal')
    nuc_profile = generate_dummy_generation_profile(nuc_capacity, 'Nuclear')
    
    total_gen_profile = solar_profile + wind_profile + geo_profile + nuc_profile
    
    # 3. Calculate Surplus/Deficit BEFORE Battery
    surplus = (total_gen_profile - total_load_profile).clip(lower=0)
    deficit = (total_load_profile - total_gen_profile).clip(lower=0)
    
    # 4. Simulate Battery
    if enable_battery:
        batt_discharge, batt_soc = simulate_battery_storage(surplus, deficit, batt_capacity, batt_duration)
    else:
        batt_discharge = pd.Series(0.0, index=range(8760))
        batt_soc = pd.Series(0.0, index=range(8760))
    
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
    col9, col10 = st.columns(2)
    col9.metric("PPA Settlement Value", f"${fin_metrics['settlement_value']:,.0f}", help="Revenue (or Cost) from PPA Settlement: (Market - Strike) * Matched Vol")
    col10.metric("REC Value", f"${fin_metrics['rec_cost']:,.0f}", help="Revenue from REC Sales (or Cost if purchasing)")
    
    # Charts
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
    fig.add_trace(go.Scatter(x=x_axis, y=matched_profile[start_hour:end_hour],
                             mode='lines', name='Hourly Matched Clean Energy', fill='tozeroy', line=dict(color='#FFA500', width=0)))
    fig.add_trace(go.Scatter(x=x_axis, y=total_gen_profile[start_hour:end_hour],
                             mode='lines', name='Total Clean Energy', line=dict(color='#2ca02c', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=x_axis, y=batt_discharge[start_hour:end_hour],
                             mode='lines', name='Battery Discharge', fill='tozeroy', line=dict(color='#1f77b4', width=1)))
    
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
    fig_bar.add_trace(go.Scatter(x=monthly_stats.index, y=monthly_stats['Load'], name='Load', mode='lines', line=dict(color='red', width=3)))
    fig_bar.add_trace(go.Bar(
        x=monthly_stats.index, 
        y=monthly_stats['Matched'], 
        name='Hourly Matched Clean Energy', 
        marker_color='#FFA500',
        text=monthly_stats['Matched_Pct'],
        textposition='outside',
        textfont=dict(
            family="Arial Black",
            size=14,
            color="white"
        )
    )) # Orange with labels
    fig_bar.add_trace(go.Bar(x=monthly_stats.index, y=monthly_stats['Generation'], name='Total Clean Energy', marker_color='#2ca02c', opacity=0.6)) # Standard Green
    fig_bar.add_trace(go.Bar(x=monthly_stats.index, y=monthly_stats['Battery'], name='Battery Discharge', marker_color='#1f77b4')) # Standard blue
    
    fig_bar.update_layout(
        title="Monthly Energy Totals", 
        xaxis_title="Month", 
        yaxis_title="Energy (MWh)", 
        barmode='overlay', 
        template=chart_template,
        paper_bgcolor=chart_bg,
        plot_bgcolor=chart_bg,
        font=dict(color=chart_font_color)
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
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=z_data,
        x=list(range(1, 366)),
        y=list(range(24)),
        colorscale='RdYlGn', # Red to Green
        zmin=0, zmax=1,
        colorbar=dict(title="Matched %"),
        hovertemplate='Day: %{x}<br>Hour: %{y}<br>Matched: %{z:.1%}<extra></extra>'
    ))
    
    fig_heat.update_layout(
        title="24/7 Matching Heatmap (Hour vs Day)",
        xaxis_title="Day of Year",
        yaxis_title="Hour of Day",
        height=400,
        template=chart_template,
        paper_bgcolor=chart_bg,
        plot_bgcolor=chart_bg,
        font=dict(color=chart_font_color)
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
        'Battery_State_of_Charge_MWh': batt_soc,
        'Grid_Deficit_MW': deficit,
        'Surplus_MW': surplus
    })
    
    csv = results_df.to_csv(index=False).encode('utf-8')

    st.write("### Preview Data")
    st.dataframe(results_df, use_container_width=True)
    
    st.download_button(
        label="Download Hourly Results (CSV)",
        data=csv,
        file_name='simulation_results_8760.csv',
        mime='text/csv',
    )

