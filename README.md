# ERCOT North Renewable Aggregation & 24/7 Matching

A Streamlit application for modeling, optimizing, and analyzing 24/7 Carbon Free Energy (CFE) portfolios in the ERCOT North region.

## Features

### 🏢 Load Aggregation
- **Participant Builder**: Combine load profiles from multiple participants (Data Centers, Offices, Industrial).
- **Random Scenarios**: Quickly generate test scenarios (>500 GWh) for stress testing.
- **Upload Support**: Import custom hourly load CSVs.
- **Synthetic Generation**: Built-in profile generation (Data Center, Office, Flat) for rapid prototyping.

### ⚡ Generation Portfolio
- **Technologies**: Solar, Wind, CCS Gas, Geothermal, Nuclear.
- **Project Matching**: Automatically matches recommended capacities to real projects in the ERCOT Interconnection Queue.
- **Detailed Settlement**: Breakdown of PPA Cost vs Market Value vs Net Settlement for each technology.
- **Custom Profiles**: Upload specific generation shapes (e.g., PVWatts or SAM outputs).

### 🔋 Battery Storage & Financials
- **Tolling Model (Buyer's P&L)**: 
    - Full "Trading House" view: Revenue (Arbitrage + Ancillary) vs Costs (Fixed Toll + Charging).
    - **Dynamic Ancillary Revenue**: Model ancillary services as a fixed monthly value OR a dynamic % of energy price.
- **Battery Owner's View**:
    - Capacity Payments (Fixed "Rent" adjusted for Availability).
    - Variable O&M Charges ("Usage Fees").
    - Round-Trip Efficiency (RTE) Performance Penalties.
- **Visualizations**: 
    - **Waterfall Charts**: Monthly P&L breakdown.
    - **Stacking Logic**: Battery visualized as "Gap Filling" (Base Load -> Battery -> VRE).

### ⚖️ Scenario Comparison
- **Visual Trade-Offs**: "Efficiency Frontier" scatter plot (CFE Score vs Total Cost) and stacked capacity charts.
- **Detailed Table**: Side-by-side comparison of multiple scenarios with gradient styling for key metrics (Net Settlement, Total Cost, etc.).
- **Scenario Manager**: Capture, name, and save different configurations to compare strategies.

### 📥 Data Export
- **Formats**: CSV (Hourly), JSON (Full Configuration), and PDF (Executive Report).
- **AI Analysis**: Lightweight JSON export optimized for AI analysis (removes large arrays, keeps key metrics & configuration).

### 📊 24/7 CFE Analytics
- **Hourly Matching**: Visualizes Load vs. Clean Generation every hour of the year.
- **Heatmaps**: 365x24 heatmap of CFE matching to identify deficits.
- **Metrics**: CFE Score, Loss of Green Hours (LoGH), Productivity (MWh/MW), and Grid Deficit.
- **Market Data**: 
    - Supports historical ERCOT HB_NORTH prices (2022, 2023, 2024).
    - Automatic fallback to synthetic "Duck Curve" if data is missing.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application locally:
```bash
streamlit run app.py
```

## Data Sources
- **Generation**: Synthetic profiles tuned for ERCOT North (Solar peak ~1PM, Wind peak ~Night/Spring).
- **Prices**: Real-Time Market (RTM) Hub North data for 2022-2024.
- **Projects**: Filtered list from valid ERCOT Interconnection Queue data.
