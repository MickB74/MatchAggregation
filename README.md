# ERCOT North Renewable Aggregation & 24/7 Matching

A Streamlit application for modeling, optimizing, and analyzing 24/7 Carbon Free Energy (CFE) portfolios in the ERCOT North region.

## Features

### 🏢 Load Aggregation
- Combine load profiles from multiple participants (Data Centers, Offices, Industrial).
- Upload custom hourly load CSVs.
- Synthetic profile generation for rapid prototyping.

### ⚡ Generation Portfolio
- **Technologies**: Solar, Wind, CCS Gas, Geothermal, Nuclear.
- **Project Matching**: Automatically matches recommended capacities to real projects in the ERCOT Interconnection Queue.
- **Custom Profiles**: Upload specific generation shapes (e.g. PVWatts or SAM outputs).

### 🔋 Battery Storage & Financials
- **Detailed Contract Modeling**:
    - Capacity Payments (Fixed "Rent" adjusted for Availability).
    - Variable O&M Charges ("Usage Fees").
    - Round-Trip Efficiency (RTE) Performance Penalties.
- **Pro Forma Analysis**: Breakdown of Revenues (Arbitrage) vs Expenses (Lease, Charging, VOM).
- **Visualization**: Waterfall charts and Settlement breakdowns.
- **Simulation**: Option to simulate random outages to stress-test financial terms.

### 📊 24/7 CFE Analytics
- **Hourly Matching**: Visualizes Load vs. Clean Generation every hour of the year.
- **Heatmaps**: 365x24 heatmap of CFE matching to identify deficits.
- **Metrics**: CFE Score, Loss of Green Hours (LoGH), Productivity (MWh/MW), and Grid Deficit.
- **Economics**: PPA vs Capture Value spreads.

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
- **Prices**: 2024 ERCOT HB_NORTH Real-Time Market data (or synthetic fallback).
- **Projects**: Filtered list from valid ERCOT Interconnection Queue data.
