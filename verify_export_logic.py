import pandas as pd
import numpy as np

# Simulate 8760 data
hours = 8760
datetime_index = pd.date_range(start='2024-01-01', periods=hours, freq='h')
dummy_data = np.random.rand(hours)

try:
    results_df = pd.DataFrame({
        'Datetime': datetime_index,
        'Load_MW': dummy_data,
        'Matched_MW': dummy_data,
        'Solar_MW': dummy_data,
        'Wind_MW': dummy_data,
        'Geothermal_MW': dummy_data,
        'Nuclear_MW': dummy_data,
        'Battery_Discharge_MW': dummy_data,
        'Battery_SoC_MWh': dummy_data,
        'Grid_Deficit_MW': dummy_data,
        'Surplus_MW': dummy_data
    })
    print("SUCCESS: DataFrame created successfully.")
    print("Columns:", results_df.columns.tolist())
    print("Shape:", results_df.shape)
except Exception as e:
    print(f"FAILURE: {e}")
