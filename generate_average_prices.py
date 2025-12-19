"""
Script to generate composite average parquet file for 2020-2024 ERCOT prices.
"""
import pandas as pd
import numpy as np
import os

# Years to average
years = [2020, 2021, 2022, 2023, 2024]
profiles = []

print("Loading historical data...")
for year in years:
    parquet_file = f'ercot_rtm_{year}.parquet'
    
    if os.path.exists(parquet_file):
        print(f"  Loading {year}...")
        df = pd.read_parquet(parquet_file)
        
        if 'Location' in df.columns and 'SPP' in df.columns:
            df_north = df[df['Location'] == 'HB_NORTH'].copy()
            
            if 'Time' in df_north.columns:
                df_north['Time'] = pd.to_datetime(df_north['Time'])
                df_north.set_index('Time', inplace=True)
                
                # Resample to Hourly Mean
                df_hourly = df_north['SPP'].resample('h').mean()
                df_hourly = df_hourly.interpolate(method='linear').bfill().ffill()
                
                # Remove Feb 29 for alignment
                df_hourly = df_hourly[~((df_hourly.index.month == 2) & (df_hourly.index.day == 29))]
                
                vals = df_hourly.values
                
                # Normalize to 8760
                if len(vals) > 8760:
                    vals = vals[:8760]
                elif len(vals) < 8760:
                    vals = np.pad(vals, (0, 8760 - len(vals)), 'constant', constant_values=vals.mean())
                
                profiles.append(vals)
                print(f"    ✓ {year}: {len(vals)} hours, avg=${np.mean(vals):.2f}/MWh")
    else:
        print(f"  ⚠️  Missing: {parquet_file}")

if profiles:
    print(f"\nCalculating composite average across {len(profiles)} years...")
    
    # Stack and average
    stack = np.vstack(profiles)
    avg_profile = np.mean(stack, axis=0)
    
    print(f"  Composite Average: ${np.mean(avg_profile):.2f}/MWh")
    print(f"  Min: ${np.min(avg_profile):.2f}, Max: ${np.max(avg_profile):.2f}")
    
    # Create DataFrame with synthetic timestamps (2024 as base year)
    dates = pd.date_range(start='2024-01-01', periods=8760, freq='h')
    
    df_output = pd.DataFrame({
        'Time': dates,
        'Location': 'HB_NORTH',
        'SPP': avg_profile
    })
    
    # Save to parquet
    output_file = 'ercot_rtm_average.parquet'
    df_output.to_parquet(output_file, index=False)
    
    print(f"\n✅ Created: {output_file}")
    print(f"   Size: {os.path.getsize(output_file) / 1024:.1f} KB")
else:
    print("\n❌ No profiles loaded. Cannot create average.")
