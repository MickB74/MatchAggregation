import gridstatus
import pandas as pd
import patch_gridstatus

def fetch_and_cache(years=[2020, 2021, 2022]):
    iso = gridstatus.Ercot()
    
    for year in years:
        cache_file = f"ercot_rtm_{year}.parquet"
        
        print(f"--- Processing {year} ---")
        if pd.io.common.file_exists(cache_file):
            print(f"{year} data already cached.")
            continue

        print(f"Fetching {year} data from ERCOT (this may take a moment)...")
        try:
            # ERCOT North Hub is typically used as the reference price
            # gridstatus returns RTM SPP (Settlement Point prices)
            # Default includes all hubs. We can filter later or just save all.
            # App.py expects columns like 'Price' (typically HB_NORTH) or a filtered frame.
            # But let's look at how app.py consumes it. 
            # It loads 'ercot_rtm_{year}.parquet' and expects a dataframe with index 'Time' and column 'Price' (or it filters for HB_NORTH)
            
            # Let's see what app.py expects exactly.
            # It seems app.py loads the file and then:
            # df_full = pd.read_parquet(file_path)
            # north_hub_price = df_full[df_full['Settlement Point Name'] == 'HB_NORTH']
            
            df = iso.get_rtm_spp(year=year)
            
            # Pre-process (timezone fix)
            if not pd.api.types.is_datetime64_any_dtype(df['Time']):
                df['Time'] = pd.to_datetime(df['Time'], utc=True)
            
            # Convert to US/Central
            df['Time_Central'] = df['Time'].dt.tz_convert('US/Central')
            
            print(f"Columns found: {df.columns.tolist()}")
            
            # Identify the correct column for Settlement Point
            sp_col = None
            possible_cols = ['Settlement Point Name', 'SettlementPointName', 'SettlementPoint', 'Location', 'Node']
            for col in possible_cols:
                if col in df.columns:
                    sp_col = col
                    break
            
            if sp_col:
                print(f"Filtering using column: {sp_col}")
                df_hubs = df[df[sp_col].isin(hubs)].copy()
            else:
                print("WARNING: Could not find Settlement Point column. Saving full dataframe.")
                df_hubs = df.copy()
            
            # Rename to match expected format if needed
            if sp_col and sp_col != 'Settlement Point Name':
                df_hubs.rename(columns={sp_col: 'Settlement Point Name'}, inplace=True)

            print(f"Successfully fetched {len(df_hubs)} rows.")
            print(df_hubs.head())
            
            # Save to cache
            df_hubs.to_parquet(cache_file)
            print(f"Saved to {cache_file}")
            
        except Exception as e:
            print(f"Error fetching {year} data: {e}")

if __name__ == "__main__":
    fetch_and_cache([2020, 2021, 2022])
