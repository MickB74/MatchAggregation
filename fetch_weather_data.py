import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Define locations
locations = {
    "Wichita_Falls_Wind": {"lat": 33.9137, "lon": -98.4934}, # Optimised for Wind
    "Graham_Solar": {"lat": 33.1070, "lon": -98.5895}       # Optimised for Solar
}

years = [2024, 2025]
cache_dir = "data_cache"
os.makedirs(cache_dir, exist_ok=True)

for name, coords in locations.items():
    for year in years:
        print(f"Fetching {name} ({year})...")
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "hourly": ["temperature_2m", "relative_humidity_2m", "shortwave_radiation", "wind_speed_10m", "wind_direction_10m"],
            "timezone": "auto"
        }
        
        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            # Process hourly data
            hourly = response.Hourly()
            hourly_data = {"date": pd.date_range(
                start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
                end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = hourly.Interval()),
                inclusive = "left"
            )}
            
            hourly_data["GHI_Wm2"] = hourly.Variables(2).ValuesAsNumpy() # GHI (Shortwave Radiation)
            hourly_data["Wind_Speed_10m_mps"] = hourly.Variables(3).ValuesAsNumpy() # WS10m
            
            df = pd.DataFrame(data = hourly_data)
            
            # Save as standard format expected by utils.py
            # Format: openmeteo_{year}_{lat}_{lon}.parquet
            filename = f"openmeteo_{year}_{coords['lat']}_{coords['lon']}.parquet"
            filepath = os.path.join(cache_dir, filename)
            
            df.to_parquet(filepath)
            print(f"Saved to {filepath}")
            
        except Exception as e:
            print(f"Error fetching {name} {year}: {e}")
