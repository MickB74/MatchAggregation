"""Test script to verify load factor profile generation"""
import sys
sys.path.insert(0, '/Users/michaelbarry/Documents/GitHub/MatchAggregation')

from utils import generate_load_factor_profile
import pandas as pd

# Test Case 1: 24-hour Data Center (from spec: Office DC, Mon-Sun 24hrs, start 0)
print("Test 1: 24/7 Data Center")
print("=" * 50)
hours_247 = [24, 24, 24, 24, 24, 24, 24]
profile_dc = generate_load_factor_profile(
    annual_kwh=6000000,
    hours_per_day=hours_247,
    start_hour=0,
    year=2024
)

print(f"Total annual kWh: {profile_dc['kW'].sum():,.0f}")
print(f"Expected: 6,000,000")
print(f"Peak kW: {profile_dc['kW'].max():,.2f}")
print(f"Avg kW: {profile_dc['kW'].mean():,.2f}")
print(f"All hours should have LF=1.0: {(profile_dc['LF'] == 1.0).all()}")
print()

# Test Case 2: Manufacturing schedule (Mon-Fri 24hrs, Sat 8hrs 0-23, Sun 0)
print("Test 2: Manufacturing Schedule")
print("=" * 50)
hours_mfg = [24, 24, 24, 24, 24, 8, 0]
profile_mfg = generate_load_factor_profile(
    annual_kwh=5000000,
    hours_per_day=hours_mfg,
    start_hour=0,
    year=2024
)

print(f"Total annual kWh: {profile_mfg['kW'].sum():,.0f}")
print(f"Expected: 5,000,000")
print(f"Peak kW: {profile_mfg['kW'].max():,.2f}")
print(f"Avg kW: {profile_mfg['kW'].mean():,.2f}")

# Check Saturday pattern - should have operating hours + ramp
saturday_data = profile_mfg[profile_mfg['Datetime'].dt.dayofweek == 5].head(24)
print(f"Saturday hours with LF=1.0: {(saturday_data['LF'] == 1.0).sum()} (expected: 8)")
print(f"Saturday hours with LF=0.1: {(saturday_data['LF'] == 0.1).sum()} (expected: 2)")
print(f"Saturday hours with LF=0.2: {(saturday_data['LF'] == 0.2).sum()} (expected: 14)")
print()

# Test Case 3: Office schedule (Mon-Fri 8hrs starting at 8am, weekends off)
print("Test 3: Office Schedule")
print("=" * 50)
hours_office = [8, 8, 8, 8, 8, 0, 0]
profile_office = generate_load_factor_profile(
    annual_kwh=500000,
    hours_per_day=hours_office,
    start_hour=8,
    year=2024
)

print(f"Total annual kWh: {profile_office['kW'].sum():,.0f}")
print(f"Expected: 500,000")
print(f"Peak kW: {profile_office['kW'].max():,.2f}")

# Check Monday pattern
monday_data = profile_office[profile_office['Datetime'].dt.dayofweek == 0].head(24)
print("\nMonday hourly pattern (first 24 hours):")
print(monday_data[['Hour', 'LF']].to_string(index=False))

print("\n" + "=" * 50)
print("All tests completed!")
