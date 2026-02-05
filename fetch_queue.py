#!/usr/bin/env python3
"""
Fetch ERCOT Interconnection Queue Data
Downloads the latest interconnection queue data from ERCOT using gridstatus library.
"""

import gridstatus
import pandas as pd
from datetime import datetime
import os

def fetch_ercot_queue():
    """
    Fetch the latest ERCOT interconnection queue data.
    
    Returns:
        pd.DataFrame: Queue data
    """
    print("Initializing ERCOT connection...")
    iso = gridstatus.Ercot()
    
    print("Fetching interconnection queue data (this may take a moment)...")
    try:
        # Use the gridstatus API to get interconnection queue
        df = iso.get_interconnection_queue()
        
        print(f"Successfully fetched {len(df)} projects from ERCOT queue.")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Error fetching queue data: {e}")
        print(f"Error type: {type(e).__name__}")
        raise

def save_queue_data(df, output_file='projects_in_queue_all_generators.csv'):
    """
    Save the queue data to CSV file.
    
    Args:
        df (pd.DataFrame): Queue data
        output_file (str): Output CSV filename
    """
    # Backup old file if it exists
    if os.path.exists(output_file):
        backup_file = f"{output_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(output_file, backup_file)
        print(f"Backed up existing file to: {backup_file}")
    
    # Save new data
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} projects to {output_file}")
    
    # Show file size
    file_size = os.path.getsize(output_file)
    print(f"File size: {file_size/1024:.1f} KB")

def main():
    """Main execution function."""
    print("=" * 60)
    print("ERCOT Interconnection Queue Data Fetch")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Fetch the data
        df = fetch_ercot_queue()
        
        # Save to CSV
        save_queue_data(df)
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS: Queue data updated successfully!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ ERROR: Failed to fetch queue data")
        print(f"Details: {e}")
        print("=" * 60)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
