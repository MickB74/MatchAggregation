# Project Matching Module for ERCOT Interconnection Queue
import pandas as pd
import numpy as np
import os
import random
import json

def load_projects(csv_path='projects_in_queue_all_generators.csv'):
    """
    Load and parse the ERCOT interconnection queue CSV.
    
    Args:
        csv_path (str): Path to the projects CSV file
        
    Returns:
        pd.DataFrame: Parsed project data
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Projects file not found at {csv_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        
        # Clean up column names
        df.columns = df.columns.str.strip()
        
        # Parse capacity as float
        if 'Capacity (MW)' in df.columns:
            df['Capacity (MW)'] = pd.to_numeric(df['Capacity (MW)'], errors='coerce')
        
        return df
    except Exception as e:
        print(f"Error loading projects: {e}")
        return pd.DataFrame()

def load_asset_registry(json_path='ercot_assets.json'):
    """Load the master asset registry with Lat/Lon data."""
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except:
        return {}

def infer_hub(lat, lon):
    """Infer ERCOT Hub based on Latitude/Longitude."""
    if not lat or not lon: return "North"
    if lon < -101: return "West"
    if lat < 30: return "South"
    if lon > -96: return "Houston"
    return "North"

def get_location_metadata(project_name, county, registry):
    """
    Enrich project with Location data (Lat, Lon, Hub) using the registry.
    Priority:
    1. Exact Project Name match in Registry
    2. County match in Registry (Average Lat/Lon of projects in that county)
    """
    # 1. Exact Name Match
    if project_name in registry:
        asset = registry[project_name]
        return asset.get('lat'), asset.get('lon'), asset.get('hub')
    
    # 2. Approximate Name Match
    # (Simple containment check)
    for key, asset in registry.items():
        if project_name.lower() in key.lower() or key.lower() in project_name.lower():
             return asset.get('lat'), asset.get('lon'), asset.get('hub')

    # 3. County Location (Fallback)
    # Find any asset in the same county to get "Representative" coordinates
    if county:
        county_clean = str(county).replace(" County", "").strip().lower()
        for asset in registry.values():
            if str(asset.get('county')).lower() == county_clean:
                # Found a project in this county, use its location as proxy
                return asset.get('lat'), asset.get('lon'), infer_hub(asset.get('lat'), asset.get('lon'))
    
    return None, None, "North" # Default

def filter_projects_by_technology(df, tech_type):
    """
    Filter projects by technology type and ERCOT North region.
    
    Args:
        df (pd.DataFrame): Projects dataframe
        tech_type (str): Technology type ('Solar', 'Wind', 'Nuclear', 'Battery', 'CCS Gas', 'Geothermal')
        
    Returns:
        pd.DataFrame: Filtered projects
    """
    if df.empty or 'Fuel' not in df.columns:
        return pd.DataFrame()
    
    # Filter by ERCOT North region first
    if 'CDR Reporting Zone' in df.columns:
        df = df[df['CDR Reporting Zone'] == 'NORTH'].copy()
    
    # Map portfolio tech types to CSV Fuel codes
    fuel_map = {
        'Solar': ['SOL'],
        'Wind': ['WIN'],
        'Nuclear': ['NUC'],
        'Battery': ['OTH'],  # Battery is OTH with Technology = BA
        'Geothermal': ['GEO']
    }
    
    # Special handling for CCS Gas - search by keywords in project name
    if tech_type == 'CCS Gas':
        if 'Project Name' not in df.columns:
            return pd.DataFrame()
        
        # CCS keywords to search for in project names
        ccs_keywords = ['ccs', 'carbon capture', 'carbon sequestration', 'low carbon', 'clean gas']
        
        # Filter gas projects first
        gas_projects = df[df['Fuel'] == 'GAS'].copy()
        
        # Search for CCS keywords in project names (case-insensitive)
        ccs_mask = gas_projects['Project Name'].str.lower().str.contains(
            '|'.join(ccs_keywords), 
            case=False, 
            na=False
        )
        
        filtered = gas_projects[ccs_mask]
        return filtered
    
    # Standard filtering for other technologies
    fuel_codes = fuel_map.get(tech_type, [])
    if not fuel_codes:
        return pd.DataFrame()
    
    # Filter by fuel
    filtered = df[df['Fuel'].isin(fuel_codes)].copy()
    
    # Additional filter for Battery (must have Technology = BA)
    if tech_type == 'Battery' and 'Technology' in filtered.columns:
        filtered = filtered[filtered['Technology'] == 'BA']
    
    return filtered


def prioritize_projects(df, target_capacity):
    """
    Score and prioritize projects based on:
    - Status (IA signed, FIS completed, etc.)
    - Capacity fit (closer to target is better)
    - Location diversity
    
    Args:
        df (pd.DataFrame): Filtered projects
        target_capacity (float): Target capacity in MW
        
    Returns:
        pd.DataFrame: Projects with priority scores, sorted by score
    """
    if df.empty:
        return df
    
    projects = df.copy()
    projects['priority_score'] = 0.0
    
    # Status scoring (higher = better)
    status_scores = {
        'IA': 100,  # IA signed
        'FIS Completed': 80,
        'FIS Started': 60,
        'SS Completed': 40,
        'SS Started': 20
    }
    
    if 'GIM Study Phase' in projects.columns:
        for status, score in status_scores.items():
            projects.loc[projects['GIM Study Phase'].str.contains(status, case=False, na=False), 'priority_score'] += score
    
    # Capacity fit scoring (prefer projects within 50% to 150% of target)
    if 'Capacity (MW)' in projects.columns and target_capacity > 0:
        projects['capacity_fit'] = 100 * np.exp(-0.5 * ((projects['Capacity (MW)'] - target_capacity) / target_capacity) ** 2)
        projects['priority_score'] += projects['capacity_fit']
    
    # Sort by priority score
    projects = projects.sort_values('priority_score', ascending=False)
    
    return projects


def match_projects_to_recommendation(recommendation, max_projects_per_tech=5):
    """
    Match recommended portfolio capacities to actual ERCOT queue projects.
    
    Args:
        recommendation (dict): Portfolio recommendation with capacities
        max_projects_per_tech (int): Maximum number of projects to return per technology
        
    Returns:
        dict: Technology -> list of matched projects
    """
    # Load projects
    df = load_projects()
    
    # Load Registry for enrichment
    registry = load_asset_registry()
    
    if df.empty:
        return {}
    
    matched_projects = {}
    
    # Technologies to match
    tech_capacities = {
        'Solar': recommendation.get('Solar', 0),
        'Wind': recommendation.get('Wind', 0),
        'CCS Gas': recommendation.get('CCS Gas', 0),
        'Geothermal': recommendation.get('Geothermal', 0),
        'Nuclear': recommendation.get('Nuclear', 0),
        'Battery': recommendation.get('Battery_MW', 0)
    }
    
    for tech, capacity in tech_capacities.items():
        if capacity > 0:
            # Filter by technology
            tech_projects = filter_projects_by_technology(df, tech)
            
            if tech_projects.empty:
                continue
            
            # Prioritize projects (Deterministic sort)
            prioritized = prioritize_projects(tech_projects, capacity)
            
            # Add Randomness: Take top 20 candidates and sample 5 (max_projects_per_tech)
            # This ensures we get high-quality projects but with variety on each run.
            candidate_pool_size = 20
            candidates = prioritized.head(candidate_pool_size)
            
            if len(candidates) > max_projects_per_tech:
                # Randomly sample from the top candidates
                # Use random.sample to avoid duplicates
                indices = random.sample(range(len(candidates)), max_projects_per_tech)
                top_projects = candidates.iloc[indices]
            else:
                top_projects = candidates
            
            # Extract relevant fields
            project_list = []
            for _, row in top_projects.iterrows():
                p_name = row.get('Project Name', 'Unknown')
                p_county = row.get('County', 'Unknown')
                
                # Enrich with Lat/Lon/Hub
                lat, lon, hub = get_location_metadata(p_name, p_county, registry)
                
                project_info = {
                    'name': p_name,
                    'capacity_mw': row.get('Capacity (MW)', 0),
                    'county': p_county,
                    'status': row.get('GIM Study Phase', 'Unknown'),
                    'projected_cod': row.get('Projected COD', 'Unknown'),
                    'owner': row.get('Interconnecting Entity', 'Unknown'),
                    'lat': lat,
                    'lon': lon,
                    'hub': hub
                }
                project_list.append(project_info)
            
            matched_projects[tech] = project_list
    
    return matched_projects
