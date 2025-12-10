# Project Matching Module for ERCOT Interconnection Queue
import pandas as pd
import numpy as np
import os

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
            
            # Prioritize projects
            prioritized = prioritize_projects(tech_projects, capacity)
            
            # Select top N projects
            top_projects = prioritized.head(max_projects_per_tech)
            
            # Extract relevant fields
            project_list = []
            for _, row in top_projects.iterrows():
                project_info = {
                    'name': row.get('Project Name', 'Unknown'),
                    'capacity_mw': row.get('Capacity (MW)', 0),
                    'county': row.get('County', 'Unknown'),
                    'status': row.get('GIM Study Phase', 'Unknown'),
                    'projected_cod': row.get('Projected COD', 'Unknown'),
                    'owner': row.get('Interconnecting Entity', 'Unknown')
                }
                project_list.append(project_info)
            
            matched_projects[tech] = project_list
    
    return matched_projects
