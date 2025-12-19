import pandas as pd
import io
import datetime

def generate_excel_report(buffer, simulation_df, config, financial_metrics, monthly_stats=None):
    """
    Generates a formatted, interactive Excel report from the simulation data.
    
    Args:
        buffer (io.BytesIO): Buffer to write the Excel file to.
        simulation_df (pd.DataFrame): The hourly simulation results.
        config (dict): The scenario configuration dictionary.
        financial_metrics (dict): Financial metrics summary.
        monthly_stats (pd.DataFrame, optional): Pre-calculated monthly statistics.
    """
    
    # Use XlsxWriter as the engine
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # --- Styles ---
        # Header Style
        header_fmt = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#4472C4', # Standard Blue
            'font_color': 'white',
            'border': 1
        })
        
        # Currency Style
        curr_fmt = workbook.add_format({'num_format': '$#,##0'})
        curr_dec_fmt = workbook.add_format({'num_format': '$#,##0.00'})
        
        # Percentage Style
        pct_fmt = workbook.add_format({'num_format': '0.0%'})
        
        # General Title Style
        title_fmt = workbook.add_format({
            'bold': True,
            'font_size': 16,
            'font_color': '#4472C4'
        })
        
        # Subtitle/Label Style
        label_fmt = workbook.add_format({'bold': True, 'align': 'right', 'bg_color': '#F2F2F2', 'border': 1})
        val_fmt = workbook.add_format({'border': 1})
        
        # --- Sheet 1: Introduction & Instructions ---
        ws_intro = workbook.add_worksheet("Start Here")
        ws_intro.hide_gridlines(2)
        
        ws_intro.write(1, 1, "Clean Energy Portfolio Report", title_fmt)
        ws_intro.write(2, 1, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        instructions = [
            "Welcome to your Interactive Portfolio Report.",
            "",
            "SHEET GUIDE:",
            "1. Dashboard: High-level executive summary of your portfolio's performance.",
            "2. Hourly Data: Complete 8760-hour simulation data in a filterable table.",
            "3. Financials: Detailed breakdown of costs, revenues, and settlements.",
            "4. Configuration: Audit log of the inputs used for this simulation.",
            "",
            "TIPS:",
            "- Use the filters in 'Hourly Data' to isolate specific days or shortage events.",
            "- Green cells in the Dashboard indicate strong performance.",
            "- This report is static; to run new scenarios, return to the Web App."
        ]
        
        for i, line in enumerate(instructions):
            ws_intro.write(4 + i, 1, line)
            
        ws_intro.set_column('B:B', 60)
        
        # --- Sheet 2: Dashboard ---
        ws_dash = workbook.add_worksheet("Dashboard")
        ws_dash.hide_gridlines(2)
        ws_dash.set_column('B:C', 20)
        ws_dash.set_column('E:L', 12)
        
        ws_dash.write(1, 1, "Executive Summary", title_fmt)
        
        # KPI Metrics Table
        metrics_data = [
            ("Clean Energy Score (CFE)", config.get('cfe_score', 0) if isinstance(config.get('cfe_score'), (int, float)) else financial_metrics.get('cfe_score', 0), pct_fmt),
            ("Total Load", financial_metrics.get('total_load_mwh', 0), workbook.add_format({'num_format': '#,##0 "MWh"'})),
            ("Avg PPA Price", financial_metrics.get('weighted_ppa_price', 0), curr_dec_fmt),
            ("Total Net Cost", financial_metrics.get('total_cost', 0), curr_fmt),
            ("Net Settlement", financial_metrics.get('settlement_value', 0), curr_fmt)
        ]
        
        row = 3
        for label, val, fmt in metrics_data:
            ws_dash.write(row, 1, label, label_fmt)
            ws_dash.write(row, 2, val, fmt)
            row += 1
            
        # Monthly Summary Table
        if monthly_stats is not None and not monthly_stats.empty:
            ws_dash.write(3, 4, "Monthly Performance Breakdown", title_fmt)
            
            # Write Headers
            headers = ['Month', 'Total Load (MWh)', 'Clean Gen (MWh)', 'Matched (MWh)', 'Deficit (MWh)', 'CFE Score %']
            for col, h in enumerate(headers):
                ws_dash.write(4, 4 + col, h, header_fmt)
            
            # Write Data
            monthly_row = 5
            # Ensure monthly_stats has month names or numbers
            # If monthly_stats index is the month name, reset index
            ms = monthly_stats.copy()
            if 'Month' not in ms.columns:
                ms = ms.reset_index() # Attempt to get Month from index
                ms.rename(columns={'index': 'Month'}, inplace=True)
                
            for idx, r in ms.iterrows():
                ws_dash.write(monthly_row, 4, r.get('Month', idx+1))
                ws_dash.write(monthly_row, 5, r.get('Load', 0), workbook.add_format({'num_format': '#,##0'}))
                ws_dash.write(monthly_row, 6, r.get('Total_Supply', 0), workbook.add_format({'num_format': '#,##0'}))
                ws_dash.write(monthly_row, 7, r.get('Matched', 0), workbook.add_format({'num_format': '#,##0'}))
                
                # Deficit = Load - Matched
                deficit = r.get('Load', 0) - r.get('Matched', 0)
                ws_dash.write(monthly_row, 8, deficit, workbook.add_format({'num_format': '#,##0', 'font_color': '#d62728'}))
                
                # CFE %
                cfe = r.get('Matched', 0) / r.get('Load', 0) if r.get('Load', 0) > 0 else 0
                ws_dash.write(monthly_row, 9, cfe, pct_fmt)
                
                monthly_row += 1
            
            # Conditional Formatting for CFE Score
            ws_dash.conditional_format(f'I6:I{monthly_row}', {'type': '3_color_scale'})
            
        # --- Sheet 3: Hourly Data ---
        # simulation_df is huge, so using direct write for speed
        simulation_df.to_excel(writer, sheet_name='Hourly Data', index=False)
        ws_data = writer.sheets['Hourly Data']
        
        # Add Excel Table structure
        (max_row, max_col) = simulation_df.shape
        column_settings = [{'header': column} for column in simulation_df.columns]
        ws_data.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings, 'style': 'TableStyleMedium2'})
        
        # Freeze panes
        ws_data.freeze_panes(1, 1) # Freeze header and first column (Datetime)
        ws_data.set_column(0, 0, 20) # Widen datetime column
        
        # --- Sheet 5: Configuration Audit ---
        ws_config = workbook.add_worksheet("Configuration")
        ws_config.write(1, 1, "Scenario Configuration", title_fmt)
        ws_config.set_column('B:C', 30)
        ws_config.write(2, 1, "Parameter", header_fmt)
        ws_config.write(2, 2, "Value", header_fmt)
        
        audit_row = 3
        
        # Clean up complex objects for display (e.g. participants list)
        audit_dict = config.copy()
        if 'participants' in audit_dict and isinstance(audit_dict['participants'], list):
            audit_dict['participants'] = f"{len(audit_dict['participants'])} Sites Loaded"
        if 'excluded_techs' in audit_dict:
             audit_dict['excluded_techs'] = ", ".join(audit_dict['excluded_techs'])
             
        # Flatten dictionary slightly or just iterate top-level
        for k, v in audit_dict.items():
            if isinstance(v, (list, dict)):
                continue # Skip massive arrays like profiles
            ws_config.write(audit_row, 1, str(k).replace('_', ' ').title(), label_fmt)
            ws_config.write(audit_row, 2, str(v), val_fmt)
            audit_row += 1
            
        # --- Sheet 4: Financials ---
        ws_fin = workbook.add_worksheet("Financials")
        ws_fin.hide_gridlines(2)
        ws_fin.set_column('B:B', 30)
        ws_fin.set_column('C:G', 15)
        
        ws_fin.write(1, 1, "Detailed Financial Performance", title_fmt)
        
        if 'tech_details' in financial_metrics:
            ws_fin.write(3, 1, "Financial Performance by Technology", header_fmt)
            fin_headers = ['Technology', 'Generated (MWh)', 'Matched (MWh)', 'PPA Cost ($)', 'Market Value ($)', 'Net Settlement ($)']
            for col, h in enumerate(fin_headers):
                 ws_fin.write(4, 1 + col, h, header_fmt)
            
            fin_row = 5
            for tech, details in financial_metrics['tech_details'].items():
                if details['Matched_MWh'] > 0:
                    ws_fin.write(fin_row, 1, tech, workbook.add_format({'border': 1}))
                    ws_fin.write(fin_row, 2, details.get('Total_MWh', 0), workbook.add_format({'num_format': '#,##0', 'border': 1}))
                    ws_fin.write(fin_row, 3, details.get('Matched_MWh', 0), workbook.add_format({'num_format': '#,##0', 'border': 1}))
                    ws_fin.write(fin_row, 4, details.get('Total_Cost', 0), workbook.add_format({'num_format': '$#,##0', 'border': 1}))
                    ws_fin.write(fin_row, 5, details.get('Market_Value', 0), workbook.add_format({'num_format': '$#,##0', 'border': 1}))
                    
                    settlement = details.get('Settlement', 0)
                    ws_fin.write(fin_row, 6, settlement, workbook.add_format({'num_format': '$#,##0', 'border': 1}))
                    fin_row += 1
            
            # Add Total Row
            ws_fin.write(fin_row, 1, "Total", workbook.add_format({'bold': True, 'bg_color': '#D9D9D9', 'border': 1}))
            ws_fin.write(fin_row, 4, financial_metrics.get('total_cost', 0), workbook.add_format({'bold': True, 'num_format': '$#,##0', 'bg_color': '#D9D9D9', 'border': 1})) 
            ws_fin.write(fin_row, 5, "N/A", workbook.add_format({'bold': True, 'bg_color': '#D9D9D9', 'border': 1}))
            ws_fin.write(fin_row, 6, financial_metrics.get('settlement_value', 0), workbook.add_format({'bold': True, 'num_format': '$#,##0', 'bg_color': '#D9D9D9', 'border': 1}))

        # --- Sheet 5: Configuration Audit ---
                    
        # Clean up
        # Writer context manager handles save/close
