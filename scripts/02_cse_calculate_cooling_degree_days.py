# -*- coding: utf-8 -*-
"""
This method is based on the work of Werning et al. 2024 Climate Solutions Explorer, 
with their data from https://zenodo.org/records/13753537

This script is modified from their "simple degree days" script
https://github.com/iiasa/cse_impact_data/blob/main/01_cse_calculate_sdd.py

Here we just separate out into Heating and Cooling degree days - Cooling in this script.

We set the cooling threshold at 22C

To run this script, download the temperature data from the Zenodo, and
edit the file location variables.

"""
import sys
sys.path.append('/nfs/annie/earcwel/isimip-climate/cse_impact_data-main')
import itertools as it
import cse_functions as cf
import cse_functions_indicators as cfi
# import cse_functions_attributes as cfa

#%% Settings
#------------------------------------------------------------------------------

# Set protocol
protocol = '3b' 
input_dir, GCMs, RCPs, timestep = cf.set_protocol(protocol)

# Manually overwrite GCMs/RCP/timestep here if required
# GCMs = ['MIROC5']
# RCPs = ['rcp26', 'rcp60', 'rcp85']
# timestep = ''

# Set output directory
output_dir = '../output'
  
# Choose required tas and set variable
input_var = 'tas' # tas, tasmin, tasmax
output_var = 'cooling_degree_days'

# Set balance temperatures
balance_temperature_cooling_low = 22.0

# Specify file with temperature anomalies, thresholds and year range
GMT_anomaly_file = '/nfs/annie/earcwel/isimip-climate/cse_impact_data-main/required_files/ISIMIP3b_GCM_GMT_1601_2100.xlsx'
GWLs = [1.2, 1.5, 2.0, 2.5, 3.0, 3.5]
year_range = 30
reference_period = [1974, 2004]

make_ref = False

#%% Calculate cooling degree days temperature
#------------------------------------------------------------------------------

# Calculate min and max years for each GCM/RCP/threshold combination
years = cf.find_year_range(GCMs, RCPs, GMT_anomaly_file, GWLs, year_range)
     
for GCM, RCP in it.product(GCMs, RCPs):
    
    if make_ref:
        
        if len([val for key,val in years.items() if f'{GCM}_{RCP}' in key and len(val) != 0]) > 0:
        
            print(f'{GCM}_{RCP}')
            
            # Load data for reference period and convert to Celsius
            file_list = cf.create_file_list(input_dir, GCM, RCP, input_var, timestep)
            tas_ref = cf.load_data(file_list, reference_period)
            tas_ref = cf.convert_to_Celsius(tas_ref)      
            
            # Calculate indicator for all 31 years and average over all years
            cooling_degree_days_ref = cfi.calculate_cooling_degree_days(tas_ref, 
                                                   balance_temperature_cooling_low)
            
            cooling_degree_days_ref_mean = cooling_degree_days_ref.mean(dim='year') 
            
            # Save averaged data
            output_file = cf.create_output_file(output_dir, GCM, RCP, 'historical', 
                                                reference_period, 
                                                f'{output_var}{input_var.replace("tas", "")}_{str(balance_temperature_cooling_low).replace(".", "p")}')
            cf.write_output(cooling_degree_days_ref_mean, output_file, output_var)
            
        else:
            continue
    
    for gwl in GWLs:
        
        combination = f'{GCM}_{RCP}_{gwl}'
        
        # Check if the current threshold exists for the GCM/RCP combination
        if not years[combination]:
            continue
            
        else:
            
            print(combination)
           
            # Load data
            file_list = cf.create_file_list(input_dir, GCM, RCP, input_var, timestep)
            tas = cf.load_data(file_list, years[combination])
            tas = cf.convert_to_Celsius(tas)      
            
            # Calculate indicator for all 31 years and average over all years
            cooling_degree_days = cfi.calculate_cooling_degree_days(tas, 
                                                    balance_temperature_cooling_low)
            cooling_degree_days_mean = cooling_degree_days.mean(dim='year') 
            
            # Save averaged data
            output_file = cf.create_output_file(output_dir, GCM, RCP, gwl, 
                                                years[combination], 
                                                f'{output_var}{input_var.replace("tas", "")}_{str(balance_temperature_cooling_low).replace(".", "p")}')
            cf.write_output(cooling_degree_days_mean, output_file, output_var)  
