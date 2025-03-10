"""

This method is based on the work of Werning et al. 2024 Climate Solutions Explorer, 
with their data from https://zenodo.org/records/13753537

This script is taken from their script 
https://github.com/iiasa/cse_impact_data/blob/main/00_cse_calculate_gcm_gmt.py
 - the only modifications are
file locations.

To run this script, download the temperature data from the Zenodo, and
edit the file location variables, including in cse_functions.py

"""

import sys
sys.path.append('/nfs/annie/earcwel/isimip-climate/cse_impact_data-main')
import os
import re
import xarray as xr
import glob
from dask.diagnostics import ProgressBar
import pandas as pd
import cse_functions as cf

# %% Settings
# -----------------------------------------------------------------------------

# Set protocol (either ISIMIP2b or ISIMIP3b)
protocol = '3b' 
years = ['1601', '2100'] # depends on protocol; 1661-2099 2b, 1601-2100 3b
input_dir, GCMs, RCPs, timestep = cf.set_protocol(protocol)

# Manually overwrite GCMs/RCP/timestep here if required
# GCMs = ['IPSL-CM5A-LR']
# RCPs = ['ssp126']
# timestep = ''

# Set output directory, input variable (near surface air temperature), and
# start and end years
output_dir = 'required_files'
var = 'tas_'
land_area_file = '../required_files/gridarea05.nc'

# %% Define required functions
#------------------------------------------------------------------------------

def create_file_list(input_dir, GCM, RCP, var):
    # Creates list with all files for the specified variable, GCM and RCP
    
    #files_path = os.path.join(input_dir, RCP, GCM) + '\\*' + var + '*.nc*'
    files_path = os.path.join(input_dir) + f'/{GCM.lower()}*{RCP}_{var}*.nc*'

    file_list = glob.glob(files_path)    
    print(files_path)
    return file_list

def select_files_by_year_range(files, year_range):
    # Selects only files within the specified year range
    
    file_list = []
    
    for i in files:
        years_in_file = re.findall("[0-9]{4,}", i)
    
        if (years_in_file[1] >= year_range[0] and years_in_file[0] <= year_range[1]):
            file_list.append(i)
            
    return file_list    

# %% Calculate GMT anomalies
# -----------------------------------------------------------------------------

# Load land area file
land_area = xr.open_dataset(land_area_file)

GMT_anomaly = pd.DataFrame()

for i in GCMs:
    print(i)
    
    # Generate file list for historical data
    #file_path_hist = os.path.join(input_dir, 'historical', i) + '\\*' + var + '*.nc*'
    file_path_hist = os.path.join(input_dir) +  f'/{i.lower()}*historical_{var}*.nc*'
    file_list_hist = glob.glob(file_path_hist)
    file_list_hist = [item for item in file_list_hist if 'landonly' not in item]
    file_list_hist_40_05 = [item for item in file_list_hist if re.findall("[0-9]{4,}", item)[0] >= '1940']
    
    # Generate file list for pre-industrial control data
    #file_path_pic = os.path.join(input_dir, 'piControl', i) + '\\*' + var + '*.nc*'
    file_path_pic = os.path.join(input_dir) + f'/{i.lower()}*picontrol_{var}*.nc*'
    print(file_path_pic)
    file_list_pic = glob.glob(file_path_pic)
    file_list_pic = [item for item in file_list_pic if 'landonly' not in item]
    
    # Load pre-industrial control data
    file_list = file_list_pic 
    file_list = select_files_by_year_range(file_list, years)    
    #print(file_list)
    tas = xr.open_mfdataset(file_list, parallel=True, 
                            chunks={'lon':720, 'lat':360, 'time':250}, use_cftime=True)
    
    # Calculate average temperature 
    tas_annual = tas.mean(dim='time', skipna=True)   
    weighted =  tas_annual['tas'] * land_area['area']
    hist_avg_temp = weighted.sum(dim=['lat', 'lon'], skipna=True) / land_area.sum()
    hist_avg_temp = hist_avg_temp.rename({'area': i})
   
    for j in RCPs:
    
        print(j)

        # Create file list for future projections and add historical data
        file_list = create_file_list(input_dir, i, j, var)
        file_list = [item for item in file_list if 'landonly' not in item]
        file_list = file_list + file_list_hist_40_05  
    
        # Load data
        tas = xr.open_mfdataset(file_list, parallel=True, 
                                chunks={'lon':720, 'lat':360, 'time':250}, use_cftime=True)
        
        # Calculate annual temperature
        tas_annual = tas.groupby('time.year').mean(dim='time', skipna=True)
        weighted =  tas_annual['tas'] * land_area['area']
        annual_avg_temp = weighted.sum(dim=['lat', 'lon'], skipna=True) / land_area.sum()        
        GCM_RCP_combo = i + '_' + j
        annual_avg_temp = annual_avg_temp.rename({'area': GCM_RCP_combo})
    
        # Calculate temperature anomaly using a 31-year rolling window
        with ProgressBar():
            temp = annual_avg_temp[GCM_RCP_combo].rolling(year=31, center=True).mean() - hist_avg_temp[i]
            temp.name = GCM_RCP_combo
            GMT_anomaly = pd.concat([GMT_anomaly, temp.to_dataframe()], 
                                    ignore_index=False, axis=1)
        
# Save output file 
with ProgressBar():
    GMT_anomaly = GMT_anomaly.dropna(how='all')
    GMT_anomaly.to_excel(os.path.join(output_dir, 
                          f'ISIMIP{protocol}_GCM_GMT_{years[0]}_{years[1]}.xlsx'), 
                          sheet_name='GCM_GMT')
