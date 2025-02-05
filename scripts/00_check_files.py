import xarray as xr
import glob
import os

# Delete picontrol files with dates prior to 1675

files_path='/nfs/annie/earcwel/isimip-climate/from_archive/*nc'
file_list = glob.glob(files_path)

for f in file_list:

    data_in = xr.open_mfdataset(f)

    y1 = int(f.split("_")[-2])
    if y1 < 1675:
        print(f)
        os.system(f'rm {f}')


