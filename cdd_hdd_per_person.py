import numpy as np
import glob
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.cm
import iris
import os
import pickle
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import copy
from sklearn.metrics import r2_score

GCMs = ['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'UKESM1-0-LL']
scens = ['ssp126', 'ssp370', 'ssp585']
varlist = ['cooling_degree_days', 'heating_degree_days']

units = {
    'cooling_degree_days':'CDD', 
    'heating_degree_days':'HDD',
    }

FIGDIR = 'figures'

regional = True
region_datadir = '../isimip-df-agri/data'
cube_r10 = iris.load_cube(os.path.join(region_datadir, 'processed', 'utils', 'r10masks_fractional.nc'))
cube_r10.coord('latitude').guess_bounds()
cube_r10.coord('longitude').guess_bounds()
region_longnames = {
    'SEAP': 'South-East Asia and developing Pacific',
    'EEWA': 'Eurasia',
    'AJNZ': 'Asia-Pacific Developed',
    'AFRC': 'Africa',
    'MDEA': 'Middle East',
    'LAMC': 'Latin America and Caribbean',
    'NAMC': 'North America',
    'EAAS': 'Eastern Asia',
    'SOAS': 'Southern Asia',
    'EURO': 'Europe',
    # 'WORLD': 'World', # use every gridcell
}

years_dim = 2006 + np.arange(95)

cmap = matplotlib.cm.YlOrRd
cmap.set_bad('lightgrey',1)

#%%
data = {}
for scen in scens:
    print(scen)
    ssp = scen[:4]
    
    data[scen] = {}
    pop_file = f'required_files/population_{ssp}soc_0p5deg_annual_2006-2100.nc4'
    pop_data_in = xr.open_dataset(pop_file, decode_times=False)
    pop_data_in['time'] = years_dim
    
    try:
       lons
    except NameError:
        lons = pop_data_in['lon'].data
        lats = pop_data_in['lat'].data

    
    for gcm in GCMs:
        print(gcm)

        data[scen][gcm] = {}
        
        clim_files = glob.glob(f'output/{gcm.lower()}_{scen}_*_cool*nc4')
        gwls = []
        for clim_file in clim_files:
            gwl_in = clim_file.split("_")[2]
            gwls.append(gwl_in)
        gwls = set(gwls)
        
        for gwl in gwls:
            print(gwl)

            data[scen][gcm][gwl] = {}
            
            fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        
            axs = axs.ravel()
            for i in np.arange(6):
                axs[i].remove()
                
            for var_i, var in enumerate(varlist):
                data[scen][gcm][gwl][var] = {}

                files_in = glob.glob(f'output/{gcm.lower()}_{scen}_{gwl}_{var}_*nc4')
                if len(files_in) != 1:
                    raise Exception(f'Need 1 file, have {len(files_in)}')
                    
                years = files_in[0].split(".nc4")[0].split("_")[-2:]
                years = [int(i) for i in years]
                    
                dd_data_in = xr.open_dataset(files_in[0])
                
                dd_data_plot = dd_data_in[var]
                dd_data_plot = np.where(dd_data_plot==0, np.nan, dd_data_plot)
                
                ax = plt.subplot(2, 3, 3*var_i+1, projection=ccrs.Robinson(
                    central_longitude=0))
                
    
                mesh_1 = ax.pcolormesh(
                        lons,
                        lats,
                        dd_data_plot,
                        cmap=cmap,
                        # vmin=-10,
                        # vmax=10,
                        transform=ccrs.PlateCarree(),
                        rasterized=True,
                    )
                cb = plt.colorbar(
                    mesh_1, extend="neither", orientation="horizontal", shrink=0.6, 
                    pad=0.05, label='Days/yr'
                    )
                ax.set_title(f'{var} {gcm} {scen} {gwl.replace("p", ".")}C')
                ax.coastlines()


                
                pop_slice = pop_data_in.where((
                    pop_data_in.time >= years[0]) & (
                    pop_data_in.time <= years[1])
                        ).mean(dim='time', skipna=True)
                        
                
                ax = plt.subplot(2, 3, 3*var_i+2, projection=ccrs.Robinson(
                    central_longitude=0))
                
                pop_slice_plot = pop_slice['number_of_people']
                pop_slice_plot = np.where(pop_slice_plot==0, 
                                          np.nan, pop_slice_plot)

    
                mesh_1 = ax.pcolormesh(
                        lons,
                        lats,
                        pop_slice_plot,
                        cmap=cmap,
                        transform=ccrs.PlateCarree(),
                        rasterized=True,
                        norm=matplotlib.colors.LogNorm(vmin=10),
                    )
                cb = plt.colorbar(
                    mesh_1, extend="neither", orientation="horizontal", shrink=0.6, 
                    pad=0.05, label='People'
                    )
                ax.set_title(f'Pop {gcm} {scen} {gwl.replace("p", ".")}C')
                ax.coastlines()

                
                        
                pop_total = pop_slice.sum()
                
                pop_wgt_exp = dd_data_in[var
                                     ]*pop_slice['number_of_people']
                
                
                pop_wgt_exp_norm = pop_wgt_exp/pop_total
                
                
                ax = plt.subplot(2, 3, 3*var_i+3, projection=ccrs.Robinson(
                    central_longitude=0))
                
                
                # pop_wgt_exp_plot = pop_wgt_exp['number_of_people']
                pop_wgt_exp_plot = np.where(pop_wgt_exp==0, 
                                          np.nan, pop_wgt_exp)

                mesh_1 = ax.pcolormesh(
                        lons,
                        lats,
                        pop_wgt_exp_plot,
                        cmap=cmap,
                        transform=ccrs.PlateCarree(),
                        rasterized=True,
                        norm=matplotlib.colors.LogNorm(vmin=100),
                    )
                cb = plt.colorbar(
                    mesh_1, extend="neither", orientation="horizontal", shrink=0.6, 
                    pad=0.05, label='People*Days/yr'
                    )
                ax.set_title(f'Weighted {gcm} {scen} {gwl.replace("p", ".")}C')
                ax.coastlines()
                
                if regional:
                    
                    for r_i, region in enumerate(region_longnames.keys()):
                        pop_reg = np.nansum(pop_slice['number_of_people']*cube_r10[r_i, ...].data)
                        
                        data[scen][gcm][gwl][var][region
                          ] = np.nansum(pop_wgt_exp*cube_r10[r_i, ...].data)/pop_reg
                        
                
                data[scen][gcm][gwl][var]['global'
                  ] = pop_wgt_exp_norm['number_of_people'].sum().data

            fig.tight_layout()
            
            os.makedirs(f'{FIGDIR}/maps/', exist_ok=True)
            fig.savefig(f'{FIGDIR}/maps/{scen}_{gcm}_{gwl}.png', dpi=100)
            fig.clf()    
            plt.close(fig)
#%%

os.makedirs('dictionaries', exist_ok=True)
with open('dictionaries/data.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
#%%

with open('dictionaries/data.pkl', 'rb') as handle:
    data = pickle.load(handle)

#%%


regs_full = ['global']
if regional:
    regs_full.extend(list(region_longnames.keys()))



def fit_o1(x, alpha, beta):
    yhat = alpha + beta*x
    return yhat

def fit_o2(x, alpha, beta_t, beta_t2):
    yhat = alpha + beta_t*x + beta_t2*x**2
    return yhat

def fit_exp(x, alpha, beta):
    yhat = alpha*np.exp(beta*x)
    return yhat


def fit_log(x, alpha, beta):
    yhat = alpha + beta*np.log(x)
    return yhat

fitnames = {
    fit_o1:'Linear', 
    fit_o2:'Quadratic', 
    fit_exp:'Exp',
    fit_log:'Log',
    }

params = {}

for region in regs_full:
    params[region] = {}

    gwl_vals = []
    dd_vals = {}
    for var in varlist:
        dd_vals[var] = []
    
    for scen in scens:
        for gcm in GCMs:
            for gwl in data[scen][gcm].keys():
                gwl_num = float(gwl.replace("p", "."))
                gwl_vals.append(gwl_num)
                
                for var in varlist:
                    dd_vals[var].append(data[scen][gcm][gwl][var][region])

    for var in varlist:
        params[region][var] = {}

        for fit in [fit_o1, fit_o2, fit_exp, fit_log]:
            fitname = fitnames[fit]
            params[region][var][fitname] = {}
            
            params_in, _ = curve_fit(
                fit, gwl_vals, dd_vals[var])
            
            params[region][var][fitname]['params'] = params_in
            
            y_pred = fit(np.array(gwl_vals), *params_in)
            r2 = r2_score(dd_vals[var], y_pred)
            
            params[region][var][fitname]['r2'] = np.around(r2, decimals=3)
 

os.makedirs('dictionaries', exist_ok=True)
with open('dictionaries/params_multi.pkl', 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL) 

#%%

with open('dictionaries/params.pkl', 'rb') as handle:
    params = pickle.load(handle)

#%%

# plot 1

markers = {
    'GFDL-ESM4':'*', 
    'IPSL-CM6A-LR':'x', 
    'MPI-ESM1-2-HR':'+', 
    'MRI-ESM2-0':'o', 
    'UKESM1-0-LL':'.',
    }

colors = {
    'ssp126':'blue', 
    'ssp370':'orange', 
    'ssp585':'red'
    }

temps_plot = np.linspace(0, 4, 100)

for region in regs_full:
    
    fig, axs = plt.subplots(4, 3, figsize=(15, 8))
    axs = axs.ravel()
    
    for scen in scens:
        for gcm in GCMs:
            for gwl in data[scen][gcm].keys():
                gwl_num = float(gwl.replace("p", "."))
                for var_i, var in enumerate(varlist):
                    axs[var_i].plot(gwl_num, data[scen][gcm][gwl][var][region],
                           color=colors[scen], marker=markers[gcm])
    
    labels = {}
    for var_i, var in enumerate(varlist):
        
        params_for_plot = params[region][var]
        
        # label = (f'{np.around(params_for_plot[0], decimals=2)} + '
        #          f'{np.around(params_for_plot[1], decimals=2)}T + '
        #          f'{np.around(params_for_plot[2], decimals=2)}T^2'
        #          )
        
        label = (f'{np.around(params_for_plot[0], decimals=2)}*'
                 f'e^({np.around(params_for_plot[1], decimals=2)}T)'
                 )
        
        labels[var] = label
        
        axs[var_i].plot(temps_plot, fit(temps_plot, *params_for_plot),
               color='green')
        
        
        
        
    axs[0].set_ylabel('CDDs/person')                
    axs[0].set_xlabel('GMST')                
    axs[0].set_title(f'{region}') 

    axs[1].set_ylabel('HDDs/person')                
    axs[1].set_xlabel('GMST') 
    axs[1].set_title(f'{region}') 

    handles = []
    for gcm in GCMs:
        handles.append(Line2D([], [], label=gcm, color='grey', marker = markers[gcm], linestyle='None', markersize=10))
    for scen in scens:
        handles.append(Line2D([0], [0], label=scen, color=colors[scen]))
    
    handles2 = copy.deepcopy(handles)
    
    handles.append(Line2D([0], [0], label=labels['cooling_degree_days'], color='green'))
    axs[0].legend(handles=handles)
    
    handles2.append(Line2D([0], [0], label=labels['heating_degree_days'], color='green'))
    axs[1].legend(handles=handles2)
    
    
    os.makedirs(f'{FIGDIR}/on_gmst/', exist_ok=True)
    fig.savefig(f'{FIGDIR}/on_gmst/scatter_gmst_exp_fit_{region}.png', dpi=100)
    fig.clf()    
    plt.close(fig)    
    
#%%

# plot multi

markers = {
    'GFDL-ESM4':'*', 
    'IPSL-CM6A-LR':'x', 
    'MPI-ESM1-2-HR':'+', 
    'MRI-ESM2-0':'o', 
    'UKESM1-0-LL':'.',
    }

colors = {
    'ssp126':'blue', 
    'ssp370':'orange', 
    'ssp585':'red'
    }

temps_plot = np.linspace(0, 4, 100)

for region in regs_full:
    
    fig, axs = plt.subplots(4, 4, figsize=(25, 15))
    axs = axs.ravel()
    
    axs_count = -2
    
    for fit in [fit_o1, fit_o2, fit_exp, fit_log]:
        fitname = fitnames[fit]
    
        for var in varlist:
            params_for_plot = params[region][var][fitname]['params']

            axs_count += 2
            for scen in scens:
                for gcm in GCMs:
                    for gwl in data[scen][gcm].keys():
                        gwl_num = float(gwl.replace("p", "."))
                        axs[axs_count].plot(gwl_num, data[scen][gcm][gwl][var][region],
                               color=colors[scen], marker=markers[gcm])
                        
                        residual = data[scen][gcm][gwl][var][region
                                     ] - fit(gwl_num, *params_for_plot)
                        
                        axs[axs_count+1].plot(gwl_num, residual,
                               color=colors[scen], marker=markers[gcm])
                        
                        
            axs[axs_count].plot(temps_plot, fit(temps_plot, *params_for_plot),
                   color='green')
            
            
            axs[axs_count].set_ylabel(f'{units[var]}/person')                
            axs[axs_count].set_xlabel('GMST')                
            axs[axs_count].set_title(f'{units[var]} {fitname} {region}; R2 = {params[region][var][fitname]["r2"]}') 
                   
            
            axs[axs_count+1].set_ylabel(f'{units[var]}/person')                
            axs[axs_count+1].set_xlabel('GMST')                
            axs[axs_count+1].set_title(f'{units[var]} {fitname} {region} Residuals') 
            axs[axs_count+1].axhline(y=0, color='grey', linestyle='--')
                        
            handles = []
            for gcm in GCMs:
                handles.append(Line2D([], [], label=gcm, color='grey', marker = markers[gcm], linestyle='None', markersize=10))
            for scen in scens:
                handles.append(Line2D([0], [0], label=scen, color=colors[scen]))
            
            handles.append(Line2D([0], [0], label=f'{fitname}', color='green'))
            axs[axs_count].legend(handles=handles)
    
    
    os.makedirs(f'{FIGDIR}/on_gmst/', exist_ok=True)
    plt.tight_layout()
    fig.savefig(f'{FIGDIR}/on_gmst/multi_fits_scatter_gmst_exp_{region}.png', dpi=100)
    fig.clf()    
    plt.close(fig)   
        
    