# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 08:43:31 2021

script to read simpeg result structure and safe them to an .xls file like the
zondTEM one. based upon a routine by marco

@author: lukas, marco
"""

# %% import modules
import os

import numpy as np
import pandas as pd

# pd.options.mode.chained_assignment = None  # default='warn'
from pandas import ExcelWriter
from glob import glob

# %% functions
def average_non_zeros(toavrg, axis=0):
    masked = np.ma.masked_equal(toavrg, 0)
    return masked.mean(axis=axis)


def create_xls_header(idx, snd_id, distance, position, rRMSs):
    xls_header_dict = {'col1':[np.nan,np.nan,np.nan,np.nan],
                       'col2':[np.nan,np.nan,np.nan,np.nan],
                       'col3':[np.nan,np.nan,np.nan,np.nan],
                       'col4':[np.nan,np.nan,np.nan,np.nan]}
    xls_header = pd.DataFrame(xls_header_dict)
    
    xls_header.iloc[0,0] = f'Station#{idx}'
    xls_header.iloc[0,1] = snd_id
    xls_header.iloc[1,0] = 'Distance:'
    xls_header.iloc[1,1] = distance
    xls_header.iloc[2,0] = 'Coordinates'
    xls_header.iloc[2,1] = position[0]
    xls_header.iloc[2,2] = position[1]
    xls_header.iloc[2,3] = position[2]
    xls_header.iloc[3,0] = 'err%'
    xls_header.iloc[3,1] = rRMSs[0]
    xls_header.iloc[3,2] = 'ro_err%'
    xls_header.iloc[3,3] = rRMSs[1]
    
    # move header to first row
    # xls_header = pd.DataFrame(np.vstack([xls_header.columns, xls_header.to_numpy()]),
    #                   columns = [f'col{i+1}' for i in range(xls_header.shape[1])])
    
    return xls_header


def create_xls_data(datafit_df):
    n_chnnls = len(datafit_df)
    nan_arr = np.empty((n_chnnls,))
    nan_arr[:] = np.nan
    xls_data_dict = {'#':nan_arr,
                     't,s':nan_arr,
                     'ro_a_o':nan_arr,
                     'ro_a_c':nan_arr,
                     'u_t_o':nan_arr,
                     'u_t_c':nan_arr}
    xls_data = pd.DataFrame(xls_data_dict)
    
    xls_data.iloc[:,0] = np.asarray(datafit_df.index) + 1
    xls_data.iloc[:,1] = datafit_df.iloc[:,0]
    xls_data.iloc[:,2] = datafit_df.iloc[:,5]
    xls_data.iloc[:,3] = datafit_df.iloc[:,4]
    xls_data.iloc[:,4] = datafit_df.iloc[:,2]
    xls_data.iloc[:,5] = datafit_df.iloc[:,1]
    
    misfit_sig_rel = (datafit_df.iloc[:,2] - datafit_df.iloc[:,1]) / datafit_df.iloc[:,2]
    misfit_roa_rel = (datafit_df.iloc[:,6] - datafit_df.iloc[:,5]) / datafit_df.iloc[:,6]
    
    rRMS_sig = np.sqrt(np.mean(misfit_sig_rel**2)) * 100
    rRMS_roa = np.sqrt(np.mean(misfit_roa_rel**2)) * 100
    
    rRMSs = np.r_[rRMS_sig, rRMS_roa]
    
    # move header to first row
    xls_data = pd.DataFrame(np.vstack([xls_data.columns, xls_data.to_numpy()]),
                      columns = [f'col{i+1}' for i in range(xls_data.shape[1])])
    
    return xls_data, rRMSs


def create_xls_model(model_df):
    mdl_prep = model_df.reset_index(drop=True).iloc[:,3:]
    if mdl_prep.iloc[-1, 0] == 0:                        # thk case, need to calculate z (depths)
        mdl_prep['#'] = list(range(1,len(mdl_prep) + 1))
        if ('Polarizability' in mdl_prep.columns) or ('mpa(rad)' in mdl_prep.columns):
            mdl_prep['Polarizability'] = model_df.iloc[:, 5]  # 0  #model_df.iloc[
            mdl_prep['Time constant'] = model_df.iloc[:, 6]  # 0.00005
            mdl_prep['C exponent'] = model_df.iloc[:, 7]  # 0.5
            mdl_prep['Magnetic permeability'] = 1
            mdl_prep['h'] = abs(mdl_prep.iloc[:, 0])
        else:
            mdl_prep['Polarizability'] = 0  #model_df.iloc[
            mdl_prep['Time constant'] = 0.00005
            mdl_prep['C exponent'] = 0.5
            mdl_prep['Magnetic permeability'] = 1
            mdl_prep['h'] = abs(mdl_prep.iloc[:, 0])

        z0 = np.r_[0, np.cumsum(mdl_prep.iloc[:, 0])[:-1]]
        mdl_prep['z'] = z0

    else:
        mdl_prep = mdl_prep.iloc[1::2]
        mdl_prep['#'] = list(range(1,len(mdl_prep) + 1))
        mdl_prep['Polarizability'] = model_df.iloc[:, 5]  # 0  #model_df.iloc[
        mdl_prep['Time constant'] = model_df.iloc[:, 6]  # 0.00005
        mdl_prep['C exponent'] = model_df.iloc[:, 7]  # 0.5
        mdl_prep['Magnetic permeability'] = 1
        mdl_prep['z'] = abs(mdl_prep['depth(m)'])

        z0 = list(mdl_prep['z'])
        z0.insert(0,0)
        
        mdl_prep['z'] = z0[:-1]
    
        thk_arr = np.zeros(len(z0))
        if len(z0) < 20:
            max_range = len(z0)
        else:
            max_range = len(len(z0) - 2)
    
        for i in range(0, max_range+1):
            thk_curr = z0[i + 1] - z0[i]
            thk_arr[i] = round(thk_curr,2)
    
        mdl_prep['h'] = thk_arr[:-1]

    xls_model = mdl_prep[['#','rho(Ohmm)','Polarizability','Time constant',
                           'C exponent','Magnetic permeability','h','z']]
    xls_model = xls_model.rename(columns = {'rho(Ohmm)': 'Resistivity'})
    xls_model.iloc[-1,-2] = ''  # empty last thickness entry
    
    # move header to first row
    xls_model = pd.DataFrame(np.vstack([xls_model.columns, xls_model.to_numpy()]),
                      columns = [f'col{i+1}' for i in range(xls_model.shape[1])])
    
    return xls_model


# %% folder structure
line_id = '2'

# type_inv = 'pyGIMLi'
type_inv = f'sb-l{line_id}_blk-6lay-mpa'
filename = f'sb-line-0{line_id}'
# version = 'v00_tr7-110us'
version = 'cs12_tr12-100us'

coord_name = f'20210716-sb_Line{line_id}-xz'
use_coord = True
use_only_xz = False  # use only xz info and keep coordinate

path_main = '../../'
path_results = path_main + f'03-inv_results/{type_inv}/{filename}/{version}/'

path_xz_coord = path_main + '05-coord/'

if use_only_xz:
    xz = np.loadtxt(path_xz_coord + 'xz.csv', skiprows=1, delimiter=',')
elif use_coord:
    df_coord = pd.read_csv(path_xz_coord + f'{coord_name}.csv')
    # xz = np.asarray(df_coord.loc[:, ['cum_sd', 'height_dem']])
    xz = np.asarray(df_coord.loc[:, ['x', 'z']])
    ids_coord = [f'{id}' for id in df_coord.sndID]
    fname_export = coord_name
else:
    print('using info from saved files')


# fids_results = glob(path_results + 'L*')[::-1]  # reverse order
fids_results = glob(path_results + '*')
fids_results = [fid for fid in fids_results if '.' not in fid.split(os.sep)[-1]]  # remove non folder fids
n_snds = len(fids_results)
# ids_snd = [fid.split('\\')[1] for fid in fids_results]
# ids_snd = [fid for fid in ids_snd if filename not in fid.split('\\')[1]]
# ids_snd = [fid.split('\\')[1] for fid in fids_results if filename not in fid.split('\\')[1]]

# %% loop over folders to extract all and write to individual excel files
for invrun in [f'{i:03d}' for i in range(0, 3)]:
    xls_set_coll = pd.DataFrame()
    if not use_coord:
        print('using coord-info from saved files, no separate coordinate file available!!')
        fname_export = filename
        for i, fid in enumerate(fids_results):
            snd_id = fid.split(os.sep)[-1]
            print('using sounding ID: ', snd_id)
    
            fid_mdl = fid + f'/csv/invrun{invrun}_{snd_id}.csv'
            fid_fit = fid + f'/csv/invrun{invrun}_{snd_id}_fit.csv'
            
            mdl = pd.read_csv(fid_mdl)
            fit = pd.read_csv(fid_fit)
            
            # # select core mesh only (and a few layers additionally)  # only for multi layered (>10) smooth solutions
            if len(mdl) > 10:
                thk_diffs = np.diff(mdl.iloc[:,3])
                thk_mean = average_non_zeros(thk_diffs[5:30])*1.1
                start_outer_mesh = np.where(thk_diffs < thk_mean)[0][-1]  # select the last where the condition is fulfilled
                mdl_core = mdl[:start_outer_mesh]
            else:
                mdl_core = mdl
            
            position = np.asarray(mdl.iloc[0,:3])
            dist = position[0]  # use the x coordinate as the distance (only for local coordinate sys)
            # TODO more general coordinate reading, read xyz and dist...!!
    
            # prepare data fit
            n_chnnls = len(fit)
            
            xls_data, rRMSs = create_xls_data(datafit_df=fit)
            
            xls_header = create_xls_header(idx=i, snd_id=snd_id,
                                           distance=dist, position=position,
                                           rRMSs=rRMSs)
            
            xls_model = create_xls_model(model_df=mdl_core)
            
            xls_full = xls_header.append([xls_data, xls_model]).reset_index(drop=True)
            
            xls_set_coll = xls_set_coll.append(xls_full)
            # if i == 0:
            #     break
    else:
        for j, id_coord in enumerate(ids_coord):
            for i, fid in enumerate(fids_results):
                snd_id = fid.split(os.sep)[-1]
                if id_coord == snd_id:
                    print('using sounding ID: ', snd_id)
                    print('with coord ID: ', id_coord)
                    fid_mdl = fid + f'/csv/invrun{invrun}_{snd_id}.csv'
                    fid_fit = fid + f'/csv/invrun{invrun}_{snd_id}_fit.csv'
                    # fid_mdl = fid + f'/csv/invrun{invrun}_L50-{i+1}.csv'
                    # fid_fit = fid + f'/csv/invrun{invrun}_L50-{i+1}_fit.csv'
                    
                    mdl = pd.read_csv(fid_mdl)
                    fit = pd.read_csv(fid_fit)
                    
                    export_mdl_csv = mdl.copy()
                    export_mdl_csv.drop(axis=0, columns='Y', inplace=True)

                    
                    # select core mesh only (and a few layers additionally)  # only for multi layered (>10) smooth solutions
                    if len(mdl) > 10:
                        thk_diffs = np.diff(mdl.iloc[:,3])
                        thk_mean = average_non_zeros(thk_diffs[5:30])*1.1
                        start_outer_mesh = np.where(thk_diffs < thk_mean)[0][-1]  # select the last where the condition is fulfilled
                        mdl_core = mdl[:start_outer_mesh]
                    else:
                        mdl_core = mdl
                    
                    position = np.asarray(mdl.iloc[0,:3])
                    dist = position[0]  # use the x coordinate as the distance (only for local coordinate sys)
                    # TODO more general coordinate reading, read xyz and dist...!!
                    
                    if use_only_xz:
                        dist = xz[j,0]  # replace distance with x value from xz file
                        position[2] = xz[j,1]  # replace z coordinate with info from xz file
                    elif use_coord:
                        dist = xz[j,0]  # replace distance with x value from xz file
                        position[2] = xz[j,1]  # replace z coordinate with info from xz file
                        position[0] = df_coord.iloc[j, 1]  # use easting
                        position[1] = df_coord.iloc[j, 2]  # use northing
                    else:
                        print('using info from saved files')
                    
                    # prepare data fit
                    n_chnnls = len(fit)
                    xls_data, rRMSs = create_xls_data(datafit_df=fit)
                    
                    # prep header
                    xls_header = create_xls_header(idx=j, snd_id=snd_id,
                                                   distance=dist, position=position,
                                                   rRMSs=rRMSs)
                    
                    # prep model
                    xls_model = create_xls_model(model_df=mdl_core)
                    
                    # merge
                    xls_full = xls_header.append([xls_data, xls_model]).reset_index(drop=True)
                    xls_set_coll = xls_set_coll.append(xls_full)
                    
                    
                    # export model with xz info
                    export_mdl_csv.iloc[:, 0:2] = xz[j, :]
                    export_mdl_csv.rename(columns={'depth(m)':'thk(m)'}, inplace=True)
                    fid_mdl_new = fid + f'/csv/invrun{invrun}_{snd_id}_mdlXZ.csv'
                    
                    export_mdl_csv.to_csv(fid_mdl_new, index=False)
                    
                    # if i == 0:
                    #     break
    
    
    # %% concatenate and export
    new_col = pd.DataFrame(columns=['col0'])
    xls_set_coll = pd.concat([new_col, xls_set_coll]).reset_index(drop=True)
    xls_set_coll.loc[-1] = np.full([xls_set_coll.shape[1], ], np.nan)
    xls_set_coll = xls_set_coll.sort_index().reset_index(drop=True)
    
    xls_set_coll.to_excel(path_results + f'{fname_export}_{invrun}_{version}.xls',
                          header=False, index=False)
    
    # xls_set_coll[]
