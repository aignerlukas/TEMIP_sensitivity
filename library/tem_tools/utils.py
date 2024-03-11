#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 18:07:45 2022

utilities and tools for tem sounding and survey classes

@author: laigner
"""

# %% import modules
import re
import logging
import numpy as np
import pandas as pd


# %% general
def get_float_from_string(string):
    numeric_const_pattern = r"""
           [-+]? # optional sign
           (?:
              (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
              |
              (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
           )
           # followed by optional exponent part if desired
           (?: [Ee] [+-]? \d+ ) ?
           """

    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    numeric_list = rx.findall(string)
    if len(numeric_list) > 1:
        return np.float_(numeric_list)
    elif len(numeric_list) == 1:
        return np.float_(numeric_list)[0]
    else:
        print('error - no numerics found in string')


# %% zond_specific
def rearr_zond_mdl(Mdl2_reArr):
    """
    function to rearrange model value structure in order to
    plot a step-model.
    """
    Mdl2_reArr = np.asarray(Mdl2_reArr, dtype='float')
    FileLen=len(Mdl2_reArr)
    MdlElev = np.zeros((FileLen*2,2)); r=1;k=0
    for i in range(0,FileLen*2):
        if i == FileLen:
            MdlElev[-1,1] = MdlElev[-2,1]
            break
        if i == 0:
            MdlElev[i,0] = Mdl2_reArr[i,4]
            MdlElev[i:i+2,1] = Mdl2_reArr[i,1]
        else:
            MdlElev[k+i:k+i+2,0] = -Mdl2_reArr[i,4]    # height
            MdlElev[r+i:r+i+2,1] = Mdl2_reArr[i,1]     # res
            k+=1; r+=1

    MdlElev = np.delete(MdlElev,-1,0) # delete last row!!
    return MdlElev


def get_zt_inv_model(dat, idcs_mdl, snd_id, remove_IP=True):
    model_df = dat.loc[idcs_mdl.start[snd_id]:idcs_mdl.end[snd_id],
                          ['c1','c2','c3','c4','c5','c6','c7','c8']]
    model_df.columns = ['ctr','Rho','Pol','Tconst','Cexpo','MagP','h','z']
    model_df = model_df.apply(pd.to_numeric)
    model_df.Pol /= 100  # remove scale from zondTEM chargeability
    model_df.dropna(axis=0, how='all', inplace=True)

    # if (model_df.loc[:, 'Pol'] == model_df.iloc[0, 2]).all():
    if remove_IP:
        logging.info('IP params are all the same, removing them from df')
        model_df.drop(['Pol','Tconst','Cexpo'], axis=1, inplace=True)
        return model_df
    else:
        return model_df


def get_zt_header(dat, idcs_hdr, snd_id):
    header = dat.loc[idcs_hdr.start[snd_id]:idcs_hdr.end[snd_id],
                     ['c1','c2','c3','c4']]
    return header


def get_zt_response(dat, idcs_resp, snd_id):
    response =  dat.loc[idcs_resp.start[snd_id]:idcs_resp.end[snd_id],
                        ['c1','c2','c3','c4','c5','c6']]
    response.columns = ['ctr','time','rho_O','rho_C','U_O','U_C']
    return response.apply(pd.to_numeric)



# %% functions_zond_xls
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
    xls_data.iloc[:,1] = datafit_df.iloc[:,0]  # time in s
    xls_data.iloc[:,2] = datafit_df.iloc[:,6]  # rhoa observed
    xls_data.iloc[:,3] = datafit_df.iloc[:,5]  # rhoa calculated
    xls_data.iloc[:,4] = datafit_df.iloc[:,2]  # signal observed
    xls_data.iloc[:,5] = datafit_df.iloc[:,1]  # signal calculated
    
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
