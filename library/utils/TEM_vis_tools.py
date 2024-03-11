# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 13:05:14 2018
collection of functions to parse and plot .xls files

@author: laigner
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import pandas as pd

import os
import sys
import math
import logging
import itertools
import matplotlib

from scipy.optimize import root
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

import matplotlib.offsetbox as offsetbox
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib import ticker
import matplotlib.colors as colors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from math import ceil
from scipy.stats import skew

from .TEM_proc_tools import parse_TEMfastFile

as_strided = np.lib.stride_tricks.as_strided



# %% setup logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)
# logger.setLevel(logging.DEBUG)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %% reading
def parse_zondMdl(filename):
    """function to parse .mdl file from zondTEM software.
    not necessary if export is done by exporting to excel only.
    """
    mdlFile = np.genfromtxt(filename,skip_header=1)
    FileLen=len(mdlFile)
    MdlElev = np.zeros((FileLen*2,2)); r=1;k=0
    for i in range(0,FileLen*2):
        if i == FileLen:
            MdlElev[-1,1] = MdlElev[-2,1]
            break
        if i == 0:
            MdlElev[i,1] = mdlFile[i,3]
            MdlElev[i:i+2,0] = mdlFile[i,0]
        else:
            MdlElev[r+i:r+i+2,0] = mdlFile[i,0]     #resistivity
            MdlElev[k+i:k+i+2,1] = -mdlFile[i,3]    #elevation
            k+=1; r+=1
    return MdlElev, filename


def parse_zondxls(path, file):
    """
    function to parse ZondTEM1d .xls file and create indices for further
    subselecting of dat file

    Keyword arguments:
    path, file -- directions and file name
    zondVersion -- version of the zondTEM software; IPversion export has more model paramters (IP params) than noIP version
    ToDo: automatize the selection of the zondVersion - load all columns, than decide which ones to use
    """
    
    # assume the zondVersion is IP and read all needed columns
    try:
        if '.xls' in file:
            raw = pd.read_excel(path + os.sep + file,
                                usecols=range(1, 9),
                                names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'],
                                header=None)
        elif '.csv' in file:
            raw = pd.read_csv(path + os.sep + file,
                              usecols=range(1, 9),
                              names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'],
                              header=None,
                              engine='python')
        else:
            print('no valid file ending in filename... exiting!')
            sys.exit(0)
        zondVersion = 'IP'

    # catch the ValueError exception (if noIP) and read only with 6 columns
    except ValueError:
        if '.xls' in file:
            raw = pd.read_excel(path + os.sep + file,
                                usecols=range(1, 7),
                                names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'],
                                header=None)
        elif '.csv' in file:
            raw = pd.read_csv(path + os.sep + file,
                              usecols=range(1, 7),
                              names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'],
                              header=None,
                              engine='python')
        else:
            print('no valid file ending in filename... exiting!')
            sys.exit(0)
        zondVersion = 'noIP'

    # print(raw)
    if '.xls' in file:
        # dat = raw.drop(labels='empty', axis=1)
        dat = raw.drop(labels=0, axis=0)
    elif '.csv' in file:
        dat = raw.drop(labels=0, axis=0)
    else:
        print('no valid file ending in filename... exiting!')
        sys.exit(0)

    indices_labels = np.flatnonzero(dat.c1 == '#') + 1
    ev2nd0 = indices_labels[::2]; ev2nd1 = indices_labels[1::2]
    end = len(dat)
    endMdl = np.append(ev2nd0[1:]-5, [end])

    indices_hdr = pd.DataFrame({'start':ev2nd0-4, 'end':ev2nd0},
                               columns=['start', 'end'])
    indices_mresp = pd.DataFrame({'start':ev2nd0+1, 'end':ev2nd1-1},
                               columns=['start', 'end'])
    indices_mdl = pd.DataFrame({'start':ev2nd1+1, 'end':endMdl},
                               columns=['start', 'end'])

    nSnds = len(indices_hdr)
    sndNames = []
    for logID in range(0,nSnds):
        hdr = dat.loc[indices_hdr.start[logID]:indices_hdr.end[logID]]
        sndNames.append(hdr.iloc[0][1])

    indices_hdr.insert(0, "sndID", sndNames)
    indices_mresp.insert(0, "sndID", sndNames)
    indices_mdl.insert(0, "sndID", sndNames)

    return dat, indices_hdr, indices_mresp, indices_mdl, zondVersion


def add_xz2dat(dat, indices_hdr, xz):
    nsnds = len(indices_hdr)
    dat_coord = dat.copy()
    for snd_id in range(0, nsnds):
        dat_coord.at[indices_hdr.start[snd_id] + 1, 'c2'] = xz[snd_id, 0]
        dat_coord.at[indices_hdr.start[snd_id] + 2,'c4'] = xz[snd_id, 1]
    return dat_coord


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %% utils
def get_response(dat, idcs_response, snd_id):
    response =  dat.loc[idcs_response.start[snd_id]:idcs_response.end[snd_id],
                        ['c1','c2','c3','c4','c5','c6']]
    response.columns = ['ctr','time','rho_O','rho_C','U_O','U_C']
    return response.apply(pd.to_numeric)


def get_header(dat, idcs_header, snd_id):
    header = dat.loc[idcs_header.start[snd_id]:idcs_header.end[snd_id],
                     ['c1','c2','c3','c4']]
    return header


def get_model(dat, idcs_model, snd_id):
    model =  dat.loc[idcs_model.start[snd_id]:idcs_model.end[snd_id],
                        ['c1','c2','c3','c4','c5','c6','c7','c8']]
    model.columns = ['ctr','rho','pol','tau','c','magP','h','z']
    model.drop(['magP'], axis=1, inplace=True)
    return model.apply(pd.to_numeric)


def zond2stepmdl(dpth_mdl):
    """
    function to rearrange model value structure in order to
    plot a step-model.

    Parameters
    ----------
    mdl_2b_rearr : np.array
        mdl array with depths (col0) and values (col1).

    Returns
    -------
    ra_mdl : np.array
        step model.

    """
    rows = len(dpth_mdl)

    ra_mdl = np.zeros((rows*2,2))
    r = 1
    k = 0
    
    for i in range(0,rows*2):
        if i == rows:
            ra_mdl[-1,1] = ra_mdl[-2, 1]
            break
        if i == 0:
            ra_mdl[i,0] = dpth_mdl[i, 0]
            ra_mdl[i:i+2,1] = dpth_mdl[i, 1]
        else:
            ra_mdl[k+i:k+i+2,0] = -dpth_mdl[i, 0]    # depths
            ra_mdl[r+i:r+i+2,1] = dpth_mdl[i, 1]     # values
            k += 1
            r += 1

    ra_mdl = np.delete(ra_mdl, -1, 0) # delete last row!!
    return ra_mdl


def get_dpthandvals(mdl_df, column='rho'):
    depth = np.asarray(mdl_df['z'], dtype=float)
    values = np.asarray(mdl_df[column], dtype=float)
    array_dpth_vals = np.column_stack((depth, values))
    return array_dpth_vals


def depth_from_thickness(thick_model, add_bottom=50):
    """routine to reshape a thickness model to a model
    containing the depth of each boundary point.

    Keyword arguments:
    thick_model -- thickness model - np.array
    add_bottom -- how many m below the last layer
    in:
    col0 ... Resistivity
    col1 ... Thickness
    out:
    col0 ... Resistivity
    col1 ... Depth
    """
    n_layers = len(thick_model)
    mdl_size = n_layers
    depth_model = np.zeros((mdl_size*2, 2))
    r0 = 1; #resistivity counter
    d1 = 0; #depth ctr
    for i in range(0,mdl_size+1):
        if i == mdl_size:
            print('end-i:',i)
            depth_model[-1,0] = thick_model[-1,0]
            depth_model[-1,1] = depth_model[-2,1] + add_bottom
            break
        if i == 0:
            print('start-i:',i)
            depth_model[i,0] = thick_model[0,0]
            depth_model[i,1] = 0
            depth_model[i+1,0] = thick_model[0,0]
            depth_model[i+1,1] = thick_model[0,1]
        else:
            print('else-i:',i)
            depth_model[r0+i:r0+i+2,0] = thick_model[r0,0]
            depth_model[d1+i:d1+i+2,1] = thick_model[d1,1]
            if d1 > 0:
                print('d1-1', d1-1)
                depth_model[d1+i:d1+i+2,1] = np.cumsum(thick_model[:,1])[d1]
            r0 += 1;
            d1 += 1;
    return depth_model


def rotate_xy(x, y, angle, origin, show_plot=False):
    ox = origin[0]
    oy = origin[1]
    x_rot = (ox + math.cos(angle) * (x - ox) - math.sin(angle) * (y - oy))
    y_rot = (oy + math.sin(angle) * (x - ox) + math.cos(angle) * (y - oy))
    
    if show_plot:
        fig, ax = plt.subplots(1,1)
        ax.plot(x, y, 'ok', label='not tilted')
        ax.plot(x_rot, y_rot, 'or', label='tilted')
        ax.plot(ox, oy, 'xg', label='origin ')
        ax.set_title(f'rotational angle: {angle} rad')
        ax.legend()
    
    # telx_curr = ox + math.cos(angle) * (telx_curr - ox) - math.sin(angle) * (telz_curr - oy)
    # telz_curr = oy + math.sin(angle) * (telx_curr - ox) + math.cos(angle) * (telz_curr - oy)
    return x_rot, y_rot


def remove_soundings(indices_hdr, indices_mresp, indices_mdl, snd2remove):
    """
    function to remove soundings from indices dataframes

    input: the 3 dataframes which contain the indices for the Header,
    the Modelresponse and the Model;
    !! List of indices which should be removed !!

    returns:
    indices dataframes which are cleared from the soundings contained in the
    snd2remove list
    """
    indHdr_clr = indices_hdr[~indices_hdr.sndID.isin(snd2remove)]
    indMre_clr = indices_mresp[~indices_hdr.sndID.isin(snd2remove)]
    indMdl_clr = indices_mdl[~indices_hdr.sndID.isin(snd2remove)]

    return indHdr_clr, indMre_clr, indMdl_clr


def select_soundings(indices_hdr, indices_mresp, indices_mdl, snd2keep):
    """
    function to remove soundings from indices dataframes

    input: the 3 dataframes which contain the indices for the Header,
    the Modelresponse and the Model;
    !! List of indices which should be selected !!

    returns:
    indices dataframes which contain only the selected ones.
    """
    indHdr_sld = indices_hdr[indices_hdr.sndID.isin(snd2keep)].copy()
    indHdr_sld['sort_cat'] = pd.Categorical(indHdr_sld['sndID'],
                                            categories=snd2keep,
                                            ordered=True)

    indHdr_sld.sort_values('sort_cat', inplace=True)
    indHdr_sld.reset_index(inplace=True)
    indHdr_sld.drop(indHdr_sld.columns[[0,-1]], axis=1, inplace=True)

    indMre_sld = indices_mresp[indices_hdr.sndID.isin(snd2keep)].copy()
    indMre_sld['sort_cat'] = pd.Categorical(indMre_sld['sndID'],
                                            categories=snd2keep,
                                            ordered=True)
    indMre_sld.sort_values('sort_cat', inplace=True)
    indMre_sld.reset_index(inplace=True)
    indMre_sld.drop(indMre_sld.columns[[0,-1]], axis=1, inplace=True)

    indMdl_sld = indices_mdl[indices_hdr.sndID.isin(snd2keep)].copy()
    indMdl_sld['sort_cat'] = pd.Categorical(indMdl_sld['sndID'],
                                            categories=snd2keep,
                                            ordered=True)
    indMdl_sld.sort_values('sort_cat', inplace=True)
    indMdl_sld.reset_index(inplace=True)
    indMdl_sld.drop(indMdl_sld.columns[[0,-1]], axis=1, inplace=True)

    return indHdr_sld, indMre_sld, indMdl_sld



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %% doi
def rho_average(doi, lr_rho, lr_thk):
    rho_int = 0
    for i in range(1,len(lr_rho)):
        # print(doi)
        # print(lr_thk[0:i])
        if all(doi > lr_thk[0:i]):
            rho_int += np.log10(lr_rho[i]) * (lr_thk[i] - lr_thk[i-1])
        else:
            rho_int += np.log10(lr_rho[i]) * (doi - lr_thk[i-1])
            break
    rho_av = 10**(rho_int / doi)
    return rho_av


def calc_doi(fname_ztres, path_ztres,
             fname_raw, path_raw,
             x0=100, verbose=False):
    """

    Parameters
    ----------
    fname_ztres : str
        DESCRIPTION.
    path_ztres : str
        DESCRIPTION.
    fname_raw : str
        DESCRIPTION.
    path_raw : str
        DESCRIPTION.
    x0 : TYPE, optional
        initial value for the root search (ie zero search). The default is 100.

    Returns
    -------
    DOIs : list
        DESCRIPTION.

    """
    (dat, indices_hdr, indices_mresp,
     indices_mdl, zondVersion) = parse_zondxls(path_ztres,
                                               fname_ztres)
    sounding_names = indices_hdr.sndID.to_list()
    signal = dat.loc[indices_mresp.start[1]:indices_mresp.end[1],
                         ['c1','c2','c3','c4','c5','c6']]
    signal.columns = ['ctr','time','rho_O','rho_C','U_O','U_C']
    signal.loc[:,'time':'U_C'] = signal.loc[:,'time':'U_C'].astype('float')

    DOIs = []
    OPTs = []

    for logID, snd_name in enumerate(sounding_names):
        dat_raw, nLogs, ind_hdr, ind_dat = parse_TEMfastFile(fname_raw,
                                                         path_raw)
        rawdata = dat_raw.loc[ind_dat.start[0]:ind_dat.end[0]-1]
        rawdata = rawdata.apply(pd.to_numeric)
        header_raw = dat_raw.loc[ind_hdr.start[0]:ind_hdr.end[0]-1]
        Curr = header_raw.iloc[3,5]
        current = float(Curr[3:6])
        tx_size = float(header_raw.iloc[4][1])
        tx_area = tx_size**2
        # print(tx_size, tx_area)
        eta = signal.iloc[-1,4]  #*1e-6  # last gate, from mcV to V
        tN = signal.iloc[-1, 1]  # time of last time gate for filtered data
        
        model = dat.loc[indices_mdl.start[logID]:indices_mdl.end[logID],
                        ['c1','c2','c3','c4','c5','c6','c7','c8']]
        model.columns = ['ctr','Rho','Pol','Tconst','Cexpo','MagP','h','z']
        model.drop(['Pol','Tconst','Cexpo'], axis=1, inplace=True)
        mdl_rz = np.asarray(model.loc[:,['Rho', 'z']], dtype='float')  # factor to test doi for higher rho * 10
        # rA_model = zond2stepmdl(model)
        # print(rA_model)
        
        doi_fun = lambda x: 0.55*(current*tx_area*rho_average(x, mdl_rz[:,0], mdl_rz[:,1]) / (eta))**(1/5) - x
        OPTs.append(root(doi_fun, x0))
        DOIs.append(OPTs[logID]['x'][0])
        
        if verbose:
            print(f'doi calc of: {snd_name}')
            print('TX-area: ', tx_area)
            print('volt-level: ', eta)
            print('at timegate (us): ', tN)
            print(OPTs[logID]['message'])
            print('doi: ', DOIs[logID])

    return DOIs, OPTs


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %% section
def get_PatchCollection(elemx, elemz, colormap='jet_r', log10=False,
                        edgecolors='None', lw=0):
    """Script to derive polygon patches collections out of grid information and
    data values; Updated version to enable easy log10 scaling!!

    Author: Jakob Gallistl
    """
    import matplotlib.colors as colors

    # size of elemx
    a, b = np.shape(elemx)
    # patches list
    patches = []

    # loop to compute polygon patches
    for elem in np.arange(b):
        elx = np.expand_dims(elemx[:, elem], axis=1)
        elz = np.expand_dims(elemz[:, elem], axis=1)

        v = np.concatenate((elx, elz), axis=1)
        poly = Polygon(v)
        patches.append(poly)

    norm = colors.Normalize() if (log10 == False) else colors.LogNorm()
    p = PatchCollection(patches,
                        edgecolors=edgecolors,
                        linewidth=lw,
                        # zorder=0,
                        cmap=colormap, #default is reversed jetMap
                        norm=norm)     #default is linear norm
    return p


def get_val_elems(dat, indices_hdr, indices_mresp, indices_mdl,
                  kind='rho'):  # 'Pol','Tconst','Cexpo'

    n_logs = len(indices_hdr)
    n_layers = np.asarray(indices_mdl.end - indices_mdl.start + 1)
    nlay_sum = np.sum(n_layers)
    nlay_ind = np.cumsum(n_layers)
    val_elems = np.zeros((1, nlay_sum))
    
    for log_id in range(0, n_logs):
        snd_df = dat.loc[indices_mdl.start[log_id]:indices_mdl.end[log_id],
                              ['c1','c2','c3','c4','c5','c6','c7','c8']]
        snd_df.columns = ['ctr','rho','pol','tau','c','magP','h','z']
        snd_df.drop(['magP'], axis=1, inplace=True)

        dpth_vals = get_dpthandvals(snd_df, column=kind)
        snd_arr = zond2stepmdl(dpth_vals)

        vals = np.copy(snd_arr[::2,1])
        n_layer = n_layers[log_id]

        if log_id == 0:
            val_elems[:, 0:n_layer] = vals
        else:
            val_elems[:, nlay_ind[log_id-1]:nlay_ind[log_id]] = vals
        # separated_logs.append([indices_hdr.sndID[log_id], snd_df, z_el])

    return val_elems.reshape((np.size(val_elems),))


def create_TEMelem(dat, indices_hdr, indices_mresp, indices_mdl,
                   log_width=2, xoffset=0, extend_bot=5,
                   zondVersion='IP'):
    """
    creates 4 corner elements for plotting.
    Needs to be combined with Jakobs get_PatchCollection
    """
    n_logs = len(indices_hdr)
    n_layers = np.asarray(indices_mdl.end - indices_mdl.start + 1)
    nlay_sum = np.sum(n_layers)
    nlay_ind = np.cumsum(n_layers)

    telx = np.zeros((4, nlay_sum))
    telz = np.zeros((4, nlay_sum))
    telRho = np.zeros((1, nlay_sum))

    frame_lowLeft = np.zeros((n_logs,3)) #create array for lower left Point of each log. add depth of log
    topo = np.zeros((n_logs,2))
    log_id = 0
    snd_names = []
    
    separated_logs = []
    
    for log_id in range(0, n_logs):
        if zondVersion == 'IP':
            snd_df = get_model(dat, indices_mdl, log_id)
            dpth_vals = get_dpthandvals(snd_df, column='rho')
            log2plot = zond2stepmdl(dpth_vals)

        elif zondVersion == 'noIP':
            snd_df = dat.loc[indices_mdl.start[log_id]:indices_mdl.end[log_id]]
            dpth_vals = get_dpthandvals(snd_df, column='rho')
            log2plot = zond2stepmdl(dpth_vals)

        snd_names.append(indices_hdr.sndID[log_id])
        nLayer = n_layers[log_id]
        
        # print(dat.loc[indices_hdr.start[log_id]+1,'c2'])
        # print(xoffset)
        dis_xpos = dat.loc[indices_hdr.start[log_id]+1,'c2'] + xoffset
        height = dat.loc[indices_hdr.start[log_id]+2,'c4']  # get height from header
        topo[log_id, 1] = height
        topo[log_id, 0] = dis_xpos

        xLog = [[dis_xpos - log_width/2],
                [dis_xpos + log_width/2],
                [dis_xpos + log_width/2],
                [dis_xpos - log_width/2]]

        z = log2plot[:, 0]
        z = np.insert(z, 0, 0, axis=0) + height
        z_el = np.copy(as_strided(z, (4, n_layers[log_id]), (8, 16)))
        z_el[-2:,-1] = z_el[0:2,-1] - extend_bot
        
        Rho = np.copy(log2plot[::2,1])
        # print(log2plot[::2,0])
        # print(nLayer)

        frame_lowLeft[log_id,0] = np.min(xLog)
        frame_lowLeft[log_id,1] = np.min(z_el)
        frame_lowLeft[log_id,2] = abs(np.min(z_el) - np.max(z_el))        # depth of log

        if log_id == 0:
            telx[:, 0:nLayer] = np.repeat(xLog, nLayer, axis=1)
            telz[:, 0:nLayer] = z_el
            telRho[:, 0:nLayer] = Rho
        else:
            telx[:,nlay_ind[log_id-1]:nlay_ind[log_id]] = np.repeat(xLog, nLayer, axis=1)
            telz[:,nlay_ind[log_id-1]:nlay_ind[log_id]] = z_el
            telRho[:,nlay_ind[log_id-1]:nlay_ind[log_id]] = Rho

        separated_logs.append([indices_hdr.sndID[log_id], log2plot, z_el])

    telRho = telRho.reshape((np.size(telRho),))

    return telx, telz, telRho, frame_lowLeft, n_logs, topo, snd_names, extend_bot, separated_logs


def tilt_1Dlogs(telx, telz, separated_logs, xz_df, frame_lowLeft,
                log_width=2, tilt_log='center'):
    # logs can be tilted based on topography
    # 'slope' uses the slope from the current sounding to the next one
    # 'center' uses the slope between the soundings around the currrent sounding
    
    telx_tilt = np.zeros_like(telx)
    telz_tilt = np.zeros_like(telz)
    tilt_angles = []
    origins = []

    for idx, sounding in enumerate(separated_logs):
        soundingID = sounding[0]

        # get the x position of sounding and create x coordinates of patch
        xPos = float(xz_df[xz_df['SoundingID'] == soundingID]['X'])
        xLog = [[xPos - log_width/2],
                [xPos + log_width/2],
                [xPos + log_width/2],
                [xPos - log_width/2]]
        telx_curr = np.repeat(xLog, np.shape(sounding[-1])[-1], axis=1)
        # telz_curr = float(xz_df[xz_df['SoundingID'] == soundingID]['Z']) + sounding[-1]
        telz_curr = sounding[-1]

        if tilt_log == 'slope':
            delta_x = np.diff(xz_df['X'])
            delta_z = np.diff(xz_df['Z'])
            xz_df['delta_x'] = np.append(delta_x, delta_x[-1])
            xz_df['delta_z'] = np.append(delta_z, 0)

        if tilt_log == 'center':
            pad_x = np.zeros((len(xz_df)+2, 1))
            pad_z = pad_x.copy()

            pad_x[1:-1, 0] = xz_df['X']
            pad_x[0, 0] = -pad_x[2, 0]
            # last value is distance between last two point added to last point
            pad_x[-1, 0] = 2*pad_x[-2, 0] - pad_x[-3, 0]

            pad_z[1:-1, 0] = xz_df['Z']
            # use first z value for 0 and last z value for -1
            pad_z[0, 0] = pad_z[1, 0]
            pad_z[-1, 0] = pad_z[-2, 0]
            delta_x = np.zeros((len(xz_df), 1))
            delta_x[0::2, 0] = np.diff(pad_x[0::2, 0])
            delta_x[1::2, 0] = np.diff(pad_x[1::2, 0])
            delta_z = np.zeros((len(xz_df), 1))
            delta_z[0::2, 0] = np.diff(pad_z[0::2, 0])
            delta_z[1::2, 0] = np.diff(pad_z[1::2, 0])
            xz_df['delta_x'] = delta_x
            xz_df['delta_z'] = delta_z

        # rotate the log around first patch as center of mass
        # simple similarity transformation
        ox, oy = np.mean(telx_curr[:, 0]), np.mean(telz_curr[:, 0])
        ox, oy = np.min(telx_curr[:, 0]), np.max(telz_curr[:, 0])
        
        angle = np.pi/2 - math.atan2(float(xz_df[xz_df['SoundingID'] == soundingID]['delta_x']),
                                     float(xz_df[xz_df['SoundingID'] == soundingID]['delta_z']))
        tilt_angles.append(angle)
        origins.append((ox, oy))

        telx_curr = ox + math.cos(angle) * (telx_curr - ox) - math.sin(angle) * (telz_curr - oy)
        telz_curr = oy + math.sin(angle) * (telx_curr - ox) + math.cos(angle) * (telz_curr - oy)
        
        frame_lowLeft[idx,0] = np.min(telx_curr)
        frame_lowLeft[idx,1] = np.min(telz_curr)
        frame_lowLeft[idx,2] = abs(np.min(telz_curr) - np.max(telz_curr))        #depth of log
        
        if idx == 0:
            telx_tilt = telx_curr
            telz_tilt = telz_curr
        else:
            telx_tilt = np.hstack((telx_tilt, telx_curr))
            telz_tilt = np.hstack((telz_tilt, telz_curr))

    origins = np.asarray(origins)

    return telx_tilt, telz_tilt, origins, tilt_angles


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %% interpolation
def create_support_pts(dat, indices_hdr, indices_mresp, indices_mdl,
                       log_width=2, xoffset=0, values='rho',
                       tilt=False, tilt_angles=None, origins=None,
                       average_thin=True, up_sample='thick', sample_factor=10,
                       use_common_depth=False):
    """
    Parameters
    ----------
    dat : pd.DataFrame
        contains all inversion results.
    indices_hdr : pd.DataFrame
        indices corresponding to the dat variable,
        for selecting the header of each sounding.
    indices_mresp : pd.DataFrame
        indices corresponding to the dat variable,
        for selecting the model response of each sounding.
    indices_mdl : pd.DataFrame
        indices corresponding to the dat variable,
        for selecting the model response of each sounding..
    log_width : TYPE, optional
        DESCRIPTION. The default is 2.
    xoffset : TYPE, optional
        DESCRIPTION. The default is 0.
    zondVersion : TYPE, optional
        DESCRIPTION. The default is 'IP'.

    Returns
    -------
    None.

    """
    n_logs = len(indices_hdr)
    bot_dep_res = np.zeros((n_logs, 2))
    n_layers = np.asarray(indices_mdl.end - indices_mdl.start + 1)

    for log_id in range(0,n_logs):
        snd_df = get_model(dat, indices_mdl, log_id)
        dpth_vals = get_dpthandvals(snd_df, column=values)
        log2plot = zond2stepmdl(dpth_vals)

        row = np.arange(1, len(log2plot), 2)
        bot_dep_res[log_id, 0] = log2plot[-1, 0]
        bot_dep_res[log_id, 1] = log2plot[-1, 1]

    if use_common_depth:
        bottom_depths = np.full((n_logs,), min(bot_dep_res[:,0]) - 10)
        # bottom_depths = min(bot_dep_res[:,0]) - 5  # minimum depth from all soundings in the profile
    else:
        bottom_depths = bot_dep_res[:, 0] - 10

    all_support_points = np.empty((0, 3), float)

    for log_id in range(0, n_logs):
        dis_xpos = dat.loc[indices_hdr.start[log_id]+1,'c2'] + xoffset
        height = dat.loc[indices_hdr.start[log_id]+2,'c4']  # get height from header
        logger.info(height)

        snd_df = get_model(dat, indices_mdl, log_id)
        dpth_vals = get_dpthandvals(snd_df, column=values)
        log2plot = zond2stepmdl(dpth_vals)
        supPts_raw = np.copy(log2plot)

        thckns = abs(supPts_raw[:-1:2,0] - supPts_raw[1::2,0])
        mean_thck = np.mean(thckns)
        for r in row:
            thickness = abs(supPts_raw[r,0] - supPts_raw[r-1,0])
            logger.info(thickness)
            shift = thickness / 50
            supPts_raw[r,0] = supPts_raw[r,0] + shift

        supPts = np.vstack((supPts_raw, [bottom_depths[log_id], bot_dep_res[log_id, 1]]))
        x_vals = np.full((len(supPts), ), dis_xpos)
        supPts_x = np.insert(supPts, 0, x_vals, axis=1)
        supPts_x[:,1] = supPts_x[:,1] + height

        if up_sample == 'all':  # upsampling all
            int_func = interp1d(supPts[:,0], supPts[:,1], kind='linear')
            int_depths = np.linspace(0, supPts[-1,0], n_layers[log_id]*sample_factor)
            int_depths = np.resize(int_depths, (len(int_depths), 1))
            supPts_int = np.hstack((int_depths, int_func(int_depths)))
            x_vals = np.full((len(supPts_int), ), dis_xpos)
            supPts_int_x = np.insert(supPts_int, 0, x_vals, axis=1)
            supPts_int_x[:,1] = supPts_int_x[:,1] + height

            if tilt and (tilt_angles is not None) and (origins is not None):
                supPts_int_x[:,0], supPts_int_x[:,1] = rotate_xy(x=supPts_int_x[:,0], y=supPts_int_x[:,1],
                                                                 angle=tilt_angles[log_id], origin=origins[log_id],
                                                                 show_plot=False)

            all_support_points = np.append(all_support_points, supPts_int_x, axis=0)

        elif up_sample == 'thick':  # upsampling only thicker (than average) layers
            logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            logger.info('entering selective upsampling...')
            for k in range(1,len(supPts)):
                thickness = abs(supPts[k,0] - supPts[k-1,0])
                logger.info(k, 'of supp points')
                logger.info('mean thickness: ', mean_thck)
                logger.info('thickness', thickness)
                if thickness > mean_thck*1.5:  
                    logger.info('thickness is larger than mean thk!!')
                    logger.info('interpolating for: ')
                    logger.info(supPts[k-1:k+1,0],
                           supPts[k-1:k+1,1])

                    int_func = interp1d(supPts[k-1:k+1,0],
                                        supPts[k-1:k+1,1], kind='linear')
                    int_depths = np.linspace(supPts[k-1,0], supPts[k,0], sample_factor)
                    int_depths = np.resize(int_depths, (len(int_depths), 1))
                    supPts_int = np.hstack((int_depths, int_func(int_depths)))
                    x_vals = np.full((len(supPts_int), ), dis_xpos)
                    supPts_int_x = np.insert(supPts_int, 0, x_vals, axis=1)
                    supPts_int_x[:,1] = supPts_int_x[:,1] + height
                    
                    if tilt and (tilt_angles is not None) and (origins is not None):
                        supPts_int_x[:,0], supPts_int_x[:,1] = rotate_xy(x=supPts_int_x[:,0], y=supPts_int_x[:,1],
                                                                         angle=tilt_angles[log_id], origin=origins[log_id])
                    
                    all_support_points = np.append(all_support_points, supPts_int_x, axis=0)

                else:
                    logger.info('no upsampling ... ')
                    logger.info(supPts[k-1:k+1,:])
                    logger.info(supPts[k-1:k+1,:].shape)
                    logger.info('calc mean: ', np.mean(supPts[k-1:k+1,:], axis=0))
                    logger.info('shape mean: ', np.mean(supPts[k-1:k+1,:], axis=0).shape)
                    if average_thin == True:
                        logger.info('averaging thin layers to reduce number of interpolation points:')
                        curr_supPts = np.mean(supPts[k-1:k+1,:], axis=0)
                        # x_vals = np.full((len(curr_supPts), ), dis_xpos)
                        supPts_x = np.r_[dis_xpos, curr_supPts].reshape((1,3))
                        # supPts_x = np.insert(curr_supPts, 0, x_vals, axis=1)

                        if k == 1:
                            curr_supPts = supPts[k-1,:]
                            curr_supPts[0] = curr_supPts[0] * 0.9
                            supPts_x_add = np.r_[dis_xpos, curr_supPts].reshape((1,3))
                            supPts_x = np.r_[supPts_x, supPts_x_add]
                            
                            supPts_x[:,1] = supPts_x[:,1] + height
                        else:
                            supPts_x[:,1] = supPts_x[:,1] + height

                    else:
                        curr_supPts = supPts[k-1:k+1,:]
                        x_vals = np.full((len(curr_supPts), ), dis_xpos)
                        supPts_x = np.insert(curr_supPts, 0, x_vals, axis=1)
                        logger.info(supPts_x)
                        supPts_x[:,1] = supPts_x[:,1] + height
                    
                    if tilt and (tilt_angles is not None) and (origins is not None):
                        supPts_x[:,0], supPts_x[:,1] = rotate_xy(x=supPts_x[:,0], y=supPts_x[:,1],
                                                                 angle=tilt_angles[log_id], origin=origins[log_id])
                    
                    all_support_points = np.append(all_support_points, supPts_x, axis=0)

        elif up_sample is None:  # no upsampling done
            logger.info('no upsampling done ...')
            supPts_x = np.insert(supPts, 0, x_vals, axis=1)
            supPts_x[:,1] = supPts_x[:,1] + height
            
            if tilt and (tilt_angles is not None) and (origins is not None):
                supPts_x[:,0], supPts_x[:,1] = rotate_xy(x=supPts_x[:,0], y=supPts_x[:,1],
                                                         angle=tilt_angles[log_id], origin=origins[log_id],
                                                         show_plot=False)
            
            all_support_points = np.append(all_support_points, supPts_x, axis=0)
        
        else:
            raise ValueError("PLease select either 'all', 'thick' or None for the up_sample kwarg")

    return all_support_points


def interp_TEMlogs(all_support_points,
                   mesh_resolution, method='linear'):
    """

    Parameters
    ----------
    all_support_points : TYPE
        DESCRIPTION.
    mesh_resolution : TYPE
        DESCRIPTION.
    method : TYPE, optional
        DESCRIPTION. The default is 'linear'.

    Returns
    -------
    None.

    """
    x = all_support_points[:,0]
    z_dep = all_support_points[:,1]
    rho = all_support_points[:,2]

    # target grid to interpolate to
    factor = 3
    xi = np.arange(np.min(x)-factor*mesh_resolution,
                   np.max(x)+factor*mesh_resolution,
                   mesh_resolution)
    zi = np.arange(np.min(z_dep)-factor*mesh_resolution,
                   np.max(z_dep)+factor*mesh_resolution,
                   mesh_resolution)
    xi_mesh, zi_mesh = np.meshgrid(xi,zi[::-1]-0.8)
    print('++++++++++++Interpolation++++++++++++++++++++++++++++++++')
    print('max_zmesh: ', np.max(zi_mesh))

    # interpolate
    rhoi = griddata((x,z_dep), rho, (xi_mesh,zi_mesh), method,
                    rescale=True)

    return xi_mesh, zi_mesh, rhoi


def kriging_TEMlogs(support_pts, mesh_resolution, mesh=None,
                    method='universal_kriging', variogram_mdl='spherical',
                    anisotropy_scaling=1, anisotropy_angle=0,
                    use_weight=True, nlags=6):

    x = support_pts[:,0]
    z_dep = support_pts[:,1]
    rho = support_pts[:,2]
    
    if mesh != None:
        xi = mesh[:, 0]
        yi = mesh[:, 1]
    else:
        xi = np.arange(np.min(x)-2*mesh_resolution,
                        np.max(x)+2*mesh_resolution,
                        mesh_resolution)
        yi = np.arange(np.min(z_dep)-2*mesh_resolution,
                        np.max(z_dep)+2*mesh_resolution,
                        mesh_resolution)

    if method == 'ordinary_kriging':
        orkr = OrdinaryKriging(x, z_dep, rho,
                                 variogram_model=variogram_mdl,
                                 weight=use_weight, nlags=nlags,
                                 exact_values=True,
                                 anisotropy_scaling=anisotropy_scaling,
                                 anisotropy_angle=anisotropy_angle,
                                 verbose=True, enable_plotting=False)
        zi, z_var = orkr.execute('grid', xi, yi,
                                   backend='vectorized',  # vectorized, loop, C
                                   ) #n_closest_points=3

    elif method == 'universal_kriging':
        unkr = UniversalKriging(x, z_dep, rho,
                                  variogram_model=variogram_mdl,
                                  drift_terms=['regional_linear'],
                                  weight=use_weight, nlags=nlags,
                                  exact_values=True,
                                  anisotropy_scaling=anisotropy_scaling,
                                  anisotropy_angle=anisotropy_angle,
                                  verbose=True, enable_plotting=False)
        zi, z_var = unkr.execute('grid', xi, yi,
                                   backend='vectorized',  # vectorized, loop, C
                                   )  # n_closest_points=3
    
    return xi, yi, zi, z_var


def interpolate_TEMelem(telx, telz, telRho,
                        mesh_resolution, method='linear',
                        useMiddle=True, moveFrac=20):
    """This routine interpolates TEM elements obtained from createTEMelem routine.

    further usage in plot section to show interpolation between multiple logs
    DoNOT use the middle of each element.
    """
    # prepare data for interpolation...
    if useMiddle == True:
        x = telx.mean(axis=0)
        z_dep = telz.mean(axis=0)
        rho = np.copy(telRho)
    else:
        x_upper = telx.mean(axis=0)
        x_lower = telx.mean(axis=0)
        x = np.hstack((x_upper, x_lower))

        z_upper = telz[0,:]
        z_lower = telz[2,:]
        el_height = z_upper-z_lower
        z_elMove = el_height/moveFrac
        z_dep = np.hstack((z_upper-z_elMove,
                           z_lower+z_elMove))

        rho = np.hstack((telRho, telRho))

    # target grid to interpolate to
    print('start at xdistance:', np.min(telx))
    xi = np.arange(np.min(telx),
                   np.max(telx)+mesh_resolution,
                   mesh_resolution)
    zi = np.arange(np.min(telz)-mesh_resolution,
                   np.max(telz)+2*mesh_resolution,
                   mesh_resolution)

    xi_mesh, zi_mesh = np.meshgrid(xi,zi[::-1])

    # interpolate
    rhoi = griddata((x,z_dep), rho, (xi_mesh,zi_mesh), method,
                    rescale=True, fill_value=25)

    return x, z_dep, xi_mesh, zi_mesh, rhoi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %% plotting
def plot_interpol_contours(axis, xi_mesh, yi_mesh, zi,
                           c_limits, lvls, cmap, log10,
                           show_clines=False, cline_freq=20):
    """
    

    Parameters
    ----------
    axis : TYPE
        DESCRIPTION.
    xi_mesh : TYPE
        DESCRIPTION.
    yi_mesh : TYPE
        DESCRIPTION.
    zi : TYPE
        DESCRIPTION.
    c_limits : TYPE
        DESCRIPTION.
    lvls : TYPE
        DESCRIPTION.
    cmap : TYPE
        DESCRIPTION.
    log10 : TYPE
        DESCRIPTION.
    show_clines : TYPE, optional
        DESCRIPTION. The default is False.
    cline_freq : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    norm = colors.Normalize() if (log10 == False) else colors.LogNorm()
    levels_cf = np.linspace(c_limits[0], c_limits[1], lvls)
    ctrf = axis.contourf(xi_mesh, yi_mesh, zi,
                        levels=levels_cf, cmap=cmap,
                        norm=norm, extend='both', zorder=0)

    if show_clines:
        levels_c = levels_cf[::cline_freq]  # select every 20th level of the filling
        ctr = axis.contour(xi_mesh, yi_mesh,
                          zi, colors='white',
                          levels=levels_c, norm=norm,
                          zorder=5)
        axis.clabel(ctr, fontsize=10, inline=True)  # add labels
        return ctrf, ctr
    else:
        return ctrf



def plot_framesandnames(axis, log_width, tilt_angles, origins,
                frame_lowLeft, xz_df, snd_names, relRMS_sig, 
                show_frames=True, show_rms=False, show_names=False, txtsize=14,
                top_space=5, extend_bot=5, rotation=0):
    """
    

    Parameters
    ----------
    axis : TYPE
        DESCRIPTION.
    log_width : TYPE
        DESCRIPTION.
    tilt_angles : TYPE
        DESCRIPTION.
    origins : TYPE
        DESCRIPTION.
    frame_lowLeft : TYPE
        DESCRIPTION.
    xz_df : TYPE
        DESCRIPTION.
    snd_names : TYPE
        DESCRIPTION.
    relRMS_sig : TYPE
        DESCRIPTION.
    show_rms : TYPE, optional
        DESCRIPTION. The default is False.
    show_names : TYPE, optional
        DESCRIPTION. The default is False.
    txtsize : TYPE, optional
        DESCRIPTION. The default is 14.
    top_space : TYPE, optional
        DESCRIPTION. The default is 5.
    extend_bot : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    None.

    """

    n_logs = len(snd_names)

    frame_upLeft = np.copy(frame_lowLeft)
    frame_upLeft[:, 1] = origins[:, 1]
    frame_upLeft[:, 0] = np.asarray(xz_df.X) - (log_width/2)
    
    for i in range(0, n_logs):
        if show_frames:
            frame = patches.Rectangle(xy=(frame_upLeft[i,0], frame_upLeft[i,1]),
                                      width=log_width,
                                      height=-frame_upLeft[i,2],
                                      linewidth=1.5,
                                      edgecolor='k',
                                      facecolor='none',
                                      angle=tilt_angles[i]*180/np.pi,
                                      zorder=10)
            axis.add_patch(frame)

        x_txt = frame_upLeft[i,0]
        z_txt = max(frame_upLeft[:, 1]) + top_space + 10

        snd_name = snd_names[i]  # snd_name = snd_names[i].upper()
        snd_rms = '\n{:.1f}%'.format(relRMS_sig[i])

        if show_rms:
            if i == 0:
                axis.text(x_txt-5, z_txt+5, '\nRMS$=$',
                      ha='right', va='center',
                      rotation=0, size=txtsize)
            axis.text(x_txt+1, z_txt+7, snd_rms,
                      ha='center', va='center',
                      rotation=rotation, size=txtsize)
        if show_names:
            axis.text(x_txt+1, z_txt+12, snd_name,
                      ha='center', va='center', color='crimson',
                      rotation=rotation, size=txtsize)
    return


def plot_TEM_cbar(fig, axis, pacoll, label, cmin, cmax, 
                  cbar_pos='right', log10_switch=True, 
                  label_fmt='%.1f', round_tick_label=1):
    """
    

    Parameters
    ----------
    fig : TYPE
        DESCRIPTION.
    axis : TYPE
        DESCRIPTION.
    pacoll : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.
    cmin : TYPE
        DESCRIPTION.
    cmax : TYPE
        DESCRIPTION.
    cbar_pos : TYPE, optional
        DESCRIPTION. The default is 'right'.
    log10_switch : TYPE, optional
        DESCRIPTION. The default is True.
    label_fmt : TYPE, optional
        DESCRIPTION. The default is '%.1f'.
    round_tick_label : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    """
    if cbar_pos == 'right':
        divider = make_axes_locatable(axis)
        cax1 = divider.append_axes("right", size="2%", pad=0.1)
        cb1 = fig.colorbar(pacoll, cax=cax1, format=label_fmt)
        # cb.ax.yaxis.set_ticks([10, 20, 40, 100, 200])
        if log10_switch is True:
            ticks = np.round(np.logspace(np.log10(cmin), np.log10(cmax), 5), round_tick_label)
            cb1.ax.yaxis.set_ticks(ticks)
        else:
            tick_locator = ticker.MaxNLocator(nbins=5)
            cb1.locator = tick_locator; cb1.update_ticks()
        cb1.ax.minorticks_off()
        cb1.set_label(label)

    elif cbar_pos == 'left':
        divider = make_axes_locatable(axis)
        cax1 = divider.append_axes("left", size="2%", pad=0.8)
        cb1 = fig.colorbar(pacoll, cax=cax1, format=label_fmt)
        if log10_switch is True:
            ticks = np.round(np.logspace(np.log10(cmin), np.log10(cmax), 5), round_tick_label)
            cb1.ax.yaxis.set_ticks(ticks)
        else:
            tick_locator = ticker.MaxNLocator(nbins=5)
            cb1.locator = tick_locator; cb1.update_ticks()
        cb1.ax.minorticks_off()
        cb1.ax.yaxis.set_ticks_position('left')
        cb1.ax.yaxis.set_label_position('left')
        # cb1.set_label('TEM: ' + labelRho)
        cb1.set_label(label)
        axis.yaxis.set_label_coords(-0.015, 0.05)
        axis.set_ylabel('H (m)', rotation=90)
    return


def plot_section(stdPath, path, file,  # TODO update to newest features, move to class structure!!
                 path_coord=None, file_coord=None,
                 log10=False, saveFig=True,
                 intMethod='linear', version='v1', filetype='.jpg', dpi=300,
                 xoffset=0, label_s=8, log_width=1.5, depth=40,
                 rmin=99, rmax=99):
    """
    this routine plots a section of TEM-models obtained from ZondTEM software

    Parameters
    ----------
    stdPath : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.
    file : TYPE
        DESCRIPTION.
    path_coord : TYPE, optional
        DESCRIPTION. The default is None.
    file_coord : TYPE, optional
        DESCRIPTION. The default is None.
    log10 : TYPE, optional
        DESCRIPTION. The default is False.
    saveFig : TYPE, optional
        DESCRIPTION. The default is True.
    intMethod : TYPE, optional
        DESCRIPTION. The default is 'linear'.
    version : TYPE, optional
        DESCRIPTION. The default is 'v1'.
    filetype : TYPE, optional
        DESCRIPTION. The default is '.jpg'.
    dpi : TYPE, optional
        DESCRIPTION. The default is 300.
    xoffset : TYPE, optional
        DESCRIPTION. The default is 0.
    label_s : TYPE, optional
        DESCRIPTION. The default is 8.
    log_width : TYPE, optional
        DESCRIPTION. The default is 1.5.
    depth : TYPE, optional
        DESCRIPTION. The default is 40.
    rmin : TYPE, optional
        DESCRIPTION. The default is 99.
    rmax : TYPE, optional
        DESCRIPTION. The default is 99.

    Returns
    -------
    None.

    """
    
    fullpath = stdPath + os.sep + path

    (telx,
     telz,
     telRho,
     frame_lowLeft,
     nLogs, topo, snd_names) = create_TEMelem(fullpath, file,
                                              path_coord=path_coord, file_coord=file_coord,
                                              log_width=log_width, xoffset=xoffset, zondVersion='IP')

    fig, ax1 = plt.subplots(1,1, figsize=(18, 5)) #figSize in inch
    #Plot1 - RHO
    if intMethod != 'No':
        lvls = 200
        if rmin == rmax:
            xi_mesh, zi_mesh, rhoi = interpolate_TEMelem(telx,
                                                         telz,
                                                         telRho,
                                                         mesh_resolution=0.1,
                                                         method=intMethod)
            if log10 == True:
                ctrf = plt.contourf(xi_mesh, zi_mesh, np.log10(rhoi),
                                    levels=lvls, cmap='jet')
            else:
                ctrf = plt.contourf(xi_mesh, zi_mesh, rhoi,
                                    levels=lvls, cmap='jet')

            upLeft = np.copy(topo[0,:] + np.array([[-5, 50]]))
            topoMask = np.insert(topo, 0, upLeft, axis=0)
            upRight = np.copy(topo[-1,:] + np.array([[5, 50]]))
            topoMask = np.insert(topoMask, len(topo)+1, upRight, axis=0)
            maskTopo  = Polygon(topoMask, facecolor='white', closed=True)
            ax1.add_patch(maskTopo)
        if rmin != rmax:
            xi_mesh, zi_mesh, rhoi = interpolate_TEMelem(telx,
                                                         telz,
                                                         telRho,
                                                         mesh_resolution=0.1,
                                                         method=intMethod)
            if log10 == True:
                ctrf = plt.contourf(xi_mesh, zi_mesh, np.log10(rhoi),
                                    levels=lvls, cmap='jet')
            else:
                ctrf = plt.contourf(xi_mesh, zi_mesh, rhoi,
                                    levels=lvls, cmap='jet')

            upLeft = np.copy(topo[0,:] + np.array([[-5, 50]]))
            topoMask = np.insert(topo, 0, upLeft, axis=0)
            upRight = np.copy(topo[-1,:] + np.array([[5, 50]]))
            topoMask = np.insert(topoMask, len(topo)+1, upRight, axis=0)
            maskTopo  = Polygon(topoMask, facecolor='white', closed=True)
            ax1.add_patch(maskTopo)

    pr = get_PatchCollection(telx, telz)
    if log10 == True:
        pr.set_array(np.log10(telRho))
        labelRho = r"$log_{10}\rho$ [$\Omega$m]"
        if not rmin == rmax:
            pr.set_clim([np.log10(rmin), np.log10(rmax)])
            ctrf.set_clim([np.log10(rmin), np.log10(rmax)])
    else:
        pr.set_array(telRho)
        labelRho = r"$\rho$ [$\Omega$m]"
        if not rmin == rmax:
            pr.set_clim([rmin, rmax])
            ctrf.set_clim([rmin, rmax])
    pr.set_cmap(cmap=matplotlib.cm.jet)

    ax1.add_collection(pr)

    #make frames around logs; add labels to soundings
    for i in range(0,nLogs):
        frame = patches.Rectangle((frame_lowLeft[i,0],frame_lowLeft[i,1]-10),
                                  log_width,
                                  frame_lowLeft[i,2]+10,
                                  linewidth=1,
                                  edgecolor='k',
                                  facecolor='none')
        ax1.add_patch(frame)

        x_txt = frame_lowLeft[i,0]
        z_txt = frame_lowLeft[i,1]+frame_lowLeft[i,2] + 3
        ax1.text(x_txt+5, z_txt+4, snd_names[i],
                 ha='center', va='center',
                 rotation=45, size=12,
                 bbox=dict(facecolor='white', alpha=1))

    ax1.plot(topo[:,0], topo[:,1]-0.5, '-k', linewidth=2.5)
    if file_coord != None:                         # plot add Info (bathymetrie)
        coord_raw = pd.read_csv(path_coord + file_coord, sep=',', engine='python',
                                header=0)
        # print(coord_raw)
        ax1.plot(coord_raw.Distance, coord_raw.Z-coord_raw.Depth, 'o:k', linewidth=1.5)

    ax1.grid(which='major', axis='y', color='grey', linestyle='--', linewidth=1)
    ax1.locator_params(axis='x', nbins=15)
    ax1.locator_params(axis='y', nbins=7)
    ax1.tick_params(axis='both', which='both', labelsize=label_s)
    ax1.set_ylabel('Z [m]', fontsize=label_s, rotation=0)
    ax1.set_xlabel('distance [m]', fontsize=label_s)
    ax1.yaxis.set_label_coords(-0.03, 1.05)

#    tickInc = 5
#    ticks = np.arange(rmin, rmax+tickInc, tickInc)
#    print('these are the planned ticks: ', ticks)
#    cb = fig.colorbar(pr,
#                      ax=ax1,
#                      orientation='vertical',
#                      aspect=10,
#                      pad=0.01,
#                      ticks=ticks,
#                      format="%.1f")
#    cb.set_label(labelRho,
#                 fontsize=label_s)
#    cb.ax.tick_params(labelsize=label_s)
#
#    cb.set_ticklabels(ticklabels=ticks)
##    cb.set_clim([rmin, rmax])

    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="3%", pad=0.2)
    cb = fig.colorbar(pr, cax=cax1, format='%d')
    cb.ax.minorticks_off()
    tick_locator = ticker.MaxNLocator(nbins=6)
    cb.locator = tick_locator; cb.update_ticks()
    cb.set_label(labelRho,
                 fontsize=label_s)
    cb.ax.tick_params(labelsize=label_s)

    ax1.set_ylim((np.max(telz)-depth, np.max(telz)+5))
    ax1.set_xlim((np.min(telx)-2, np.max(telx)+2))
#    ax1.set_aspect(2, adjustable='box')
    ax1.set_aspect('equal')

    if saveFig == True:
        savepath = fullpath + os.sep + file[:-4] + intMethod + '_int_' + version + filetype
        print('saving figure to:',
              savepath)
        fig.savefig(savepath,
                    dpi=dpi,
                    bbox_inches='tight')
    else:
        plt.show()
    return



def compare_model_response(path, file, path_savefig, plottitle=None,
                           fig=None, axes=None, snd_info=None,
                           save_fig=False, filetype='.png', dpi=300,
                           show_rawdata=True, show_all=True,
                           xoffset=0, snd_indxs=np.r_[0],
                           log_rhoa=False, log_res=False,
                           sign_lim=(1e-8, 1e3), rhoa_lim=(0, 500),
                           time_lim=(2, 15500), res_lim=(1e0, 1e4), dep_lim=(0, 40),
                           linewidth=2, markerSize=3):
    """
    routine to plot single or multiple model responses in order to check
    how well the data is represented by the model.
    """
    # plt.close('all')
    # fullpath = stdPath + os.sep + path
    fullpath = path

    savePath = path_savefig + os.sep + 'dataFit'
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    dat, indices_hdr, indices_mresp, indices_mdl, zondVersion = parse_zondxls(fullpath, file)
    nLogs = len(indices_hdr)

    if show_rawdata == True:
        inv_type = file.split('_')[-1][0:2]
        inv_setup = file.split('_')[-2]
        inv_info = f"{inv_setup}_{inv_type}"
        file_raw = file.replace(inv_info, 'raw')
        (dat_raw, indices_hdr_raw, indices_mresp_raw,
         indices_mdl_raw, zondVersion) = parse_zondxls(fullpath, file_raw)

    if show_all == True:
        logIndx = range(0,nLogs)

    for i in snd_indxs:
        # preparation for plotting part
        header = dat.loc[indices_hdr.start[i]:indices_hdr.end[i], ['c1','c2','c3','c4']]
        logID = header.iloc[0,1]
        #coordinates = header.iloc[2,1:4]
        distance = header.iloc[1,1]
        rRMS_sig = header.iloc[3,1]
        rRMS_roa = header.iloc[3,3]

        signal = get_response(dat, idcs_response=indices_mresp, snd_id=i)

        if show_rawdata == True:
            signal_raw = get_response(dat_raw,
                                      idcs_response=indices_mresp_raw,
                                      snd_id=i)

        if zondVersion == 'IP':
            model = dat.loc[indices_mdl.start[i]:indices_mdl.end[i],
                                  ['c1','c2','c3','c4','c5','c6','c7','c8']]
            model.columns = ['ctr','Rho','Pol','Tconst','Cexpo','MagP','h','z']
            model.drop(['Pol','Tconst','Cexpo'], axis=1, inplace=True)

        elif zondVersion == 'noIP':
            model = dat.loc[indices_mdl.start[i]:indices_mdl.end[i], ['c1','c2','c3','c4','c5']]
            model.columns = ['ctr','Rho','MagP','h','z']

        # Preparing figure and axis objects
        if fig == None:
            fig = plt.figure(figsize=(8, 6)) #figSize in inch; Invoke fig!
        else:
            print('using provided figure object ...')
        if axes == None:
            gs = gridspec.GridSpec(2, 4) #define grid for subplots
            ax_sig = fig.add_subplot(gs[0, 0:3])
            ax_roa = fig.add_subplot(gs[1, 0:3])
            ax_mdl = fig.add_subplot(gs[0:2, 3])
        else:
            print('using 3 (?) provided axes objects ...')
            # TODO add check if really three of them where provided ...

        # ~~~~~~~~~~~~~~~~~~~~~~~ dBz/dt plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if show_rawdata == True:
            ax_sig.loglog(signal_raw.time*10**6, signal_raw.U_O*10**-6, 'D',
                        color='darkgray',
                        lw=linewidth, ms=markerSize)

        ax_sig.loglog(signal.time*10**6, signal.U_C*10**-6, '-k',   # from uV to V
                      signal.time*10**6, signal.U_O*10**-6,  'Dg',
                      lw=linewidth, ms=markerSize)

        if show_rawdata == False:
            ax_sig.xaxis.set_minor_formatter(ScalarFormatter())
            ax_sig.xaxis.set_major_formatter(ScalarFormatter())
            print('overriding limits')
            time_lim = None
            sign_lim = None

        if time_lim is not None:
            ax_sig.set_xlim(time_lim)

        if sign_lim is not None:
            ax_sig.set_ylim(sign_lim)

        # ax_sig.set_xlabel('time ($\mu$s)')
        ax_sig.set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V)")
        ax_sig.tick_params(axis='x',          # changes apply to the x-axis
                           which='both',      # both major and minor ticks are affected
                           bottom=False,      # ticks along the bottom edge are off
                           top=False,         # ticks along the top edge are off
                           labelbottom=False) # labels along the bottom edge are off

        ax_sig.grid(which='major', color='white', linestyle='-')
        ax_sig.grid(which='minor', color='white', linestyle=':')

        obRMS = offsetbox.AnchoredText(f'rRMS: {rRMS_sig}%', loc='upper left')
        ax_sig.add_artist(obRMS)

        if snd_info is not None:
            text_props = {'fontsize': 12, 'fontfamily':'monospace'}
            ob = offsetbox.AnchoredText(snd_info, loc=1, prop=text_props)
            ax_sig.add_artist(ob)


        # ~~~~~~~~~~~~~~~~~~~~~~~ rhoa plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if show_rawdata == True:
            ax_roa.semilogx(signal_raw.time*10**6, signal_raw.rho_O, 'D',
                            color='darkgray', label='raw data',
                            lw=linewidth, ms=markerSize)

        ax_roa.semilogx(signal.time*10**6, signal.rho_C, '-k',
                        lw=linewidth, ms=markerSize,
                        label='model_response')
        
        ax_roa.semilogx(signal.time*10**6, signal.rho_O, 'Dg',
                        lw=linewidth, ms=markerSize,
                        label='selected data')

        if show_rawdata == False:
            ax_roa.xaxis.set_minor_formatter(ScalarFormatter())
            ax_roa.xaxis.set_major_formatter(ScalarFormatter())
            print('overriding limits')
            time_lim = None
            sign_lim = None
        if time_lim is not None:
            ax_roa.set_xlim(time_lim)
        if rhoa_lim is not None:
            ax_roa.set_ylim(rhoa_lim)
        if log_rhoa == True:
            ax_roa.set_yscale('log')

        ax_roa.grid(which='major', color='white', linestyle='-')
        ax_roa.grid(which='minor', color='white', linestyle=':')

        ax_roa.set_xlabel('time ($\mu$s)')
        ax_roa.set_ylabel(r'$\rho_a$ ' + r'($\Omega$m)')

        obRMS = offsetbox.AnchoredText(f'rRMS: {rRMS_roa}%', loc='upper left')
        ax_roa.add_artist(obRMS)

        ax_roa.legend(loc='best')


        # ~~~~~~~~~~~~~~~~~~~~~~~ inv model plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        rA_model = zond2stepmdl(model)

        ax_mdl.plot(rA_model[:,1], rA_model[:,0], '-ko',
                 linewidth=linewidth, markerSize=markerSize,
                 label='model')

        if res_lim is not None:
            ax_mdl.set_xlim(res_lim)
        if dep_lim is not None:
            ax_mdl.set_ylim(dep_lim)
        if log_res == True:
            ax_mdl.set_xscale('log')

        ax_mdl.set_xlabel(r'$\rho$ ($\Omega$m)')
        ax_mdl.set_ylabel('height (m)')
        ax_mdl.legend()
        
        ax_mdl.yaxis.set_label_position("right")
        ax_mdl.yaxis.tick_right()

        logName = logID #+ ' at ' + str(distance+xoffset) + 'm'
        if plottitle != None:
            logName = plottitle

        fig.suptitle(logName, fontsize=14)

        if save_fig == True:
            fig.tight_layout()
            plt.subplots_adjust(top=0.95)
            if show_rawdata == False:
                full_savePath = savePath + os.sep + 'zoom_' + file.replace('.xls', '') + '_' + logID + filetype
            else:
                full_savePath = savePath + os.sep + file.replace('.xls', '') + '_' + logID + filetype
            fig.savefig(full_savePath,
                        dpi=dpi,
                        bbox_inches='tight')
            print('saving figure to:\n', full_savePath)
            # plt.close('all')
        else:
            print("Show figure...")
            fig.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.show()

    return fig, ax_sig, ax_roa, ax_mdl


def plot_doi(axis, DOIs, distances, **kwargs):
    axis.plot(distances, np.asarray(DOIs),
              '--.', **kwargs)
    return axis


def plot_intfc(axis, depths, distances, **kwargs):
    axis.plot(distances, depths,
              '--.', **kwargs)
    return axis


def put_blanking(ax, telx, telz, depth):
    """
    adapted from Jakobs routine to blank part of a section with a constant depth below
    the surface;
    ToDO: improve to longer polygon with a point at constant depth below each tem-sounding"""
    maxx = np.max(telx)
    minx = np.min(telx)
    init_z = np.min(telz[:,20])
    final_z = np.min(telz[:,-20])

    ax.add_patch(Polygon([(minx, init_z-depth),
                          (maxx, final_z-depth),
                          (maxx, final_z-300),
                          (minx, init_z-300)],
                         closed=True,
                         facecolor='white'))


def get_depths_of_inflictions(mdl, min_depth, max_depth,
                              approach='max-d1', show_plot=True):
    """
    

    Parameters
    ----------
    model : np.array(nx2), depth, 
        col0 depth -, col1 values. step model
    min_depth : float
        minimum depth.
    max_depth : TYPE
        maximum depth. (sub zero)
    approach : str
        use either the maximum of the first derivative or the min of the second
    show_plot : bool
        decide whether you want to see the plot ...

    Returns
    -------
    depth : float
        depth to infliction pont in the given range
    value : float
        value of the model at the retrieved depth

    """
    model = mdl.copy()
    
    _, idx = np.unique(model[:,0], return_index=True)
    unique_depths = model[:,0][np.sort(idx)]
    
    _, idx = np.unique(model[:,1], return_index=True)
    unique_vals = model[:,1][np.sort(idx)]
    
    print(unique_depths[:10])
    print(unique_depths.shape)
    print(unique_vals[:10])
    print(unique_vals.shape)
    
    model_unique = np.column_stack((unique_depths[:100],
                                    unique_vals[:100]))  # TODO fix this terrible hack here!!
    print(model_unique[:10])
    
    mask = (model_unique[:,0] < min_depth) & (model_unique[:,0] > max_depth)
    sub_model = model_unique[mask]

    d1 = np.gradient(sub_model[:,1])
    d2 = np.gradient(d1)
    if approach == 'max-d1':
        value_d1 = max(abs(d1))  # 1st derivative maximum
        print('min of d1: ', value_d1)
        
        print(any((d1 == value_d1)))
        value = sub_model[(abs(d1) == value_d1), 1].item()
        print('value of d1: ', value)
        
        depth = sub_model[(abs(d1) == value_d1), 0].item()
        print('depth of d1: ', depth)
    elif approach == '0-d2':

        value_d2 = min(abs(d2))  # 2nd derivative closest to 0
        print('min of d2: ', value_d2)
        
        print(any((d2 == value_d2)))
        value = sub_model[(abs(d2) == value_d2), 1].item()
        print('value of d2: ', value)
        
        depth = sub_model[(abs(d2) == value_d2), 0].item()
        print('depth of d2: ', depth)
    else:
        raise ValueError('please select either "max-d1", or "0-d2" for the approach.')

    if show_plot:
        fig, ax = plt.subplots(1,1)
        ax.plot(model[:,1], model[:,0], '-k', label='model')
        ax.hlines(y=min_depth,
                  xmin=min(sub_model[:,1]),
                  xmax=max(sub_model[:,1]),
                  color='crimson', label='minimum depth')
        ax.hlines(y=max_depth,
                  xmin=min(sub_model[:,1]),
                  xmax=max(sub_model[:,1]),
                  color='crimson', label='maximum depth')

        ax.plot(value, depth, 'oc', label='infliction point')
        ax.set_xlim((1, max(sub_model[:,1])*1.5))
        ax.legend(fontsize=10, loc='lower left')

        ax1 = ax.twiny()
        ax1.plot(d1, sub_model[:,0], 'b:', label='first derivative')
        ax1.plot(d2, sub_model[:,0], 'g:', label='second derivative')
        
        ax1.legend(fontsize=10, loc='lower right')

    # return value
    return depth, value    

# %% dual moment routiners (old and outdated - update only if necessary!!)
def parse_zondxls_DM(path, file):
    """
    function to parse ZondTEM1d .xls file and create indices for further
    subselecting of dat file
    version for Dual Moment TEM!!
    ToCode: zondVersion!!

    """
    raw = pd.read_excel(path + os.sep + file,
                        names = ['empty','c1','c2','c3','c4','c5','c6', 'c7'],
                        header=None)

    dat = raw.drop(labels='empty', axis=1)
    dat = dat.drop(labels=0, axis=0)
    indices_labels = np.flatnonzero(dat.c1 == '#') + 1

    ev2nd0 = indices_labels[::2]
    ev2nd1 = indices_labels[1::2]
    end = len(dat)
    endMdl = np.append(ev2nd0[1:]-5, [end])

    indices_hdr = pd.DataFrame({'start':ev2nd0-4, 'end':ev2nd0},
                               columns=['start', 'end'])

    indices_mresp = pd.DataFrame({'start':ev2nd0+1, 'end':ev2nd1-1},
                               columns=['start', 'end'])

    indices_mdl = pd.DataFrame({'start':ev2nd1+1, 'end':endMdl},
                               columns=['start', 'end'])
    return dat, indices_hdr, indices_mresp, indices_mdl


def comp_Mdlresp2_DMdata(stdPath, path, file, plottitle=None,
                         saveFig=False, filetype='.png',
                         show_rawdata=True, show_all=True,
                         xoffset=0, logIndx = np.array([0]), rho2log=False,
                         set_appResLim=False, minApRes=0, maxApRes=500,
                         set_signalLim=False, minSig=10e-8, maxSig=10e4,
                         linewidth=2, markerSize=3):
    """
    DualMoment TEM version - eg WalkTEM
    routine to plot single or multiple model responses in order to check
    how well the data is represented by the model.
    """
    plt.close('all')
    fullpath = stdPath + os.sep + path

    savePath = fullpath + os.sep + 'dataFit'
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    dat, indices_hdr, indices_mresp, indices_mdl = parse_zondxls(fullpath, file)
    nLogs = len(indices_hdr)

    if show_rawdata == True:
        file_raw = file.replace('s1', 'raw')
        dat_raw, indices_hdr_raw, indices_mresp_raw, indices_mdl_raw = parse_zondxls(fullpath, file_raw)

    if show_all == True:
        logIndx = range(0,nLogs)

    for i in logIndx:
        # preparation for plotting part
        header = dat.loc[indices_hdr.start[i]:indices_hdr.end[i], ['c1','c2','c3','c4']]
        logID = header.iloc[0,1]
        #coordinates = header.iloc[2,1:4]
        distance = header.iloc[1,1]
        #error = header.iloc[3,1]

        signal = dat.loc[indices_mresp.start[i]:indices_mresp.end[i],
                     ['c1','c2','c3','c4','c5','c6','c7']]
        signal.columns = ['ctr','time','rho_O','rho_C','U_O','U_C','Ts']

        id_Ts1 = np.asarray(signal.loc[:,'Ts'] == 1).squeeze()
        id_Ts2 = np.asarray(signal.loc[:,'Ts'] == 2).squeeze()

        if show_rawdata == True:
            signal_raw = dat_raw.loc[indices_mresp_raw.start[i]:indices_mresp_raw.end[i],
                                     ['c1','c2','c3','c4','c5','c6','c7']]
            signal_raw.columns = ['ctr','time','rho_O','rho_C','U_O','U_C','Ts']
            id_Ts1r = np.asarray(signal_raw.loc[:,'Ts'] == 1).squeeze()
            id_Ts2r = np.asarray(signal_raw.loc[:,'Ts'] == 2).squeeze()

        model = dat.loc[indices_mdl.start[i]:indices_mdl.end[i], ['c1','c2','c3','c4','c5']]
        model.columns = ['ctr','Rho','MagP','h','z']


        # Plotting part
        fig = plt.figure(figsize=(12, 8)) #figSize in inch; Invoke fig!
        gs = gridspec.GridSpec(2, 4) #define grid for subplots

        appRes = fig.add_subplot(gs[0, 0:3])
        if show_rawdata == True:
            appRes.semilogx(signal_raw.loc[id_Ts1r, 'time']*10**6,
                            signal_raw.loc[id_Ts1r, 'rho_O'],
                            'D', color='darkgray', markerSize=markerSize)
            appRes.semilogx(signal_raw.loc[id_Ts2r, 'time']*10**6,
                            signal_raw.loc[id_Ts2r, 'rho_O'],
                            '*', color='darkgray', markerSize=markerSize)

        appRes.semilogx(signal.loc[id_Ts1, 'time']*10**6,
                        signal.loc[id_Ts1, 'rho_C'], '-',
                        color='lightgreen', linewidth=linewidth)
        appRes.semilogx(signal.loc[id_Ts1, 'time']*10**6,
                        signal.loc[id_Ts1, 'rho_O'],  'D',
                        color='green', markerSize=markerSize)
        appRes.semilogx(signal.loc[id_Ts2, 'time']*10**6,
                        signal.loc[id_Ts2, 'rho_C'], ':',
                        color='salmon', linewidth=linewidth)
        appRes.semilogx(signal.loc[id_Ts2, 'time']*10**6,
                        signal.loc[id_Ts2, 'rho_O'],  '*',
                        color='darkred', markerSize=markerSize)

        if show_rawdata == False:
            appRes.xaxis.set_minor_formatter(ScalarFormatter())
            appRes.xaxis.set_major_formatter(ScalarFormatter())

        plt.grid(which='major', color='lightgray', linestyle='-')
        plt.grid(which='minor', color='lightgray', linestyle=':')

        appRes.set_xlabel('time [$\mu$s]')
        appRes.set_ylabel(r'$\rho_a$' + r'[$\Omega$m]')

        if set_appResLim == True:
            appRes.set_ylim((minApRes,maxApRes))


#        minApRes = np.floor(np.min(signal.rho_O)*0.9)
#        maxApRes = np.ceil(np.max(signal.rho_O)*1.01)
#
#        minTime = np.floor(np.min(signal_raw.time)*0.9)
#        maxTime = np.ceil(np.max(signal_raw.time)*1.01)
#
#        appRes.set_ylim((minApRes,maxApRes))
#        appRes.set_xlim((minTime,maxTime))

        if show_rawdata == True:
            #appRes.legend(['excluded data', 'model response', 'used Data'])
            appRes.legend(['excluded data-1A', 'excluded data-7A',
                           'model response-1A', 'used data-1A',
                           'model response-7A', 'used data-7A'])
        else:
            #appRes.legend(['model response', 'measured'])
            appRes.legend(['model response-1A', 'measured-1A',
                           'model response-7A', 'measured-7A',])


        volt = fig.add_subplot(gs[1, 0:3])
        if show_rawdata == True:
            volt.loglog(signal_raw.loc[id_Ts1r, 'time']*10**6,
                        signal_raw.loc[id_Ts1r, 'U_O']*10**-8,
                        'D', color='darkgray',
                        linewidth=0.9, markerSize=3)
            volt.loglog(signal_raw.loc[id_Ts2r, 'time']*10**6,
                        signal_raw.loc[id_Ts2r, 'U_O']*10**-8,
                        '*', color='darkgray',
                        linewidth=0.9, markerSize=3)


#            volt.loglog(signal_raw.time*10**6, signal_raw.U_O, 'D',
#                        color='darkgray',
#                        linewidth=0.9, markerSize=3)

        volt.loglog(signal.loc[id_Ts1, 'time']*10**6,
                    signal.loc[id_Ts1, 'U_C']*10**-8, '-',
                    color='lightgreen', linewidth=linewidth)
        volt.loglog(signal.loc[id_Ts1, 'time']*10**6,
                    signal.loc[id_Ts1, 'U_O']*10**-8,  'D',
                    color='green', markerSize=markerSize)
        volt.loglog(signal.loc[id_Ts2, 'time']*10**6,
                    signal.loc[id_Ts2, 'U_C']*10**-8, ':',
                    color='salmon', linewidth=linewidth)
        volt.loglog(signal.loc[id_Ts2, 'time']*10**6,
                    signal.loc[id_Ts2, 'U_O']*10**-8,  '*',
                    color='darkred', markerSize=markerSize)

#        volt.loglog(signal.time*10**6, signal.U_C, '-k',
#                    signal.time*10**6, signal.U_O,  'Dg',linewidth=0.9, markerSize=3)

        volt.set_xlabel('time [$\mu$s]')
        volt.set_ylabel('U/I [V/A]')

        if set_signalLim == True:
            volt.set_ylim((minSig,maxSig))


        if show_rawdata == True:
            #volt.legend(['excluded data', 'model response', 'used Data'])
            volt.legend(['excluded data-1A', 'excluded data-7A',
                         'model response-1A', 'used data-1A',
                         'model response-7A', 'used data-7A'])
        else:
            #volt.legend(['model response', 'measured'])
            volt.legend(['model response-1A', 'measured-1A',
                         'model response-7A', 'measured-7A',])

        if show_rawdata == False:
            volt.xaxis.set_minor_formatter(ScalarFormatter())
            volt.xaxis.set_major_formatter(ScalarFormatter())

        volt.grid(which='major', color='lightgray', linestyle='-')
        volt.grid(which='minor', color='lightgray', linestyle=':')

        rA_model = zond2stepmdl(model)

        mdl = fig.add_subplot(gs[0:2, 3])
        mdl.plot(rA_model[:,1], rA_model[:,0], '-ko',
                 linewidth=linewidth, markerSize=markerSize)

        if rho2log == True:
            mdl.set_xscale('log')
            xlabel = r'$log_{\rho}$ [$\Omega$m]'
        else:
            xlabel = r'$\rho$ [$\Omega$m]'

        mdl.set_xlabel(xlabel)
        mdl.set_ylabel('depth [m]')
        mdl.legend(['model'])

        logName = logID + ' at ' + str(distance) + 'm'
        if plottitle != None:
            logName = plottitle


        fig.suptitle('Dual-Moment (Walk-TEM) ' + logName, fontsize=14)
        if saveFig == True:
            fig.tight_layout()
            plt.subplots_adjust(top=0.95)
            if show_rawdata == False:
                full_savePath = savePath + os.sep + 'zoom' + file.replace('.xls', '') + '_' + logID + filetype
                fig.savefig(full_savePath,
                            dpi=300,
                            bbox_inches='tight')
                print('saving figure to:\n', full_savePath)
            else:
                full_savePath = savePath + os.sep + file.replace('.xls', '') + '_' + logID + filetype
                fig.savefig(full_savePath,
                            dpi=300,
                            bbox_inches='tight')
                print('saving figure to:\n', full_savePath)
            plt.close('all')
        else:
            fig.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.show()
    return


def sturges_theorem(n_data):
    return ceil(np.log2(n_data)) + 1


def doanes_theorem(data):
    n = len(data)
    s_g1 = np.sqrt((6 * (n-2)) / ((n+1)*(n+3)))
    n_bins = 1 + np.log2(n) + np.log2(1 + (abs(skew(data)) / s_g1))
    return ceil(n_bins)





# # -*- coding: utf-8 -*-
# """
# Created on Thu Sep 27 13:05:14 2018
# collection of functions to parse and plot .xls files
# obtained from zondTEM software

# @author: laigner
# """

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import matplotlib.gridspec as gridspec
# import pandas as pd


# import os
# import sys
# import math
# import itertools
# import matplotlib

# from scipy.optimize import root
# from scipy.interpolate import griddata
# from scipy.interpolate import interp1d

# import matplotlib.offsetbox as offsetbox
# from matplotlib.ticker import ScalarFormatter
# from matplotlib.ticker import MaxNLocator
# from matplotlib import ticker
# from matplotlib.patches import Polygon
# from matplotlib.collections import PatchCollection
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# from pykrige.ok import OrdinaryKriging
# from pykrige.uk import UniversalKriging

# from math import ceil
# from scipy.stats import skew

# from TEM_tools import parse_TEMfastFile

# as_strided = np.lib.stride_tricks.as_strided


# def rho_average(doi, lr_rho, lr_thk):
#     rho_int = 0
#     for i in range(1,len(lr_rho)):
#         # print(doi)
#         # print(lr_thk[0:i])
#         if all(doi > lr_thk[0:i]):
#             rho_int += np.log10(lr_rho[i]) * (lr_thk[i] - lr_thk[i-1])
#         else:
#             rho_int += np.log10(lr_rho[i]) * (doi - lr_thk[i-1])
#             break
#     rho_av = 10**(rho_int / doi)
#     return rho_av


# def calc_doi(fname_ztres, path_ztres,
#              fname_raw, path_raw,
#              x0=100, verbose=False):
#     """

#     Parameters
#     ----------
#     fname_ztres : str
#         DESCRIPTION.
#     path_ztres : str
#         DESCRIPTION.
#     fname_raw : str
#         DESCRIPTION.
#     path_raw : str
#         DESCRIPTION.
#     x0 : TYPE, optional
#         initial value for the root search (ie zero search). The default is 100.

#     Returns
#     -------
#     DOIs : list
#         DESCRIPTION.

#     """
#     (dat, indices_Hdr, indices_Mresp,
#      indices_Mdl, zondVersion) = parse_zondxls(path_ztres,
#                                                fname_ztres)
#     sounding_names = indices_Hdr.sndID.to_list()
#     signal = dat.loc[indices_Mresp.start[1]:indices_Mresp.end[1],
#                          ['c1','c2','c3','c4','c5','c6']]
#     signal.columns = ['ctr','time','rho_O','rho_C','U_O','U_C']
#     signal.loc[:,'time':'U_C'] = signal.loc[:,'time':'U_C'].astype('float')

#     DOIs = []
#     OPTs = []

#     for logID, snd_name in enumerate(sounding_names):
#         dat_raw, nLogs, ind_hdr, ind_dat = parse_TEMfastFile(fname_raw,
#                                                          path_raw)
#         rawdata = dat_raw.loc[ind_dat.start[0]:ind_dat.end[0]-1]
#         rawdata = rawdata.apply(pd.to_numeric)
#         header_raw = dat_raw.loc[ind_hdr.start[0]:ind_hdr.end[0]-1]
#         Curr = header_raw.iloc[3,5]
#         current = float(Curr[3:6])
#         tx_size = float(header_raw.iloc[4][1])
#         tx_area = tx_size**2
#         # print(tx_size, tx_area)
#         eta = signal.iloc[-1,4]  #*1e-6  # last gate, from mcV to V
#         tN = signal.iloc[-1, 1]  # time of last time gate for filtered data
        
#         model = dat.loc[indices_Mdl.start[logID]:indices_Mdl.end[logID],
#                         ['c1','c2','c3','c4','c5','c6','c7','c8']]
#         model.columns = ['ctr','Rho','Pol','Tconst','Cexpo','MagP','h','z']
#         model.drop(['Pol','Tconst','Cexpo'], axis=1, inplace=True)
#         mdl_rz = np.asarray(model.loc[:,['Rho', 'z']], dtype='float')  # factor to test doi for higher rho * 10
#         # rA_model = reArr_zondMdl(model)
#         # print(rA_model)
        
#         doi_fun = lambda x: 0.55*(current*tx_area*rho_average(x, mdl_rz[:,0], mdl_rz[:,1]) / (eta))**(1/5) - x
#         OPTs.append(root(doi_fun, x0))
#         DOIs.append(OPTs[logID]['x'][0])
        
#         if verbose:
#             print(f'doi calc of: {snd_name}')
#             print('TX-area: ', tx_area)
#             print('volt-level: ', eta)
#             print('at timegate (us): ', tN)
#             print(OPTs[logID]['message'])
#             print('doi: ', DOIs[logID])

#     return DOIs, OPTs


# def parse_zondMdl(filename):
#     """function to parse .mdl file from zondTEM software.
#     not necessary if export is done by exporting to excel only.
#     """
#     mdlFile = np.genfromtxt(filename,skip_header=1)
#     FileLen=len(mdlFile)
#     MdlElev = np.zeros((FileLen*2,2)); r=1;k=0
#     for i in range(0,FileLen*2):
#         if i == FileLen:
#             MdlElev[-1,1] = MdlElev[-2,1]
#             break
#         if i == 0:
#             MdlElev[i,1] = mdlFile[i,3]
#             MdlElev[i:i+2,0] = mdlFile[i,0]
#         else:
#             MdlElev[r+i:r+i+2,0] = mdlFile[i,0]     #resistivity
#             MdlElev[k+i:k+i+2,1] = -mdlFile[i,3]    #elevation
#             k+=1; r+=1
#     return MdlElev, filename


# def reArr_zondMdl(Mdl2_reArr):
#     """
#     function to rearrange model value structure in order to
#     plot a step-model.
#     """
#     Mdl2_reArr = np.asarray(Mdl2_reArr, dtype='float')
#     FileLen=len(Mdl2_reArr)
#     MdlElev = np.zeros((FileLen*2,2)); r=1;k=0
#     for i in range(0,FileLen*2):
#         if i == FileLen:
#             MdlElev[-1,1] = MdlElev[-2,1]
#             break
#         if i == 0:
#             MdlElev[i,0] = Mdl2_reArr[i,4]
#             MdlElev[i:i+2,1] = Mdl2_reArr[i,1]
#         else:
#             MdlElev[k+i:k+i+2,0] = -Mdl2_reArr[i,4]    #elevation
#             MdlElev[r+i:r+i+2,1] = Mdl2_reArr[i,1]     #resistivity
#             k+=1; r+=1

#     MdlElev = np.delete(MdlElev,-1,0) # delete last row!!
#     return MdlElev


# def depth_from_thickness(thick_model, add_bottom=50):
#     """routine to reshape a thickness model to a model
#     containing the depth of each boundary point.

#     Keyword arguments:
#     thick_model -- thickness model - np.array
#     add_bottom -- how many m below the last layer
#     in:
#     col0 ... Resistivity
#     col1 ... Thickness
#     out:
#     col0 ... Resistivity
#     col1 ... Depth
#     """
#     nLayers = len(thick_model); mdl_size = nLayers
#     depth_model = np.zeros((mdl_size*2, 2))
#     r0 = 1; #resistivity counter
#     d1 = 0; #depth ctr
#     for i in range(0,mdl_size+1):
#         if i == mdl_size:
#             print('end-i:',i)
#             depth_model[-1,0] = thick_model[-1,0]
#             depth_model[-1,1] = depth_model[-2,1] + add_bottom
#             break
#         if i == 0:
#             print('start-i:',i)
#             depth_model[i,0] = thick_model[0,0]
#             depth_model[i,1] = 0
#             depth_model[i+1,0] = thick_model[0,0]
#             depth_model[i+1,1] = thick_model[0,1]
#         else:
#             print('else-i:',i)
#             depth_model[r0+i:r0+i+2,0] = thick_model[r0,0]
#             depth_model[d1+i:d1+i+2,1] = thick_model[d1,1]
#             if d1 > 0:
#                 print('d1-1', d1-1)
#                 depth_model[d1+i:d1+i+2,1] = np.cumsum(thick_model[:,1])[d1]
#             r0 += 1;
#             d1 += 1;
#     return depth_model


# def get_PatchCollection(elemx, elemz, colormap='jet_r', log10=False):
#     """Script to derive polygon patches collections out of grid information and
#     data values; Updated version to enable easy log10 scaling!!

#     Author: Jakob Gallistl
#     """
#     import matplotlib.colors as colors

#     # size of elemx
#     a, b = np.shape(elemx)
#     # patches list
#     patches = []

#     # loop to compute polygon patches
#     for elem in np.arange(b):
#         elx = np.expand_dims(elemx[:, elem], axis=1)
#         elz = np.expand_dims(elemz[:, elem], axis=1)

#         v = np.concatenate((elx, elz), axis=1)

#         poly = Polygon(v)

#         patches.append(poly)

#     norm = colors.Normalize() if (log10 == False) else colors.LogNorm()

#     p = PatchCollection(patches,
#                         edgecolors='None',
#                         linewidth=0,
#                         zorder=0,
#                         cmap=colormap, #default is reversed jetMap
#                         norm=norm)     #default is linear norm
#     return p


# def parse_zondxls(path, file,
#                   path_coord=None, file_coord=None):
#     """
#     function to parse ZondTEM1d .xls file and create indices for further
#     subselecting of dat file

#     Keyword arguments:
#     path, file -- directions and file name
#     zondVersion -- version of the zondTEM software; IPversion export has more model paramters (IP params) than noIP version
#     ToDo: automatize the selection of the zondVersion - load all columns, than decide which ones to use
#     """
    
#     # assume the zondVersion is IP and read all needed columns
#     try:
#         if '.xls' in file:
#             raw = pd.read_excel(path + os.sep + file,
#                                 usecols=range(1, 9),
#                                 names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'],
#                                 header=None)
#         elif '.csv' in file:
#             raw = pd.read_csv(path + os.sep + file,
#                               usecols=range(1, 9),
#                               names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'],
#                               header=None,
#                               engine='python')
#         else:
#             print('no valid file ending in filename... exiting!')
#             sys.exit(0)
#         zondVersion = 'IP'

#     # catch the ValueError exception (if noIP) and read only with 6 columns
#     except ValueError:
#         if '.xls' in file:
#             raw = pd.read_excel(path + os.sep + file,
#                                 usecols=range(1, 7),
#                                 names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'],
#                                 header=None)
#         elif '.csv' in file:
#             raw = pd.read_csv(path + os.sep + file,
#                               usecols=range(1, 7),
#                               names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'],
#                               header=None,
#                               engine='python')
#         else:
#             print('no valid file ending in filename... exiting!')
#             sys.exit(0)
#         zondVersion = 'noIP'

#     # print(raw)
#     if '.xls' in file:
#         # dat = raw.drop(labels='empty', axis=1)
#         dat = raw.drop(labels=0, axis=0)
#     elif '.csv' in file:
#         dat = raw.drop(labels=0, axis=0)
#     else:
#         print('no valid file ending in filename... exiting!')
#         sys.exit(0)

#     indices_labels = np.flatnonzero(dat.c1 == '#') + 1
#     ev2nd0 = indices_labels[::2]; ev2nd1 = indices_labels[1::2]
#     end = len(dat)
#     endMdl = np.append(ev2nd0[1:]-5, [end])

#     indices_Hdr = pd.DataFrame({'start':ev2nd0-4, 'end':ev2nd0},
#                                columns=['start', 'end'])
#     indices_Mresp = pd.DataFrame({'start':ev2nd0+1, 'end':ev2nd1-1},
#                                columns=['start', 'end'])
#     indices_Mdl = pd.DataFrame({'start':ev2nd1+1, 'end':endMdl},
#                                columns=['start', 'end'])

#     nSnds = len(indices_Hdr)
#     sndNames = []
#     for logID in range(0,nSnds):
#         hdr = dat.loc[indices_Hdr.start[logID]:indices_Hdr.end[logID]]
#         sndNames.append(hdr.iloc[0][1])

#     indices_Hdr.insert(0, "sndID", sndNames)
#     indices_Mresp.insert(0, "sndID", sndNames)
#     indices_Mdl.insert(0, "sndID", sndNames)

#     return dat, indices_Hdr, indices_Mresp, indices_Mdl, zondVersion


# def remove_soundings(indices_Hdr, indices_Mresp, indices_Mdl, snd2remove):
#     """
#     function to remove soundings from indices dataframes

#     input: the 3 dataframes which contain the indices for the Header,
#     the Modelresponse and the Model;
#     !! List of indices which should be removed !!

#     returns:
#     indices dataframes which are cleared from the soundings contained in the
#     snd2remove list
#     """
#     indHdr_clr = indices_Hdr[~indices_Hdr.sndID.isin(snd2remove)]
#     indMre_clr = indices_Mresp[~indices_Hdr.sndID.isin(snd2remove)]
#     indMdl_clr = indices_Mdl[~indices_Hdr.sndID.isin(snd2remove)]

#     return indHdr_clr, indMre_clr, indMdl_clr


# def select_soundings(indices_Hdr, indices_Mresp, indices_Mdl, snd2keep):
#     """
#     function to remove soundings from indices dataframes

#     input: the 3 dataframes which contain the indices for the Header,
#     the Modelresponse and the Model;
#     !! List of indices which should be selected !!

#     returns:
#     indices dataframes which contain only the selected ones.
#     """
#     indHdr_sld = indices_Hdr[indices_Hdr.sndID.isin(snd2keep)].copy()
#     indHdr_sld['sort_cat'] = pd.Categorical(indHdr_sld['sndID'],
#                                             categories=snd2keep,
#                                             ordered=True)

#     indHdr_sld.sort_values('sort_cat', inplace=True)
#     indHdr_sld.reset_index(inplace=True)
#     indHdr_sld.drop(indHdr_sld.columns[[0,-1]], axis=1, inplace=True)

#     indMre_sld = indices_Mresp[indices_Hdr.sndID.isin(snd2keep)].copy()
#     indMre_sld['sort_cat'] = pd.Categorical(indMre_sld['sndID'],
#                                             categories=snd2keep,
#                                             ordered=True)
#     indMre_sld.sort_values('sort_cat', inplace=True)
#     indMre_sld.reset_index(inplace=True)
#     indMre_sld.drop(indMre_sld.columns[[0,-1]], axis=1, inplace=True)

#     indMdl_sld = indices_Mdl[indices_Hdr.sndID.isin(snd2keep)].copy()
#     indMdl_sld['sort_cat'] = pd.Categorical(indMdl_sld['sndID'],
#                                             categories=snd2keep,
#                                             ordered=True)
#     indMdl_sld.sort_values('sort_cat', inplace=True)
#     indMdl_sld.reset_index(inplace=True)
#     indMdl_sld.drop(indMdl_sld.columns[[0,-1]], axis=1, inplace=True)

#     return indHdr_sld, indMre_sld, indMdl_sld


# def create_TEMelem(dat, indices_Hdr, indices_Mresp, indices_Mdl,
#                    path_coord=None, file_coord=None,
#                    log_width=2, xoffset=0, extend_bot=5,
#                    zondVersion='IP'):
#     """ creates 4 corner elements for plotting.
#      Needs to be combined with Jakobs get_PatchCollection

#     To Do:
#     improve the position of data points for interpolation,
#     tilt logs according to slope of the sections;
#     log should be normal to the TEM-loop plaen
#     """
#     nLogs = len(indices_Hdr)
#     nLayers = np.asarray(indices_Mdl.end - indices_Mdl.start + 1)
#     nLay_sum = np.sum(nLayers)
#     nLay_ind = np.cumsum(nLayers)

#     telx = np.zeros((4,nLay_sum))
#     telz = np.zeros((4,nLay_sum))
#     telRho = np.zeros((1,nLay_sum))

#     frame_lowLeft = np.zeros((nLogs,3)) #create array for lower left Point of each log. add depth of log
#     topo = np.zeros((nLogs,2))
#     log_id = 0
#     snd_names = []
    
#     separated_logs = []
    
#     for log_id in range(0,nLogs):
#         if zondVersion == 'IP':
#             log2plot_df = dat.loc[indices_Mdl.start[log_id]:indices_Mdl.end[log_id],
#                                   ['c1','c2','c3','c4','c5','c6','c7','c8']]
#             log2plot_df.columns = ['ctr','Rho','Pol','Tconst','Cexpo','MagP','h','z']
#             log2plot_df.drop(['Pol','Tconst','Cexpo'], axis=1, inplace=True)
#             # print(log2plot_df)
#             log2plot = reArr_zondMdl(log2plot_df)
#             # print(log2plot)

#         elif zondVersion == 'noIP':
#             log2plot_df = dat.loc[indices_Mdl.start[log_id]:indices_Mdl.end[log_id]]
#             log2plot = reArr_zondMdl(log2plot_df)

#         snd_names.append(indices_Hdr.sndID[log_id])
#         nLayer = nLayers[log_id]
        
#         # print(dat.loc[indices_Hdr.start[log_id]+1,'c2'])
#         # print(xoffset)
#         dis_xpos = dat.loc[indices_Hdr.start[log_id]+1,'c2'] + xoffset
#         height = dat.loc[indices_Hdr.start[log_id]+2,'c4'] #get height from header
#         topo[log_id, 1] = height
#         topo[log_id, 0] = dis_xpos

#         xLog = [[dis_xpos - log_width/2],
#                 [dis_xpos + log_width/2],
#                 [dis_xpos + log_width/2],
#                 [dis_xpos - log_width/2]]

#         z = log2plot[:,0]
#         z = np.insert(z,0,0,axis=0) + height
#         z_el = np.copy(as_strided(z, (4,nLayers[log_id]), (8,16)))
#         z_el[-2:,-1] = z_el[0:2,-1] - extend_bot
        
#         Rho = np.copy(log2plot[::2,1])
#         # print(log2plot[::2,0])
#         # print(nLayer)

#         frame_lowLeft[log_id,0] = np.min(xLog)
#         frame_lowLeft[log_id,1] = np.min(z)
#         frame_lowLeft[log_id,2] = abs(np.min(z) - np.max(z))        #depth of log

#         if log_id == 0:
#             telx[:,0:nLayer] = np.repeat(xLog, nLayer, axis=1)
#             telz[:,0:nLayer] = z_el
#             telRho[:,0:nLayer] = Rho
#         else:
#             telx[:,nLay_ind[log_id-1]:nLay_ind[log_id]] = np.repeat(xLog, nLayer, axis=1)
#             telz[:,nLay_ind[log_id-1]:nLay_ind[log_id]] = z_el
#             telRho[:,nLay_ind[log_id-1]:nLay_ind[log_id]] = Rho

#         separated_logs.append([indices_Hdr.sndID[log_id], log2plot, z_el])

#     telRho = telRho.reshape((np.size(telRho),))

#     return telx, telz, telRho, frame_lowLeft, nLogs, topo, snd_names, extend_bot, separated_logs


# def rotate_xy(x, y, angle, origin, show_plot=False):
#     ox = origin[0]
#     oy = origin[1]
#     x_rot = (ox + math.cos(angle) * (x - ox) - math.sin(angle) * (y - oy))
#     y_rot = (oy + math.sin(angle) * (x - ox) + math.cos(angle) * (y - oy))
    
#     if show_plot:
#         fig, ax = plt.subplots(1,1)
#         ax.plot(x, y, 'ok', label='not tilted')
#         ax.plot(x_rot, y_rot, 'or', label='tilted')
#         ax.plot(ox, oy, 'xg', label='origin ')
#         ax.set_title(f'rotational angle: {angle} rad')
#         ax.legend()
    
#     # telx_curr = ox + math.cos(angle) * (telx_curr - ox) - math.sin(angle) * (telz_curr - oy)
#     # telz_curr = oy + math.sin(angle) * (telx_curr - ox) + math.cos(angle) * (telz_curr - oy)
#     return x_rot, y_rot


# def create_suppPts(dat, indices_Hdr, indices_Mresp, indices_Mdl,
#                    path_coord=None, file_coord=None, zondVersion='IP',
#                    verbose=False, log_width=2, xoffset=0,
#                    tilt=False, tilt_angles=None, origins=None,
#                    average_thin=True, up_sample='thick', sample_factor=10):
#     """
#     Parameters
#     ----------
#     dat : pd.DataFrame
#         contains all inversion results.
#     indices_Hdr : pd.DataFrame
#         indices corresponding to the dat variable,
#         for selecting the header of each sounding.
#     indices_Mresp : pd.DataFrame
#         indices corresponding to the dat variable,
#         for selecting the model response of each sounding.
#     indices_Mdl : pd.DataFrame
#         indices corresponding to the dat variable,
#         for selecting the model response of each sounding..
#     path_coord : TYPE, optional
#         DESCRIPTION. The default is None.
#     file_coord : TYPE, optional
#         DESCRIPTION. The default is None.
#     log_width : TYPE, optional
#         DESCRIPTION. The default is 2.
#     xoffset : TYPE, optional
#         DESCRIPTION. The default is 0.
#     zondVersion : TYPE, optional
#         DESCRIPTION. The default is 'IP'.

#     Returns
#     -------
#     None.

#     """
#     nLogs = len(indices_Hdr)
#     bot_dep_res = np.zeros((nLogs,2))

#     for log_id in range(0,nLogs):
#         nLayers = np.asarray(indices_Mdl.end - indices_Mdl.start + 1)
#         log2plot_df = dat.loc[indices_Mdl.start[log_id]:indices_Mdl.end[log_id],
#                               ['c1','c2','c3','c4','c5','c6','c7','c8']]
#         log2plot_df.columns = ['ctr','Rho','Pol','Tconst','Cexpo','MagP','h','z']
#         log2plot_df.drop(['Pol','Tconst','Cexpo'], axis=1, inplace=True)
#         log2plot = reArr_zondMdl(log2plot_df)

#         row = np.arange(1, len(log2plot), 2)
#         bot_dep_res[log_id,0] = log2plot[-1,0]
#         bot_dep_res[log_id,1] = log2plot[-1,1]

#     bot_comMin = min(bot_dep_res[:,0]) - 15

#     supPts_allLogs = np.empty((0,3), float)

#     for log_id in range(0,nLogs):
#         nLayers = np.asarray(indices_Mdl.end - indices_Mdl.start + 1)
#         dis_xpos = dat.loc[indices_Hdr.start[log_id]+1,'c2'] + xoffset
#         height = dat.loc[indices_Hdr.start[log_id]+2,'c4'] #get height from header
#         # print(height)

#         log2plot_df = dat.loc[indices_Mdl.start[log_id]:indices_Mdl.end[log_id],
#                               ['c1','c2','c3','c4','c5','c6','c7','c8']]
#         log2plot_df.columns = ['ctr','Rho','Pol','Tconst','Cexpo','MagP','h','z']
#         log2plot_df.drop(['Pol','Tconst','Cexpo'], axis=1, inplace=True)
#         log2plot = reArr_zondMdl(log2plot_df)
#         supPts_raw = np.copy(log2plot)
        
#         thckns = abs(supPts_raw[:-1:2,0] - supPts_raw[1::2,0])
#         mean_thck = np.mean(thckns)

#         for r in row:
#             thickness = abs(supPts_raw[r,0] - supPts_raw[r-1,0])
#             # print(thickness)
#             shift = thickness / 50
#             supPts_raw[r,0] = supPts_raw[r,0] + shift

#         supPts = np.vstack((supPts_raw, [bot_comMin, bot_dep_res[log_id,1]]))
#         x_vals = np.full((len(supPts), ), dis_xpos)
#         supPts_x = np.insert(supPts, 0, x_vals, axis=1)
#         supPts_x[:,1] = supPts_x[:,1] + height

#         if up_sample == 'all':  # upsampling all
#             int_func = interp1d(supPts[:,0], supPts[:,1], kind='linear')
#             int_depths = np.linspace(0, supPts[-1,0], nLayers[0]*sample_factor)
#             int_depths = np.resize(int_depths, (len(int_depths), 1))
#             supPts_int = np.hstack((int_depths, int_func(int_depths)))
#             x_vals = np.full((len(supPts_int), ), dis_xpos)
#             supPts_int_x = np.insert(supPts_int, 0, x_vals, axis=1)
#             supPts_int_x[:,1] = supPts_int_x[:,1] + height

#             if tilt and (tilt_angles is not None) and (origins is not None):
#                 supPts_int_x[:,0], supPts_int_x[:,1] = rotate_xy(x=supPts_int_x[:,0], y=supPts_int_x[:,1],
#                                                                  angle=tilt_angles[log_id], origin=origins[log_id],
#                                                                  show_plot=False)

#             supPts_allLogs = np.append(supPts_allLogs, supPts_int_x, axis=0)

#         elif  up_sample == 'thick':  # upsampling only thicker (than average) layers
#             if verbose:
#                 print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#                 print('entering selective upsampling...')
#             for k in range(1,len(supPts)):
#                 thickness = abs(supPts[k,0] - supPts[k-1,0])
#                 if verbose:
#                     print(k, 'of supp points')
#                     print('mean thickness: ', mean_thck)
#                     print('thickness', thickness)
#                 if thickness > mean_thck*1.5:  
#                     if verbose:
#                         print('thickness is larger than mean thk!!')
#                         print('interpolating for: ')
#                         print(supPts[k-1:k+1,0],
#                                supPts[k-1:k+1,1])

#                     int_func = interp1d(supPts[k-1:k+1,0],
#                                         supPts[k-1:k+1,1], kind='linear')
#                     int_depths = np.linspace(supPts[k-1,0], supPts[k,0], sample_factor)
#                     int_depths = np.resize(int_depths, (len(int_depths), 1))
#                     supPts_int = np.hstack((int_depths, int_func(int_depths)))
#                     x_vals = np.full((len(supPts_int), ), dis_xpos)
#                     supPts_int_x = np.insert(supPts_int, 0, x_vals, axis=1)
#                     supPts_int_x[:,1] = supPts_int_x[:,1] + height
                    
#                     if tilt and (tilt_angles is not None) and (origins is not None):
#                         supPts_int_x[:,0], supPts_int_x[:,1] = rotate_xy(x=supPts_int_x[:,0], y=supPts_int_x[:,1],
#                                                                          angle=tilt_angles[log_id], origin=origins[log_id])
                    
#                     supPts_allLogs = np.append(supPts_allLogs, supPts_int_x, axis=0)

#                 else:
#                     if verbose:
#                         print('no upsampling ... ')
#                         print(supPts[k-1:k+1,:])
#                         print(supPts[k-1:k+1,:].shape)
#                         print('calc mean: ', np.mean(supPts[k-1:k+1,:], axis=0))
#                         print('shape mean: ', np.mean(supPts[k-1:k+1,:], axis=0).shape)
#                     if average_thin == True:
#                         if verbose:
#                             print('averaging thin layers to reduce number of interpolation points:')
#                         curr_supPts = np.mean(supPts[k-1:k+1,:], axis=0)
#                         # x_vals = np.full((len(curr_supPts), ), dis_xpos)
#                         supPts_x = np.r_[dis_xpos, curr_supPts].reshape((1,3))
#                         # supPts_x = np.insert(curr_supPts, 0, x_vals, axis=1)

#                         if k == 1:
#                             curr_supPts = supPts[k-1,:]
#                             curr_supPts[0] = curr_supPts[0] * 0.9
#                             supPts_x_add = np.r_[dis_xpos, curr_supPts].reshape((1,3))
#                             supPts_x = np.r_[supPts_x, supPts_x_add]
                            
#                             supPts_x[:,1] = supPts_x[:,1] + height
#                         else:
#                             supPts_x[:,1] = supPts_x[:,1] + height

#                     else:
#                         curr_supPts = supPts[k-1:k+1,:]
#                         x_vals = np.full((len(curr_supPts), ), dis_xpos)
#                         supPts_x = np.insert(curr_supPts, 0, x_vals, axis=1)
#                         # print(supPts_x)
#                         supPts_x[:,1] = supPts_x[:,1] + height
                    
#                     if tilt and (tilt_angles is not None) and (origins is not None):
#                         supPts_int_x[:,0], supPts_int_x[:,1] = rotate_xy(x=supPts_int_x[:,0], y=supPts_int_x[:,1],
#                                                                          angle=tilt_angles[log_id], origin=origins[log_id])
                    
#                     supPts_allLogs = np.append(supPts_allLogs, supPts_x, axis=0)

#         elif up_sample is None:  # no upsampling done
#             if verbose:
#                 print('no upsampling done ...')
#             supPts_x = np.insert(supPts, 0, x_vals, axis=1)
#             supPts_x[:,1] = supPts_x[:,1] + height
            
#             if tilt and (tilt_angles is not None) and (origins is not None):
#                 supPts_int_x[:,0], supPts_int_x[:,1] = rotate_xy(x=supPts_int_x[:,0], y=supPts_int_x[:,1],
#                                                                  angle=tilt_angles[log_id], origin=origins[log_id],
#                                                                  show_plot=False)
            
#             supPts_allLogs = np.append(supPts_allLogs, supPts_x, axis=0)
        
#         else:
#             raise ValueError("PLease select either 'all', 'thick' or None for the up_sample kwarg")

#     return supPts_allLogs


# def interp_TEMlogs(supPts_allLogs,
#                    mesh_resolution, method='linear'):
#     """

#     Parameters
#     ----------
#     supPts_allLogs : TYPE
#         DESCRIPTION.
#     mesh_resolution : TYPE
#         DESCRIPTION.
#     method : TYPE, optional
#         DESCRIPTION. The default is 'linear'.

#     Returns
#     -------
#     None.

#     """
#     x = supPts_allLogs[:,0]
#     z_dep = supPts_allLogs[:,1]
#     rho = supPts_allLogs[:,2]

#     # target grid to interpolate to
#     factor = 3
#     xi = np.arange(np.min(x)-factor*mesh_resolution,
#                    np.max(x)+factor*mesh_resolution,
#                    mesh_resolution)
#     zi = np.arange(np.min(z_dep)-factor*mesh_resolution,
#                    np.max(z_dep)+factor*mesh_resolution,
#                    mesh_resolution)
#     xi_mesh, zi_mesh = np.meshgrid(xi,zi[::-1]-0.8)
#     print('++++++++++++Interpolation++++++++++++++++++++++++++++++++')
#     print('max_zmesh: ', np.max(zi_mesh))

#     # interpolate
#     rhoi = griddata((x,z_dep), rho, (xi_mesh,zi_mesh), method,
#                     rescale=True)

#     return x, z_dep, xi_mesh, zi_mesh, rhoi


# # TODO fix kriging option
# # update and downsample interpolation points first...

# def kriging_TEMlogs(supPts_allLogs, mesh_resolution,
#                     method='universal_kriging',
#                     variogram_mdl='spherical',
#                     anisotropy_scaling=1,
#                     use_weight=True):
#     x = supPts_allLogs[:,0]
#     z_dep = supPts_allLogs[:,1]
#     rho = supPts_allLogs[:,2]
    
#     # target meshgrid to interpolate to
#     factor = 2
#     xi = np.arange(np.min(x)-factor*mesh_resolution,
#                     np.max(x)+factor*mesh_resolution,
#                     mesh_resolution)
#     zi = np.arange(np.min(z_dep)-factor*mesh_resolution,
#                     np.max(z_dep)+factor*mesh_resolution,
#                     mesh_resolution)
#     # xi_mesh, zi_mesh = np.meshgrid(xi,zi[::-1])
    
#     if method == 'ordinary_kriging':
#         OK_rho = OrdinaryKriging(x, z_dep, rho,
#                                   variogram_model=variogram_mdl,
#                                   weight=use_weight,
#                                   exact_values=True,  # False, use exact values at interpolation position
#                                   anisotropy_scaling=anisotropy_scaling,
#                                   verbose=True,
#                                   enable_plotting=False)
#         rhoi, rho_variance = OK_rho.execute('grid', xi, zi,
#                                             backend='C',  #vectorized, loop
#                                             ) #n_closest_points=3

#     elif method == 'universal_kriging':
#         UK_rho = UniversalKriging(x, z_dep, rho,
#                                   variogram_model=variogram_mdl,
#                                   drift_terms=['regional_linear'],
#                                   weight=use_weight,
#                                   exact_values=True,  # False, use exact values at interpolation position
#                                   anisotropy_scaling=anisotropy_scaling,
#                                   verbose=True,
#                                   enable_plotting=False)
#         rhoi, rho_variance = UK_rho.execute('grid', xi, zi,
#                                             backend='vectorized',
#                                             )  # n_closest_points=3
    
#     return xi, zi, rhoi, rho_variance


# def interpolate_TEMelem(telx, telz, telRho,
#                         mesh_resolution, method='linear',
#                         useMiddle=True, moveFrac=20):
#     """This routine interpolates TEM elements obtained from createTEmelem routine.

#     further usage in plot section to show interpolation between multiple logs
#     DoNOT use the middle of each element.
#     """
#     # prepare data for interpolation...
#     if useMiddle == True:
#         x = telx.mean(axis=0)
#         z_dep = telz.mean(axis=0)
#         rho = np.copy(telRho)
#     else:
#         x_upper = telx.mean(axis=0)
#         x_lower = telx.mean(axis=0)
#         x = np.hstack((x_upper, x_lower))

#         z_upper = telz[0,:]
#         z_lower = telz[2,:]
#         el_height = z_upper-z_lower
#         z_elMove = el_height/moveFrac
#         z_dep = np.hstack((z_upper-z_elMove,
#                            z_lower+z_elMove))

#         rho = np.hstack((telRho, telRho))

#     # target grid to interpolate to
#     print('start at xdistance:', np.min(telx))
#     xi = np.arange(np.min(telx),
#                    np.max(telx)+mesh_resolution,
#                    mesh_resolution)
#     zi = np.arange(np.min(telz)-mesh_resolution,
#                    np.max(telz)+2*mesh_resolution,
#                    mesh_resolution)

#     xi_mesh, zi_mesh = np.meshgrid(xi,zi[::-1])

#     # interpolate
#     rhoi = griddata((x,z_dep), rho, (xi_mesh,zi_mesh), method,
#                     rescale=True, fill_value=25)

#     return x, z_dep, xi_mesh, zi_mesh, rhoi


# def plot_section(stdPath, path, file,
#                  path_coord=None, file_coord=None,
#                  log10=False, saveFig=True,
#                  intMethod='linear', version='v1', filetype='.jpg', dpi=300,
#                  xoffset=0, label_s=8, log_width=1.5, depth=40,
#                  rmin=99, rmax=99):
#     """ this routine plots a section of TEM-models obtained from ZondTEM software
#     """
#     fullpath = stdPath + os.sep + path

#     (telx,
#      telz,
#      telRho,
#      frame_lowLeft,
#      nLogs, topo, snd_names) = create_TEMelem(fullpath, file,
#                                               path_coord=path_coord, file_coord=file_coord,
#                                               log_width=log_width, xoffset=xoffset, zondVersion='IP')

#     fig, ax1 = plt.subplots(1,1, figsize=(18, 5)) #figSize in inch
#     #Plot1 - RHO
#     if intMethod != 'No':
#         lvls = 200
#         if rmin == rmax:
#             xi_mesh, zi_mesh, rhoi = interpolate_TEMelem(telx,
#                                                          telz,
#                                                          telRho,
#                                                          mesh_resolution=0.1,
#                                                          method=intMethod)
#             if log10 == True:
#                 ctrf = plt.contourf(xi_mesh, zi_mesh, np.log10(rhoi),
#                                     levels=lvls, cmap='jet')
#             else:
#                 ctrf = plt.contourf(xi_mesh, zi_mesh, rhoi,
#                                     levels=lvls, cmap='jet')

#             upLeft = np.copy(topo[0,:] + np.array([[-5, 50]]))
#             topoMask = np.insert(topo, 0, upLeft, axis=0)
#             upRight = np.copy(topo[-1,:] + np.array([[5, 50]]))
#             topoMask = np.insert(topoMask, len(topo)+1, upRight, axis=0)
#             maskTopo  = Polygon(topoMask, facecolor='white', closed=True)
#             ax1.add_patch(maskTopo)
#         if rmin != rmax:
#             xi_mesh, zi_mesh, rhoi = interpolate_TEMelem(telx,
#                                                          telz,
#                                                          telRho,
#                                                          mesh_resolution=0.1,
#                                                          method=intMethod)
#             if log10 == True:
#                 ctrf = plt.contourf(xi_mesh, zi_mesh, np.log10(rhoi),
#                                     levels=lvls, cmap='jet')
#             else:
#                 ctrf = plt.contourf(xi_mesh, zi_mesh, rhoi,
#                                     levels=lvls, cmap='jet')

#             upLeft = np.copy(topo[0,:] + np.array([[-5, 50]]))
#             topoMask = np.insert(topo, 0, upLeft, axis=0)
#             upRight = np.copy(topo[-1,:] + np.array([[5, 50]]))
#             topoMask = np.insert(topoMask, len(topo)+1, upRight, axis=0)
#             maskTopo  = Polygon(topoMask, facecolor='white', closed=True)
#             ax1.add_patch(maskTopo)

#     pr = get_PatchCollection(telx, telz)
#     if log10 == True:
#         pr.set_array(np.log10(telRho))
#         labelRho = r"$log_{10}\rho$ [$\Omega$m]"
#         if not rmin == rmax:
#             pr.set_clim([np.log10(rmin), np.log10(rmax)])
#             ctrf.set_clim([np.log10(rmin), np.log10(rmax)])
#     else:
#         pr.set_array(telRho)
#         labelRho = r"$\rho$ [$\Omega$m]"
#         if not rmin == rmax:
#             pr.set_clim([rmin, rmax])
#             ctrf.set_clim([rmin, rmax])
#     pr.set_cmap(cmap=matplotlib.cm.jet)

#     ax1.add_collection(pr)

#     #make frames around logs; add labels to soundings
#     for i in range(0,nLogs):
#         frame = patches.Rectangle((frame_lowLeft[i,0],frame_lowLeft[i,1]-10),
#                                   log_width,
#                                   frame_lowLeft[i,2]+10,
#                                   linewidth=1,
#                                   edgecolor='k',
#                                   facecolor='none')
#         ax1.add_patch(frame)

#         x_txt = frame_lowLeft[i,0]
#         z_txt = frame_lowLeft[i,1]+frame_lowLeft[i,2] + 3
#         ax1.text(x_txt+5, z_txt+4, snd_names[i],
#                  ha='center', va='center',
#                  rotation=45, size=12,
#                  bbox=dict(facecolor='white', alpha=1))

#     ax1.plot(topo[:,0], topo[:,1]-0.5, '-k', linewidth=2.5)
#     if file_coord != None:                         # plot add Info (bathymetrie)
#         coord_raw = pd.read_csv(path_coord + file_coord, sep=',', engine='python',
#                                 header=0)
#         # print(coord_raw)
#         ax1.plot(coord_raw.Distance, coord_raw.Z-coord_raw.Depth, 'o:k', linewidth=1.5)

#     ax1.grid(which='major', axis='y', color='grey', linestyle='--', linewidth=1)
#     ax1.locator_params(axis='x', nbins=15)
#     ax1.locator_params(axis='y', nbins=7)
#     ax1.tick_params(axis='both', which='both', labelsize=label_s)
#     ax1.set_ylabel('Z [m]', fontsize=label_s, rotation=0)
#     ax1.set_xlabel('distance [m]', fontsize=label_s)
#     ax1.yaxis.set_label_coords(-0.03, 1.05)

# #    tickInc = 5
# #    ticks = np.arange(rmin, rmax+tickInc, tickInc)
# #    print('these are the planned ticks: ', ticks)
# #    cb = fig.colorbar(pr,
# #                      ax=ax1,
# #                      orientation='vertical',
# #                      aspect=10,
# #                      pad=0.01,
# #                      ticks=ticks,
# #                      format="%.1f")
# #    cb.set_label(labelRho,
# #                 fontsize=label_s)
# #    cb.ax.tick_params(labelsize=label_s)
# #
# #    cb.set_ticklabels(ticklabels=ticks)
# ##    cb.set_clim([rmin, rmax])

#     divider = make_axes_locatable(ax1)
#     cax1 = divider.append_axes("right", size="3%", pad=0.2)
#     cb = fig.colorbar(pr, cax=cax1, format='%d')
#     cb.ax.minorticks_off()
#     tick_locator = ticker.MaxNLocator(nbins=6)
#     cb.locator = tick_locator; cb.update_ticks()
#     cb.set_label(labelRho,
#                  fontsize=label_s)
#     cb.ax.tick_params(labelsize=label_s)

#     ax1.set_ylim((np.max(telz)-depth, np.max(telz)+5))
#     ax1.set_xlim((np.min(telx)-2, np.max(telx)+2))
# #    ax1.set_aspect(2, adjustable='box')
#     ax1.set_aspect('equal')

#     if saveFig == True:
#         savepath = fullpath + os.sep + file[:-4] + intMethod + '_int_' + version + filetype
#         print('saving figure to:',
#               savepath)
#         fig.savefig(savepath,
#                     dpi=dpi,
#                     bbox_inches='tight')
#     else:
#         plt.show()
#     return

# def get_response(dat, idcs_response, snd_id):
#     response =  dat.loc[idcs_response.start[snd_id]:idcs_response.end[snd_id],
#                         ['c1','c2','c3','c4','c5','c6']]
#     response.columns = ['ctr','time','rho_O','rho_C','U_O','U_C']
#     return response.apply(pd.to_numeric)


# def get_header(dat, idcs_header, snd_id):
#     header = dat.loc[idcs_header.start[snd_id]:idcs_header.end[snd_id],
#                      ['c1','c2','c3','c4']]
#     return header


# def compare_model_response(path, file, path_savefig, plottitle=None,
#                            fig=None, axes=None, snd_info=None,
#                            save_fig=False, filetype='.png', dpi=300,
#                            show_rawdata=True, show_all=True,
#                            xoffset=0, snd_indxs=np.r_[0],
#                            log_rhoa=False, log_res=False,
#                            sign_lim=(1e-8, 1e3), rhoa_lim=(0, 500),
#                            time_lim=(2, 15500), res_lim=(1e0, 1e4), dep_lim=(0, 40),
#                            linewidth=2, markerSize=3):
#     """
#     routine to plot single or multiple model responses in order to check
#     how well the data is represented by the model.
#     """
#     # plt.close('all')
#     # fullpath = stdPath + os.sep + path
#     fullpath = path

#     savePath = path_savefig + os.sep + 'dataFit'
#     if not os.path.exists(savePath):
#         os.makedirs(savePath)

#     dat, indices_Hdr, indices_Mresp, indices_Mdl, zondVersion = parse_zondxls(fullpath, file)
#     nLogs = len(indices_Hdr)

#     if show_rawdata == True:
#         inv_type = file.split('_')[-1][0:2]
#         inv_setup = file.split('_')[-2]
#         inv_info = f"{inv_setup}_{inv_type}"
#         file_raw = file.replace(inv_info, 'raw')
#         (dat_raw, indices_Hdr_raw, indices_Mresp_raw,
#          indices_Mdl_raw, zondVersion) = parse_zondxls(fullpath, file_raw)

#     if show_all == True:
#         logIndx = range(0,nLogs)

#     for i in snd_indxs:
#         # preparation for plotting part
#         header = dat.loc[indices_Hdr.start[i]:indices_Hdr.end[i], ['c1','c2','c3','c4']]
#         logID = header.iloc[0,1]
#         #coordinates = header.iloc[2,1:4]
#         distance = header.iloc[1,1]
#         rRMS_sig = header.iloc[3,1]
#         rRMS_roa = header.iloc[3,3]

#         signal = get_response(dat, idcs_response=indices_Mresp, snd_id=i)

#         if show_rawdata == True:
#             signal_raw = get_response(dat_raw,
#                                       idcs_response=indices_Mresp_raw,
#                                       snd_id=i)

#         if zondVersion == 'IP':
#             model = dat.loc[indices_Mdl.start[i]:indices_Mdl.end[i],
#                                   ['c1','c2','c3','c4','c5','c6','c7','c8']]
#             model.columns = ['ctr','Rho','Pol','Tconst','Cexpo','MagP','h','z']
#             model.drop(['Pol','Tconst','Cexpo'], axis=1, inplace=True)

#         elif zondVersion == 'noIP':
#             model = dat.loc[indices_Mdl.start[i]:indices_Mdl.end[i], ['c1','c2','c3','c4','c5']]
#             model.columns = ['ctr','Rho','MagP','h','z']

#         # Preparing figure and axis objects
#         if fig == None:
#             fig = plt.figure(figsize=(8, 6)) #figSize in inch; Invoke fig!
#         else:
#             print('using provided figure object ...')
#         if axes == None:
#             gs = gridspec.GridSpec(2, 4) #define grid for subplots
#             ax_sig = fig.add_subplot(gs[0, 0:3])
#             ax_roa = fig.add_subplot(gs[1, 0:3])
#             ax_mdl = fig.add_subplot(gs[0:2, 3])
#         else:
#             print('using 3 (?) provided axes objects ...')
#             # TODO add check if really three of them where provided ...

#         # ~~~~~~~~~~~~~~~~~~~~~~~ dBz/dt plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         if show_rawdata == True:
#             ax_sig.loglog(signal_raw.time*10**6, signal_raw.U_O*10**-6, 'D',
#                         color='darkgray',
#                         lw=linewidth, ms=markerSize)

#         ax_sig.loglog(signal.time*10**6, signal.U_C*10**-6, '-k',   # from uV to V
#                       signal.time*10**6, signal.U_O*10**-6,  'Dg',
#                       lw=linewidth, ms=markerSize)

#         if show_rawdata == False:
#             ax_sig.xaxis.set_minor_formatter(ScalarFormatter())
#             ax_sig.xaxis.set_major_formatter(ScalarFormatter())
#             print('overriding limits')
#             time_lim = None
#             sign_lim = None

#         if time_lim is not None:
#             ax_sig.set_xlim(time_lim)

#         if sign_lim is not None:
#             ax_sig.set_ylim(sign_lim)

#         # ax_sig.set_xlabel('time ($\mu$s)')
#         ax_sig.set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V)")
#         ax_sig.tick_params(axis='x',          # changes apply to the x-axis
#                            which='both',      # both major and minor ticks are affected
#                            bottom=False,      # ticks along the bottom edge are off
#                            top=False,         # ticks along the top edge are off
#                            labelbottom=False) # labels along the bottom edge are off

#         ax_sig.grid(which='major', color='white', linestyle='-')
#         ax_sig.grid(which='minor', color='white', linestyle=':')

#         obRMS = offsetbox.AnchoredText(f'rRMS: {rRMS_sig}%', loc='upper left')
#         ax_sig.add_artist(obRMS)

#         if snd_info is not None:
#             text_props = {'fontsize': 12, 'fontfamily':'monospace'}
#             ob = offsetbox.AnchoredText(snd_info, loc=1, prop=text_props)
#             ax_sig.add_artist(ob)


#         # ~~~~~~~~~~~~~~~~~~~~~~~ rhoa plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         if show_rawdata == True:
#             ax_roa.semilogx(signal_raw.time*10**6, signal_raw.rho_O, 'D',
#                             color='darkgray', label='raw data',
#                             lw=linewidth, ms=markerSize)

#         ax_roa.semilogx(signal.time*10**6, signal.rho_C, '-k',
#                         lw=linewidth, ms=markerSize,
#                         label='model_response')
        
#         ax_roa.semilogx(signal.time*10**6, signal.rho_O, 'Dg',
#                         lw=linewidth, ms=markerSize,
#                         label='selected data')

#         if show_rawdata == False:
#             ax_roa.xaxis.set_minor_formatter(ScalarFormatter())
#             ax_roa.xaxis.set_major_formatter(ScalarFormatter())
#             print('overriding limits')
#             time_lim = None
#             sign_lim = None
#         if time_lim is not None:
#             ax_roa.set_xlim(time_lim)
#         if rhoa_lim is not None:
#             ax_roa.set_ylim(rhoa_lim)
#         if log_rhoa == True:
#             ax_roa.set_yscale('log')

#         ax_roa.grid(which='major', color='white', linestyle='-')
#         ax_roa.grid(which='minor', color='white', linestyle=':')

#         ax_roa.set_xlabel('time ($\mu$s)')
#         ax_roa.set_ylabel(r'$\rho_a$ ' + r'($\Omega$m)')

#         obRMS = offsetbox.AnchoredText(f'rRMS: {rRMS_roa}%', loc='upper left')
#         ax_roa.add_artist(obRMS)

#         ax_roa.legend(loc='best')


#         # ~~~~~~~~~~~~~~~~~~~~~~~ inv model plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         rA_model = reArr_zondMdl(model)

#         ax_mdl.plot(rA_model[:,1], rA_model[:,0], '-ko',
#                  linewidth=linewidth, markerSize=markerSize,
#                  label='model')

#         if res_lim is not None:
#             ax_mdl.set_xlim(res_lim)
#         if dep_lim is not None:
#             ax_mdl.set_ylim(dep_lim)
#         if log_res == True:
#             ax_mdl.set_xscale('log')

#         ax_mdl.set_xlabel(r'$\rho$ ($\Omega$m)')
#         ax_mdl.set_ylabel('height (m)')
#         ax_mdl.legend()
        
#         ax_mdl.yaxis.set_label_position("right")
#         ax_mdl.yaxis.tick_right()

#         logName = logID #+ ' at ' + str(distance+xoffset) + 'm'
#         if plottitle != None:
#             logName = plottitle

#         fig.suptitle(logName, fontsize=14)

#         if save_fig == True:
#             fig.tight_layout()
#             plt.subplots_adjust(top=0.95)
#             if show_rawdata == False:
#                 full_savePath = savePath + os.sep + 'zoom_' + file.replace('.xls', '') + '_' + logID + filetype
#             else:
#                 full_savePath = savePath + os.sep + file.replace('.xls', '') + '_' + logID + filetype
#             fig.savefig(full_savePath,
#                         dpi=dpi,
#                         bbox_inches='tight')
#             print('saving figure to:\n', full_savePath)
#             # plt.close('all')
#         else:
#             print("Show figure...")
#             fig.tight_layout()
#             plt.subplots_adjust(top=0.95)
#             plt.show()

#     return fig, ax_sig, ax_roa, ax_mdl


# def plot_doi(axis, DOIs, distances, **kwargs):
#     axis.plot(distances, np.asarray(DOIs),
#               '--.', **kwargs)
#     return axis


# def plot_intfc(axis, depths, distances, **kwargs):
#     axis.plot(distances, depths,
#               '--.', **kwargs)
#     return axis


# def put_blanking(ax, telx, telz, depth):
#     """
#     adapted from Jakobs routine to blank part of a section with a constant depth below
#     the surface;
#     ToDO: improve to longer polygon with a point at constant depth below each tem-sounding"""
#     maxx = np.max(telx)
#     minx = np.min(telx)
#     init_z = np.min(telz[:,20])
#     final_z = np.min(telz[:,-20])

#     ax.add_patch(Polygon([(minx, init_z-depth),
#                           (maxx, final_z-depth),
#                           (maxx, final_z-300),
#                           (minx, init_z-300)],
#                          closed=True,
#                          facecolor='white'))


# def get_depths_of_inflictions(mdl, min_depth, max_depth,
#                               approach='max-abs(d1)',
#                               show_plot=True, return_axis=False,
#                               verbose=False):
#     """
    

#     Parameters
#     ----------
#     model : np.array(nx2), depth, 
#         col0 depth -, col1 values. step model
#     min_depth : float
#         minimum depth.
#     max_depth : TYPE
#         maximum depth. (sub zero)
#     approach : str
#         use either the maximum of the first derivative or the min of the second
#     show_plot : bool
#         decide whether you want to see the plot ...

#     Returns
#     -------
#     depth : float
#         depth to infliction pont in the given range
#     value : float
#         value of the model at the retrieved depth

#     """
#     model = mdl.copy()
    
#     _, idx = np.unique(model[:,0], return_index=True)
#     unique_depths = model[:,0][np.sort(idx)]
    
#     _, idx = np.unique(model[:,1], return_index=True)
#     unique_vals = model[:,1][np.sort(idx)]
    
#     if verbose:
#         print(unique_depths[:10])
#         print(unique_depths.shape)
#         print(unique_vals[:10])
#         print(unique_vals.shape)
    
#     # model_unique = np.column_stack((unique_depths[:40],
#     #                                 unique_vals[:40]))  # TODO fix this terrible hack here!!
    
#     model_unique = np.column_stack((unique_depths,
#                                     unique_vals))
    
#     if verbose:
#         print('selecting first part of model')
#         print(model_unique[:])
#         print(model_unique.shape)
    
#     mask = (model_unique[:,0] < min_depth) & (model_unique[:,0] > max_depth)
#     sub_model = model_unique[mask]

#     d1 = np.gradient(sub_model[:,1])
#     d2 = np.gradient(d1)
#     if approach == 'max-abs(d1)':
#         value_d1 = max(abs(d1))  # 1st derivative maximum
#         print('max of abs(d1): ', value_d1)
#         # value_d1 = min(d1)  # 1st derivative minimum
#         # print('min of d1: ', value_d1)
        
#         print(any((abs(d1) == value_d1)))
#         # value = sub_model[(d1 == value_d1), 1].item()
#         value = sub_model[(abs(d1) == value_d1), 1].item()
#         print('value of d1: ', value)
        
#         # depth = sub_model[(d1 == value_d1), 0].item()
#         depth = sub_model[(abs(d1) == value_d1), 0].item()
#         print('depth of d1: ', depth)
#     elif approach == 'min-d1':
#         # value_d1 = max(abs(d1))  # 1st derivative maximum
#         # print('max of abs(d1): ', value_d1)
#         value_d1 = min(d1)  # 1st derivative minimum
#         print('min of d1: ', value_d1)
        
#         print(any((d1 == value_d1)))
#         value = sub_model[(d1 == value_d1), 1].item()
#         # depth = sub_model[(abs(d1) == value_d1), 1].item()
#         print('value of d1: ', value)
        
#         depth = sub_model[(d1 == value_d1), 0].item()
#         # depth = sub_model[(abs(d1) == value_d1), 0].item()
#         print('depth of d1: ', depth)
#     elif approach == '0-d2':
#         value_d2 = min(abs(d2))  # 2nd derivative closest to 0
#         print('min of d2: ', value_d2)
        
#         print(any((d2 == value_d2)))
#         value = sub_model[(abs(d2) == value_d2), 1].item()
#         print('value of d2: ', value)
        
#         depth = sub_model[(abs(d2) == value_d2), 0].item()
#         print('depth of d2: ', depth)
#     else:
#         raise ValueError('please select either "max-d1", or "0-d2" for the approach.')

#     if show_plot:
#         fig, ax = plt.subplots(1,1)
#         ax.plot(model[:,1], model[:,0], '-k', label='model')
#         ax.hlines(y=min_depth,
#                   xmin=min(sub_model[:,1]),
#                   xmax=max(sub_model[:,1]),
#                   color='crimson', label='minimum depth')
#         ax.hlines(y=max_depth,
#                   xmin=min(sub_model[:,1]),
#                   xmax=max(sub_model[:,1]),
#                   color='crimson', label='maximum depth')

#         ax.plot(value, depth, 'oc', label='infliction point')
#         ax.set_xlim((0.5*min(sub_model[:,1]), max(sub_model[:,1])*1.5))
#         ax.legend(fontsize=10, loc='lower left')

#         ax1 = ax.twiny()
#         ax1.plot(d1, sub_model[:,0], 'b:', label='first derivative')
#         ax1.plot(d2, sub_model[:,0], 'g:', label='second derivative')
        
#         ax1.legend(fontsize=10, loc='lower right')

#     if return_axis:
#         return ax, depth, value
#     else:
#         return depth, value



# def parse_zondxls_DM(path, file):
#     """
#     function to parse ZondTEM1d .xls file and create indices for further
#     subselecting of dat file
#     version for Dual Moment TEM!!
#     ToCode: zondVersion!!

#     """
#     raw = pd.read_excel(path + os.sep + file,
#                         names = ['empty','c1','c2','c3','c4','c5','c6', 'c7'],
#                         header=None)

#     dat = raw.drop(labels='empty', axis=1)
#     dat = dat.drop(labels=0, axis=0)
#     indices_labels = np.flatnonzero(dat.c1 == '#') + 1

#     ev2nd0 = indices_labels[::2]
#     ev2nd1 = indices_labels[1::2]
#     end = len(dat)
#     endMdl = np.append(ev2nd0[1:]-5, [end])

#     indices_Hdr = pd.DataFrame({'start':ev2nd0-4, 'end':ev2nd0},
#                                columns=['start', 'end'])

#     indices_Mresp = pd.DataFrame({'start':ev2nd0+1, 'end':ev2nd1-1},
#                                columns=['start', 'end'])

#     indices_Mdl = pd.DataFrame({'start':ev2nd1+1, 'end':endMdl},
#                                columns=['start', 'end'])
#     return dat, indices_Hdr, indices_Mresp, indices_Mdl


# def comp_Mdlresp2_DMdata(stdPath, path, file, plottitle=None,
#                          saveFig=False, filetype='.png',
#                          show_rawdata=True, show_all=True,
#                          xoffset=0, logIndx = np.array([0]), rho2log=False,
#                          set_appResLim=False, minApRes=0, maxApRes=500,
#                          set_signalLim=False, minSig=10e-8, maxSig=10e4,
#                          linewidth=2, markerSize=3):
#     """
#     DualMoment TEM version - eg WalkTEM
#     routine to plot single or multiple model responses in order to check
#     how well the data is represented by the model.
#     """
#     plt.close('all')
#     fullpath = stdPath + os.sep + path

#     savePath = fullpath + os.sep + 'dataFit'
#     if not os.path.exists(savePath):
#         os.makedirs(savePath)

#     dat, indices_Hdr, indices_Mresp, indices_Mdl = parse_zondxls(fullpath, file)
#     nLogs = len(indices_Hdr)

#     if show_rawdata == True:
#         file_raw = file.replace('s1', 'raw')
#         dat_raw, indices_Hdr_raw, indices_Mresp_raw, indices_Mdl_raw = parse_zondxls(fullpath, file_raw)

#     if show_all == True:
#         logIndx = range(0,nLogs)

#     for i in logIndx:
#         # preparation for plotting part
#         header = dat.loc[indices_Hdr.start[i]:indices_Hdr.end[i], ['c1','c2','c3','c4']]
#         logID = header.iloc[0,1]
#         #coordinates = header.iloc[2,1:4]
#         distance = header.iloc[1,1]
#         #error = header.iloc[3,1]

#         signal = dat.loc[indices_Mresp.start[i]:indices_Mresp.end[i],
#                      ['c1','c2','c3','c4','c5','c6','c7']]
#         signal.columns = ['ctr','time','rho_O','rho_C','U_O','U_C','Ts']

#         id_Ts1 = np.asarray(signal.loc[:,'Ts'] == 1).squeeze()
#         id_Ts2 = np.asarray(signal.loc[:,'Ts'] == 2).squeeze()

#         if show_rawdata == True:
#             signal_raw = dat_raw.loc[indices_Mresp_raw.start[i]:indices_Mresp_raw.end[i],
#                                      ['c1','c2','c3','c4','c5','c6','c7']]
#             signal_raw.columns = ['ctr','time','rho_O','rho_C','U_O','U_C','Ts']
#             id_Ts1r = np.asarray(signal_raw.loc[:,'Ts'] == 1).squeeze()
#             id_Ts2r = np.asarray(signal_raw.loc[:,'Ts'] == 2).squeeze()

#         model = dat.loc[indices_Mdl.start[i]:indices_Mdl.end[i], ['c1','c2','c3','c4','c5']]
#         model.columns = ['ctr','Rho','MagP','h','z']


#         # Plotting part
#         fig = plt.figure(figsize=(12, 8)) #figSize in inch; Invoke fig!
#         gs = gridspec.GridSpec(2, 4) #define grid for subplots

#         appRes = fig.add_subplot(gs[0, 0:3])
#         if show_rawdata == True:
#             appRes.semilogx(signal_raw.loc[id_Ts1r, 'time']*10**6,
#                             signal_raw.loc[id_Ts1r, 'rho_O'],
#                             'D', color='darkgray', markerSize=markerSize)
#             appRes.semilogx(signal_raw.loc[id_Ts2r, 'time']*10**6,
#                             signal_raw.loc[id_Ts2r, 'rho_O'],
#                             '*', color='darkgray', markerSize=markerSize)

#         appRes.semilogx(signal.loc[id_Ts1, 'time']*10**6,
#                         signal.loc[id_Ts1, 'rho_C'], '-',
#                         color='lightgreen', linewidth=linewidth)
#         appRes.semilogx(signal.loc[id_Ts1, 'time']*10**6,
#                         signal.loc[id_Ts1, 'rho_O'],  'D',
#                         color='green', markerSize=markerSize)
#         appRes.semilogx(signal.loc[id_Ts2, 'time']*10**6,
#                         signal.loc[id_Ts2, 'rho_C'], ':',
#                         color='salmon', linewidth=linewidth)
#         appRes.semilogx(signal.loc[id_Ts2, 'time']*10**6,
#                         signal.loc[id_Ts2, 'rho_O'],  '*',
#                         color='darkred', markerSize=markerSize)

#         if show_rawdata == False:
#             appRes.xaxis.set_minor_formatter(ScalarFormatter())
#             appRes.xaxis.set_major_formatter(ScalarFormatter())

#         plt.grid(which='major', color='lightgray', linestyle='-')
#         plt.grid(which='minor', color='lightgray', linestyle=':')

#         appRes.set_xlabel('time [$\mu$s]')
#         appRes.set_ylabel(r'$\rho_a$' + r'[$\Omega$m]')

#         if set_appResLim == True:
#             appRes.set_ylim((minApRes,maxApRes))


# #        minApRes = np.floor(np.min(signal.rho_O)*0.9)
# #        maxApRes = np.ceil(np.max(signal.rho_O)*1.01)
# #
# #        minTime = np.floor(np.min(signal_raw.time)*0.9)
# #        maxTime = np.ceil(np.max(signal_raw.time)*1.01)
# #
# #        appRes.set_ylim((minApRes,maxApRes))
# #        appRes.set_xlim((minTime,maxTime))

#         if show_rawdata == True:
#             #appRes.legend(['excluded data', 'model response', 'used Data'])
#             appRes.legend(['excluded data-1A', 'excluded data-7A',
#                            'model response-1A', 'used data-1A',
#                            'model response-7A', 'used data-7A'])
#         else:
#             #appRes.legend(['model response', 'measured'])
#             appRes.legend(['model response-1A', 'measured-1A',
#                            'model response-7A', 'measured-7A',])


#         volt = fig.add_subplot(gs[1, 0:3])
#         if show_rawdata == True:
#             volt.loglog(signal_raw.loc[id_Ts1r, 'time']*10**6,
#                         signal_raw.loc[id_Ts1r, 'U_O']*10**-8,
#                         'D', color='darkgray',
#                         linewidth=0.9, markerSize=3)
#             volt.loglog(signal_raw.loc[id_Ts2r, 'time']*10**6,
#                         signal_raw.loc[id_Ts2r, 'U_O']*10**-8,
#                         '*', color='darkgray',
#                         linewidth=0.9, markerSize=3)


# #            volt.loglog(signal_raw.time*10**6, signal_raw.U_O, 'D',
# #                        color='darkgray',
# #                        linewidth=0.9, markerSize=3)

#         volt.loglog(signal.loc[id_Ts1, 'time']*10**6,
#                     signal.loc[id_Ts1, 'U_C']*10**-8, '-',
#                     color='lightgreen', linewidth=linewidth)
#         volt.loglog(signal.loc[id_Ts1, 'time']*10**6,
#                     signal.loc[id_Ts1, 'U_O']*10**-8,  'D',
#                     color='green', markerSize=markerSize)
#         volt.loglog(signal.loc[id_Ts2, 'time']*10**6,
#                     signal.loc[id_Ts2, 'U_C']*10**-8, ':',
#                     color='salmon', linewidth=linewidth)
#         volt.loglog(signal.loc[id_Ts2, 'time']*10**6,
#                     signal.loc[id_Ts2, 'U_O']*10**-8,  '*',
#                     color='darkred', markerSize=markerSize)

# #        volt.loglog(signal.time*10**6, signal.U_C, '-k',
# #                    signal.time*10**6, signal.U_O,  'Dg',linewidth=0.9, markerSize=3)

#         volt.set_xlabel('time [$\mu$s]')
#         volt.set_ylabel('U/I [V/A]')

#         if set_signalLim == True:
#             volt.set_ylim((minSig,maxSig))


#         if show_rawdata == True:
#             #volt.legend(['excluded data', 'model response', 'used Data'])
#             volt.legend(['excluded data-1A', 'excluded data-7A',
#                          'model response-1A', 'used data-1A',
#                          'model response-7A', 'used data-7A'])
#         else:
#             #volt.legend(['model response', 'measured'])
#             volt.legend(['model response-1A', 'measured-1A',
#                          'model response-7A', 'measured-7A',])

#         if show_rawdata == False:
#             volt.xaxis.set_minor_formatter(ScalarFormatter())
#             volt.xaxis.set_major_formatter(ScalarFormatter())

#         volt.grid(which='major', color='lightgray', linestyle='-')
#         volt.grid(which='minor', color='lightgray', linestyle=':')

#         rA_model = reArr_zondMdl(model)

#         mdl = fig.add_subplot(gs[0:2, 3])
#         mdl.plot(rA_model[:,1], rA_model[:,0], '-ko',
#                  linewidth=linewidth, markerSize=markerSize)

#         if rho2log == True:
#             mdl.set_xscale('log')
#             xlabel = r'$log_{\rho}$ [$\Omega$m]'
#         else:
#             xlabel = r'$\rho$ [$\Omega$m]'

#         mdl.set_xlabel(xlabel)
#         mdl.set_ylabel('depth [m]')
#         mdl.legend(['model'])

#         logName = logID + ' at ' + str(distance) + 'm'
#         if plottitle != None:
#             logName = plottitle


#         fig.suptitle('Dual-Moment (Walk-TEM) ' + logName, fontsize=14)
#         if saveFig == True:
#             fig.tight_layout()
#             plt.subplots_adjust(top=0.95)
#             if show_rawdata == False:
#                 full_savePath = savePath + os.sep + 'zoom' + file.replace('.xls', '') + '_' + logID + filetype
#                 fig.savefig(full_savePath,
#                             dpi=300,
#                             bbox_inches='tight')
#                 print('saving figure to:\n', full_savePath)
#             else:
#                 full_savePath = savePath + os.sep + file.replace('.xls', '') + '_' + logID + filetype
#                 fig.savefig(full_savePath,
#                             dpi=300,
#                             bbox_inches='tight')
#                 print('saving figure to:\n', full_savePath)
#             plt.close('all')
#         else:
#             fig.tight_layout()
#             plt.subplots_adjust(top=0.95)
#             plt.show()
#     return


# def sturges_theorem(n_data):
#     return ceil(np.log2(n_data)) + 1


# def doanes_theorem(data):
#     n = len(data)
#     s_g1 = np.sqrt((6 * (n-2)) / ((n+1)*(n+3)))
#     n_bins = 1 + np.log2(n) + np.log2(1 + (abs(skew(data)) / s_g1))
#     return ceil(n_bins)