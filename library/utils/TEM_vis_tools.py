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
    """
    Function to derive polygon patches collections out of grid/TEM position
    information and data values; Updated version to enable log10 scaling!

    Author: Jakob Gallistl, used to map the TEM results into an IP-grid in:
        Gallistl et al., (2022), 'Quantification of soil textural and hydraulic 
        properties in a complex conductivity imaging framework: Results from the Wolfsegg slope''
        10.3389/feart.2022.911611

    Parameters
    ----------
    elemx : array-like
        x positions.
    elemz : array-like
        y positions (depth info, actually y in the plot).
    colormap : str, optional
        short name of the colormap. The default is 'jet_r'.
    log10 : boolean, optional
        Switch for log10 scaling of the colors. The default is False.
    edgecolors : str, optional
        Color of the polygon edges. The default is 'None'.
    lw : float, optional
        linewidth of the polygon edges. The default is 0.

    Returns
    -------
    p : PatchCollection
        collection of polygone patches.

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
    Function to create 4 corner elements for plotting.
    Needs to be combined with Jakobs get_PatchCollection to obtain the polygone
    collection for plotting TEM sounding results.

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


def tilt_1Dlogs(telx, telz, separated_logs, xz_df, 
                log_width=2, tilt_log='center'):
    """
    Function to tilt the x and z elements according to the topography of the
    TEM profile

    Author: Jakob Gallistl, used to map the TEM results into an IP-grid in:
        Gallistl et al., (2022), 'Quantification of soil textural and hydraulic 
        properties in a complex conductivity imaging framework: Results from the Wolfsegg slope''
        10.3389/feart.2022.911611

    Parameters
    ----------
    telx : array-like
        x positions.
    telz : array-like
        z positions, depths (y-axis in the plot).
    separated_logs : np.ndarray
        contains information on the sounding positions and sounding IDs.
    xz_df : pd.DataFrame
        contains the x and z positions of the sounding positions along a profile.
    log_width : float, optional
        width of the plotted logs. The default is 2.
    tilt_log : str, optional
        method for tilting the logs, either 'center' or 'slope'.
        The default is 'center', which takes the slope at the center between 
        3 sounding positions. 'slope' uses the slope from one sounding to the next one.

    Raises
    ------
    ValueError
        if tilt_log does not equal either 'slope' or 'center'.

    Returns
    -------
    telx_tilt : np.ndarray
        rotated x-values.
    telz_tilt : np.ndarray
        rotated z-values.
    origins : np.ndarray
        origins of the similarity transformation used to rotate the elements.
    tilt_angles : list
        with the tilt angles for each position.

    """

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

        elif tilt_log == 'center':
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

        else:
            raise ValueError('please use either "slope" or "center" for the tilt_log kwarg.')

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
    Function to create support points for the interpolation
    
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
    Function to interpolate TEM sounding results

    Parameters
    ----------
    all_support_points : np.ndarray (n, 3)
        contains all support points.
        column 0: x positions
        column 1: z (y) positions
        column 2: values for the interpolation (e.g., rho)
    mesh_resolution : float
        resolution of the mesh to which the interpolation will be done.
    method : string, optional
        interpolation methodology, see griddata doc for details.
        The default is 'linear'.

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


def plot_doi(axis, DOIs, distances, **kwargs):
    axis.plot(distances, np.asarray(DOIs),
              '--.', **kwargs)
    return axis


def plot_intfc(axis, depths, distances, **kwargs):
    axis.plot(distances, depths,
              '--.', **kwargs)
    return axis


def sturges_theorem(n_data):
    return ceil(np.log2(n_data)) + 1


def doanes_theorem(data):
    n = len(data)
    s_g1 = np.sqrt((6 * (n-2)) / ((n+1)*(n+3)))
    n_bins = 1 + np.log2(n) + np.log2(1 + (abs(skew(data)) / s_g1))
    return ceil(n_bins)
