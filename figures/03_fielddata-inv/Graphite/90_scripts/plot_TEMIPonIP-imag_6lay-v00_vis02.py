# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 2021
@author: laigner
"""


#%% import of necessary modules
import os
import sys
import math
import copy
from glob import glob
from scipy import interpolate

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors

from matplotlib import ticker
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import LogLocator, LogFormatterSciNotation as LogFormatter

# custom modules
from TEM_invvis_tools import compare_model_response
from TEM_invvis_tools import remove_soundings
from TEM_invvis_tools import select_soundings
from TEM_invvis_tools import parse_zondxls
from TEM_invvis_tools import create_TEMelem
from TEM_invvis_tools import get_PatchCollection
from TEM_invvis_tools import create_support_pts
from TEM_invvis_tools import interp_TEMlogs
from TEM_invvis_tools import plot_doi
from TEM_invvis_tools import kriging_TEMlogs
from TEM_invvis_tools import doanes_theorem
from TEM_invvis_tools import tilt_1Dlogs
from TEM_invvis_tools import get_val_elems
from TEM_invvis_tools import plot_framesandnames
from TEM_invvis_tools import plot_TEM_cbar
from TEM_invvis_tools import add_xz2dat
from TEM_invvis_tools import plot_interpol_contours

from plot_inv_tools import plot_ERT_con
from plot_inv_tools import plot_ERT_res
from plot_inv_tools import plot_IP_pha
from plot_inv_tools import plot_IP_imag
from plot_inv_tools import plot_ERT_cbar
from plot_inv_tools import read_grid_triang as rgt


# %% plot style
plt.style.use('seaborn-ticks')
# sns.set(context="notebook", style="whitegrid",
#         rc={"axes.axisbelow": False})

plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 18

xtick_spacing_major = 25
xtick_spacing_minor = 5
ytick_spacing_major = 10
ytick_spacing_minor = 2


# %% path structure ERT
# read IP-inversion (define some parameters first)
#inv structure_Adrian
dir_main_sl = '../../../../../'
dir_main_sl = '../../../../../../../../../../PotGRAF/Hengstberg/'
sip_dir = dir_main_sl + "20220421/GBA_profile/"

path_grids = sip_dir + "grids/"
grid_id = "SIP_P10"

path_inv = sip_dir + "/inv_SIP_10"
lid = "10"; 
inv = "sip_p1_1_250_f5";
    
finInv = path_inv + os.sep + lid + os.sep + inv
finGrids = path_grids + '/' + grid_id

print('reading grids from:\n', finGrids)
elec, elemx, elemz = rgt(finGrids + '.elc',
                         finGrids + '.elm')
# mean_height_ERT = np.mean(elec[:,2])
# mean_height_TEM = np.mean(xz_tem[:,1])
# height_diff = mean_height_ERT - mean_height_TEM


# %% main setup ERT
x_shift = 0
height_diff = 0

plot_s = False
sens_blank = -3

cbar_pos_ERT = 'right'
cmap_ert = 'Spectral_r'
cmap_imag = 'Spectral_r'

clims_pha = (25, 100)
clims_imag = (25, 2000)

# plot boundaries
# rmin = 10; rmax = 40
# s1_min = 10; s1_max = 150
# s2_min = -0.0030; s2_max = 0.0030
# zondVersion='noIP'
# zondVersion='IP'
# path_coord = stdPath + r"data/coord/"
    # file_coord = "lg-P3_coord_GKeast.csv"


# %% TEM main boolean switches
saveFig = True
# saveFig = False

use_only_elec_topo = True
# use_only_elec_topo = False
use_coord = False  # use the xz information to replace existing information in header of TEM result file
use_elec_heights = True

tilt_log = None
tilt_log = 'center'
extend_bot = 15

log10_switch = True
# log10_switch = False

res2con = True
# res2con = False

rRMS_sig_th = 8  # RMS threshhold in %
rRMS_sig_th = None


# %% interpolation setup:
interpol_method = 'No'
# intMethod='linear'
lvls=1000; mesh_resolution=0.5
up_sample='thick'; sample_factor=10
up_sample=None

# intMethod='kriging'; lvls=100; mesh_resolution=2
kr_mthd = 'ordinary_kriging'
variogram_mdl = 'linear'
use_weights = True

show_intPts = True
show_intPts = False


# %% plot setup
patch_ec = 'k'  # 'none'
patch_lw = 0.8  # 0

top_space = 10
bot_space = 5
side_space = 5

# override_xlim = (30, 68)
override_xlim = None

col_map = 'Spectral_r'  # 'viridis_r', plasma_r, inferno_r, magma_r, cividis_r, jet, jet_r
if res2con == True:
    col_map = 'Spectral_r'

norm = colors.Normalize() if (log10_switch == False) else colors.LogNorm()
scale = 'lin' if (log10_switch == False) else 'log10'
locator = LogLocator()
formatter = LogFormatter()


# %% colorbar and limits
xoffset = 25; log_width=3; depth=60
rmin=1000/1000; rmax=1000/3
rmin=10; rmax=1000

elPara = 'Res'
if res2con == True:
    elPara = 'Con'
    cmin_tem = 1000/rmax
    cmax_tem = 1000/rmin # to mS/m
    print('min and max cbar limits:')
    print(cmin_tem, cmax_tem)
    clims_ert = np.r_[cmin_tem, cmax_tem]
    clims_ert = (1, 100)                               # override colorbar limits for ERT (different limits than for TEM)
else:
    cmin_tem, cmax_tem = rmin, rmax
    clims_ert = np.r_[rmin, rmax]
    clims_ert = (1, 100)                                 # override colorbar limits for ERT (different limits than for TEM)

filetype='.png'; dpi=200; label_s=12


# %% path structure TEM
# version = 'v01_tr5-300us'  # v02_tr5-400us, v02_tr5-500us, v03_tr6-400us
version = 'v00_tr5-200us'  # version = 'v00_tr5-200us'

for invrun in [f'{i:03d}' for i in range(0, 3)]:
# invrun = '002'
    fname_raw = 'hb-p10-sel.tem'
    lid = 'hb-p10_6lay-mpa'
    version = 'v00_tr5-200us'

    path_main = '../../../'
    path = path_main + f'03-inv_results/pyGIMLi/{lid}/{version}/'
    fids = glob(path + f'*{invrun}_{version}.xls')
    
    path_xz_coord = path_main + '05-coord/'
    fname_coord = 'p10_xz.csv'

    path_raw = path_main + "01-data/"
    
    # additional data, e.g.:
    # path_lakedepth = stdPath + f"03-inv/as/{version}/depths/"
    # fname_lakedepth = 'ProfilOW-Tiefen.txt'
    # lakedepths = pd.read_csv(path_lakedepth + fname_lakedepth,
    #                          sep=' ', engine='python', names=['id', 'depth'])

    scriptname = os.path.basename(sys.argv[0])
    print(f'running {scriptname} ...')
    vis_version = scriptname.split('.')[0].split('_')[-1]
    
    path_savefig = path_main + f'04-vis/pyGIMLi/TEMonIP/{lid}/{version}/{vis_version}/'
    if not os.path.exists(path_savefig):
        os.makedirs(path_savefig)


    # %% loop over files in path that end with *.xls
    for fid in fids[0:]:
        file = fid.split(os.sep)[-1]
    
        # %% load data and select subset
        (dat, ind_hdr, ind_mresp, ind_mdl,
         zondVersion) = parse_zondxls(path, file)
        nSnds = len(ind_hdr)
        
        relRMS_sig = []
        relRMS_rhoa = []
        dstncs = []
        hgths = []
        coords = []
        for logID in range(0,nSnds):
            hdr = dat.loc[ind_hdr.start[logID]:ind_hdr.end[logID], :]
            # print(hdr)
            dstncs.append(float(hdr.iloc[1,1]))
            relRMS_sig.append(float(hdr.iloc[3,1]))
            relRMS_rhoa.append(float(hdr.iloc[3,1]))
            hgths.append(float(hdr.iloc[2,3]))
            coords.append((hdr.iloc[2, 1], hdr.iloc[2, 2], hdr.iloc[2, 3]))
        
        heights = np.array(hgths)
        distances = np.array(dstncs)
        snd_names = ind_hdr.sndID
        relRMS_sig = np.asarray(relRMS_sig)
        coords = np.asarray(coords)

        if not use_coord:
            xz = np.column_stack((distances, heights))
            xz[:, 0] += xoffset
            distances = xz[:, 0]
            if use_elec_heights:
                print('using elec topo also for tem')
                f = interpolate.interp1d(elec[:, 1], elec[:, 2], fill_value="extrapolate", kind='cubic')
                tem_heights_interp = f(distances)
                xz = np.column_stack((distances, tem_heights_interp))
                heights = xz[:, 1]
                dat = add_xz2dat(dat, ind_hdr, xz)
        else:
            df_coord = pd.read_csv(path_xz_coord + fname_coord)
            xz = np.asarray(df_coord.loc[:, ['cum_sd', 'Z']])
            xz[:, 0] += xoffset
            distances = xz[:, 0]
            if use_elec_heights:
                print('using elec topo also for tem')
                f = interpolate.interp1d(elec[:, 1], elec[:, 2])
                tem_heights_interp = f(distances)
                xz = np.column_stack((distances, tem_heights_interp))
            heights = xz[:, 1]
            dat = add_xz2dat(dat, ind_hdr, xz)
    
        xz_df = pd.DataFrame(xz, columns=['X', 'Z'])
        xz_df['SoundingID'] = ind_hdr.sndID
        
    
        # %% filter according to RMS
        if not rRMS_sig_th == None:
            # n_bins = sturges_theorem(len(relRMS_sig))
            n_bins = doanes_theorem(relRMS_sig)
            
            fig_hist, ax_hist = plt.subplots(1, 2, figsize=(12,5), sharey=True)
            ax_hist[0].hist(relRMS_sig, bins=doanes_theorem(relRMS_sig))
            
            ax_hist[0].set_xlabel('rRMSE of signal (%)')
            ax_hist[0].set_ylabel('count')
            ax_hist[0].legend()
            
            ax_hist[1].hist(relRMS_rhoa, bins=doanes_theorem(relRMS_rhoa))
            ax_hist[1].set_xlabel('rRMSE of rhoa (%)')
            plt.tight_layout()
            
            ax_hist[0].axvline(x=rRMS_sig_th, color='crimson', label='threshhold')
            mask_rms = np.asarray(relRMS_sig) < rRMS_sig_th
            snd_names_filt = snd_names[mask_rms]
            heights = heights[mask_rms]
            distances = distances[mask_rms]
            coords = coords[mask_rms]
            
            nSnds = len(snd_names_filt)
            
            xz = xz[mask_rms]
            xz_df = xz_df[mask_rms]
            relRMS_sig = relRMS_sig[mask_rms]
            
            ind_hdr, ind_mresp, ind_mdl = select_soundings(ind_hdr,
                                                           ind_mresp,
                                                           ind_mdl,
                                                           snd_names_filt)
            selected_snds = ind_hdr.sndID.values.tolist()
            print(selected_snds)


        # %% Create TEM elements
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9),
                                 constrained_layout=False) #figSize in inch

        (telx, telz, telRho, frame_lowLeft,
         nLogs, topo, snd_names, extend_bot,
         separated_logs) = create_TEMelem(dat, ind_hdr, ind_mresp, ind_mdl,
                                          log_width=log_width, xoffset=xoffset, extend_bot=extend_bot,
                                          zondVersion='IP')
        topo[:, 0] = topo[:, 0] - log_width/2
        
        if tilt_log is not None:
            (telx_tilt, telz_tilt, origins,
             tilt_angles) = tilt_1Dlogs(telx, telz, separated_logs,
                                        xz_df, frame_lowLeft,
                                        log_width=log_width,
                                        tilt_log='center')
    
            pr = get_PatchCollection(telx_tilt, telz_tilt,
                                     colormap=col_map,
                                     log10=log10_switch,
                                     edgecolors=patch_ec, lw=patch_lw)
    
        else:
            print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('you decided to keep the logs vertical... ')
            tilt_angles = np.zeros((nSnds,))
            origins = xz
            pr = get_PatchCollection(telx, telz,
                                     colormap=col_map,
                                     log10=log10_switch,
                                     edgecolors=patch_ec, lw=patch_lw)
    
        pr_2 = copy.deepcopy(pr)
        
        if res2con == True:
            pr.set_array(1000/telRho)
            pr_2.set_array(1000/telRho)
            label_cbar_tem = r"$\sigma_0$ ($mS/m$)"
            label_cbar_ert = r"$\sigma$ ($mS/m$)"
            label_cbar_pha = r"$\sigma''$ ($\mu$S)"
            pr.set_clim([cmin_tem, cmax_tem])
            pr_2.set_clim([cmin_tem, cmax_tem])
        else:
            pr.set_array(telRho)
            pr_2.set_array(telRho)
            label_cbar_tem = r"$\rho$ ($\Omega$m)"
            label_cbar_ert = r"$\rho$ ($\Omega$m)"
            label_cbar_pha = r"$-\phi$ (mrad)"
            pr.set_clim([cmin_tem, cmax_tem])
            pr_2.set_clim([cmin_tem, cmax_tem])


        # %% plot IP
        ax1, prERT = plot_ERT_con(fig1, ax1, fid_inv=finInv, fid_grids=finGrids,
                                  x_shift=x_shift, z_shift=height_diff, log10=True, expand_topo=15,
                                  colMap=cmap_ert, c_lims=clims_ert, sens_blank=sens_blank)
        
        round_tick_label = 1
        label_fmt = f'%.{round_tick_label}f'
    
        divider = make_axes_locatable(ax1)
        cax2 = divider.append_axes("right", size="2%", pad=0.3)
        cb2 = fig1.colorbar(prERT, cax=cax2, format=label_fmt)
        # cb.ax.yaxis.set_ticks([10, 20, 40, 100, 200])
        if log10_switch is True:
            ticks = np.round(np.logspace(np.log10(clims_ert[0]), np.log10(clims_ert[1]), 5), round_tick_label)
            cb2.ax.yaxis.set_ticks(ticks)
        else:
            tick_locator = ticker.MaxNLocator(nbins=5)
            cb2.locator = tick_locator; cb2.update_ticks()
        cb2.ax.minorticks_off()
        cb2.set_label('IP: ' + label_cbar_ert)
        
        
        ax2, prPHA = plot_IP_imag(fig1, ax2, fid_inv=finInv, fid_grids=finGrids,
                                  x_shift=x_shift, z_shift=height_diff, log10=True, expand_topo=15,
                                  colMap=cmap_imag, c_lims=clims_imag, sens_blank=sens_blank)
    
        round_tick_label = 1
        label_fmt = f'%.{round_tick_label}f'
    
        divider2 = make_axes_locatable(ax2)
        cax2_1 = divider2.append_axes("right", size="2%", pad=0.3)
        cb2_1 = fig1.colorbar(prPHA, cax=cax2_1, format=label_fmt)
        # cb.ax.yaxis.set_ticks([10, 20, 40, 100, 200])
        if log10_switch is True:
            ticks = np.round(np.logspace(np.log10(clims_imag[0]), np.log10(clims_imag[1]), 5), round_tick_label)
            cb2_1.ax.yaxis.set_ticks(ticks)
        else:
            tick_locator = ticker.MaxNLocator(nbins=5)
            cb2_1.locator = tick_locator; cb2_1.update_ticks()
        cb2_1.ax.minorticks_off()
        cb2_1.set_label('IP: ' + label_cbar_pha)


        # %% add logs after interpolation
        pr.set_cmap(cmap=col_map)
        ax1.add_collection(pr)
        
        pr_2.set_cmap(cmap=col_map)
        ax2.add_collection(pr_2)

        # %% Axis labels, tick frequency, etc
        show_names = True
        show_rms = True
        plot_framesandnames(ax1, log_width, tilt_angles, origins,
                    frame_lowLeft, xz_df, snd_names, relRMS_sig, 
                    show_frames=False, show_rms=show_rms, show_names=show_names, txtsize=14,
                    top_space=6, extend_bot=extend_bot, rotation=45)

        # %% Axis labels, tick frequency, etc
        ax1.locator_params(axis='x', nbins=15)
        ax1.locator_params(axis='y', nbins=7)
        # ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True, zorder=10)
        ax1.tick_params(axis='both', which='both', direction='inout', top=False, right=False, zorder=10)
        ax1.tick_params(axis='both', which='major', pad=10, width=1.3, length=8, color='k')
        ax1.tick_params(axis='both', which='minor', width=1, length=6, color='k')
    
        # ax1.grid(which='major', axis='y', color='white', linestyle='--', linewidth=0.8)
        ax1.xaxis.set_major_locator(MultipleLocator(xtick_spacing_major))
        # ax1.xaxis.set_major_formatter('{x:.1f}')
        ax1.xaxis.set_minor_locator(MultipleLocator(xtick_spacing_minor))
    
        ax1.yaxis.set_major_locator(MultipleLocator(ytick_spacing_major))
        # ax1.xaxis.set_major_formatter('{x:.1f}')
        ax1.yaxis.set_minor_locator(MultipleLocator(ytick_spacing_minor))
    
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Height (m)')

        # %% colorbars ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("left", size="2%", pad=0.8)
        cb1 = fig1.colorbar(pr, cax=cax1, format=label_fmt)
        if log10_switch is True:
            ticks = np.round(np.logspace(np.log10(cmin_tem), np.log10(cmax_tem), 5), round_tick_label)
            cb1.ax.yaxis.set_ticks(ticks)
        else:
            tick_locator = ticker.MaxNLocator(nbins=5)
            cb1.locator = tick_locator; cb1.update_ticks()
        cb1.ax.minorticks_off()
        cb1.ax.yaxis.set_ticks_position('left')
        cb1.ax.yaxis.set_label_position('left')
        # cb1.set_label('TEM: ' + labelRho)
        cb1.set_label('TEM: ' + label_cbar_tem)
        ax1.yaxis.set_label_coords(-0.015, -0.1)
        ax1.set_ylabel('H (m)', rotation=90)

        cax1_2 = divider2.append_axes("left", size="2%", pad=0.8)
        cb1_2 = fig1.colorbar(pr_2, cax=cax1_2, format=label_fmt)
        if log10_switch is True:
            ticks = np.round(np.logspace(np.log10(cmin_tem), np.log10(cmax_tem), 5), round_tick_label)
            cb1_2.ax.yaxis.set_ticks(ticks)
        else:
            tick_locator = ticker.MaxNLocator(nbins=5)
            cb1_2.locator = tick_locator; cb1_2.update_ticks()
        cb1_2.ax.minorticks_off()
        cb1_2.ax.yaxis.set_ticks_position('left')
        cb1_2.ax.yaxis.set_label_position('left')

        cb1_2.set_label('TEM: ' + label_cbar_tem)
        ax1.yaxis.set_label_coords(-0.015, -0.1)
        ax1.set_ylabel('H (m)', rotation=90)


        # %% set limits for plots
        if depth != 'auto':
            if tilt_log is not None:
                ax1.set_ylim((np.max(telz_tilt) - depth, np.max(telz_tilt) + top_space))
                ax1.set_xlim((np.min(telx_tilt) - side_space, 
                              np.max(telx_tilt) + side_space))
            else:
                ax1.set_ylim((np.max(telz) - depth, np.max(telz) + top_space))
                ax1.set_xlim((np.min(telx) - side_space, np.max(telx) + side_space))
        else:
            if tilt_log is not None:
                ax1.set_ylim((np.min(telz_tilt), np.max(telz_tilt) + top_space))
                ax1.set_xlim((np.min(telx_tilt) - side_space, 
                              np.max(telx_tilt) + side_space))
            else:
                ax1.set_ylim((np.min(telz), np.max(telz) + top_space))
                ax1.set_xlim((np.min(telx) - side_space, np.max(telx) + side_space))
        
        if depth != 'auto':
            if tilt_log is not None:
                ax2.set_ylim((np.max(telz_tilt) - depth, np.max(telz_tilt) + top_space))
                ax2.set_xlim((np.min(telx_tilt) - side_space, 
                              np.max(telx_tilt) + side_space))
            else:
                ax2.set_ylim((np.max(telz) - depth, np.max(telz) + top_space))
                ax2.set_xlim((np.min(telx) - side_space, np.max(telx) + side_space))
        else:
            if tilt_log is not None:
                ax2.set_ylim((np.min(telz_tilt), np.max(telz_tilt) + top_space))
                ax2.set_xlim((np.min(telx_tilt) - side_space, 
                              np.max(telx_tilt) + side_space))
            else:
                ax2.set_ylim((np.min(telz), np.max(telz) + top_space))
                ax2.set_xlim((np.min(telx) - side_space, np.max(telx) + side_space))

        # %% add Info
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        topo[:,1] = topo[:,1]
        expand_topo = 5
        # ax1.plot(topo[:,0], topo[:,1], '-k', lw='2')
    
        if use_only_elec_topo:
            # ax1.plot(np.r_[topo[0, 0] - 15, topo[0, 0]],
            #          np.r_[topo[0, 1], topo[0, 1]],
            #           '-', color='k')
        
            # ax1.plot(np.r_[topo[-1, 0] + 15, topo[-1, 0]],
            #          np.r_[topo[-1, 1], topo[-1, 1]],
            #           '-', color='k')
            pass
        else:
            ax1.plot(np.r_[topo[0,0] - 15, topo[:, 0], topo[-1, 0] + 15],
                      np.r_[topo[0,1], topo[:, 1], topo[-1, 1]] - 0.1,
                      '-', color='k')
        
        # if 'depth' in coord_raw.columns.values:                         # plot add Info (bathymetrie)
        #     print('about to plot water depth...')
        #     ax1 = plot_intfc(ax1, coord_raw.z-coord_raw.depth.values-height,
        #                      dstncs.values, color='white', zorder=10, ls='-')
    
        ##################### CALC and plot DOI ###################################
        # DOIs, OPTs = calc_doi(fname_ztres=file, path_ztres=path,
        #                       fname_raw=fname_raw, path_raw=path_raw,
        #                       x0=30, verbose=True)
        # ax1 = plot_doi(ax1, heights-DOIs,
        #                distances, color='r', zorder=10)
    
    
    
        # intfc2_h = np.r_[468, 468, 463, 460, 461.5, 461.5,
        #                  461, 463.2, 463.8, 462.5, 461.5, 463.4]
        # ax1 = plot_intfc(ax1, intfc2_h-height, dstncs.values, color='white', zorder=10)
        
        # intfc1_h = np.r_[np.nan, (coord_raw.z-coord_raw.depth)[1], 468, 465.5, 467.5, 467.5,
        #                      466.5, 468.5, 468, 466, 467.5, 468.5]
        # ax1 = plot_intfc(ax1, intfc1_h-height, dstncs.values, color='white', zorder=10)
    
        # ax1.text(350, 445-height, 'DOI', color='r',
        #          ha='center', va='center',
        #          rotation=0, size=26)
        
        # ax1.text(350, 475-height, 'water', color='white',
        #          ha='center', va='center',
        #          rotation=0, size=22)
        
        # ax1.text(350, 470-height, 'fluvial sediments', color='white',
        #          ha='center', va='center',
        #          rotation=0, size=20)
        
        # ax1.text(350, 464-height, 'clay rich', color='white',
        #          ha='center', va='center',
        #          rotation=0, size=22)
        
        # ax1.text(350, 464-height, 'highest $\sigma$', color='white',
        #          ha='center', va='center',
        #          rotation=0, size=22)
        
        if override_xlim is not None:
            ax1.set_xlim(override_xlim)
        
        # ax1.grid(which='major', color='lightgray', linestyle='--', zorder=10)
        # ax1.grid(which='minor', color='lightgray', linestyle=':', zorder=10)
        plt.tight_layout()
        
        if saveFig == True:
            logname = f'{vis_version}__{interpol_method[:3]}_{col_map}_{elPara}_{scale}.log'
            with open(f"{path_savefig}/{logname}", 'w') as fp:
                pass
            savefid = f"{path_savefig}/{file[:-4]}_{inv}_{vis_version}{filetype}"
            print('saving figure to:',
                  savefid)
            fig1.savefig(savefid,
                        dpi=dpi,
                        bbox_inches='tight')
        else:
            plt.show()
            
        # break
    
    
    # %% old snippets
    # adjustFigAspect(fig1,aspect=4) ##??
    
    
    # %% add Coordinates and distances to dat (selected Snds only)
    # nSnds = len(ind_Hdr)
    # dstncs = coord_raw.distance
    # dstncs = coord_raw.dist_prj
    # height = coord_raw.z[0]
    
    # relRMS = []
    # for logID in range(0,nSnds):
    #     hdr = dat.loc[ind_Hdr.start[logID]:ind_Hdr.end[logID]]
    #     relRMS.append(float(hdr.iloc[3,1]))
    #     # print(coord_raw)
    #     for i in range(0,len(coord_raw)):
    #         if coord_raw.name[i] == hdr.iloc[0][1].upper():
    #             hdr.iat[1,1] = dstncs[i]
    #             hdr.iat[2,1] = coord_raw.x[i]
    #             hdr.iat[2,2] = coord_raw.y[i]
    #             hdr.iat[2,3] = 0 # coord_raw.z[i]
    
        # divider = make_axes_locatable(ax1)
        # cax1 = divider.append_axes("right", size="2%", pad=0.3)
        # cb = fig1.colorbar(pr, cax=cax1, format='%.2f')
        # # cb.ax.yaxis.set_ticks([10, 20, 40, 100, 200])
        # if log10_switch is True:
        #     ticks = np.round(np.logspace(np.log10(cmin), np.log10(cmax), 6), 3)
        #     cb.ax.yaxis.set_ticks(ticks)
        # else:
        #     tick_locator = ticker.MaxNLocator(nbins=6)
        #     cb.locator = tick_locator; cb.update_ticks()
        # # cb.locator = locator
        # # cb.formatter = formatter
        # cb.ax.minorticks_off()
    
        # cb.set_label(labelRho)
        # # cb.ax.tick_params(labelsize=label_s)