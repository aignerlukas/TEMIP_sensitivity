# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 2021

script to visualize a TEMIP section

TODOs:
    [] plot only selected parameters
    [] update kriging parameters
        [] directional kriging!!
    [] add doi!
        [] calc from res
        [] plot to all??

@author: laigner
"""

# %% import of necessary modules
import os
import sys
from glob import glob

rel_path_to_libs = '../../../../../'
if not rel_path_to_libs in sys.path:
    sys.path.append(rel_path_to_libs)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib.patches import Polygon
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import LogLocator, LogFormatterSciNotation as LogFormatter
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# custom modules
from library.utils.TEM_vis_tools import compare_model_response
from library.utils.TEM_vis_tools import parse_zondxls
from library.utils.TEM_vis_tools import select_soundings
from library.utils.TEM_vis_tools import create_TEMelem
from library.utils.TEM_vis_tools import get_PatchCollection
from library.utils.TEM_vis_tools import calc_doi
from library.utils.TEM_vis_tools import create_support_pts
from library.utils.TEM_vis_tools import interp_TEMlogs
from library.utils.TEM_vis_tools import kriging_TEMlogs
from library.utils.TEM_vis_tools import doanes_theorem
from library.utils.TEM_vis_tools import tilt_1Dlogs
from library.utils.TEM_vis_tools import get_val_elems
from library.utils.TEM_vis_tools import plot_framesandnames
from library.utils.TEM_vis_tools import plot_TEM_cbar
from library.utils.TEM_vis_tools import add_xz2dat
from library.utils.TEM_vis_tools import plot_interpol_contours

from library.utils.TEM_vis_tools import plot_doi


# %% plot style
plt.style.use('seaborn-ticks')
# sns.set(context="notebook", style="whitegrid",
#         rc={"axes.axisbelow": False})

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 18

xtick_spacing_major = 20
xtick_spacing_minor = 5
ytick_spacing_major = 20
ytick_spacing_minor = 5

filetype='.png'
dpi=200
label_s=12


# %% boolean switches
use_coord = False
# use_coord = True

saveFig = True
# saveFig = False

tilt_log = None
tilt_log = 'center'
tilt_switch = True if (tilt_log != None) else False

log10_switch = True
# log10_switch = False

res2con = True
res2con = False

rRMS_sig_th = 8  # RMS threshhold in %
rRMS_sig_th = None

plot_comparison = False
# extract_boundaries = True
extract_boundaries = False
# save_extraction = True
save_extraction = False

show_intPts = True
show_intPts = False

TEM_cbar_pos = 'right'

# %% interpolation setup:
# int_method = None
# int_method = 'linear'
# lvls=1000
# mesh_resolution=1
# up_sample = 'thick'
# sample_factor = 10
# up_sample = None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ KRIGING ~~~~~~~~~~~~~~~~
int_method='kriging'; lvls=300; mesh_resolution=1
up_sample='all'; sample_factor = 8
kr_mthd = 'universal_kriging'  # ordinary_kriging, universal_kriging
variogram_mdl = 'exponential'  # linear, power, gaussian, spherical, exponential, hole-effect

style = 'grid' # grid, points
nlags = 6
anisotropy_scaling = 5
anisotropy_angle = -17.6  # check tilt angles: np.array(tilt_angles) * 180 / np.pi, use median
use_weights = True


# %% adjust plot
xoffset = 0
log_width = 2.5
patch_ec = 'k'  # 'none'
patch_lw = 0.5  # 0
topo_lw = 2.0

rmin = 250
rmax = 5000
depth = 50
# depth='auto'

top_space = 5
bot_space = 5
side_space = 5
extend_bot = 20

# %% plot appearance
col_map_r = 'viridis_r'  # 'viridis_r', plasma_r, inferno_r, magma_r, cividis_r, jet, jet_r
if res2con == True:
    col_map_r = 'Spectral_r'

norm = colors.Normalize() if (log10_switch == False) else colors.LogNorm()
scale = 'lin' if (log10_switch == False) else 'log10'
locator = LogLocator()
formatter = LogFormatter()

elPara = 'Res'
if res2con == True:
    elPara = 'Con'
    cmin = 1000/rmax
    cmax = 1000/rmin # to mS/m
    print('min and max cbar limits:')
    print(cmin, cmax)
else:
    cmin = rmin
    cmax = rmax


# CC param setup
# CC_keys = ['pol', 'tau', 'c']
# CC_col_maps = ['YlOrBr', 'YlGn', 'YlGnBu']
# # CC_labels = ['m ()', r'$\tau$ (s)', 'c ()']  # pelton model
# CC_labels = ['$\phi_{max}$ (rad)', r'$\tau_{phi}$ (s)', 'c ()']  # mpa model
# CC_limits = [(0.05, 1.0), (1e-5, 5e-4), (0.05, 1)]
# CC_label_fmts = ['%.1f', '%.1e', '%.1f']
# CC_log_switches = [False, True, False]
# CC_scales = ['lin', 'log10', 'lin']
# CC_rnd_ticks = [1, 8, 1]

CC_keys = ['pol', 'c']
CC_col_maps = ['YlOrBr', 'YlGnBu']
# CC_labels = ['m ()', r'$\tau$ (s)', 'c ()']  # pelton model
CC_labels = ['$\phi_{max}$ (rad)', 'c ()']  # mpa model
CC_limits = [(0.1, 1.0), (0.1, 1)]
CC_label_fmts = ['%.1f', '%.1f']
CC_log_switches = [False, False]
CC_scales = ['lin', 'lin']
CC_rnd_ticks = [1, 1]

int_vals = ['rho'] + CC_keys
int_cmaps = [col_map_r] + CC_col_maps
int_climits = [(cmin, cmax)] + CC_limits
int_log10 = [scale] + CC_scales

n_rows = len(int_vals)


# %% path structure
line_id = '2'

# type_inv = 'pyGIMLi'
type_inv = f'sb-l{line_id}_blk-6lay-mpa'
lid = f'sb-line-0{line_id}'
version = 'cs12_tr12-100us'

for invrun in [f'{i:03d}' for i in range(0, 3)]:
    # invrun = '000'
    fname_result = f'20210716-sb_Line{line_id}-xz_{invrun}_{version}'

    scriptname = os.path.basename(sys.argv[0])
    print(f'running {scriptname} ...')
    vis_version = scriptname.split('.')[0].split('_')[-1]

    path_main = '../../'
    path = path_main + f'30_TEM-results/{type_inv}/{lid}/{version}/'
    fids = glob(path + fname_result + '.xls')

    path_xz_coord = path_main + '50_coord/'
    fname_coord = f'20210716-sb_Line{line_id}-xz'

    path_raw = path_main + "00_data/selected/"
    fname_raw = f'sb-line-0{line_id}.tem'

    # additional data, e.g.:
    # path_lakedepth = stdPath + f"03-inv/as/{version}/depths/"
    # fname_lakedepth = 'ProfilOW-Tiefen.txt'
    # lakedepths = pd.read_csv(path_lakedepth + fname_lakedepth,
    #                          sep=' ', engine='python', names=['id', 'depth'])

    path_savefig = path_main + f'40_vis/{lid}/{vis_version}/'
    if not os.path.exists(path_savefig):
        os.makedirs(path_savefig)

    # %% read coordinates if used:
    if use_coord:
        df_coord = pd.read_csv(path_xz_coord + fname_coord)
        # xz = np.asarray(df_coord.loc[:, ['cum_sd', 'height_dem']])
        xz = np.asarray(df_coord.loc[:, ['cum_sd', 'Z']])


    # %% loop over files in path that end with *.xls
    for fid in fids[0:]:
        file = fid.split('/')[-1]

        DOIs, OPTs = calc_doi(fname_ztres=file, path_ztres=path,
                              fname_raw=fname_raw, path_raw=path_raw,
                              x0=100, verbose=False)

        if plot_comparison == True:
            compare_model_response(path, file, path_savefig=path_savefig, plottitle=None,
                                   saveFig=True, filetype='.png',
                                   show_rawdata=True, show_all=True,
                                   xoffset=0, logIndx = np.array([1]), rho2log=False,
                                   set_appResLim=False, minApRes=0, maxApRes=500,
                                   set_signalLim=False, minSig=10e-8, maxSig=10e4,
                                   set_timeRange=False, minT=2, maxT=15500,
                                   linewidth=2, markerSize=3, labelsize=12)

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
        else:
            heights = xz[:, 1]

        xz_df = pd.DataFrame(xz, columns=['X', 'Z'])
        xz_df['SoundingID'] = ind_hdr.sndID


        # %% add coordinates to dataframe:
        if use_coord:
            dat_crd = add_xz2dat(dat, ind_hdr, xz)


        # %% filter according to RMS
        if not rRMS_sig_th == None:
            # n_bins = sturges_theorem(len(relRMS_sig))
            n_bins = doanes_theorem(relRMS_sig)

            fig_hist, ax_hist = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
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
            DOIs = np.asarray(DOIs)[mask_rms]
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



        # %% Setup figure
        fig, axes = plt.subplots(n_rows, 1, figsize=(8, 9))


        # %% Create TEM elements
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
                                     colormap=col_map_r,
                                     log10=log10_switch,
                                     edgecolors=patch_ec, lw=patch_lw)

        else:
            print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('you decided to keep the logs vertical... ')
            tilt_angles = np.zeros((nSnds,))
            origins = xz
            pr = get_PatchCollection(telx, telz,
                                     colormap=col_map_r,
                                     log10=log10_switch,
                                     edgecolors=patch_ec, lw=patch_lw)

        if res2con == True:
            pr.set_array(1000/telRho)
            labelRho = r"$\sigma_0$ ($mS/m$)"
            pr.set_clim([cmin, cmax])
        else:
            pr.set_array(telRho)
            labelRho = r"$\rho_0$ ($\Omega$m)"
            pr.set_clim([cmin, cmax])


        # %% prepare CC params as logs
        CC_vals = []
        CC_prs = []

        for idx, CC_label in enumerate(CC_labels):
            CC_vals.append(get_val_elems(dat, ind_hdr,
                                         ind_mresp, ind_mdl, kind=CC_keys[idx]))
            if tilt_log is not None:
                CC_prs.append(get_PatchCollection(telx_tilt, telz_tilt,
                                         colormap=CC_col_maps[idx],
                                         log10=CC_log_switches[idx],
                                         edgecolors=patch_ec, lw=patch_lw)
                                         )
            else:
                CC_prs.append(get_PatchCollection(telx, telz,
                                         colormap=CC_col_maps[idx],
                                         log10=CC_log_switches[idx],
                                         edgecolors=patch_ec, lw=patch_lw)
                                         )


        # %% Interpolation part,         # TODO all to a plotting function!
        if int_method != None:                                                           # Interpolation
            for i in range(0, n_rows):
                support_pts = create_support_pts(dat, ind_hdr, ind_mresp, ind_mdl,
                                                 log_width=log_width, xoffset=xoffset, values=int_vals[i],
                                                 tilt=tilt_switch, tilt_angles=tilt_angles,
                                                 origins=origins, up_sample=up_sample,
                                                 sample_factor=sample_factor, average_thin=True,
                                                 use_common_depth=False)

                if int_method == 'kriging':
                    (xi, yi, zi,
                     z_var) = kriging_TEMlogs(support_pts, mesh_resolution, mesh=None,
                                              method=kr_mthd, variogram_mdl=variogram_mdl,
                                              anisotropy_scaling=anisotropy_scaling,
                                              anisotropy_angle=anisotropy_angle,
                                              use_weight=use_weights, nlags=nlags)

                else:
                     (xi_mesh, yi_mesh,
                      zi) = interp_TEMlogs(support_pts,
                                           mesh_resolution,
                                           method=int_method)

                if style == 'grid':
                    xi_mesh, yi_mesh = np.meshgrid(xi, yi)
                    frame =  0
                    minX = np.min(xi_mesh); maxX = np.max(xi_mesh)
                    minY = np.min(yi_mesh); maxY = np.max(yi_mesh)
                    extent = (minX-frame, maxX+frame,
                              minY-frame, maxY+frame)
                    if res2con == True:
                        ctrs = plot_interpol_contours(axes[i], xi_mesh, yi_mesh, 1000/zi,
                                                      int_climits[i], lvls, int_cmaps[i], int_log10[i],
                                                      show_clines=False, cline_freq=20)
                    else:
                        ctrs = plot_interpol_contours(axes[i], xi_mesh, yi_mesh, zi,
                                                      int_climits[i], lvls, int_cmaps[i], int_log10[i],
                                                      show_clines=False, cline_freq=20)
                elif style == 'points':
                    print('creating a scatter plot based upon the mesh that was provided')
                    if res2con == True:
                        axes[i].scatter(xi, yi, c=1000/zi, cmap=int_cmaps[i])
                    else:
                        axes[i].scatter(xi, yi, c=zi, cmap=int_cmaps[i],
                                        vmin=int_climits[i][0], vmax=int_climits[i][1])

                upLeft = np.copy(topo[0,:]) + np.r_[-15, 50]
                loLeft = np.copy(topo[0,:]) + np.r_[-15, 0]
                upRight = np.copy(topo[-1,:]) + np.r_[15, 50]
                loRight = np.copy(topo[-1,:]) + np.r_[15, 0]
                topoMask = np.vstack((upLeft, loLeft, topo,
                                      loRight, upRight))
                maskTopo  = Polygon(topoMask, facecolor='white', closed=True)
                axes[i].add_patch(maskTopo)

                if show_intPts:
                    x = support_pts[:,0]
                    z_dep = support_pts[:,1]
                    axes[i].plot(x, z_dep, 'x', color='white', ms=0.8)
                    axes[i].plot(xi_mesh, yi_mesh, '.k', ms=0.5)
                    axes[i].plot(origins[:, 0], origins[:, 1], 'xr', ms=2)


        # %% add logs after interpolation
        pr.set_cmap(cmap=col_map_r)
        axes[0].add_collection(pr)

        for idx, CC_pr in enumerate(CC_prs):
            CC_pr.set_clim(CC_limits[idx])
            CC_pr.set_array(CC_vals[idx])
            CC_pr.set_cmap(cmap=CC_col_maps[idx])
            axes[idx+1].add_collection(CC_pr)


        # %% make frames around logs; add labels to soundings
        for i in range(0, n_rows):
            if i == 0:
                show_names = True
                show_rms = True
            else:
                show_names = False
                show_rms = False

            plot_framesandnames(axes[i], log_width, tilt_angles, origins,
                                frame_lowLeft, xz_df, snd_names, relRMS_sig,
                                show_frames=False, show_rms=show_rms, show_names=show_names, txtsize=12,
                                top_space=6, extend_bot=extend_bot, rotation=45)


            # %% Axis labels, tick frequency, etc
            axes[i].locator_params(axis='x', nbins=15)
            axes[i].locator_params(axis='y', nbins=7)
            # axes[i].tick_params(axis='both', which='both', direction='in', top=True, right=True, zorder=10)
            axes[i].tick_params(axis='both', which='both', direction='inout', top=False, right=False, zorder=10)
            axes[i].tick_params(axis='both', which='major', pad=10, width=1.3, length=8, color='k')
            axes[i].tick_params(axis='both', which='minor', width=1, length=6, color='k')

            # axes[i].grid(which='major', axis='y', color='white', linestyle='--', linewidth=0.8)
            axes[i].xaxis.set_major_locator(MultipleLocator(xtick_spacing_major))
            # axes[i].xaxis.set_major_formatter('{x:.1f}')
            axes[i].xaxis.set_minor_locator(MultipleLocator(xtick_spacing_minor))

            axes[i].yaxis.set_major_locator(MultipleLocator(ytick_spacing_major))
            # axes[i].xaxis.set_major_formatter('{x:.1f}')
            axes[i].yaxis.set_minor_locator(MultipleLocator(ytick_spacing_minor))

            axes[i].set_xlabel('Distance (m)')
            axes[i].set_ylabel('Height (m)')


            # %% colorbars
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if i == 0:  # plot resistivity
                plot_TEM_cbar(fig, axes[i], pacoll=pr, label=labelRho,
                              cmin=cmin, cmax=cmax, round_tick_label=1,
                              cbar_pos='right', log10_switch=True)
            else:  # plot CC params
                limits = CC_limits[i-1]
                plot_TEM_cbar(fig, axes[i], pacoll=CC_prs[i-1], label=CC_labels[i-1],
                              cmin=limits[0], cmax=limits[1], cbar_pos='right', round_tick_label=CC_rnd_ticks[i-1],
                              log10_switch=CC_log_switches[i-1], label_fmt=CC_label_fmts[i-1])


            # %% set limits for plots
            if depth != 'auto':
                if tilt_log is not None:
                    axes[i].set_ylim((np.max(telz_tilt) - depth, np.max(telz_tilt) + top_space))
                    axes[i].set_xlim((np.min(telx_tilt) - side_space,
                                  np.max(telx_tilt) + side_space))
                else:
                    axes[i].set_ylim((np.max(telz) - depth, np.max(telz) + top_space))
                    axes[i].set_xlim((np.min(telx) - side_space, np.max(telx) + side_space))
            else:
                if tilt_log is not None:
                    axes[i].set_ylim((np.min(telz_tilt), np.max(telz_tilt) + top_space))
                    axes[i].set_xlim((np.min(telx_tilt) - side_space,
                                  np.max(telx_tilt) + side_space))
                else:
                    axes[i].set_ylim((np.min(telz), np.max(telz) + top_space))
                    axes[i].set_xlim((np.min(telx) - side_space, np.max(telx) + side_space))


        # %% add Topo line and DOI
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            expand_topo = 15
            axes[i].plot(np.r_[topo[0,0] - expand_topo, topo[:, 0], topo[-1, 0] + expand_topo],
                         np.r_[topo[0,1], topo[:, 1], topo[-1, 1]] - 0.1,
                         '-k', lw=topo_lw)

        # plot_doi(axes[0], heights-DOIs,
        #           distances, color='black', zorder=10, label='DOI')


        # %% add label tags
        all_axs = fig.get_axes()
        # tags = ['(a)', '(b)', '(c)', '(d)', '(e)']
        tags = ['(a)', '(b)', '(c)']
        for idx, tag in enumerate(tags):
            at = AnchoredText(tag,
                              prop={'color': 'k', 'fontsize': 16}, frameon=True,
                              loc='upper right')
            at.patch.set_boxstyle("round, pad=0.0, rounding_size=0.2")
            all_axs[idx].add_artist(at)

        fig.tight_layout()

        if saveFig == True:
            logname = f'{vis_version}_{int_method[:3]}_{col_map_r}_{elPara}_{scale}.log'
            with open(f"{path_savefig}/{logname}", 'w') as fp:
                pass
            savefid = f"{path_savefig}/{vis_version}_{fname_result}{filetype}"
            print('saving figure to:',
                  savefid)
            fig.savefig(savefid,
                        dpi=dpi,
                        bbox_inches='tight')
        else:
            plt.show()

        # break
