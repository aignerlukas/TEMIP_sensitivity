#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:54:03 2022

uses MPA model for creating the dispersion of el. resistivity
uses inversion class from joint inversion:
    https://github.com/florian-wagner/four-phase-inversion/blob/master/code/fpinv/lsqrinversion.py

TODO:
    [Done] add versioning
    [Done] test other startmodel
    [Done] abort criteria
    [Done] constraints!!
    [] add log

    [done] iterate over two different tau values and plot into two rows!
    [done] save stuff
    [done] reload for plotting

    [] increase error around negative peak!?!
    [] estimate tau from peak position?!
        [] function estimate_tau()
    [] add doi
    [done] save results for dgsa


generate data without noise, add noise, trafo afterwards
    [done] generate clean data
    [done] add noise
    [done] trafo afterwards

test new data transformations
    [done] kth root
    [done] arsinh

Graphite test case with negative and positive IP effect
(few readings with reversed sign, two different tau values!)

script to combine plots of inversion and dgsa

@author: laigner
"""

# %% import modules
import os
import sys
# import logging

rel_path_to_libs = '../../../'
if not rel_path_to_libs in sys.path:
    sys.path.append('../../../')  # add relative path to folder that contains all custom modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors as clr
# import seaborn as sns

# from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.offsetbox import AnchoredText
# from numpy.linalg import norm
# from math import sqrt

# import pygimli as pg
# from pygimli.utils import boxprint

# import custom modules from additional sys.path
# from library.TEM_frwrd.empymod_frwrd_ip import empymod_frwrd
# from library.TEM_frwrd.utils import arsinh
# from library.TEM_frwrd.utils import kth_root


# from library.utils.TEM_ip_tools import prep_mdl_para_names
# from library.utils.TEM_ip_tools import get_phimax_from_CCR
# from library.utils.TEM_ip_tools import get_tauphi_from_tr
# from library.utils.TEM_ip_tools import plot_diffs

from library.utils.universal_tools import plot_signal
# from library.utils.universal_tools import plot_rhoa
# from library.utils.universal_tools import calc_rhoa
# from library.utils.universal_tools import simulate_error
# from library.utils.universal_tools import query_yes_no

# from library.utils.TEM_inv_tools import vecMDL2mtrx
# from library.utils.TEM_inv_tools import mtrxMDL2vec
# from library.utils.TEM_inv_tools import save_result_and_fit

from library.utils.timer import Timer

# from pg_temip_invtools import lsqr
# from library.TEM_inv.pg_temip_inv import temip_block1D_fwd
# from library.TEM_inv.pg_temip_inv import LSQRInversion

from library.utils.TEM_ip_tools import plot_pem_stepmodel
from library.utils.TEM_ip_tools import plot_mpa_stepmodel

from library.utils.TEM_sean_tools import vert_pareto_plot


# %% plot appearance
dpi = 200
plt.style.use('ggplot')

fs_shift = -7
plt.rcParams['axes.labelsize'] = 17 + fs_shift
plt.rcParams['axes.titlesize'] = 17 + fs_shift
plt.rcParams['xtick.labelsize'] = 16 + fs_shift
plt.rcParams['ytick.labelsize'] = 16 + fs_shift
plt.rcParams['legend.fontsize'] = 16 + fs_shift


lims_time = (2e-6, 3e-3)
lims_signal = (1e-9, 1e-2)
lims_rhoa = (1e0, 1e4)
lims_depth = (40, 0)

# lims_rho0 = (0.5, 1e3)
lims_rho0 = (1, 1e3)
lims_mpa = (-0.1, 1.1)
lims_tau = (5e-7, 8e-2)
lims_c = (-0.1, 1.1)


# %% setup script
savefigs = True
# savefigs = False

show_simulated_data = True
# show_simulated_data = False

start_inv = True
# start_inv = False

max_depth = 55
nlayers = 3


scriptname = os.path.basename(sys.argv[0])
print(f'running {scriptname} ...')
version_vis = scriptname.split('.')[0].split('_')[-1]
version_inv = scriptname.split('_')[-3].split('-')[-1]
version_dgsa = scriptname.split('_')[-2].split('-')[-1]

plot_typ = scriptname.split('_')[0]
inv_typ = 'inv-mpa-cstr-lin'
case = scriptname.split('_')[-4]

main = './'
savepath = main + f'04-vis/{plot_typ}_{case}/inv-{version_inv}_dgsa-{version_dgsa}/{version_vis}/'
if not os.path.exists(savepath):
    os.makedirs(savepath)

savepath_data = main + f'03-inv_results/{inv_typ}/{version_inv}/'
# if not os.path.exists(savepath_data):
#     os.makedirs(savepath_data)

savepath_dgsa = ('../../04_DGSA-figures/TEMIP/results_dgsa/graph-12-03lay_norm-taucomp' +
                 f'/inv_{version_inv}/{version_dgsa}/')

t = Timer()


# %% setup solver
device = 'TEMfast'
setup_device = {"timekey": 5,
                "currentkey": 4,
                "txloop": 12.5,  #6.25, 12.5, 25
                "rxloop": 12.5,
                "current_inj": 4.0,
                "filter_powerline": 50,
                "ramp_data": 'donauinsel'}

# 'ftarg': 'key_81_CosSin_2009', 'key_201_CosSin_2012', 'ftarg': 'key_601_CosSin_2009'
setup_solver = {'ft': 'dlf',                     # type of fourier trafo
                  'ftarg': 'key_601_CosSin_2009',  # ft-argument; filter type # https://empymod.emsig.xyz/en/stable/api/filters.html#module-empymod.filters -- for filter names
                  'verbose': 0,                    # level of verbosity (0-4) - larger, more info
                  'srcpts': 3,                     # Approx. the finite dip. with x points. Number of integration points for bipole source/receiver, default is 1:, srcpts/recpts < 3 : bipole, but calculated as dipole at centre
                  'recpts': 3,                     # Approx. the finite dip. with x points. srcpts/recpts >= 3 : bipole
                  'ht': 'dlf',                     # type of fourier trafo
                  'htarg': 'key_401_2009',         # hankel transform filter type, 'key_401_2009', 'key_101_2009'
                  'nquad': 3,                     # Number of Gauss-Legendre points for the integration. Default is 3.
                  'cutoff_f': 1e8,               # cut-off freq of butterworthtype filter - None: No filter applied, WalkTEM 4.5e5
                  'delay_rst': 0,                 # ?? unknown para for walktem - keep at 0 for fasttem
                  'rxloop': 'vert. dipole'}       # or 'same as txloop' - not yet operational


# %% setup inversion parameters
max_iter = 25  # 25
lam = 1000
cooling_factor = 0.8
noise_floor = 0.04  # lowest relative error accepted in the error vector (%/100)
my = 1e2

relerr = 0.04
abserr = 1e-10
noise_factor = 1

tx_loop = 12.5

# resp_trafo = 'oddroot_7'
# resp_trafo = 'areasinhyp'
resp_trafo = None

return_rhoa = False


# %% setup figure! (two rows)
fig_result = plt.figure(figsize=(7, 12), constrained_layout=False)
gs = fig_result.add_gridspec(4, 8)

ax_fit = fig_result.add_subplot(gs[0, 0:5])
ax_rho = fig_result.add_subplot(gs[1, 0:2])
ax_mpa = fig_result.add_subplot(gs[1, 2:4])
ax_tau = fig_result.add_subplot(gs[1, 4:6])
ax_c = fig_result.add_subplot(gs[1, 6:8])

ax_sens = fig_result.add_subplot(gs[0, 5:8])

ax_fit1 = fig_result.add_subplot(gs[2, 0:5])
ax_rho1 = fig_result.add_subplot(gs[3, 0:2])
ax_mpa1 = fig_result.add_subplot(gs[3, 2:4])
ax_tau1 = fig_result.add_subplot(gs[3, 4:6])
ax_c1 = fig_result.add_subplot(gs[3, 6:8])

ax_sens1 = fig_result.add_subplot(gs[2, 5:8])

axes_fit = np.array([ax_fit, ax_fit1])
axes_mdl = np.array([[ax_rho, ax_mpa, ax_tau, ax_c],
                     [ax_rho1, ax_mpa1, ax_tau1, ax_c1]])
axes_sens = np.array([ax_sens, ax_sens1])


ip_effect_names = ['IP_p', 'IP_m']
ip_effect_types = ['$\oplus$IP, ', '$\ominus$IP, ']

tau_vals = [5e-2, 5e-4]

inv_run = 0
ip_modeltype = 'mpa'


# %% iterate over two different tau values
for j, tau_val in enumerate(tau_vals):

    name_snd = ip_effect_names[j]
    
    savename = ('invrun{:03d}_{:s}'.format(inv_run, name_snd))
    print(f'saving data from inversion run: {inv_run}')
    position = (0, 0, 0)  # location of sounding

    savepath_csv = savepath_data + f'{name_snd}/csv/'
    snd_prot_fid = savepath_csv.replace('csv/', '') + f'{case}_snd{name_snd}.log'

    # %% read result
    
    # read log files:
    read_log = np.genfromtxt(snd_prot_fid,
                             skip_header=1, delimiter='\t')

    if ip_modeltype == 'pelton':
        labels_CC = ['chargeability m ()', r'rel. time $\tau$ (s)']
    elif ip_modeltype == 'mpa':
        labels_CC = ['mpa (rad)', r'rel. time $\tau_{\phi}$ (s)']
    elif ip_modeltype == None:
        pass
        # TODO adjust here to read results correctly!!
    else:
        raise ValueError('this ip modeltype is not implemented here ...')
    
    savename = ('invrun{:03d}_{:s}'.format(inv_run, name_snd))
    fid_results = savepath_csv + savename + '.csv'
    fid_sm = savepath_csv + savename + '_startmodel.csv'
    fid_tm = savepath_csv + savename + '_truemodel.csv'
    fid_fit = savepath_csv + savename + '_fit.csv'
    fid_jac = savepath_csv + savename + '_jac.csv'

    invmdl_arr = np.genfromtxt(fid_results,
                               skip_header=1, delimiter=',')[:,3:]

    initmdl_arr = np.genfromtxt(fid_sm,
                                skip_header=1, delimiter=',')[:,3:]

    truemdl_arr = np.genfromtxt(fid_tm,
                                skip_header=1, delimiter=',')[:,3:]

    read_fit = np.genfromtxt(fid_fit,
                             skip_header=1, delimiter=',')

    jac_df = pd.read_csv(fid_jac, index_col=0)

    # create variables for plotting
    if len(read_log.shape) == 1:
        log_i = read_log
    else:
        log_i = read_log[inv_run]

    # sndname = log_i[0]  not working - non numeric!!!
    lam = log_i[4]
    my = log_i[5]
    cf = log_i[6]
    nf = log_i[7]
    max_iter = log_i[8]
    n_iter = log_i[9]
    absrms = log_i[10]
    relrms = log_i[11]
    chi2 = log_i[12]
    runtime = log_i[13]

    times_rx = read_fit[:,0]
    pred_data = read_fit[:,1]
    obs_dat_sub = read_fit[:,2]
    obs_error = read_fit[:,3]
    est_error = read_fit[:,4]

    pred_rhoa = read_fit[:,5]
    obs_rhoa = read_fit[:,6]

    # %% plot the data fit - without trafo into result plot
    _ = plot_signal(axes_fit[j], time=times_rx, signal=obs_dat_sub, label='data',
                    marker='o', color='k', sub0color='gray',
                    sub0marker='d', sub0label='negative data')
    # axes_fit[j].loglog(times_rx, abs(est_error), color='gray', ls='--', label='noise floor')

    _ = plot_signal(axes_fit[j], time=times_rx, signal=pred_data, label='response',
                    marker='.', ls='--', color='dodgerblue', sub0color='orange',
                    sub0marker='s', sub0label='negative response')
    
    lines, labels = axes_fit[j].get_legend_handles_labels()

    axes_fit[j].legend(loc='lower left')
    axes_fit[j].set_xlabel('Time (s)')
    axes_fit[j].set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
    axes_fit[j].set_xlim(lims_time)
    axes_fit[j].set_ylim(lims_signal)
    axes_fit[j].set_title(f'$\chi_2$ = {chi2:0.1f}, rRMS = {relrms:0.1f}%')


    # %% plot the model
    if ip_modeltype == 'pelton':
        plot_pem_stepmodel(axes=axes_mdl[j, :], model2d=truemdl_arr, label='true',
                           color='black', ls='-', depth_limit=lims_depth)
        plot_pem_stepmodel(axes=axes_mdl[j, :], model2d=initmdl_arr, label='initial',
                           color='gray', ls='--', marker='.', depth_limit=lims_depth)
        plot_pem_stepmodel(axes=axes_mdl[j, :], model2d=invmdl_arr, label='inverted',
                           color='dodgerblue', ls='--', marker='.', depth_limit=lims_depth)

    elif ip_modeltype == 'mpa':
        plot_mpa_stepmodel(axes=axes_mdl[j, :], model2d=truemdl_arr, label='true',
                           color='black', ls='-', depth_limit=lims_depth)
        plot_mpa_stepmodel(axes=axes_mdl[j, :], model2d=initmdl_arr, label='initial',
                           color='gray', ls='--', marker='.', depth_limit=lims_depth)
        plot_mpa_stepmodel(axes=axes_mdl[j, :], model2d=invmdl_arr, label='inverted',
                           color="dodgerblue", ls='--', marker='None', depth_limit=lims_depth)

    else:
        raise ValueError('this ip modeltype is not implemented here ...')
    
    axes_mdl[j, 0].set_xlim()
    axes_mdl[j, 1].set_xlim(lims_mpa)
    axes_mdl[j, 2].set_xlim(lims_tau)
    axes_mdl[j, 3].set_xlim(lims_c)

    gs.tight_layout(fig_result)


    # %% plot the DGSA result
    
    sens_logscale_x = True
    n_ticks = 4
    
    cluster = 3

    # params_per_col = np.r_[7, 8]  # thk, rho, all mpa params
    params_per_col = None
    # param_names_long = ['thickness (thk)', r'DC-resistivity ($\rho$)']
    n_cols = 1  # len(params_per_col)
    
    fid = f'sens_{name_snd}_{cluster}clustr.csv'

    axes_sens[j].set_title(f'{ip_effect_types[j]}' + '$\\tau_{\phi} =' + f' ${tau_val} s')
    sens = pd.read_csv(savepath_dgsa + fid, index_col=0)

    vert_pareto_plot(sens, ax=axes_sens[j], np_plot='all', fmt=None,
                    colors=None, confidence=True, sort_by_sens=False,
                    n_cols=n_cols, params_per_col=params_per_col,
                    add_empty_to_col=False)
    # axes_sens[j, 0].set_ylabel(f'{loop_sizes[jdx]:.1f} m loop')
    
    # for kdx, name in enumerate(param_names_long):
    #     axes_sens[j, kdx].set_title(f'{name}')

    axes_sens[j].grid(which='minor', axis='x', visible=True,
                    color='white', linestyle=':', linewidth=1.5)
    axes_sens[j].grid(which='major', axis='x', visible=True,
                    color='white', linestyle='--', linewidth=1.5)
    
    axes_sens[j].set_xlabel('Sensitivity ()')

    if sens_logscale_x == True:
        print('setting xaxis to log_10 scale...')
        axes_sens[j].set_xscale('log')
    
    


# add labels:
all_axs = fig_result.get_axes()
tags = ['(a)', '(c)', '(d)', '(e)', '(f)', '(b)', '(g)', '(i)', '(j)', '(k)', '(l)', '(h)']
for idx, tag in enumerate(tags):
    at = AnchoredText(tag,
                      prop={'color': 'k', 'fontsize': 12}, frameon=True,
                      loc='upper right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    all_axs[idx].add_artist(at)

if savefigs:
    savename = f'temIP_numdata-result_{ip_modeltype}.png'
    print('saving to: ', savepath + savename)
    fig_result.savefig(savepath + savename, dpi=dpi, bbox_inches='tight')









