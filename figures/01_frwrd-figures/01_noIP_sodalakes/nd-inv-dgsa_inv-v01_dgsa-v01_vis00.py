#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 18:12:19 2022

fielddata inversion, sodalalkes, 12 and 50 m loop
blocky inversion
10 layers up to same depth

v00:
    [done] same initial model for 12 and 50 m loop
    [done] log scaled thk values increasing to max depth
    
    [done] new starting model shape:
        [done] conductive anomalies increase in resistivity with depth
        [done] slightly more resistive bottom layer

    [] add bottom resistivity

@author: laigner
"""


# %% import modules
import os
import sys

rel_path_to_libs = '../../../'
if not rel_path_to_libs in sys.path:
    sys.path.append('../../../')  # add realtive path to folder that contains all custom modules

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from library.utils.TEM_inv_tools import calc_doi
from library.utils.TEM_inv_tools import mtrxMDL2vec
from library.utils.TEM_sean_tools import vert_pareto_plot
from library.tem_tools.survey import Survey

from library.utils.timer import Timer
t = Timer()


# %% save path structure
scriptname = os.path.basename(sys.argv[0])
print(f'running {scriptname} ...')
version_vis = scriptname.split('.')[0].split('_')[-1]
version_inv = scriptname.split('_')[-3].split('-')[-1]
version_dgsa = scriptname.split('_')[-2].split('-')[-1]

typ = scriptname.split('_')[0]


savepath = f'./04-vis/{typ}/inv-{version_inv}_dgsa-{version_dgsa}/{version_vis}/'
if not os.path.exists(savepath):
    os.makedirs(savepath)


# %% setup plotting
dpi = 200
plt.style.use('ggplot')

fs_shift = -4
plt.rcParams['axes.labelsize'] = 18 + fs_shift
plt.rcParams['axes.titlesize'] = 18 + fs_shift
plt.rcParams['xtick.labelsize'] = 16 + fs_shift
plt.rcParams['ytick.labelsize'] = 16 + fs_shift
plt.rcParams['legend.fontsize'] = 16 + fs_shift

lims_time = (1e-6, 1e-3)
lims_sign = (1e-10, 1e0)
lims_rhoa = (1e0, 1e2)
lims_rho = (1e0, 2e2)
lims_depth = (90, -2)

min_time = 3e-6
lw = 2.5


# %% setup script
save_fig = True
# save_fig = False

show_simulated_data = True
# show_simulated_data = False

start_inv = True
# start_inv = False

show_results = True
# show_results = False

save_data = True
# save_data = False

#  (inversion settings)
lambdas = [50]  # in blocky case: initial lambdas
cooling_factor = [0.9]
max_iter = 25
mys = [1e2]  # regularization for parameter constraint
noise_floors = [0.015]  # 1.0 %

thk = np.r_[4, 10, 15, 25, 0]
res = np.r_[30, 120, 35, 100, 55]

mdl_true = np.column_stack((thk, res))


# %% path tem data and inversion results
main = './'
name_file = 'test-loops_v00'
rawdata_path = main + '01-data/modelled/sodalakes/'
rawdata_fname = f'{name_file}.tem'

savepath_result = main + f'03-inv_results/nd-loopcomp-12-50/{version_inv}/'
savepath_dgsa = f'../../04_DGSA-figures/noIP/results_dgsa/num-12-50/{version_dgsa}/'


fig_result = plt.figure(figsize=(14, 8), constrained_layout=False)
gs = fig_result.add_gridspec(2, 6)

ax_fit = fig_result.add_subplot(gs[0, 0:2])
ax_rho = fig_result.add_subplot(gs[0, 2:4])
ax_fit1 = fig_result.add_subplot(gs[1, 0:2])
ax_rho1 = fig_result.add_subplot(gs[1, 2:4])

ax_thk = fig_result.add_subplot(gs[0, 4:5])
ax_res = fig_result.add_subplot(gs[0, 5:6])
ax_thk1 = fig_result.add_subplot(gs[1, 4:5])
ax_res1 = fig_result.add_subplot(gs[1, 5:6])

ax = np.array([[ax_fit, ax_rho],
               [ax_fit1, ax_rho1]])
axes_dgsa = np.array([[ax_thk, ax_res],
                      [ax_thk1, ax_res1]])


# %% load inv results for plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if show_results:
    invrun = '000'

    survey = Survey()
    survey.parse_temfast_data(filename=rawdata_fname, path=rawdata_path)
    survey.parse_inv_results(result_folder=savepath_result, invrun=invrun)
    survey.select_soundings_by(prop='tx_side1', vals=[12.5, 50.0])

    snd12 = survey.soundings[survey.sounding_names[0]]
    filter_12 = np.r_[np.asarray(snd12.time_f)[0]*0.99, np.asarray(snd12.time_f)[-1]*1.1]
    
    log_12 = survey.invlog_info[survey.invlog_info.name == snd12.name.strip()]
    rrms_12 = log_12.rRMS.values[0]
    chi2_12 = log_12.chi2.values[0]
    lam_12 = log_12.lam.values[0]
    lamfin_12 = log_12.lam_fin.values[0]
    cf_12 = log_12.cf.values[0]
    noifl_12 = log_12.noifl.values[0]
    niter_12 = log_12.n_iter.values[0]

    model_inv = mtrxMDL2vec(snd12.inv_model)
    nlayer = int(len(model_inv) / 2)

    res_inv, thk_inv = model_inv[nlayer-1:nlayer*2-1], model_inv[0:nlayer-1]
    mdl4doi = np.column_stack((np.r_[0, thk_inv], res_inv))
    doi12 = calc_doi(current=snd12.current,
                    tx_area=snd12.tx_loop**2,
                    eta=snd12.sgnl_c.iloc[-1],
                    mdl_rz=mdl4doi, x0=30,
                    verbose=True)
    
    
    snd12.plot_dBzdt(which='observed', ax=ax[0, 0], xlimits=(3e-6, 5e-3),
                          ylimits=(1e-10, 1e-1), label='data', color='k', show_sub0_label=False)
    snd12.plot_dBzdt(which='calculated', ax=ax[0, 0], color='dodgerblue',
                         marker='x', ls=':', label='response', show_sub0_label=False)
    ax[0, 0].set_title(f'numerical data: 12.5 m loop\n$\chi^2$ = {chi2_12:0.1f}, rRMS = {rrms_12:0.1f}%')
    ax[0, 0].legend()


    snd12.plot_model(mdl_true, ax=ax[0, 1], color="black", ls='--', label="true model")
    snd12.plot_initial_model(ax=ax[0, 1], color='green', 
                             marker='.', ls='--', label='init. model')
    snd12.plot_inv_model(ax=ax[0, 1], color='dodgerblue',
                               marker='.', label='inv. model'
                               )
    ax[0, 1].axhline(y=doi12[0], ls='--', color='magenta', label='DOI')

    ax[0, 1].set_title((f'init. lam {lam_12:.1f} cooling: {cf_12}, iters {niter_12}\n' +
                        f'fin. lam {lamfin_12:.1f}, noise floor = {noifl_12}%'))
    
    ax[0, 1].invert_yaxis()
    ax[0, 1].set_xlim(lims_rho)
    ax[0, 1].set_ylim(lims_depth)
    ax[0, 1].set_xlabel(r'$\rho$ ($\Omega$m)')
    ax[0, 1].set_ylabel('z (m)')
    ax[0, 1].legend(loc='center right')
    
    tag = f'DOI = {doi12[0]:.1f} m'
    at = AnchoredText(tag, prop={'color': 'k', 'fontsize': 14},
                        frameon=True, loc='lower left')
    at.patch.set_boxstyle("round, pad=0.0, rounding_size=0.2")
    ax[0, 1].add_artist(at)

    snd50 = survey.soundings[survey.sounding_names[1]]
    filter_50 = np.r_[np.asarray(snd50.time_f)[0]*0.99, np.asarray(snd50.time_f)[-1]*1.1]
    
    model_inv = mtrxMDL2vec(snd50.inv_model)
    res_inv, thk_inv = model_inv[nlayer-1:nlayer*2-1], model_inv[0:nlayer-1]
    mdl4doi = np.column_stack((np.r_[0, thk_inv], res_inv))
    doi50 = calc_doi(current=snd50.current,
                    tx_area=snd50.tx_loop**2,
                    eta=snd50.sgnl_c.iloc[-1],
                    mdl_rz=mdl4doi, x0=30,
                    verbose=True)
    
    
    log_50 = survey.invlog_info[survey.invlog_info.name == snd50.name.strip()]
    rrms_50 = log_50.rRMS.values[0]
    chi2_50 = log_50.chi2.values[0]
    lam_50 = log_50.lam.values[0]
    lamfin_50 = log_50.lam_fin.values[0]
    cf_50 = log_50.cf.values[0]
    noifl_50 = log_50.noifl.values[0]
    niter_50 = log_50.n_iter.values[0]
    
    snd50.plot_dBzdt(which='observed', ax=ax[1, 0], xlimits=(3e-6, 5e-3),
                          ylimits=(1e-10, 1e-1), label='field data 50.0 m loop',
                          color='k', show_sub0_label=False
                          )
    snd50.plot_dBzdt(which='calculated', ax=ax[1, 0],
                           color='dodgerblue', marker='x', ls=':',
                           show_sub0_label=False
                           )
    ax[1, 0].set_title(f'numerical data: 50.0 m loop\n$\chi^2$ = {chi2_50:0.1f}, rRMS = {rrms_50:0.1f}%')
    
    snd50.plot_model(mdl_true, ax=ax[1, 1], color="black", ls='--')
    snd50.plot_initial_model(ax=ax[1, 1], color='green', marker='.', ls='--')
    snd50.plot_inv_model(ax=ax[1, 1], color='dodgerblue', marker='.')
    ax[1, 1].axhline(y=doi50[0], ls='--', color='magenta', label='DOI')
    
    tag = f'DOI = {doi50[0]:.1f} m'
    at = AnchoredText(tag, prop={'color': 'k', 'fontsize': 14},
                        frameon=True, loc='lower left')
    at.patch.set_boxstyle("round, pad=0.0, rounding_size=0.2")
    ax[1, 1].add_artist(at)
    
    
    ax[1, 1].set_title(f'lambda: {lam_50}')
    ax[1, 1].set_title((f'init. lam {lam_50:.1f} cooling: {cf_50}, iters {niter_50}\n' +
                        f'fin. lam {lamfin_50:.1f}, noise floor = {noifl_50}%'))
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_xlim(lims_rho)
    ax[1, 1].set_ylim(lims_depth)
    ax[1, 1].set_xlabel(r'$\rho$ ($\Omega$m)')
    ax[1, 1].set_ylabel('z (m)')


# %% DGSA plot
sens_logscale_x = True
n_ticks = 4

cluster = 3

loop_sizes = [12.5, 50.0]
params_per_col=np.r_[7, 8]  # thk, rho, all mpa params
param_names_long = ['thickness (thk)', r'DC-resistivity ($\rho$)']
n_cols = len(params_per_col)

n_cols = len(params_per_col)
skip_rows_from_col = np.zeros((2, max(params_per_col))).astype(bool)
skip_rows_from_col[0, -1] = True  # no thickness in bottom layer

fids = [f'sens_12mloop_{cluster}clustr.csv', f'sens_50mloop_{cluster}clustr.csv']

axes_dgsa[0, 0].get_shared_x_axes().join(axes_dgsa[0, 0], axes_dgsa[0, 1], 
                                         axes_dgsa[1, 0], axes_dgsa[1, 1])

for jdx, fid in enumerate(fids):
    sens = pd.read_csv(savepath_dgsa + fid, index_col=0)

    vert_pareto_plot(sens, ax=axes_dgsa[jdx], np_plot='all', fmt=None,
                    colors=None, confidence=True, sort_by_sens=False,
                    n_cols=n_cols, params_per_col=params_per_col,
                    add_empty_to_col=skip_rows_from_col)
    axes_dgsa[jdx, 0].set_ylabel(f'{loop_sizes[jdx]:.1f} m loop')
    
    for kdx, name in enumerate(param_names_long):
        axes_dgsa[jdx, kdx].set_title(f'{name}')

        axes_dgsa[jdx, kdx].grid(which='minor', axis='x', visible=True,
                        color='white', linestyle=':', linewidth=1.5)
        axes_dgsa[jdx, kdx].grid(which='major', axis='x', visible=True,
                        color='white', linestyle='--', linewidth=1.5)
        
        axes_dgsa[jdx, kdx].set_xlabel('Sensitivity ()')

        if sens_logscale_x == True:
            print('setting xaxis to log_10 scale...')
            axes_dgsa[jdx, kdx].set_xscale('log')



# %% add labels and save fig
all_axs = np.r_[ax.flatten()[0:2], axes_dgsa.flatten()[0:2],
                ax.flatten()[2:], axes_dgsa.flatten()[2:]]
tags = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

for i, tag in enumerate(tags):
    at = AnchoredText(tag,
                      prop={'color': 'k', 'fontsize': 14}, frameon=True,
                      loc='lower right')
    at.patch.set_boxstyle("round, pad=0.0, rounding_size=0.2")
    all_axs[i].add_artist(at)

fig_result.tight_layout()

if save_fig:
    fig_result.savefig(savepath + f'nd-inv-dgsa_{version_vis}.png', dpi=300)