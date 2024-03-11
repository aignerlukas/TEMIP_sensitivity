# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:03:31 2020

script to compare forward solutions of zondTEM and empymod

homogenous model and salt lake test cases
10, 100, 1000 Ohmm and Saltlake example

compare to simpeg and zondtem

4 by 4 grid


@author: lukas
"""
# %% modules
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from glob import glob

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# from matplotlib.offsetbox import AnchoredText
# from matplotlib.ticker import LogLocator, NullFormatter
# from scipy.constants import epsilon_0

# custom
# from TEM_frwrd.empymod_frwrd_ip import empymod_frwrd
from TEM_frwrd.empymod_frwrd import empymod_frwrd
from TEM_frwrd.simpeg_frwrd import simpeg_frwrd

from TEM_frwrd.BEL1D_TEM_tools import mtrxMdl2vec
from TEM_frwrd.BEL1D_TEM_tools import reshape_model
from TEM_frwrd.BEL1D_TEM_tools import plot_signal
from TEM_frwrd.BEL1D_TEM_tools import plot_rhoa
from TEM_frwrd.BEL1D_TEM_tools import reArr_mdl
from TEM_frwrd.BEL1D_TEM_tools import mdl2steps

from TEM_frwrd.TEMIP_tools import plot_ip_model
from TEM_frwrd.TEMIP_tools import PEM
from TEM_frwrd.TEMIP_tools import CCM
from TEM_frwrd.TEMIP_tools import CCM_K
from TEM_frwrd.TEMIP_tools import get_omega_peak_PE
from TEM_frwrd.TEMIP_tools import save_as_tem
from TEM_frwrd.TEMIP_tools import simulate_error
from TEM_frwrd.TEMIP_tools import calc_rhoa

from tem_tools.sounding import Sounding


# %% plot setup
plt.style.use('ggplot')

shift_fontsize = 1
plt.rcParams['axes.labelsize'] = 16 - shift_fontsize
plt.rcParams['axes.titlesize'] = 16 - shift_fontsize
plt.rcParams['xtick.labelsize'] = 14 - shift_fontsize
plt.rcParams['ytick.labelsize'] = 14 - shift_fontsize
plt.rcParams['legend.fontsize'] = 13 - shift_fontsize

lim_sig = (1e-10, 1e0)
lim_time = (1e-6, 1e-3)
lim_diff = (0, 80)

save_figs = True
# save_figs = False


# %% empymod params
cutoff_f = 4.5e5
cutoff_f = 4e6
cutoff_f = 1e8
# cutoff_f = None


# %% path setup
scriptname = os.path.basename(sys.argv[0])
print(f'running {scriptname} ...')
test_vers = scriptname.split('.')[0].split('_')[-1]
type_vers = scriptname.split('.')[0].split('_')[-2]

main = '..'
# versions = ['v10-hm', 'v10-hm', 'v10-hm', 'v00-sl']
# resis = ['10', '100', '1000', 'sl']  # homogeneous model resistivity

versions = ['v00-sl']
resis = ['sl']  # homogeneous model resistivity


fig, axes = plt.subplots(2, 2, figsize=(9, 6.6))

savepath_plots = f'../plots/{type_vers}/{test_vers}/'
if not os.path.exists(savepath_plots):
    os.makedirs(savepath_plots)

row_id = 0
for i, version in enumerate(versions):
    res = resis[i]

    path_zond_modeling = f'{main}/selected_data/modelled_zt/{version}/'
    path_tem_file = f'{main}/selected_data/raw_files/{version}/'

    fids_zt = glob(path_zond_modeling + 'SL-*.xls')
    fids_raw = glob(path_tem_file + 'SL-*.tem')

    for idx, fid_raw in enumerate(fids_raw):
        # %% read the result and the .tem file
        snd_result_zt = Sounding()
        snd_result_zt.parse_sndng(fid_raw)
        snd_result_zt.parse_zond_result(fids_zt[idx])
    
        snd_raw = Sounding()
        snd_raw.parse_sndng(fid_raw)
    
        loop = int(snd_raw.tx_loop)
        print(f'preparing plot for {int(snd_raw.tx_loop)}')
    
        # %% extract the model and the settings
        thk = np.asarray(snd_result_zt._model_df.h, dtype=float)
        thk[-1] = 0
        resistivity = np.asarray(snd_result_zt._model_df.Rho, dtype=float)
        model = np.column_stack((thk, resistivity))

        nlay = model.shape[0]
        nparas = model.shape[1]

        # %% calculate the forward with empymod
        settings = {"timekey": snd_raw.metainfo['timekey'],
                    "currentkey": snd_raw.metainfo['currentkey'],
                    "txloop": snd_raw.tx_loop,  #6.25, 12.5, 25
                    "rxloop": snd_raw.metainfo['rx_loop'],
                    "current_inj": float(snd_raw.current),
                    "filter_powerline": 50}
        with open(f'{savepath_plots}/sl-{loop:02d}_set-device_{test_vers}.yml', 'w') as file:
            dump(settings, file, Dumper)
    
        # 'ftarg': 'key_201_CosSin_2012', 'ftarg': 'key_601_CosSin_2009'
        setup_empymod = {'ft': 'dlf',                     # type of fourier trafo
                         'ftarg': 'key_601_CosSin_2009',  # ft-argument; filter type # https://empymod.emsig.xyz/en/stable/api/filters.html#module-empymod.filters -- for filter names
                         'verbose': 4,                    # level of verbosity (0-4) - larger, more info
                         'srcpts': 3,                     # Approx. the finite dip. with x points. Number of integration points for bipole source/receiver, default is 1:, srcpts/recpts < 3 : bipole, but calculated as dipole at centre
                         'recpts': 3,                     # Approx. the finite dip. with x points. srcpts/recpts >= 3 : bipole
                         'ht': 'dlf',                     # type of fourier trafo
                         'htarg': 'key_401_2009',         # hankel transform filter type, 'key_401_2009',
                         'nquad': 3,                     # Number of Gauss-Legendre points for the integration. Default is 3.
                         'cutoff_f': cutoff_f,               # cut-off freq of butterworthtype filter - None: No filter applied, WalkTEM 4.5e5
                         'delay_rst': 0,                 # ?? unknown para for walktem - keep at 0 for fasttem
                         'rxloop': 'vert. dipole'}       # or 'same as txloop' - not yet operational
        with open(f'{savepath_plots}/sl-{loop:02d}_set-empymod_{test_vers}.yml', 'w') as file:
            dump(setup_empymod, file, Dumper)
    
    
        # run modeling
        frwrd = empymod_frwrd(setup_device=settings,
                              setup_solver=setup_empymod,
                              filter_times=None, device='TEMfast',
                              nlayer=nlay, nparam=nparas)
        t_mdld = frwrd.times_rx
        mdld_sig = frwrd.calc_response(model)
        mdld_rhoa = frwrd.calc_rhoa()
    
    
        # %% setup simpeg forward
        setup_simpeg = {'coredepth': 50,
                        'csz': 2,
                        'relerr': 0.001,
                        'abserr': 1e-15}

        frwrd_sp = simpeg_frwrd(setup_device=settings,
                           setup_simpeg=setup_simpeg,
                           device='TEMfast',
                           nlayer=model.shape[0], nparam=model.shape[1])
        frwrd_sp.infer_layer2mesh(model, show_mesh=False)
        frwrd_sp.calc_response(model)
        frwrd_sp.calc_rhoa()
    
        # %% calculate differences (relative)
        # sig_calc_norm = snd_result_zt.sgnl_c * 1e-6 / snd_raw.rx_area * snd_raw.current
        diff_sig_zt = abs(snd_result_zt.sgnl_c - mdld_sig)
        # diff_rhoa = abs(frwrd_zt.rhoa - mdld_rhoa)
        diff_rel_sig_zt = (diff_sig_zt / snd_result_zt.sgnl_c) * 100
        # diff_rel_rhoa = (diff_rhoa / mdld_rhoa) * 100
        rrms_sig_zt = np.sqrt(np.mean(np.square(diff_rel_sig_zt)))
        # rrms_roa = np.sqrt(np.mean(np.square(diff_rel_rhoa)))

        
        diff_sig_sp = abs(frwrd_sp.response - mdld_sig)
        # diff_rhoa = abs(frwrd_sp.rhoa - mdld_rhoa)
        diff_rel_sig_sp = (diff_sig_sp / mdld_sig) * 100
        # diff_rel_rhoa = (diff_rhoa / mdld_rhoa) * 100
        rrms_sig_sp = np.sqrt(np.mean(np.square(diff_rel_sig_sp)))
        # rrms_roa = np.sqrt(np.mean(np.square(diff_rel_rhoa)))


        # %% plotting
        # # first plot the zondTEM modeling results
        snd_result_zt.plot_dBzdt('calculated', ax=axes[row_id, idx],
                                  label='zondTEM', color='dodgerblue', zorder=5)
        # plot_signal(time=t_mdld, signal=sig_calc_norm, axis=axes[0], label='zondTEM')
        # snd_result_zt.plot_rhoa('calculated', ax=axes[1, idx],
        #                         label='zondTEM', color='deeppink', zorder=5)
    
        # plot the simpeg data first
        plot_signal(time=t_mdld, signal=frwrd_sp.response, axis=axes[row_id+1, idx], label='SimPEG',
                    color='peru', marker='d', ls=':', zorder=5)
        axes[row_id+1, idx].set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
        # plot_rhoa(time=t_mdld, rhoa=frwrd_sp.rhoa, axis=axes[1, idx], label='SimPEG',
        #           color='deeppink', marker='d', ls=':', zorder=5)

        # empymod after
        plot_signal(time=t_mdld, signal=mdld_sig, axis=axes[row_id, idx], label='em$\Pi$mod',
                    color='darkblue', lw=1.8, zorder=10)
        plot_signal(time=t_mdld, signal=mdld_sig, axis=axes[row_id+1, idx], label='em$\Pi$mod',
                    color='darkblue', lw=1.8, zorder=10)
        # plot_rhoa(time=t_mdld, rhoa=mdld_rhoa, axis=axes[1, idx], label='em$\Pi$mod',
        #           color='maroon', lw=1.8, zorder=10)


        axes[row_id, idx].set_xlabel(r'time (s)')

        # axt = axes[row_id, idx].twinx()
        # axt.plot(t_mdld, diff_rel_sig_zt, '.:', color='gray', zorder=0, label='rel. diff')
        # axt.set_ylabel('diff zt-empy (%)')
        # axt.grid(False)
        # axt.set_ylim(lim_diff)
        
        if (idx == 1):
            # lines, labels = axes[row_id, idx].get_legend_handles_labels()
            # lines2, labels2 = axt.get_legend_handles_labels()
            # axes[row_id, idx].legend(lines + lines2, labels + labels2,
            #                               loc='upper right')
            # lines, labels = axes[row_id+1, idx].get_legend_handles_labels()
            # axes[row_id+1, idx].legend(lines + lines2, labels + labels2,
            #                                 loc='upper right')

            axes[row_id, idx].legend(loc='lower left')
            axes[row_id+1, idx].legend(loc='lower left')

        # add rms
        at = AnchoredText(f'rRMS = {rrms_sig_zt:.1f}',  # f'{settings["txloop"]} m loop\n' + f'rRMS = {rrms_sig_zt:.1f}'
                          prop={'color': 'k', 'fontsize': 14}, frameon=True,
                          loc='upper right')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axes[row_id, idx].add_artist(at)

        # axt = axes[row_id+1, idx].twinx()
        # axt.plot(t_mdld, diff_rel_sig_sp, '.:', color='gray', zorder=0)
        # axt.set_ylabel('diff sp-empy (%)')
        # axt.grid(False)
        # axt.set_ylim(lim_diff)

        # add rms
        at = AnchoredText(f'rRMS = {rrms_sig_sp:.1f}',  # f'{settings["txloop"]} m loop\n' + f'rRMS = {rrms_sig_zt:.1f}'
                          prop={'color': 'k', 'fontsize': 14}, frameon=True,
                          loc='upper right')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axes[row_id+1, idx].add_artist(at)
        
        if row_id == 0:
            axes[row_id, idx].set_title(f'{settings["txloop"]} m loop')

    # break
        
    row_id += 1

# %% add lettering and axes limits
characters = [chr(i) for i in range(ord('a'), ord('d') + 1)]
for j, ax in enumerate(axes.flatten()):
    print(j)
    
    ax.set_xlim(lim_time)
    ax.set_ylim(lim_sig)
    
    # add rms
    at = AnchoredText(f'({characters[j]})',
                      prop={'color': 'k', 'fontsize': 14}, frameon=True,
                      loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

plt.tight_layout()

# %%
if save_figs:
    # f2plot = f'{cutoff_f:.1e} Hz' if cutoff_f is not None else 'None'
    # suptitle = (f'Saltlakes model: thk: {thk}, res {resistivity}\n' +
    #             f'cut-off F: {f2plot}, ft: {setup_empymod["ftarg"]}, ht: {setup_empymod["htarg"]}')
    # fig.suptitle(suptitle, fontsize=16)
    # fig.suptitle(f'homogeneous model {res} $\Omega$m, cut-off F: {cutoff_f}Hz', fontsize=16)
    fig.savefig(savepath_plots + f'comp-frwrd_{type_vers}_{test_vers}.png', dpi=200)
