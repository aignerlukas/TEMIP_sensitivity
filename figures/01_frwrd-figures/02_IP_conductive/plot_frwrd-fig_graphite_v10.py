#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:41:45 2022

script to plot a figure which shows the foward solution for a 
TEM-FAST system with a 12.5 m loop for thre cases:
    [] without IP effect
    [] with (+) IP effect
    [] with (-) IP effect

v00: three subplots in a row

@author: laigner
"""

# %% modules
import os
import sys

rel_path_to_libs = '../../../'
if not rel_path_to_libs in sys.path:
    sys.path.append('../../../')  # add realtive path to folder that contains all custom mudules

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.offsetbox import AnchoredText

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from library.TEM_frwrd.empymod_frwrd_ip import empymod_frwrd
from library.utils.universal_tools import plot_signal

from library.utils.TEM_ip_tools import plot_ip_model
from library.utils.TEM_ip_tools import PEM_res

from library.utils.TEM_ip_tools import CC_MPA
from library.utils.TEM_ip_tools import get_m_taur_MPA
from library.utils.TEM_ip_tools import get_phimax_from_CCR
from library.utils.TEM_ip_tools import get_tauphi_from_tr


# %% plot style
plt.style.use('ggplot')

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


# %% main settings
rho_min, rho_max = 5, 2000
rho_ticks = (1e1, 1e2, 1e3)
# z_max, z_min = -40, 0
z_max, z_min = -2, 30
z_ticks = np.arange(0, 35, 5)

sig_limits = (1e-11, 1e-2)
sig_limits = (1e-13, 1e-1)
time_limits = (1e-6, 2e-3)

savefigs = True
# savefigs = False


# %% directions
scriptname = os.path.basename(sys.argv[0])
print(f'running {scriptname} ...')
version = scriptname.split('.')[0].split('_')[-1]
model_type = scriptname.split('.')[0].split('_')[-2]

modelname = f'frwrdIP_{model_type}_{version}.png'

savepath = f'./{version}/'
if not os.path.exists(savepath):
    os.makedirs(savepath)


# %% setup model
# ip_modeltype = 'pelton'  # mpa
ip_modeltype = 'mpa'  # mpa
layer_ip = 1  #.which layer has the IP effect? - automatise!!

thks = np.r_[8, 12, 0]
rho_0 = np.r_[50, 10, 500]

charg = np.r_[0, 0.9, 0]
taus = np.r_[1e-6, 5e-2, 1e-6]
cs = np.r_[0.01, 0.9, 0.01]

phi_max = abs(get_phimax_from_CCR(rho_0=rho_0, m=charg, tau_rho=taus, c=cs))
tau_phi = get_tauphi_from_tr(m=charg, tau_rho=taus, c=cs)

phi_max = np.r_[0.0, 0.8, 0.0]
tau_phi = np.r_[1e-6, 5e-2, 1e-6]
tau_phi_n = np.r_[1e-6, 5e-4, 1e-6]


con_0 = 1 / rho_0
rho_8 = rho_0 - (charg * rho_0)
con_8 = 1 / rho_8

# pel_model = np.column_stack((thks, rho_0, charg, taus, cs))
mdl_noIP = np.column_stack((thks, rho_0, np.zeros_like(phi_max), tau_phi, cs))
mdl_posIP  = np.column_stack((thks, rho_0, phi_max, tau_phi, cs))
mdl_negIP  = np.column_stack((thks, rho_0, phi_max, tau_phi_n, cs))

nlay = mdl_noIP.shape[0]
nparas = mdl_noIP.shape[1]


# %% setup TEM
settings = {"timekey": 5,
            "currentkey": 4,
            "txloop": 12.5,  #6.25, 12.5, 25
            "rxloop": 12.5,
            "current_inj": 4.0,
            "filter_powerline": 50,
            "ramp_data": 'donauinsel'}
with open(f'{savepath}/tf_setup.yml', 'w') as file:
    dump(settings, file, Dumper)

# 'ftarg': 'key_201_CosSin_2012', 'ftarg': 'key_601_CosSin_2009'
setup_empymod = {'ft': 'dlf',                     # type of fourier trafo
                 'ftarg': 'key_601_CosSin_2009',  # ft-argument; filter type # https://empymod.emsig.xyz/en/stable/api/filters.html#module-empymod.filters -- for filter names      
                 'verbose': 4,                    # level of verbosity (0-4) - larger, more info
                 'srcpts': 3,                     # Approx. the finite dip. with x points. Number of integration points for bipole source/receiver, default is 1:, srcpts/recpts < 3 : bipole, but calculated as dipole at centre
                 'recpts': 3,                     # Approx. the finite dip. with x points. srcpts/recpts >= 3 : bipole
                 'ht': 'dlf',                     # type of fourier trafo
                 'htarg': 'key_401_2009',         # hankel transform filter type, 'key_401_2009', anderson_801_1982
                 'nquad': 3,                      # Number of Gauss-Legendre points for the integration. Default is 3.
                 'cutoff_f': 1e8,    # cut-off freq of butterworthtype filter - None: No filter applied, WalkTEM 4.5e5
                 'delay_rst': 0,                  # ?? unknown para for walktem - keep at 0 for fasttem
                 'rxloop': 'vert. dipole'}        # or 'same as txloop' - not yet operational
with open(f'{savepath}/empymod_setup.yml', 'w') as file:
    dump(setup_empymod, file, Dumper)

frwrd = empymod_frwrd(setup_device=settings,
                      setup_solver=setup_empymod,
                      time_range=None, device='TEMfast',
                      nlayer=nlay, nparam=nparas)



# %% model the responses
simd_noIP = frwrd.calc_response(model=mdl_noIP,
                                ip_modeltype=ip_modeltype,  # 'cole_cole', 'cc_kozhe'
                                show_wf=False)

simd_posIP = frwrd.calc_response(model=mdl_posIP,
                                 ip_modeltype=ip_modeltype,  # 'cole_cole', 'cc_kozhe'
                                 show_wf=False)

simd_negIP = frwrd.calc_response(model=mdl_negIP,
                                 ip_modeltype=ip_modeltype,  # 'cole_cole', 'cc_kozhe'
                                 show_wf=False)
timegates = frwrd.times_rx

# %% plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5),
                               sharex=True, sharey=False,
                               constrained_layout=True)


plot_signal(axis=ax1,                                               # NO IP
            time=timegates,
            signal=simd_noIP,
            color='k',
            marker='.',
            sub0color='k',
            sub0mfc='white',
            label='model response')  #no IP effect
ax1.set_ylim(sig_limits)
ax1.set_xlim(time_limits)
ax1.set_xlabel('time (s)')
ax1.set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
# ax1.legend()

# mdl1 = ax1.inset_axes([0.55, 0.55, 0.44, 0.44])
mdl1 = ax1.inset_axes([0.03, 0.03, 0.48, 0.48])
mdl1, coleparams = plot_ip_model(axis=mdl1, ip_model=mdl_noIP,
                                 ip_modeltype=None,
                                 layer_ip=layer_ip, rho2log=True)
mdl1.set_xlim((rho_min, rho_max))
mdl1.set_ylim((z_max, z_min))
mdl1.set_xticks(rho_ticks)
mdl1.set_yticks(z_ticks)
mdl1.invert_yaxis()
# mdl1.tick_params(axis='x', which='minor', bottom=True)

mdl1.xaxis.tick_top()
mdl1.xaxis.set_label_position('top')
mdl1.yaxis.tick_right()
mdl1.yaxis.set_label_position('right')

mdl1.xaxis.set_ticks_position('both')
mdl1.yaxis.set_ticks_position('both')

mdl1.grid(axis='both', which='major', alpha=0.9, ls='-')
# mdl1.grid(which='minor', alpha=0.75, ls=':')

ax1.set_title('without IP-effects')

at = AnchoredText('(a)',
                  prop={'color': 'k', 'fontsize': 16}, frameon=True,
                  loc='upper left')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)


plot_signal(axis=ax2,                                               # positive IP
            time=timegates,
            signal=simd_posIP,
            color='k',
            marker='.',
            sub0color='k',
            sub0mfc='white',
            label='model response')  # positive IP effect
ax2.set_ylim(sig_limits)
ax2.set_xlabel('time (s)')
# ax2.legend()

# mdl2 = ax2.inset_axes([0.55, 0.55, 0.44, 0.44])
mdl2 = ax2.inset_axes([0.03, 0.03, 0.48, 0.48])
mdl2, coleparams = plot_ip_model(axis=mdl2, ip_model=mdl_posIP,
                                 ip_modeltype=ip_modeltype,
                                 layer_ip=layer_ip, rho2log=True)
mdl2.set_xlim((rho_min, rho_max))
mdl2.set_ylim((z_max, z_min))
mdl2.set_xticks(rho_ticks)
mdl2.set_yticks(z_ticks)
mdl2.invert_yaxis()

mdl2.xaxis.tick_top()
mdl2.xaxis.set_label_position('top')
mdl2.yaxis.tick_right()
mdl2.yaxis.set_label_position('right')

mdl2.xaxis.set_ticks_position('both')
mdl2.yaxis.set_ticks_position('both')

mdl2.grid(axis='both', which='major', alpha=0.9, ls='-')
# mdl2.grid(which='minor', alpha=0.75, ls=':')

ax2.set_title('$\oplus$IP effect')

at = AnchoredText('(b)',
                  prop={'color': 'k', 'fontsize': 16}, frameon=True,
                  loc='upper left')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at)



plot_signal(axis=ax3,                                               # negative IP
            time=timegates,
            signal=simd_negIP,
            color='k',
            marker='.',
            sub0color='k',
            sub0mfc='white',
            label='model response',
            sub0label='negative readings')  # negative IP effect
ax3.set_ylim(sig_limits)
ax3.set_xlabel('time (s)')
ax3.legend()

# mdl3 = ax3.inset_axes([0.55, 0.55, 0.44, 0.44])
mdl3 = ax3.inset_axes([0.03, 0.03, 0.48, 0.48])
mdl3, coleparams = plot_ip_model(axis=mdl3, ip_model=mdl_negIP,
                                 ip_modeltype=ip_modeltype,
                                 layer_ip=layer_ip, rho2log=True)
mdl3.set_xlim((rho_min, rho_max))
mdl3.set_ylim((z_max, z_min))
mdl3.set_xticks(rho_ticks)
mdl3.set_yticks(z_ticks)
mdl3.invert_yaxis()

mdl3.xaxis.tick_top()
mdl3.xaxis.set_label_position('top')
mdl3.yaxis.tick_right()
mdl3.yaxis.set_label_position('right')

mdl3.xaxis.set_ticks_position('both')
mdl3.yaxis.set_ticks_position('both')

mdl3.grid(axis='both', which='major', alpha=0.9, ls='-')
# mdl3.grid(which='minor', alpha=0.75, ls=':')

ax3.set_title('$\ominus$IP effect')

at = AnchoredText('(c)',
                  prop={'color': 'k', 'fontsize': 16}, frameon=True,
                  loc='upper left')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at)


if savefigs:
    fig.savefig(savepath + modelname, dpi=300)